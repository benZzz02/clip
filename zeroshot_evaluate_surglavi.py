import argparse
import json
import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from surgclip.surgclip.config import get_config
from surgclip.surgclip.model import SurgCLIP
from zeroshot_evaluate import (
    DATASET_CONFIGS,
    build_dataloader,
    evaluation_wrapper,
    to_builtin,
)


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.weight = nn.Parameter(base_layer.weight.detach().clone(), requires_grad=False)
        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_hidden = F.linear(self.dropout(x), self.lora_A, bias=None)
        lora_out = F.linear(lora_hidden, self.lora_B, bias=None)
        return base_out + self.scale * lora_out


def parse_comma_separated_list(spec: str):
    return [item.strip() for item in spec.split(",") if item.strip()]


def freeze_module_parameters(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def apply_lora_to_linear_layers(module: nn.Module, target_substrings, rank: int, alpha: float, dropout: float, prefix: str = ""):
    replaced = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and any(pattern in full_name for pattern in target_substrings):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(full_name)
            continue
        replaced.extend(
            apply_lora_to_linear_layers(
                child,
                target_substrings=target_substrings,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                prefix=full_name,
            )
        )
    return replaced


class SurgCLIPAdapter(torch.nn.Module):
    """
    Adapt SurgCLIP to the evaluation interface expected by zeroshot_evaluate.py:
      - encode_image(images) -> [B, D]
      - encode_text(input_ids, attention_mask) -> [B, D]
      - logit_scale (nn.Parameter)
    """

    def __init__(self, surgclip_model: SurgCLIP):
        super().__init__()
        self.surgclip = surgclip_model

        with torch.no_grad():
            init_temp = float(self.surgclip.temp.detach().cpu().item())
            init_logit_scale = math.log(1.0 / max(init_temp, 1e-6))
        self.logit_scale = torch.nn.Parameter(
            torch.tensor(init_logit_scale, dtype=torch.float32)
        )

    def _sync_temp_from_logit_scale(self):
        temp_value = (1.0 / self.logit_scale.exp()).clamp(min=1e-6, max=100.0)
        self.surgclip.temp.data.copy_(temp_value)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        self._sync_temp_from_logit_scale()

        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(
                f"Expected [B, C, H, W] or [B, T, C, H, W], got {tuple(images.shape)}"
            )

        _, pooled = self.surgclip.encode_vision(images)

        if pooled.ndim == 3:
            pooled = pooled.mean(dim=1)
        elif pooled.ndim != 2:
            raise ValueError(
                f"Expected pooled vision features to be 2D or 3D, got {tuple(pooled.shape)}"
            )

        feats = self.surgclip.vision_proj(pooled)
        feats = F.normalize(feats, dim=-1)
        return feats

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._sync_temp_from_logit_scale()

        text = SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)
        _, pooled = self.surgclip.encode_text(text)

        feats = self.surgclip.text_proj(pooled)
        feats = F.normalize(feats, dim=-1)
        return feats


def configure_finetuning(model: "SurgCLIPAdapter", args):
    finetune_mode = args.finetune_mode.lower()
    if finetune_mode not in {"full", "lora"}:
        raise ValueError(f"Unsupported finetune_mode: {finetune_mode}")

    if finetune_mode == "full":
        return {"mode": "full", "replaced_modules": []}

    freeze_module_parameters(model)
    replaced = apply_lora_to_linear_layers(
        model,
        target_substrings=parse_comma_separated_list(args.lora_targets),
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    if not replaced:
        raise RuntimeError(f"No linear layers matched LoRA targets: {args.lora_targets}")

    model.surgclip.vision_proj.weight.requires_grad = True
    if model.surgclip.vision_proj.bias is not None:
        model.surgclip.vision_proj.bias.requires_grad = True
    model.surgclip.text_proj.weight.requires_grad = True
    if model.surgclip.text_proj.bias is not None:
        model.surgclip.text_proj.bias.requires_grad = True
    model.logit_scale.requires_grad = True

    return {"mode": "lora", "replaced_modules": replaced}


def load_model_checkpoint(model, ckpt_path, device):
    print(f"正在加载模型权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print("检测到 checkpoint 格式: model_state_dict")
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print("检测到 checkpoint 格式: state_dict")
    else:
        state_dict = ckpt
        print("检测到纯 state_dict 格式")

    normalized_state_dict = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]
        normalized_state_dict[key] = value

    msg = model.load_state_dict(normalized_state_dict, strict=False)
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint 与当前模型不匹配。\n"
            f"Missing keys: {msg.missing_keys}\n"
            f"Unexpected keys: {msg.unexpected_keys}"
        )

    return model


def build_model(args, tokenizer, device):
    model_num_frames = args.model_num_frames or args.num_frames

    config = get_config(
        args.surgclip_model_name,
        overrides={
            "device": str(device),
            "num_frames": int(model_num_frames),
            "inputs": {
                "image_res": int(args.image_size),
                "video_input": {
                    "num_frames_test": int(model_num_frames),
                },
            },
            "model": {
                "temporal_modeling": {
                    "enabled": int(model_num_frames) > 1,
                },
                "text_encoder": {
                    "pretrained": args.tokenizer_name,
                },
            },
        },
    )

    surgclip_core = SurgCLIP(config=config, tokenizer=tokenizer, is_pretrain=False)
    model = SurgCLIPAdapter(surgclip_core)
    finetune_info = configure_finetuning(model, args)
    model = model.to(device)
    model = load_model_checkpoint(model, args.ckpt, device)
    print(f"评测模型 finetune_mode: {args.finetune_mode}")
    if finetune_info["replaced_modules"]:
        print(f"评测模型 LoRA modules: {len(finetune_info['replaced_modules'])}")
    model.eval()
    return model


def evaluate_zero_shot(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print("正在加载 SurgCLIP 结构...")
    model = build_model(args, tokenizer, device)

    data_loader, _ = build_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
    )
    data_loader.dataset.name = args.dataset

    results = evaluation_wrapper(
        model=model,
        data_loader=data_loader,
        tokenizer=tokenizer,
        device=device,
        output_dir=args.output_dir,
        prefix=args.dataset,
        expected_num_frames=args.num_frames,
    )

    result_path = os.path.join(args.output_dir, f"results_{args.dataset}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(results), f, ensure_ascii=False, indent=2)

    print(json.dumps(to_builtin(results), ensure_ascii=False, indent=2))
    print(f"\n结果已保存到: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation for SurgCLIP")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--surgclip_model_name", type=str, default="SurgCLIP-B")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./eval_outputs_surglavi")
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--model_num_frames", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--finetune_mode", type=str, default=os.environ.get("FINETUNE_MODE", "lora"), choices=("full", "lora"))
    parser.add_argument("--lora_rank", type=int, default=int(os.environ.get("LORA_RANK", 8)))
    parser.add_argument("--lora_alpha", type=float, default=float(os.environ.get("LORA_ALPHA", 16)))
    parser.add_argument("--lora_dropout", type=float, default=float(os.environ.get("LORA_DROPOUT", 0.05)))
    parser.add_argument(
        "--lora_targets",
        type=str,
        default=os.environ.get(
            "LORA_TARGETS",
            "text_encoder.encoder.layer.,vision_encoder.model.blocks."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_zero_shot(args)
