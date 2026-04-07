import argparse
import json
import math
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from eval_report_utils import export_evaluation_reports
from surgclip.surgclip.config import get_config
from surgclip.surgclip.model import SurgCLIP
from zeroshot_evaluate import (
    DATASET_CONFIGS,
    build_dataloader,
    evaluation_wrapper,
    to_builtin,
)


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
    model = SurgCLIPAdapter(surgclip_core).to(device)
    model = load_model_checkpoint(model, args.ckpt, device)
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

    report_paths = export_evaluation_reports(
        results=to_builtin(results),
        dataset=args.dataset,
        output_dir=args.output_dir,
        metadata={
            "dataset": args.dataset,
            "model_family": "surgclip",
            "ckpt": args.ckpt,
            "tokenizer_name": args.tokenizer_name,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_frames": args.num_frames,
            "model_num_frames": args.model_num_frames,
            "frame_stride": args.frame_stride,
            "image_size": args.image_size,
            "output_dir": args.output_dir,
            "result_json": result_path,
        },
        sota_file=args.sota_file,
    )

    print(json.dumps(to_builtin(results), ensure_ascii=False, indent=2))
    print(f"\n结果已保存到: {result_path}")
    print(f"指标表已保存到: {report_paths['summary_metrics_csv']}")
    if "sota_comparison_csv" in report_paths:
        print(f"SOTA 对比表已保存到: {report_paths['sota_comparison_csv']}")


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
    parser.add_argument("--sota_file", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_zero_shot(args)
