import os
import sys
import json
import math
import random
import argparse
import contextlib
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from pretrain_dataset import PretrainDataset

# 你复制 surgclip/ 到仓库根目录后，这个 import 才会生效
from surgclip.surgclip.config import get_config
from surgclip.surgclip.model import SurgCLIP


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_worker_init_fn(base_seed: int, rank: int):
    def _worker_init_fn(worker_id: int):
        worker_seed = base_seed + rank * 1000 + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init_fn


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


def configure_finetuning(
    model: "SurgCLIPAdapter",
    finetune_mode: str,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_targets,
):
    finetune_mode = finetune_mode.lower()
    if finetune_mode not in {"full", "lora"}:
        raise ValueError(f"Unsupported finetune_mode: {finetune_mode}")

    if finetune_mode == "full":
        return {"mode": "full", "replaced_modules": []}

    freeze_module_parameters(model)
    replaced = apply_lora_to_linear_layers(
        model,
        target_substrings=lora_targets,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    if not replaced:
        raise RuntimeError(f"No linear layers matched LoRA targets: {lora_targets}")

    model.surgclip.vision_proj.weight.requires_grad = True
    if model.surgclip.vision_proj.bias is not None:
        model.surgclip.vision_proj.bias.requires_grad = True
    model.surgclip.text_proj.weight.requires_grad = True
    if model.surgclip.text_proj.bias is not None:
        model.surgclip.text_proj.bias.requires_grad = True
    model.logit_scale.requires_grad = True

    return {"mode": "lora", "replaced_modules": replaced}


def count_parameters(module: nn.Module):
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total, trainable


def collate_fn_skip_corrupted(batch):
    # 兼容你原来的坏样本过滤逻辑（images 全 0 时跳过）
    batch = [item for item in batch if not torch.all(item[0].eq(0))]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor):
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


class SurgCLIPAdapter(torch.nn.Module):
    """
    把 SurgLaVi 的 SurgCLIP 适配成你训练代码期望的接口：
      - encode_image(images) -> [B, D]
      - encode_text(input_ids, attention_mask) -> [B, D]
      - logit_scale (nn.Parameter), 并与 SurgCLIP.temp 保持严格等价关系：logit_scale = 1/temp
    """

    def __init__(self, surgclip_model: SurgCLIP):
        super().__init__()
        self.surgclip = surgclip_model

        # SurgCLIP 内部是 temp 参数（用于除法）
        # 为了兼容你现有 loss 写法，我们引入 logit_scale，并强制 temp = 1/logit_scale
        # 初始化时对齐：temp = surgclip.temp（一个标量 Parameter）
        with torch.no_grad():
            init_temp = float(self.surgclip.temp.detach().cpu().item())
            init_logit_scale = math.log(1.0 / max(init_temp, 1e-6))
        self.logit_scale = torch.nn.Parameter(torch.tensor(init_logit_scale, dtype=torch.float32))

    def _sync_temp_from_logit_scale(self):
        # temp = 1 / exp(logit_scale)
        # 注意：SurgCLIP.temp 是 nn.Parameter，直接赋值会断开参数；我们用 data.copy_ 写回它的值
        temp_value = (1.0 / self.logit_scale.exp()).clamp(min=1e-6, max=100.0)
        self.surgclip.temp.data.copy_(temp_value)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images:
          [B,3,H,W] or [B,T,3,H,W]
        SurgCLIP.forward/encode_vision 期望的是 video，形状 [B,T,C,H,W]
        """
        self._sync_temp_from_logit_scale()

        if images.ndim == 4:
            images = images.unsqueeze(1)  # [B,1,3,H,W]
        if images.ndim != 5:
            raise ValueError(f"Expected [B,3,H,W] or [B,T,3,H,W], got {tuple(images.shape)}")

        # SurgCLIP.encode_vision 返回 (vision_embeds, pooled_vision_embeds)
        _, pooled = self.surgclip.encode_vision(images)

        # 当前 TimeSformer 返回的是逐帧 pooled 特征 [B, T, C]；
        # 这里先沿时间维做 clip-level pooling，再投影成对比学习特征。
        if pooled.ndim == 3:
            pooled = pooled.mean(dim=1)
        elif pooled.ndim != 2:
            raise ValueError(f"Expected pooled vision features to be 2D or 3D, got {tuple(pooled.shape)}")

        feats = self.surgclip.vision_proj(pooled)
        feats = F.normalize(feats, dim=-1)
        return feats

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self._sync_temp_from_logit_scale()

        text = SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)
        _, pooled = self.surgclip.encode_text(text)

        feats = self.surgclip.text_proj(pooled)
        feats = F.normalize(feats, dim=-1)
        return feats


def clip_contrastive_loss(model, images, input_ids, attention_mask):
    image_features = model.module.encode_image(images)  # [B, D]
    text_features = model.module.encode_text(input_ids, attention_mask)  # [B, D]

    rank = dist.get_rank()
    batch_size = image_features.size(0)

    gathered_image = concat_all_gather(image_features.detach())
    gathered_text = concat_all_gather(text_features.detach())

    gathered_image[rank] = image_features
    gathered_text[rank] = text_features

    all_image_features = torch.cat(gathered_image, dim=0)  # [B*W, D]
    all_text_features = torch.cat(gathered_text, dim=0)    # [B*W, D]

    logit_scale = model.module.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ all_text_features.t()
    logits_per_text = logit_scale * text_features @ all_image_features.t()

    labels = torch.arange(batch_size, device=images.device) + rank * batch_size
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2.0


def parse_args():
    p = argparse.ArgumentParser("SurgLaVi (SurgCLIP) DDP Training using existing pretrain_dataset.py")

    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 20)))
    p.add_argument("--learning_rate", type=float, default=float(os.environ.get("LR", 1e-5)))
    p.add_argument("--weight_decay", type=float, default=float(os.environ.get("WD", 0.02)))
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)

    p.add_argument("--per_gpu_batch_size", type=int, default=int(os.environ.get("PER_GPU_BATCH_SIZE", 2)))
    p.add_argument("--accum_steps", type=int, default=int(os.environ.get("ACCUM_STEPS", 1)))
    p.add_argument("--num_workers", type=int, default=int(os.environ.get("NUM_WORKERS", 4)))

    p.add_argument("--image_size", type=int, default=int(os.environ.get("IMAGE_SIZE", 224)))
    p.add_argument("--max_length", type=int, default=int(os.environ.get("MAX_LENGTH", 256)))
    p.add_argument("--num_frames", type=int, default=int(os.environ.get("NUM_FRAMES", 16)))

    # dataset paths
    p.add_argument("--video_root_folder", type=str, default=os.environ.get(
        "PRETRAIN_VIDEO_ROOT_FOLDER", "downloaded_video_224_test"
    ))
    p.add_argument("--assume_resized_video", type=int, default=int(os.environ.get("PRETRAIN_VIDEO_ALREADY_RESIZED", "0")))

    p.add_argument("--main_csv_path", type=str, default=os.environ.get(
        "PRETRAIN_MAIN_CSV_PATH", "surglavi_level_csv/all_video.csv"
    ))
    p.add_argument("--annotations_folder", type=str, default=os.environ.get(
        "PRETRAIN_ANNOTATIONS_FOLDER", "surglavi_level_csv/fine"
    ))
    p.add_argument("--annotations_root", type=str, default=os.environ.get("PRETRAIN_ANNOTATIONS_ROOT", ""))
    p.add_argument("--annotation_levels", type=str, default=os.environ.get("PRETRAIN_ANNOTATION_LEVELS", ""))
    p.add_argument("--level_mix", type=str, default=os.environ.get("PRETRAIN_LEVEL_MIX", "concat"),
                   choices=("concat", "balanced"))
    p.add_argument("--level_seed", type=int, default=int(os.environ.get("PRETRAIN_LEVEL_SEED", 42)))
    p.add_argument("--sample_mode", type=str, default=os.environ.get("PRETRAIN_SAMPLE_MODE", "random"),
                   choices=("random", "center"))

    # tokenizer: 用于 caption -> input_ids/attention_mask，同时对齐 SurgCLIP 文本编码器权重
    p.add_argument("--tokenizer_name", type=str, default=os.environ.get("TOKENIZER_NAME", "bert-base-uncased"))

    # SurgLaVi model config
    p.add_argument("--surgclip_model_name", type=str, default=os.environ.get("SURGCLIP_MODEL_NAME", "SurgCLIP-B"))

    # checkpoint saving
    p.add_argument("--save_dir", type=str, default=os.environ.get("SAVE_DIR", "."))
    p.add_argument("--save_every", type=int, default=int(os.environ.get("SAVE_EVERY", 0)),
                   help="Save checkpoint every N epochs. 0 = only save final.")
    p.add_argument("--save_name", type=str, default=os.environ.get("SAVE_NAME", "surglavi_final.pt"))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    p.add_argument("--finetune_mode", type=str, default=os.environ.get("FINETUNE_MODE", "lora"),
                   choices=("full", "lora"))
    p.add_argument("--lora_rank", type=int, default=int(os.environ.get("LORA_RANK", 8)))
    p.add_argument("--lora_alpha", type=float, default=float(os.environ.get("LORA_ALPHA", 16)))
    p.add_argument("--lora_dropout", type=float, default=float(os.environ.get("LORA_DROPOUT", 0.05)))
    p.add_argument(
        "--gradient_checkpointing",
        type=lambda x: str(x).lower() in {"1", "true", "t", "yes", "y"},
        default=os.environ.get("GRADIENT_CHECKPOINTING", "1").lower() in {"1", "true", "t", "yes", "y"},
    )
    p.add_argument(
        "--lora_targets",
        type=str,
        default=os.environ.get(
            "LORA_TARGETS",
            "text_encoder.encoder.layer.,vision_encoder.model.blocks."
        ),
        help="Comma-separated substrings used to select Linear layers for LoRA wrapping.",
    )

    return p.parse_args()


def train():
    args = parse_args()

    rank = setup_ddp()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    seed_everything(args.seed + rank)

    # perf
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    amp_dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))

    if rank == 0:
        print(f"[rank0] world_size={world_size} num_frames={args.num_frames} per_gpu_batch_size={args.per_gpu_batch_size} amp_dtype={amp_dtype}")
        print(f"[rank0] save_dir={args.save_dir} save_every={args.save_every} save_name={args.save_name}")
        print(f"[rank0] surgclip_model_name={args.surgclip_model_name} tokenizer_name={args.tokenizer_name}")
        print(f"[rank0] seed={args.seed}")
        print(f"[rank0] gradient_checkpointing={args.gradient_checkpointing}")
        if args.annotations_root:
            levels = args.annotation_levels if args.annotation_levels else "coarse,mid,fine"
            print(f"[rank0] annotations_root={args.annotations_root} annotation_levels={levels} level_mix={args.level_mix} sample_mode={args.sample_mode}")
        else:
            print(f"[rank0] annotations_folder={args.annotations_folder} sample_mode={args.sample_mode}")

    # tokenizer（用于你的 PretrainDataset）
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    train_dataset = PretrainDataset(
        main_csv_path=args.main_csv_path,
        annotations_folder=(args.annotations_folder or None),
        tokenizer=tokenizer,
        image_size=args.image_size,
        max_length=args.max_length,
        sample_mode=args.sample_mode,
        video_root_folder=args.video_root_folder,
        assume_resized_video=(args.assume_resized_video == 1),
        num_frames=args.num_frames,
        annotations_root=(args.annotations_root or None),
        annotation_levels=(args.annotation_levels or None),
        level_mix=args.level_mix,
        level_seed=args.level_seed,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        collate_fn=collate_fn_skip_corrupted,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        worker_init_fn=build_worker_init_fn(args.seed, rank),
    )

    config = get_config(
        args.surgclip_model_name,
        overrides={
            "device": str(device),
            "num_frames": int(args.num_frames),
            "inputs": {
                "image_res": int(args.image_size),
                "video_input": {
                    "num_frames_test": int(args.num_frames),
                },
            },
            "model": {
                "temporal_modeling": {
                    "enabled": int(args.num_frames) > 1,
                },
                "text_encoder": {
                    "pretrained": args.tokenizer_name,
                },
            },
            "gradient_checkpointing": bool(args.gradient_checkpointing),
        },
    )

    surgclip_core = SurgCLIP(config=config, tokenizer=tokenizer, is_pretrain=True)
    model = SurgCLIPAdapter(surgclip_core).to(device)
    finetune_info = configure_finetuning(
        model,
        finetune_mode=args.finetune_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=parse_comma_separated_list(args.lora_targets),
    )

    total_params, trainable_params = count_parameters(model)

    if os.environ.get("USE_COMPILE", "0") == "1":
        try:
            model = torch.compile(model, mode=os.environ.get("COMPILE_MODE", "max-autotune"))
        except Exception as e:
            if rank == 0:
                print(f"torch.compile failed: {e}")

    model = DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        bucket_cap_mb=int(os.environ.get("DDP_BUCKET_MB", 64)),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    total_update_steps = math.ceil(len(train_loader) / args.accum_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_update_steps))

    writer = SummaryWriter(log_dir=os.environ.get("TB_LOGDIR", "runs/SurgLaVi")) if rank == 0 else None

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"[rank0] finetune_mode={args.finetune_mode}")
        print(f"[rank0] trainable_params={trainable_params:,}/{total_params:,}")
        if finetune_info["replaced_modules"]:
            print(f"[rank0] lora_wrapped_modules={len(finetune_info['replaced_modules'])}")
            preview = finetune_info["replaced_modules"][:10]
            print(f"[rank0] lora_preview={preview}")
        with open(os.path.join(args.save_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.save_dir, "train_command.txt"), "w", encoding="utf-8") as f:
            f.write("python " + " ".join(sys.argv) + "\n")
            f.write(f"cwd={os.getcwd()}\n")
            f.write(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n")
            f.write(f"TB_LOGDIR={os.environ.get('TB_LOGDIR', '')}\n")

    start_epoch = 0
    global_step = 0

    def _save_checkpoint(path: str, epoch: int):
        if rank != 0:
            return
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        }
        torch.save(ckpt, path)
        print(f"[rank0] saved checkpoint to: {path}")

    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        if rank == 0:
            print(f"Resuming from: {args.resume_from_checkpoint}")
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        if scaler.is_enabled() and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        progress = tqdm(train_loader, disable=(rank != 0), desc=f"Epoch {epoch+1}/{args.epochs}")

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress):
            if batch is None:
                continue

            images, input_ids, attention_mask = [b.to(device, non_blocking=True) for b in batch]

            micro_step = (step % args.accum_steps) + 1
            should_update = (micro_step == args.accum_steps)

            sync_ctx = model.no_sync() if not should_update else contextlib.nullcontext()

            with sync_ctx:
                with autocast(dtype=amp_dtype):
                    raw_loss = clip_contrastive_loss(model, images, input_ids, attention_mask)
                    loss = raw_loss / args.accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if should_update:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if writer is not None:
                    writer.add_scalar("train/loss", float(raw_loss.item()), global_step)
                    writer.add_scalar("train/lr", float(scheduler.get_last_lr()[0]), global_step)
                global_step += 1

            if rank == 0:
                progress.set_postfix(loss=float(raw_loss.item()), lr=float(scheduler.get_last_lr()[0]))

        # periodic save
        if args.save_every and ((epoch + 1) % args.save_every == 0):
            path = os.path.join(args.save_dir, f"surglavi_epoch_{epoch + 1}.pt")
            _save_checkpoint(path, epoch + 1)

    # final save
    final_path = os.path.join(args.save_dir, args.save_name)
    _save_checkpoint(final_path, args.epochs)

    if writer is not None:
        writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    train()
