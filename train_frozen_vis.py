import os
import math
import argparse
import contextlib
import time
from collections import Counter

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import swanlab
except ImportError:
    swanlab = None

from model import VLP
from pretrain_dataset import PretrainDataset
from mixed_level_batch_sampler import DistributedMixedLevelBatchSampler


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor):
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


def _unwrap_state_io_module(module):
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def _normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]
        normalized[key] = value
    return normalized


def _export_plain_state_dict_from_ddp(model):
    return _unwrap_state_io_module(model.module).state_dict()


def _load_normalized_state_dict(module, state_dict, source="checkpoint"):
    target_module = _unwrap_state_io_module(module)
    normalized_state_dict = _normalize_state_dict_keys(state_dict)
    msg = target_module.load_state_dict(normalized_state_dict, strict=False)

    allowed_missing_prefixes = (
        "frame_local_projection.",
        "frame_score_head.",
        "video_gate_head.",
    )
    allowed_unexpected_prefixes = (
        "token_local_projection.",
        "token_score_head.",
        "text_gate_head.",
        "frame_score_head.",
        "video_gate_head.",
    )

    disallowed_missing = [
        key for key in msg.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    disallowed_unexpected = [
        key for key in msg.unexpected_keys
        if not key.startswith(allowed_unexpected_prefixes)
    ]

    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            f"{source} 与当前模型不匹配。\n"
            f"Missing keys: {msg.missing_keys}\n"
            f"Unexpected keys: {msg.unexpected_keys}"
        )

    if msg.missing_keys and dist.get_rank() == 0:
        print(
            f"{source} 缺少新方法模块参数，将使用当前随机初始化继续训练: "
            f"{msg.missing_keys}",
            flush=True,
        )

    return msg


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_float_list(spec: str, expected_len: int = None):
    values = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if expected_len is not None and len(values) != expected_len:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_len} comma-separated floats, got {len(values)} from: {spec}"
        )
    return tuple(values)


def parse_args():
    parser = argparse.ArgumentParser(description="VLP Frozen-Visual DDP Training")

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)

    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=int(os.environ.get("PER_GPU_BATCH_SIZE", 16)),
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=int(os.environ.get("ACCUM_STEPS", 1)),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(os.environ.get("NUM_WORKERS", 2)),
    )

    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--num_frames",
        type=int,
        default=int(os.environ.get("NUM_FRAMES", 4)),
    )

    parser.add_argument("--text_model_name", type=str, default="marcobombieri/surgicberta")
    parser.add_argument("--vision_pretrained_weights", type=str, default="lemonfm.pth")

    parser.add_argument(
        "--video_root_folder",
        type=str,
        default=os.environ.get(
            "PRETRAIN_VIDEO_ROOT_FOLDER",
            "/mnt/mydisk/CLIP/downloaded_video_224_test",
        ),
    )
    parser.add_argument("--ffmpeg_timeout", type=int, default=10)
    parser.add_argument("--max_retry", type=int, default=5)
    parser.add_argument(
        "--video_reader_threads",
        type=int,
        default=int(os.environ.get("VIDEO_READER_THREADS", 1)),
    )
    parser.add_argument(
        "--video_reader_cache_size",
        type=int,
        default=int(os.environ.get("VIDEO_READER_CACHE_SIZE", 16)),
    )
    parser.add_argument(
        "--assume_resized_video",
        type=str2bool,
        default=os.environ.get("PRETRAIN_VIDEO_ALREADY_RESIZED", "0") == "1",
    )

    parser.add_argument(
        "--main_csv_path",
        type=str,
        default=os.environ.get(
            "PRETRAIN_MAIN_CSV_PATH",
            "/mnt/mydisk/CLIP/surglavi_level_csv/all_video.csv",
        ),
    )
    parser.add_argument(
        "--annotations_folder",
        type=str,
        default=os.environ.get(
            "PRETRAIN_ANNOTATIONS_FOLDER",
            "/mnt/mydisk/CLIP/surglavi_level_csv/fine",
        ),
    )
    parser.add_argument("--annotations_root", type=str, default=None)
    parser.add_argument(
        "--annotation_levels",
        type=str,
        default=None,
        help="Comma-separated levels, e.g. coarse,mid,fine",
    )
    parser.add_argument(
        "--level_mix",
        type=str,
        default="concat",
        choices=["concat", "balanced"],
    )
    parser.add_argument(
        "--level_batch_sizes",
        type=str,
        default="fine:80,mid:32,coarse:16",
        help="Per-rank batch composition, e.g. fine:80,mid:32,coarse:16",
    )

    parser.add_argument(
        "--samples_cache_dir",
        type=str,
        default="/mnt/mydisk/CLIP/.cache/pretrain_samples",
    )
    parser.add_argument(
        "--use_samples_cache",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--rebuild_samples_cache",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--samples_cache_version",
        type=str,
        default="v1",
    )

    parser.add_argument(
        "--use_swanlab",
        type=str2bool,
        default=os.environ.get("USE_SWANLAB", "1") == "1",
    )
    parser.add_argument("--local_temperature", type=float, default=0.07)
    parser.add_argument(
        "--selection_pooling",
        type=str,
        default=os.environ.get("SELECTION_POOLING", "similarity"),
        choices=["similarity", "xpool"],
        help="Frame selection pooling strategy for text-conditioned video reweighting",
    )
    parser.add_argument(
        "--level_frame_temperatures",
        type=lambda s: parse_float_list(s, expected_len=3),
        default=(0.6, 0.9, 1.2),
        help="Comma-separated fine,mid,coarse frame temperatures",
    )
    parser.add_argument(
        "--train_window_expand_ratio",
        type=float,
        default=2.0,
        help="Expand the annotated training interval by this ratio for train-time soft frame selection",
    )
    parser.add_argument(
        "--selection_loss_weight",
        type=float,
        default=0.5,
        help="Weight for the train-time expanded-window selection contrastive loss",
    )
    parser.add_argument(
        "--hierarchical_consistency_weight",
        type=float,
        default=0.1,
        help="Weight for same-video adjacent-level consistency on selected views",
    )

    return parser.parse_args()


def parse_level_batch_sizes(spec: str):
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        level, count = item.split(":")
        out[level.strip()] = int(count.strip())
    if not out:
        raise ValueError("level_batch_sizes is empty")
    return out


def collate_fn_expanded_frames_only(batch):
    compact_batch = []
    for item in batch:
        if item is None:
            continue
        if len(item) != 6:
            raise ValueError(
                "Expected dataset items as (images, selection_images, input_ids, attention_mask, level_ids, sample_indices)."
            )
        _, selection_images, input_ids, attention_mask, level_ids, sample_indices = item
        compact_batch.append((selection_images, input_ids, attention_mask, level_ids, sample_indices))

    if not compact_batch:
        return None

    return torch.utils.data.dataloader.default_collate(compact_batch)


LEVEL_NAME_BY_ID = {
    0: "fine",
    1: "mid",
    2: "coarse",
}


def _reduce_mean_tensor(value: torch.Tensor):
    value = value.detach().float()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value


def _collect_debug_stats(model_module, level_ids):
    debug_sources = {
        "pair_confidence": model_module.last_pair_confidence,
        "frame_entropy": model_module.last_frame_entropy,
        "frame_peak": model_module.last_frame_peak,
    }

    available = {
        name: value.detach().float().reshape(-1)
        for name, value in debug_sources.items()
        if value is not None
    }
    if not available:
        return {}

    level_ids = level_ids.detach().reshape(-1)
    level_ids = level_ids.to(device=next(iter(available.values())).device, dtype=torch.long)

    stats = {}
    for name, values in available.items():
        total = values.sum()
        count = torch.tensor(float(values.numel()), device=values.device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        stats[name] = (total / count.clamp(min=1.0)).item()

        if values.numel() != level_ids.numel():
            continue

        for level_id, level_name in LEVEL_NAME_BY_ID.items():
            mask = level_ids == level_id
            if mask.any():
                level_total = values[mask].sum()
                level_count = torch.tensor(float(mask.sum().item()), device=values.device)
            else:
                level_total = torch.zeros((), device=values.device)
                level_count = torch.zeros((), device=values.device)

            dist.all_reduce(level_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(level_count, op=dist.ReduceOp.SUM)
            if level_count.item() > 0:
                stats[f"{level_name}/{name}"] = (
                    level_total / level_count.clamp(min=1.0)
                ).item()

    return stats


def clip_contrastive_loss(
    model,
    selection_images,
    input_ids,
    attention_mask,
    level_ids,
    sample_indices,
    dataset_samples,
    selection_loss_weight=0.5,
    hierarchical_consistency_weight=0.1,
):
    image_features, selected_image_features, text_features = model.module.encode_training_pair(
        image=selection_images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        level_ids=level_ids,
        selection_image=selection_images,
    )
    
    rank = dist.get_rank()
    batch_size = image_features.size(0)
    device = image_features.device
    logit_scale = model.module.logit_scale.exp()
    labels = torch.arange(batch_size, device=device) + rank * batch_size

    def _gather_features(image_feats, text_feats):
        gathered_image = concat_all_gather(image_feats.detach())
        gathered_text = concat_all_gather(text_feats.detach())

        gathered_image[rank] = image_feats
        gathered_text[rank] = text_feats

        all_image_features = torch.cat(gathered_image, dim=0)
        all_text_features = torch.cat(gathered_text, dim=0)

        return all_image_features, all_text_features

    def _symmetric_contrastive(image_feats, text_feats):
        all_image_features, all_text_features = _gather_features(image_feats, text_feats)

        logits_per_image = logit_scale * image_feats @ all_text_features.t()
        logits_per_text = logit_scale * text_feats @ all_image_features.t()
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return 0.5 * (loss_i + loss_t)

    base_loss = _symmetric_contrastive(image_features, text_features)
    hierarchical_loss = torch.zeros((), device=device)
    if selected_image_features is None:
        total_loss = base_loss
    else:
        selection_loss = _symmetric_contrastive(selected_image_features, text_features)
        if hierarchical_consistency_weight > 0:
            hierarchical_loss = compute_hierarchical_consistency_loss(
                selected_image_features=selected_image_features,
                sample_indices=sample_indices,
                dataset_samples=dataset_samples,
            )
        total_loss = (
            base_loss
            + selection_loss_weight * selection_loss
            + hierarchical_consistency_weight * hierarchical_loss
        )

    return total_loss


def compute_hierarchical_consistency_loss(selected_image_features, sample_indices, dataset_samples):
    """
    Lightweight adjacent-level consistency on same-video selected views:
      fine <-> mid and mid <-> coarse.
    This preserves the current train-time denoising design while making
    hierarchy more than a temperature-only prior.
    """
    if sample_indices is None:
        return torch.zeros((), device=selected_image_features.device)

    by_video = {}
    for batch_pos, sample_idx in enumerate(sample_indices.tolist()):
        if sample_idx < 0:
            continue
        sample = dataset_samples[int(sample_idx)]
        video_path = sample.get("video_path")
        level = str(sample.get("level", "")).lower()
        if not video_path or level not in {"fine", "mid", "coarse"}:
            continue
        by_video.setdefault(video_path, {})[level] = batch_pos

    losses = []
    for level_map in by_video.values():
        if "fine" in level_map and "mid" in level_map:
            fine_feat = selected_image_features[level_map["fine"]]
            mid_feat = selected_image_features[level_map["mid"]]
            losses.append(1.0 - F.cosine_similarity(fine_feat.unsqueeze(0), mid_feat.unsqueeze(0)).mean())
        if "mid" in level_map and "coarse" in level_map:
            mid_feat = selected_image_features[level_map["mid"]]
            coarse_feat = selected_image_features[level_map["coarse"]]
            losses.append(1.0 - F.cosine_similarity(mid_feat.unsqueeze(0), coarse_feat.unsqueeze(0)).mean())

    if not losses:
        return torch.zeros((), device=selected_image_features.device)
    return torch.stack(losses).mean()


def train():
    args = parse_args()

    rank = setup_ddp()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    amp_dtype = (
        torch.bfloat16
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        else torch.float16
    )
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))

    PER_GPU_BATCH_SIZE = args.per_gpu_batch_size
    ACCUM_STEPS = args.accum_steps
    USE_SWANLAB = args.use_swanlab
    LEVEL_BATCH_SIZES = parse_level_batch_sizes(args.level_batch_sizes)
    DEBUG_LOG_INTERVAL = int(os.environ.get("DEBUG_LOG_INTERVAL", 20))
    PERF_LOG_INTERVAL = int(os.environ.get("PERF_LOG_INTERVAL", DEBUG_LOG_INTERVAL))
    anchor_same_video_triplets = all(
        LEVEL_BATCH_SIZES.get(level, 0) > 0 for level in ("fine", "mid", "coarse")
    )

    if sum(LEVEL_BATCH_SIZES.values()) != PER_GPU_BATCH_SIZE:
        raise ValueError(
            f"Sum of level_batch_sizes must equal per_gpu_batch_size. "
            f"Got {sum(LEVEL_BATCH_SIZES.values())} vs {PER_GPU_BATCH_SIZE}"
        )

    CONFIG = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_betas": (args.adam_beta1, args.adam_beta2),
        "num_workers": args.num_workers,
        "embed_dim": args.embed_dim,
        "image_size": args.image_size,
        "max_length": args.max_length,
        "text_model_name": args.text_model_name,
        "vision_pretrained_weights": args.vision_pretrained_weights,
        "video_root_folder": args.video_root_folder,
        "ffmpeg_timeout": args.ffmpeg_timeout,
        "max_retry": args.max_retry,
        "video_reader_threads": args.video_reader_threads,
        "video_reader_cache_size": args.video_reader_cache_size,
        "assume_resized_video": args.assume_resized_video,
        "num_frames": args.num_frames,
        "annotations_root": args.annotations_root,
        "annotation_levels": args.annotation_levels,
        "level_mix": args.level_mix,
        "level_batch_sizes": LEVEL_BATCH_SIZES,
        "samples_cache_dir": args.samples_cache_dir,
        "use_samples_cache": args.use_samples_cache,
        "rebuild_samples_cache": args.rebuild_samples_cache,
        "samples_cache_version": args.samples_cache_version,
        "local_temperature": args.local_temperature,
        "selection_pooling": args.selection_pooling,
        "level_frame_temperatures": args.level_frame_temperatures,
        "train_window_expand_ratio": args.train_window_expand_ratio,
        "selection_loss_weight": args.selection_loss_weight,
        "hierarchical_consistency_weight": args.hierarchical_consistency_weight,
        "anchor_same_video_triplets": anchor_same_video_triplets,
    }

    MAIN_CSV_PATH = args.main_csv_path
    ANNOTATIONS_FOLDER = args.annotations_folder

    if rank == 0:
        print("正在初始化 VLP 模型...")

    model = VLP(
        embed_dim=CONFIG["embed_dim"],
        text_model_name=CONFIG["text_model_name"],
        vision_pretrained_weights=CONFIG["vision_pretrained_weights"],
        num_frames=CONFIG["num_frames"],
        local_temperature=CONFIG["local_temperature"],
        selection_pooling=CONFIG["selection_pooling"],
        level_frame_temperatures=CONFIG["level_frame_temperatures"],
    ).to(device)

    model.freeze_encoders_train_projections()
    model.set_frozen_modules_eval()

    if rank == 0:
        visual_total = sum(p.numel() for p in model.visual.parameters())
        visual_trainable = sum(p.numel() for p in model.visual.parameters() if p.requires_grad)

        text_backbone_total = sum(p.numel() for p in model.text.backbone.parameters())
        text_backbone_trainable = sum(p.numel() for p in model.text.backbone.parameters() if p.requires_grad)

        text_proj_total = sum(p.numel() for p in model.text.text_projection.parameters())
        text_proj_trainable = sum(p.numel() for p in model.text.text_projection.parameters() if p.requires_grad)

        video_proj_total = sum(p.numel() for p in model.video_projection.parameters())
        video_proj_trainable = sum(p.numel() for p in model.video_projection.parameters() if p.requires_grad)

        frame_pool_total = 0
        frame_pool_trainable = 0
        if model.frame_pool is not None:
            frame_pool_total = sum(p.numel() for p in model.frame_pool.parameters())
            frame_pool_trainable = sum(p.numel() for p in model.frame_pool.parameters() if p.requires_grad)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"视觉编码器冻结: trainable {visual_trainable}/{visual_total}")
        print(f"文本backbone冻结: trainable {text_backbone_trainable}/{text_backbone_total}")
        print(f"文本投影层可训练: trainable {text_proj_trainable}/{text_proj_total}")
        print(f"视频投影层可训练: trainable {video_proj_trainable}/{video_proj_total}")
        print(f"帧池化模块可训练: trainable {frame_pool_trainable}/{frame_pool_total}")
        print(f"logit_scale requires_grad: {model.logit_scale.requires_grad}")
        print(f"模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params_num:,}")
        print(f"帧池化模块：{model.frame_pool}")

    if os.environ.get("USE_COMPILE", "0") == "1":
        try:
            compile_mode = os.environ.get("COMPILE_MODE", "max-autotune")
            model = torch.compile(model, mode=compile_mode)
            if rank == 0:
                print(f"已启用 torch.compile(mode={compile_mode})")
        except Exception as e:
            if rank == 0:
                print(f"torch.compile 启用失败：{e}")

    bucket_mb = int(os.environ.get("DDP_BUCKET_MB", 64))
    model = DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        bucket_cap_mb=bucket_mb,
    )

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model_name"])

    if rank == 0:
        print("正在加载预训练数据集...")
        print(f"视频目录: {CONFIG['video_root_folder']}")
        print(f"假设视频已预缩放: {CONFIG['assume_resized_video']}")
        print(f"每个样本抽帧数: {CONFIG['num_frames']}")
        if args.annotations_root:
            print(f"标注根目录: {args.annotations_root}")
            print(f"标注层级: {args.annotation_levels}")
            print(f"层级混合方式: {args.level_mix}")
        else:
            print(f"标注目录: {ANNOTATIONS_FOLDER}")
        print(f"batch层级配比: {LEVEL_BATCH_SIZES}")
        print(f"local_temperature: {CONFIG['local_temperature']}")
        print(f"selection_pooling: {CONFIG['selection_pooling']}")
        print(f"level_frame_temperatures: {CONFIG['level_frame_temperatures']}")
        print(f"train_window_expand_ratio: {CONFIG['train_window_expand_ratio']}")
        print(f"selection_loss_weight: {CONFIG['selection_loss_weight']}")
        print(f"hierarchical_consistency_weight: {CONFIG['hierarchical_consistency_weight']}")
        print(f"anchor_same_video_triplets: {CONFIG['anchor_same_video_triplets']}")
        print(f"samples cache目录: {CONFIG['samples_cache_dir']}")
        print(f"use_samples_cache: {CONFIG['use_samples_cache']}")
        print(f"rebuild_samples_cache: {CONFIG['rebuild_samples_cache']}")
        print(f"samples_cache_version: {CONFIG['samples_cache_version']}")
        print(f"video_reader_threads: {CONFIG['video_reader_threads']}")
        print(f"video_reader_cache_size: {CONFIG['video_reader_cache_size']}")

    train_dataset = PretrainDataset(
        main_csv_path=MAIN_CSV_PATH,
        annotations_folder=ANNOTATIONS_FOLDER if not args.annotations_root else None,
        annotations_root=args.annotations_root,
        annotation_levels=args.annotation_levels,
        level_mix=args.level_mix,
        tokenizer=tokenizer,
        image_size=CONFIG["image_size"],
        max_length=CONFIG["max_length"],
        sample_mode="random",
        ffmpeg_timeout=CONFIG["ffmpeg_timeout"],
        max_retry=CONFIG["max_retry"],
        video_root_folder=CONFIG["video_root_folder"],
        assume_resized_video=CONFIG["assume_resized_video"],
        num_frames=CONFIG["num_frames"],
        return_level_id=True,
        return_sample_index=True,
        return_expanded_frames=True,
        expanded_window_ratio=CONFIG["train_window_expand_ratio"],
        samples_cache_dir=CONFIG["samples_cache_dir"],
        use_samples_cache=CONFIG["use_samples_cache"],
        rebuild_samples_cache=CONFIG["rebuild_samples_cache"],
        samples_cache_version=CONFIG["samples_cache_version"],
        video_reader_threads=CONFIG["video_reader_threads"],
        video_reader_cache_size=CONFIG["video_reader_cache_size"],
    )

    if rank == 0:
        level_counter = Counter(
            str(sample.get("level", "unknown")).lower()
            for sample in train_dataset.samples
        )
        print(f"训练样本总数: {len(train_dataset)}")
        print(
            "各层级样本数: "
            + ", ".join(
                f"{level}={level_counter.get(level, 0)}"
                for level in ("fine", "mid", "coarse")
            )
        )

    train_sampler = DistributedMixedLevelBatchSampler(
        train_dataset,
        batch_size=PER_GPU_BATCH_SIZE,
        level_batch_sizes=LEVEL_BATCH_SIZES,
        anchor_same_video_triplets=CONFIG["anchor_same_video_triplets"],
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn_expanded_frames_only,
        persistent_workers=(CONFIG["num_workers"] > 0),
        prefetch_factor=2 if CONFIG["num_workers"] > 0 else None,
    )

    num_batches = len(train_loader)
    if num_batches == 0:
        raise ValueError(
            "train_loader is empty. Check dataset size, batch size, level_batch_sizes, and drop_last settings."
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=CONFIG["learning_rate"],
        betas=CONFIG["adam_betas"],
        weight_decay=CONFIG["weight_decay"],
    )

    updates_per_epoch = math.ceil(num_batches / ACCUM_STEPS)
    total_update_steps = updates_per_epoch * CONFIG["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_update_steps))

    writer = None
    swanlab_run = None
    if rank == 0:
        log_dir = os.environ.get("TB_LOGDIR", "runs/VLP_frozen_vis")
        writer = SummaryWriter(log_dir=log_dir)

        if USE_SWANLAB:
            if swanlab is None:
                print("swanlab 未安装，跳过 SwanLab 日志。")
            else:
                swanlab_config = {
                    "epochs": CONFIG["epochs"],
                    "learning_rate": CONFIG["learning_rate"],
                    "weight_decay": CONFIG["weight_decay"],
                    "adam_betas": list(CONFIG["adam_betas"]),
                    "num_workers": CONFIG["num_workers"],
                    "embed_dim": CONFIG["embed_dim"],
                    "image_size": CONFIG["image_size"],
                    "max_length": CONFIG["max_length"],
                    "text_model_name": CONFIG["text_model_name"],
                    "vision_pretrained_weights": CONFIG["vision_pretrained_weights"],
                    "video_root_folder": CONFIG["video_root_folder"],
                    "ffmpeg_timeout": CONFIG["ffmpeg_timeout"],
                    "max_retry": CONFIG["max_retry"],
                    "assume_resized_video": CONFIG["assume_resized_video"],
                    "num_frames": CONFIG["num_frames"],
                    "per_gpu_batch_size": PER_GPU_BATCH_SIZE,
                    "accum_steps": ACCUM_STEPS,
                    "world_size": world_size,
                    "main_csv_path": MAIN_CSV_PATH,
                    "annotations_folder": ANNOTATIONS_FOLDER,
                    "annotations_root": args.annotations_root,
                    "annotation_levels": args.annotation_levels,
                    "level_mix": args.level_mix,
                    "level_batch_sizes": args.level_batch_sizes,
                    "samples_cache_dir": args.samples_cache_dir,
                    "use_samples_cache": args.use_samples_cache,
                    "rebuild_samples_cache": args.rebuild_samples_cache,
                    "samples_cache_version": args.samples_cache_version,
                    "selection_pooling": args.selection_pooling,
                    "train_window_expand_ratio": args.train_window_expand_ratio,
                    "selection_loss_weight": args.selection_loss_weight,
                    "hierarchical_consistency_weight": args.hierarchical_consistency_weight,
                    "anchor_same_video_triplets": CONFIG["anchor_same_video_triplets"],
                    "resume_from_checkpoint": args.resume_from_checkpoint,
                    "tb_logdir": log_dir,
                }

                swanlab_kwargs = {
                    "project": os.environ.get("SWANLAB_PROJECT", "CLIP"),
                    "experiment_name": os.environ.get("SWANLAB_EXPERIMENT_NAME", "VLP_frozen_vis"),
                    "config": swanlab_config,
                    "logdir": os.environ.get("SWANLAB_LOGDIR", "swanlog/VLP_frozen_vis"),
                }

                swanlab_workspace = os.environ.get("SWANLAB_WORKSPACE")
                if swanlab_workspace:
                    swanlab_kwargs["workspace"] = swanlab_workspace

                swanlab_mode = os.environ.get("SWANLAB_MODE")
                if swanlab_mode:
                    swanlab_kwargs["mode"] = swanlab_mode

                swanlab_run_id = os.environ.get("SWANLAB_RUN_ID")
                if swanlab_run_id:
                    swanlab_kwargs["id"] = swanlab_run_id

                swanlab_resume = os.environ.get("SWANLAB_RESUME")
                if swanlab_resume:
                    swanlab_kwargs["resume"] = swanlab_resume

                swanlab_run = swanlab.init(**swanlab_kwargs)
                swanlab.sync_tensorboard_torch()

    start_epoch = 0
    global_step = 0
    save_prefix = os.environ.get("SAVE_PREFIX", "")
    save_dir = os.path.dirname(save_prefix)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def _build_ckpt_path(filename: str) -> str:
        if not save_prefix:
            return filename
        if save_prefix.endswith(os.sep):
            return os.path.join(save_prefix, filename)
        return f"{save_prefix}{filename}"

    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        if rank == 0:
            print(f"正在从检查点恢复训练: {args.resume_from_checkpoint}")

        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)

        _load_normalized_state_dict(
            model.module,
            checkpoint["model_state_dict"],
            source=args.resume_from_checkpoint,
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]

        scaler_state_dict = checkpoint.get("scaler_state_dict")
        if scaler_state_dict is not None and scaler.is_enabled():
            scaler.load_state_dict(scaler_state_dict)

        if rank == 0:
            print(f"恢复成功，将从 epoch {start_epoch + 1} 开始。")

    if rank == 0:
        print("配置完成，开始训练...")

    for epoch in range(start_epoch, CONFIG["epochs"]):
        train_sampler.set_epoch(epoch)
        _unwrap_state_io_module(model.module).set_frozen_modules_eval()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [GPU {rank}]",
            position=rank,
            disable=(rank != 0),
        )

        optimizer.zero_grad(set_to_none=True)
        last_loss_value = None
        accum_loss_sum = torch.zeros((), device=device)
        accum_data_time_sum = 0.0
        accum_step_time_sum = 0.0
        batch_fetch_start = time.perf_counter()

        for step, batch in enumerate(progress_bar):
            data_time = time.perf_counter() - batch_fetch_start
            if batch is None:
                batch_fetch_start = time.perf_counter()
                continue

            step_start = time.perf_counter()
            selection_images_cpu, input_ids, attention_mask, level_ids, sample_indices = batch
            selection_images = selection_images_cpu.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            level_ids = level_ids.to(device, non_blocking=True)

            micro_step = (step % ACCUM_STEPS) + 1
            is_last_batch = (step + 1) == num_batches
            current_accum_steps = micro_step if (is_last_batch and micro_step != ACCUM_STEPS) else ACCUM_STEPS
            should_update = (micro_step == ACCUM_STEPS) or is_last_batch

            sync_ctx = model.no_sync() if not should_update else contextlib.nullcontext()

            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    raw_loss = clip_contrastive_loss(
                        model,
                        selection_images,
                        input_ids,
                        attention_mask,
                        level_ids,
                        sample_indices,
                        train_dataset.samples,
                        selection_loss_weight=CONFIG["selection_loss_weight"],
                        hierarchical_consistency_weight=CONFIG["hierarchical_consistency_weight"],
                    )
                    loss = raw_loss / current_accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            accum_loss_sum += raw_loss.detach()
            step_time = time.perf_counter() - step_start
            accum_data_time_sum += data_time
            accum_step_time_sum += step_time

            if should_update:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                window_loss = accum_loss_sum / current_accum_steps
                loss_log = window_loss.clone()
                dist.all_reduce(loss_log, op=dist.ReduceOp.SUM)
                loss_avg = loss_log.item() / world_size
                last_loss_value = loss_avg
                accum_loss_sum.zero_()
                data_time_avg = _reduce_mean_tensor(
                    torch.tensor(accum_data_time_sum / current_accum_steps, device=device)
                ).item()
                step_time_avg = _reduce_mean_tensor(
                    torch.tensor(accum_step_time_sum / current_accum_steps, device=device)
                ).item()
                accum_data_time_sum = 0.0
                accum_step_time_sum = 0.0
                debug_stats = _collect_debug_stats(_unwrap_state_io_module(model.module), level_ids)

                if writer is not None:
                    writer.add_scalar("train/loss", loss_avg, global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    writer.add_scalar("train/epoch", epoch + 1, global_step)
                    writer.add_scalar("perf/data_time", data_time_avg, global_step)
                    writer.add_scalar("perf/step_time", step_time_avg, global_step)
                    for stat_name, stat_value in debug_stats.items():
                        writer.add_scalar(f"debug/{stat_name}", stat_value, global_step)

                global_step += 1

            if rank == 0:
                postfix = {
                    "loss": f"{raw_loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "data": f"{data_time:.2f}s",
                    "step": f"{step_time:.2f}s",
                }
                if should_update:
                    if "pair_confidence" in debug_stats:
                        postfix["conf"] = f"{debug_stats['pair_confidence']:.3f}"
                    if "frame_entropy" in debug_stats:
                        postfix["fH"] = f"{debug_stats['frame_entropy']:.3f}"
                progress_bar.set_postfix(**postfix)

                if should_update and DEBUG_LOG_INTERVAL > 0 and global_step % DEBUG_LOG_INTERVAL == 0:
                    debug_parts = []
                    for level_name in ("fine", "mid", "coarse"):
                        frame_key = f"{level_name}/frame_entropy"
                        conf_key = f"{level_name}/pair_confidence"
                        if frame_key in debug_stats and conf_key in debug_stats:
                            debug_parts.append(
                                f"{level_name}: conf={debug_stats[conf_key]:.3f}, "
                                f"fH={debug_stats[frame_key]:.3f}"
                            )
                    if debug_parts:
                        print(
                            f"[debug step {global_step}] " + " | ".join(debug_parts),
                            flush=True,
                        )
                if should_update and PERF_LOG_INTERVAL > 0 and global_step % PERF_LOG_INTERVAL == 0:
                    print(
                        f"[perf step {global_step}] data={data_time_avg:.2f}s | step={step_time_avg:.2f}s",
                        flush=True,
                    )

            del input_ids, attention_mask, level_ids, sample_indices
            batch_fetch_start = time.perf_counter()

        if rank == 0:
            print(
                f"Epoch {epoch + 1} 完成。最后记录损失: "
                f"{last_loss_value if last_loss_value is not None else 'N/A'}"
            )

            checkpoint_data = {
                "epoch": epoch + 1,
                "model_state_dict": _export_plain_state_dict_from_ddp(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            }
            epoch_ckpt_path = _build_ckpt_path(f"vlp_epoch_{epoch + 1}.pt")
            torch.save(checkpoint_data, epoch_ckpt_path)
            print(f"已保存检查点: {epoch_ckpt_path}")

            if writer is not None:
                writer.flush()

    if rank == 0:
        print("训练完成。")
        final_checkpoint_data = {
            "epoch": CONFIG["epochs"],
            "model_state_dict": _export_plain_state_dict_from_ddp(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        }
        final_ckpt_path = _build_ckpt_path("vlp_final.pt")
        torch.save(final_checkpoint_data, final_ckpt_path)
        print(f"已保存最终检查点: {final_ckpt_path}")

        if writer is not None:
            writer.close()

        if swanlab_run is not None:
            swanlab.finish()

    cleanup_ddp()


if __name__ == "__main__":
    train()
