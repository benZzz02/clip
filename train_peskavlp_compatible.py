import argparse
import contextlib
import math
import os
import time

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import swanlab
except ImportError:
    swanlab = None

from model import VLP
from peskavlp_compatible_dataset import (
    PeskaVLPCompatibleDataset,
    collate_peskavlp_compatible,
)
from peskavlp_compatible_loss import PeskaVLPCompatibleLoss


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_float_list(spec, expected_len=None):
    values = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if expected_len is not None and len(values) != expected_len:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_len} comma-separated floats, got {len(values)} from: {spec}"
        )
    return tuple(values)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PeskaVLP-compatible hierarchy loop for the local VLP model"
    )

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=6)

    parser.add_argument("--per_gpu_batch_size", type=int, default=64)
    parser.add_argument("--action_batch_size", type=int, default=None)
    parser.add_argument("--mid_batch_size", type=int, default=None)
    parser.add_argument("--coarse_batch_size", type=int, default=None)

    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--text_model_name", type=str, default="marcobombieri/surgicberta")
    parser.add_argument(
        "--vision_backbone",
        type=str,
        default="convnext_lemonfm",
        choices=["convnext_lemonfm", "gsvit_m5"],
    )
    parser.add_argument("--vision_pretrained_weights", type=str, default="lemonfm.pth")

    parser.add_argument(
        "--video_root_folder",
        type=str,
        default="/data/surglavi_video/downloaded_video_224_test",
    )
    parser.add_argument("--main_csv_path", type=str, default="surglavi_level_csv/all_video.csv")
    parser.add_argument("--annotations_root", type=str, default="surglavi_level_csv")
    parser.add_argument("--annotation_levels", type=str, default="coarse,mid,fine")
    parser.add_argument("--level_mix", type=str, default="concat", choices=["concat", "balanced"])

    parser.add_argument("--ffmpeg_timeout", type=int, default=10)
    parser.add_argument("--max_retry", type=int, default=5)
    parser.add_argument("--video_reader_threads", type=int, default=1)
    parser.add_argument("--video_reader_cache_size", type=int, default=1)
    parser.add_argument("--assume_resized_video", type=str2bool, default=True)

    parser.add_argument("--samples_cache_dir", type=str, default=".cache/pretrain_samples")
    parser.add_argument("--use_samples_cache", type=str2bool, default=True)
    parser.add_argument("--rebuild_samples_cache", type=str2bool, default=False)
    parser.add_argument("--samples_cache_version", type=str, default="v1")

    parser.add_argument("--max_candidates", type=int, default=8)
    parser.add_argument("--mid_interval", type=int, default=3)
    parser.add_argument("--coarse_interval", type=int, default=5)
    parser.add_argument("--action_view_flip_prob", type=float, default=0.5)
    parser.add_argument("--action_view_noise_prob", type=float, default=0.25)
    parser.add_argument("--action_view_noise_std", type=float, default=0.01)

    parser.add_argument("--local_temperature", type=float, default=0.07)
    parser.add_argument("--selection_pooling", type=str, default="similarity", choices=["similarity", "xpool"])
    parser.add_argument(
        "--level_frame_temperatures",
        type=lambda value: parse_float_list(value, expected_len=3),
        default=(0.6, 0.9, 1.2),
    )

    parser.add_argument("--use_swanlab", type=str2bool, default=True)
    parser.add_argument("--peskavlp_temperature", type=float, default=0.1)
    parser.add_argument("--peskavlp_alpha_weight", type=float, default=0.75)
    parser.add_argument("--peskavlp_dtw_beta", type=float, default=0.0)
    parser.add_argument("--peskavlp_dtw_ratio", type=float, default=0.5)
    parser.add_argument("--peskavlp_dtw_scale_factor", type=float, default=0.01)

    return parser.parse_args()


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _unwrap_state_io_module(module):
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def _normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod.") :]
        normalized[key] = value
    return normalized


def _export_plain_state_dict_from_ddp(model):
    return _unwrap_state_io_module(model.module).state_dict()


def _load_normalized_state_dict(module, state_dict, source="checkpoint"):
    target_module = _unwrap_state_io_module(module)
    normalized_state_dict = _normalize_state_dict_keys(state_dict)
    msg = target_module.load_state_dict(normalized_state_dict, strict=False)
    if dist.get_rank() == 0 and (msg.missing_keys or msg.unexpected_keys):
        print(
            f"{source} loaded with missing={msg.missing_keys} unexpected={msg.unexpected_keys}",
            flush=True,
        )
    return msg


def _move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _reduce_mean(value):
    value = value.detach().float()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value


def active_levels_for_epoch(epoch, mid_interval, coarse_interval):
    levels = ["fine"]
    if mid_interval > 0 and epoch != 0 and epoch % mid_interval == 0:
        levels.append("mid")
    if coarse_interval > 0 and epoch != 0 and epoch % coarse_interval == 0:
        levels.append("coarse")
    return levels


def make_loader(dataset, batch_size, num_workers, world_size, rank, seed):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "sampler": sampler,
        "drop_last": True,
        "num_workers": int(num_workers),
        "pin_memory": True,
        "collate_fn": collate_peskavlp_compatible,
        "persistent_workers": int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs), sampler


def estimate_total_update_steps(loaders, args):
    total = 0
    for epoch in range(args.epochs):
        for level_name in active_levels_for_epoch(
            epoch,
            args.mid_interval,
            args.coarse_interval,
        ):
            total += math.ceil(len(loaders[level_name]) / max(1, args.accum_steps))
    return max(1, total)


def train_one_level(
    level_name,
    loader,
    sampler,
    model,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    global_step,
    args,
    writer,
    rank,
    amp_dtype,
):
    if len(loader) == 0:
        if rank == 0:
            print(
                f"Skipping level={level_name}: dataloader is empty. "
                "Lower the level batch size or check dataset size.",
                flush=True,
            )
        return global_step, None

    level_offset = {"fine": 0, "mid": 1, "coarse": 2}[level_name]
    sampler.set_epoch(epoch * 10 + level_offset)
    _unwrap_state_io_module(model.module).set_frozen_modules_eval()

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{args.epochs} {level_name} [GPU {rank}]",
        disable=(rank != 0),
    )

    optimizer.zero_grad(set_to_none=True)
    accum_loss_sum = torch.zeros((), device=device)
    accum_data_time_sum = 0.0
    accum_step_time_sum = 0.0
    last_loss_value = None
    batch_fetch_start = time.perf_counter()

    for step, raw_batch in enumerate(progress):
        data_time = time.perf_counter() - batch_fetch_start
        if raw_batch is None:
            batch_fetch_start = time.perf_counter()
            continue

        batch = _move_batch(raw_batch, device)
        step_start = time.perf_counter()
        micro_step = (step % args.accum_steps) + 1
        is_last_batch = (step + 1) == len(loader)
        current_accum_steps = (
            micro_step
            if is_last_batch and micro_step != args.accum_steps
            else args.accum_steps
        )
        should_update = micro_step == args.accum_steps or is_last_batch
        sync_ctx = model.no_sync() if not should_update else contextlib.nullcontext()

        with sync_ctx:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                if level_name == "fine":
                    raw_loss, loss_parts = criterion.forward_action(
                        model=model,
                        video=batch["video"],
                        video_aug1=batch["video_aug1"],
                        video_aug2=batch["video_aug2"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        candidate_input_ids=batch["candidate_input_ids"],
                        candidate_attention_mask=batch["candidate_attention_mask"],
                    )
                else:
                    raw_loss, loss_parts = criterion.forward_hierarchy(
                        model=model,
                        level_name=level_name,
                        video=batch["video"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        candidate_input_ids=batch["candidate_input_ids"],
                        candidate_attention_mask=batch["candidate_attention_mask"],
                        pos_step=batch["pos_step"],
                    )
                loss = raw_loss / current_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        accum_loss_sum += raw_loss.detach()
        accum_data_time_sum += data_time
        accum_step_time_sum += time.perf_counter() - step_start

        if should_update:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            window_loss = accum_loss_sum / current_accum_steps
            loss_avg = _reduce_mean(window_loss).item()
            data_time_avg = _reduce_mean(
                torch.tensor(accum_data_time_sum / current_accum_steps, device=device)
            ).item()
            step_time_avg = _reduce_mean(
                torch.tensor(accum_step_time_sum / current_accum_steps, device=device)
            ).item()
            last_loss_value = loss_avg
            accum_loss_sum.zero_()
            accum_data_time_sum = 0.0
            accum_step_time_sum = 0.0

            if writer is not None:
                writer.add_scalar(f"train/{level_name}_loss", loss_avg, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/epoch", epoch + 1, global_step)
                writer.add_scalar(f"perf/{level_name}_data_time", data_time_avg, global_step)
                writer.add_scalar(f"perf/{level_name}_step_time", step_time_avg, global_step)
                for stat_name, stat_value in loss_parts.items():
                    writer.add_scalar(stat_name, stat_value.item(), global_step)

            global_step += 1

        if rank == 0:
            progress.set_postfix(
                loss=f"{raw_loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                data=f"{data_time:.2f}s",
            )

        del batch
        batch_fetch_start = time.perf_counter()

    return global_step, last_loss_value


def build_datasets(args, tokenizer):
    common_kwargs = {
        "main_csv_path": args.main_csv_path,
        "annotations_folder": None,
        "annotations_root": args.annotations_root,
        "annotation_levels": args.annotation_levels,
        "level_mix": args.level_mix,
        "tokenizer": tokenizer,
        "image_size": args.image_size,
        "max_length": args.max_length,
        "sample_mode": "random",
        "ffmpeg_timeout": args.ffmpeg_timeout,
        "max_retry": args.max_retry,
        "video_root_folder": args.video_root_folder,
        "assume_resized_video": args.assume_resized_video,
        "num_frames": args.num_frames,
        "samples_cache_dir": args.samples_cache_dir,
        "use_samples_cache": args.use_samples_cache,
        "rebuild_samples_cache": args.rebuild_samples_cache,
        "samples_cache_version": args.samples_cache_version,
        "video_reader_threads": args.video_reader_threads,
        "video_reader_cache_size": args.video_reader_cache_size,
    }
    return {
        "fine": PeskaVLPCompatibleDataset(
            target_level="fine",
            max_candidates=args.max_candidates,
            action_view_flip_prob=args.action_view_flip_prob,
            action_view_noise_prob=args.action_view_noise_prob,
            action_view_noise_std=args.action_view_noise_std,
            **common_kwargs,
        ),
        "mid": PeskaVLPCompatibleDataset(
            target_level="mid",
            max_candidates=args.max_candidates,
            **common_kwargs,
        ),
        "coarse": PeskaVLPCompatibleDataset(
            target_level="coarse",
            max_candidates=args.max_candidates,
            **common_kwargs,
        ),
    }


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

    action_batch_size = args.action_batch_size or args.per_gpu_batch_size
    mid_batch_size = args.mid_batch_size or args.per_gpu_batch_size
    coarse_batch_size = args.coarse_batch_size or args.per_gpu_batch_size

    amp_dtype = (
        torch.bfloat16
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        else torch.float16
    )
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))

    model = VLP(
        embed_dim=args.embed_dim,
        text_model_name=args.text_model_name,
        vision_backbone=args.vision_backbone,
        vision_pretrained_weights=args.vision_pretrained_weights,
        num_frames=args.num_frames,
        local_temperature=args.local_temperature,
        selection_pooling=args.selection_pooling,
        level_frame_temperatures=args.level_frame_temperatures,
    ).to(device)
    model.freeze_encoders_train_projections()
    model.set_frozen_modules_eval()

    model = DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        bucket_cap_mb=int(os.environ.get("DDP_BUCKET_MB", 64)),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    datasets = build_datasets(args, tokenizer)
    loaders = {}
    samplers = {}
    batch_sizes = {
        "fine": action_batch_size,
        "mid": mid_batch_size,
        "coarse": coarse_batch_size,
    }
    for idx, level_name in enumerate(("fine", "mid", "coarse")):
        loaders[level_name], samplers[level_name] = make_loader(
            datasets[level_name],
            batch_size=batch_sizes[level_name],
            num_workers=args.num_workers,
            world_size=world_size,
            rank=rank,
            seed=42 + idx,
        )

    if rank == 0:
        print("PeskaVLP-compatible training setup", flush=True)
        print(f"world_size={world_size}", flush=True)
        print(
            "dataset sizes: "
            + ", ".join(f"{level}={len(dataset)}" for level, dataset in datasets.items()),
            flush=True,
        )
        print(
            "loader batches: "
            + ", ".join(f"{level}={len(loader)}" for level, loader in loaders.items()),
            flush=True,
        )
        print(
            "batch sizes: "
            + ", ".join(f"{level}={batch_sizes[level]}" for level in ("fine", "mid", "coarse")),
            flush=True,
        )
        print(
            f"schedule: fine every epoch, mid every {args.mid_interval}, "
            f"coarse every {args.coarse_interval}",
            flush=True,
        )

    if len(loaders["fine"]) == 0:
        raise ValueError(
            "fine/action dataloader is empty. Lower ACTION_BATCH_SIZE/PER_GPU_BATCH_SIZE "
            "or check the fine-level dataset."
        )

    criterion = PeskaVLPCompatibleLoss(
        temperature=args.peskavlp_temperature,
        alpha_weight=args.peskavlp_alpha_weight,
        dtw_beta=args.peskavlp_dtw_beta,
        dtw_ratio=args.peskavlp_dtw_ratio,
        dtw_scale_factor=args.peskavlp_dtw_scale_factor,
    ).to(device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    total_update_steps = estimate_total_update_steps(loaders, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_update_steps)

    writer = None
    swanlab_run = None
    if rank == 0:
        log_dir = os.environ.get("TB_LOGDIR", "runs/VLP_peskavlp_compatible")
        writer = SummaryWriter(log_dir=log_dir)

        if args.use_swanlab:
            if swanlab is None:
                print("swanlab is not installed; skipping SwanLab logging.", flush=True)
            else:
                swanlab_config = vars(args).copy()
                swanlab_config.update(
                    {
                        "world_size": world_size,
                        "action_batch_size": action_batch_size,
                        "mid_batch_size": mid_batch_size,
                        "coarse_batch_size": coarse_batch_size,
                        "total_update_steps": total_update_steps,
                        "tb_logdir": log_dir,
                    }
                )
                swanlab_kwargs = {
                    "project": os.environ.get("SWANLAB_PROJECT", "CLIP"),
                    "experiment_name": os.environ.get(
                        "SWANLAB_EXPERIMENT_NAME",
                        "VLP_peskavlp_compatible",
                    ),
                    "config": swanlab_config,
                    "logdir": os.environ.get(
                        "SWANLAB_LOGDIR",
                        "swanlog/VLP_peskavlp_compatible",
                    ),
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

    save_prefix = os.environ.get("SAVE_PREFIX", "")
    save_dir = os.path.dirname(save_prefix)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def build_ckpt_path(filename):
        if not save_prefix:
            return filename
        if save_prefix.endswith(os.sep) or save_prefix.endswith("/"):
            return os.path.join(save_prefix, filename)
        return f"{save_prefix}{filename}"

    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}", flush=True)
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        _load_normalized_state_dict(
            model.module,
            checkpoint["model_state_dict"],
            source=args.resume_from_checkpoint,
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        scaler_state_dict = checkpoint.get("scaler_state_dict")
        if scaler_state_dict is not None and scaler.is_enabled():
            scaler.load_state_dict(scaler_state_dict)

    for epoch in range(start_epoch, args.epochs):
        levels = active_levels_for_epoch(
            epoch,
            args.mid_interval,
            args.coarse_interval,
        )
        if rank == 0:
            print(f"Epoch {epoch + 1}: active levels={','.join(levels)}", flush=True)

        epoch_losses = {}
        for level_name in levels:
            global_step, last_loss = train_one_level(
                level_name=level_name,
                loader=loaders[level_name],
                sampler=samplers[level_name],
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                epoch=epoch,
                global_step=global_step,
                args=args,
                writer=writer,
                rank=rank,
                amp_dtype=amp_dtype,
            )
            epoch_losses[level_name] = last_loss

        if rank == 0:
            print(
                f"Epoch {epoch + 1} complete: "
                + ", ".join(f"{level}={loss}" for level, loss in epoch_losses.items()),
                flush=True,
            )
            checkpoint_data = {
                "epoch": epoch + 1,
                "model_state_dict": _export_plain_state_dict_from_ddp(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "args": vars(args),
            }
            epoch_ckpt_path = build_ckpt_path(f"vlp_peskavlp_compatible_epoch_{epoch + 1}.pt")
            torch.save(checkpoint_data, epoch_ckpt_path)
            print(f"Saved checkpoint: {epoch_ckpt_path}", flush=True)
            if writer is not None:
                writer.flush()

    if rank == 0:
        final_checkpoint_data = {
            "epoch": args.epochs,
            "model_state_dict": _export_plain_state_dict_from_ddp(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "args": vars(args),
        }
        final_ckpt_path = build_ckpt_path("vlp_peskavlp_compatible_final.pt")
        torch.save(final_checkpoint_data, final_ckpt_path)
        print(f"Saved final checkpoint: {final_ckpt_path}", flush=True)

        if writer is not None:
            writer.close()
        if swanlab_run is not None:
            swanlab.finish()

    cleanup_ddp()


if __name__ == "__main__":
    train()
