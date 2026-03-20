# train_frozen_vis.py

import os
import math
import argparse
import contextlib

import torch
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
import swanlab

from model import VLP
from pretrain_dataset import PretrainDataset


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


def clip_contrastive_loss(model, images, input_ids, attention_mask):
    image_features = model.module.encode_image(images)
    text_features = model.module.encode_text(input_ids, attention_mask)

    rank = dist.get_rank()
    batch_size = image_features.size(0)

    gathered_image = concat_all_gather(image_features.detach())
    gathered_text = concat_all_gather(text_features.detach())

    gathered_image[rank] = image_features
    gathered_text[rank] = text_features

    all_image_features = torch.cat(gathered_image, dim=0)
    all_text_features = torch.cat(gathered_text, dim=0)

    logit_scale = model.module.logit_scale.exp()

    logits_per_image = logit_scale * image_features @ all_text_features.t()
    logits_per_text = logit_scale * text_features @ all_image_features.t()

    labels = torch.arange(batch_size, device=images.device) + rank * batch_size

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2.0


def train():
    parser = argparse.ArgumentParser(description="VLP Frozen-Visual DDP Training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

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

    PER_GPU_BATCH_SIZE = int(os.environ.get("PER_GPU_BATCH_SIZE", 16))
    ACCUM_STEPS = int(os.environ.get("ACCUM_STEPS", 1))

    CONFIG = {
        "epochs": 20,
        "learning_rate": 1e-4,
        "weight_decay": 0.02,
        "adam_betas": (0.9, 0.999),
        "num_workers": int(os.environ.get("NUM_WORKERS", 2)),
        "embed_dim": 512,
        "image_size": 224,
        "max_length": 256,
        "text_model_name": "marcobombieri/surgicberta",
        "vision_pretrained_weights": "lemonfm.pth",
        "video_root_folder": os.environ.get(
            "PRETRAIN_VIDEO_ROOT_FOLDER",
            "/mnt/mydisk/CLIP/downloaded_video_224_test",
        ),
        "ffmpeg_timeout": 10,
        "max_retry": 5,
        "assume_resized_video": os.environ.get(
            "PRETRAIN_VIDEO_ALREADY_RESIZED",
            "0",
        ) == "1",
        "num_frames": int(os.environ.get("NUM_FRAMES", 4)),
    }

    MAIN_CSV_PATH = os.environ.get(
        "PRETRAIN_MAIN_CSV_PATH",
        "/mnt/mydisk/CLIP/surglavi_level_csv/all_video.csv",
    )
    ANNOTATIONS_FOLDER = os.environ.get(
        "PRETRAIN_ANNOTATIONS_FOLDER",
        "/mnt/mydisk/CLIP/surglavi_level_csv/fine",
    )

    if rank == 0:
        print("正在初始化 VLP 模型...")

    model = VLP(
        embed_dim=CONFIG["embed_dim"],
        text_model_name=CONFIG["text_model_name"],
        vision_pretrained_weights=CONFIG["vision_pretrained_weights"],
        num_frames=CONFIG["num_frames"],
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

    train_dataset = PretrainDataset(
        main_csv_path=MAIN_CSV_PATH,
        annotations_folder=ANNOTATIONS_FOLDER,
        tokenizer=tokenizer,
        image_size=CONFIG["image_size"],
        max_length=CONFIG["max_length"],
        sample_mode="random",
        ffmpeg_timeout=CONFIG["ffmpeg_timeout"],
        max_retry=CONFIG["max_retry"],
        video_root_folder=CONFIG["video_root_folder"],
        assume_resized_video=CONFIG["assume_resized_video"],
        num_frames=CONFIG["num_frames"],
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=PER_GPU_BATCH_SIZE,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        persistent_workers=(CONFIG["num_workers"] > 0),
        prefetch_factor=2 if CONFIG["num_workers"] > 0 else None,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=CONFIG["learning_rate"],
        betas=CONFIG["adam_betas"],
        weight_decay=CONFIG["weight_decay"],
    )

    total_update_steps = math.ceil(len(train_loader) / ACCUM_STEPS) * CONFIG["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=total_update_steps)

    writer = None
    swanlab_run = None
    if rank == 0:
        log_dir = os.environ.get("TB_LOGDIR", "runs/VLP_frozen_vis")

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

        writer = SummaryWriter(log_dir=log_dir)

    start_epoch = 0
    global_step = 0

    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        if rank == 0:
            print(f"正在从检查点恢复训练: {args.resume_from_checkpoint}")

        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)

        model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]

        if rank == 0:
            print(f"恢复成功，将从 epoch {start_epoch + 1} 开始。")

    if rank == 0:
        print("配置完成，开始训练...")

    for epoch in range(start_epoch, CONFIG["epochs"]):
        train_sampler.set_epoch(epoch)
        model.module.set_frozen_modules_eval()

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [GPU {rank}]",
            position=rank,
            disable=(rank != 0),
        )

        optimizer.zero_grad(set_to_none=True)
        step = 0
        last_loss_value = None

        for batch in progress_bar:
            images, input_ids, attention_mask = [b.to(device, non_blocking=True) for b in batch]

            sync_ctx = (
                model.no_sync()
                if ((step + 1) % ACCUM_STEPS != 0)
                else contextlib.nullcontext()
            )

            with sync_ctx:
                with autocast(dtype=amp_dtype):
                    loss = clip_contrastive_loss(model, images, input_ids, attention_mask)
                    loss = loss / ACCUM_STEPS

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            did_step = False
            if (step + 1) % ACCUM_STEPS == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                did_step = True

            if did_step:
                full_loss = (loss.detach() * ACCUM_STEPS).to(device)
                loss_log = full_loss.clone()
                dist.all_reduce(loss_log, op=dist.ReduceOp.SUM)
                loss_avg = loss_log.item() / world_size
                last_loss_value = loss_avg

                if writer is not None:
                    writer.add_scalar("train/loss", loss_avg, global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    writer.add_scalar("train/epoch", epoch + 1, global_step)

                global_step += 1

            if rank == 0:
                progress_bar.set_postfix(
                    loss=(loss.item() * ACCUM_STEPS),
                    lr=scheduler.get_last_lr()[0],
                )

            del images, input_ids, attention_mask
            step += 1

        if rank == 0:
            print(
                f"Epoch {epoch + 1} 完成。最后记录损失: "
                f"{last_loss_value if last_loss_value is not None else 'N/A'}"
            )

            checkpoint_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
            }
            torch.save(checkpoint_data, f"vlp_epoch_{epoch + 1}.pt")

            if writer is not None:
                writer.flush()

    if rank == 0:
        print("训练完成。")
        final_checkpoint_data = {
            "epoch": CONFIG["epochs"],
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
        }
        torch.save(final_checkpoint_data, "vlp_final.pt")

        if writer is not None:
            writer.close()

        if swanlab_run is not None:
            swanlab.finish()

    cleanup_ddp()


if __name__ == "__main__":
    train()
