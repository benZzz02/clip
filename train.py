# train.py

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

from model import VLP
from pretrain_dataset import PretrainDataset


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def collate_fn_skip_corrupted(batch):
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


def clip_contrastive_loss(model, images, input_ids, attention_mask):
    """
    DDP 下的 CLIP-style 对比损失：
    - 本卡 image_features 与全局 text_features 做 CE
    - 本卡 text_features 与全局 image_features 做 CE
    """
    image_features = model.module.encode_image(images)       # [B, D]
    text_features = model.module.encode_text(input_ids, attention_mask)  # [B, D]

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    batch_size = image_features.size(0)

    gathered_image = concat_all_gather(image_features.detach())
    gathered_text = concat_all_gather(text_features.detach())

    # 保留本卡梯度
    gathered_image[rank] = image_features
    gathered_text[rank] = text_features

    all_image_features = torch.cat(gathered_image, dim=0)   # [B*W, D]
    all_text_features = torch.cat(gathered_text, dim=0)     # [B*W, D]

    logit_scale = model.module.logit_scale.exp()

    logits_per_image = logit_scale * image_features @ all_text_features.t()   # [B, B*W]
    logits_per_text = logit_scale * text_features @ all_image_features.t()    # [B, B*W]

    labels = torch.arange(batch_size, device=images.device) + rank * batch_size

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2.0

    return loss


def train():
    parser = argparse.ArgumentParser(description="VLP DDP Training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    rank = setup_ddp()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)

    # 性能开关
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    amp_dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))

    PER_GPU_BATCH_SIZE = int(os.environ.get("PER_GPU_BATCH_SIZE", 32))
    ACCUM_STEPS = int(os.environ.get("ACCUM_STEPS", 1))

    CONFIG = {
        "epochs": 20,
        "learning_rate": 1e-4,
        "weight_decay": 0.02,
        "adam_betas": (0.9, 0.999),
        "num_workers": int(os.environ.get("NUM_WORKERS", 8)),
        "embed_dim": 512,
        "image_size": 224,
        "max_length": 256,
        "text_model_name": "marcobombieri/surgicberta",
        "vision_pretrained_weights": "lemonfm.pth",
    }

    MAIN_CSV_PATH = "/mnt/mydisk/CLIP/summary_csv/all_videos.csv"
    ANNOTATIONS_FOLDER = "/mnt/mydisk/CLIP/csv_outputs"

    if rank == 0:
        print("正在初始化 VLP 模型...")

    model = VLP(
        embed_dim=CONFIG["embed_dim"],
        text_model_name=CONFIG["text_model_name"],
        vision_pretrained_weights=CONFIG["vision_pretrained_weights"],
    ).to(device)

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

    train_dataset = PretrainDataset(
        main_csv_path=MAIN_CSV_PATH,
        annotations_folder=ANNOTATIONS_FOLDER,
        tokenizer=tokenizer,
        image_size=CONFIG["image_size"],
        max_length=CONFIG["max_length"],
        sample_mode="random",
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
        collate_fn=collate_fn_skip_corrupted,
        persistent_workers=(CONFIG["num_workers"] > 0),
        prefetch_factor=4 if CONFIG["num_workers"] > 0 else None,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        betas=CONFIG["adam_betas"],
        weight_decay=CONFIG["weight_decay"],
    )

    total_update_steps = math.ceil(len(train_loader) / ACCUM_STEPS) * CONFIG["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=total_update_steps)

    writer = None
    if rank == 0:
        log_dir = os.environ.get("TB_LOGDIR", "runs/VLP")
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
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [GPU {rank}]",
            position=rank,
            disable=(rank != 0),
        )

        optimizer.zero_grad(set_to_none=True)
        step = 0
        last_loss_value = None

        for batch in progress_bar:
            has_batch = torch.tensor(0 if batch is None else 1, device=device)
            dist.all_reduce(has_batch, op=dist.ReduceOp.MIN)
            if has_batch.item() == 0:
                continue

            images, input_ids, attention_mask = [b.to(device, non_blocking=True) for b in batch]

            sync_ctx = model.no_sync() if ((step + 1) % ACCUM_STEPS != 0) else contextlib.nullcontext()

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
            print(f"Epoch {epoch+1} 完成。最后记录损失: {last_loss_value if last_loss_value is not None else 'N/A'}")

            checkpoint_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
            }
            torch.save(checkpoint_data, f"vlp_epoch_{epoch+1}.pt")

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

    cleanup_ddp()


if __name__ == "__main__":
    train()