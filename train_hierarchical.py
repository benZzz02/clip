"""
Hierarchical VLP Training with PeskaVLP-style multi-dataloader schedule.

Reuses CLIP repo patterns from train.py:
  - DDP setup/cleanup (setup_ddp, cleanup_ddp)
  - AMP autocast + GradScaler
  - AdamW + CosineAnnealingLR
  - PretrainDataset + DistributedSampler
  - VLP model (encode_image, encode_text)
  - zeroshot_evaluate for in-training evaluation

Adds PeskaVLP-style:
  - 3 dataloaders at fine/mid/coarse annotation levels
  - Frequency-modulated training: action every epoch, keystep every 3, abstract every 5
  - Triple video augmentation (SimCLR-style) + multi-text MILNCE loss
  - In-training zero-shot evaluation on surgical datasets

Usage:
    torchrun --nproc_per_node=4 train_hierarchical.py \
        --main_csv_path /data/all_videos.csv \
        --annotations_root /data/annotations \
        --video_root_folder /data/videos
"""

import os
import math
import argparse
import contextlib
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from model import VLP
from pretrain_dataset import PretrainDataset
from peska_dataset import PeskaAugmentedDataset, peska_collate_fn
from peska_losses import HierarchicalLossAction, HierarchicalLossPhase
from peska_all_gather import all_gather

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ==============================================================================
# DDP (from train.py)
# ==============================================================================

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ==============================================================================
# Per-Batch Training
# ==============================================================================

def train_action_batch(model, batch, loss_fn, world_size, rank,
                       accum_step, total_accum, embed_dim, amp_dtype):
    """Action level: triple video augmentation + InfoNCE + MILNCE + SimCLR."""
    video = batch["video"].cuda(rank, non_blocking=True)
    aug1 = batch["video_aug1"].cuda(rank, non_blocking=True)
    aug2 = batch["video_aug2"].cuda(rank, non_blocking=True)
    bs, T, C, H, W = video.shape

    # Concatenate 3 views for single forward pass
    all_vid = torch.cat([video.reshape(-1, C, H, W),
                         aug1.reshape(-1, C, H, W),
                         aug2.reshape(-1, C, H, W)], dim=0)

    p_ids = batch["primary_text"]["input_ids"].cuda(rank, non_blocking=True)
    p_mask = batch["primary_text"]["attention_mask"].cuda(rank, non_blocking=True)
    c_ids = batch["candidate_texts"]["input_ids"].cuda(rank, non_blocking=True)
    c_mask = batch["candidate_texts"]["attention_mask"].cuda(rank, non_blocking=True)
    n_cand = batch["num_candidates"]

    sync_ctx = model.no_sync() if (accum_step + 1) % total_accum != 0 else contextlib.nullcontext()

    with sync_ctx:
        with autocast(dtype=amp_dtype):
            emb_all = model.module.encode_image(all_vid)
            bt = bs * T
            v_emb = emb_all[:bt].view(bs, T, -1).mean(1) if T > 1 else emb_all[:bt]
            a1_emb = emb_all[bt:2*bt].view(bs, T, -1).mean(1) if T > 1 else emb_all[bt:2*bt]
            a2_emb = emb_all[2*bt:].view(bs, T, -1).mean(1) if T > 1 else emb_all[2*bt:]

            t_emb = model.module.encode_text(p_ids, p_mask)
            c_emb = model.module.encode_text(c_ids, c_mask) if n_cand > 0 and c_ids.shape[0] > 0 \
                else torch.zeros(0, embed_dim, device=video.device)

            if world_size > 1:
                v_emb = all_gather(v_emb, world_size, rank)
                a1_emb = all_gather(a1_emb, world_size, rank)
                a2_emb = all_gather(a2_emb, world_size, rank)
                t_emb = all_gather(t_emb, world_size, rank)
                if c_emb.shape[0] > 0:
                    c_emb = all_gather(c_emb, world_size, rank)

            logit_scale = model.module.logit_scale.exp() if hasattr(model.module, 'logit_scale') else None
            loss = loss_fn(v_emb, a1_emb, a2_emb, t_emb, c_emb, logit_scale) / total_accum

        return loss


def train_phase_batch(model, batch, loss_fn, world_size, rank,
                      accum_step, total_accum, embed_dim, amp_dtype):
    """Keystep/Abstract level: single video + InfoNCE + cross-text alignment."""
    video = batch["video"].cuda(rank, non_blocking=True)
    bs, T, C, H, W = video.shape

    p_ids = batch["primary_text"]["input_ids"].cuda(rank, non_blocking=True)
    p_mask = batch["primary_text"]["attention_mask"].cuda(rank, non_blocking=True)
    c_ids = batch["candidate_texts"]["input_ids"].cuda(rank, non_blocking=True)
    c_mask = batch["candidate_texts"]["attention_mask"].cuda(rank, non_blocking=True)
    n_cand = batch["num_candidates"]

    sync_ctx = model.no_sync() if (accum_step + 1) % total_accum != 0 else contextlib.nullcontext()

    with sync_ctx:
        with autocast(dtype=amp_dtype):
            v_emb = model.module.encode_image(video.reshape(-1, C, H, W))
            v_emb = v_emb.view(bs, T, -1).mean(1) if T > 1 else v_emb
            t_emb = model.module.encode_text(p_ids, p_mask)
            c_emb = model.module.encode_text(c_ids, c_mask) if n_cand > 0 and c_ids.shape[0] > 0 \
                else torch.zeros(0, embed_dim, device=video.device)

            if world_size > 1:
                v_emb = all_gather(v_emb, world_size, rank)
                t_emb = all_gather(t_emb, world_size, rank)
                if c_emb.shape[0] > 0:
                    c_emb = all_gather(c_emb, world_size, rank)

            if n_cand > 0 and c_emb.shape[0] > 0:
                c_emb = c_emb.view(t_emb.shape[0], n_cand, -1)

            logit_scale = model.module.logit_scale.exp() if hasattr(model.module, 'logit_scale') else None
            loss = loss_fn(v_emb, t_emb, c_emb, logit_scale) / total_accum

        return loss


# ==============================================================================
# Training Loop
# ==============================================================================

def train_epoch(loader, model, loss_fn, optimizer, scheduler, scaler,
                args, rank, world_size, epoch, name, train_fn):
    """Train one epoch on a single dataloader."""
    model.train()
    running_loss = 0.0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc=f"E{epoch+1:02d} {name:8s}",
                                      disable=(rank != 0), leave=False)):
        if batch is None:
            continue

        loss = train_fn(model, batch, loss_fn, world_size, rank,
                        step, args.accum_steps, args.embed_dim, args.amp_dtype)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.accum_steps == 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running_loss += loss.item() * args.accum_steps

        if rank == 0 and (step + 1) % args.log_interval == 0:
            avg = running_loss / args.log_interval
            lr = scheduler.get_last_lr()[0]
            if args.writer:
                gs = epoch * len(loader) + step
                args.writer.add_scalar(f"train/{name}_loss", avg, gs)
                args.writer.add_scalar(f"train/{name}_lr", lr, gs)
            running_loss = 0.0


# ==============================================================================
# Cosine warmup scheduler (from PeskaVLP: get_cosine_schedule_with_warmup)
# ==============================================================================

def get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay, same as PeskaVLP."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ==============================================================================
# In-Training Eval (reuses zeroshot_evaluate.py)
# ==============================================================================

def run_eval(model, tokenizer, epoch, args, device):
    """Run zero-shot evaluation using existing zeroshot_evaluate infrastructure."""
    from zeroshot_evaluate import build_dataloader, evaluation_wrapper

    model.eval()
    datasets = [d.strip() for d in args.eval_datasets.split(",") if d.strip()]

    # Add triplet eval if requested
    if args.eval_triplet:
        datasets.append("cholect50_triplet")

    for ds in datasets:
        try:
            loader, _ = build_dataloader(ds, batch_size=16, num_workers=2,
                                         num_frames=args.num_frames)
            loader.dataset.name = ds
            out = os.path.join(args.output_dir, f"eval_epoch_{epoch+1}")
            result = evaluation_wrapper(model.module, loader, tokenizer, device,
                                        out, ds, args.num_frames)
            if args.writer and result:
                for k, v in result.items():
                    if isinstance(v, (int, float)):
                        args.writer.add_scalar(f"eval/{ds}/{k}", v, epoch)
            if dist.get_rank() == 0:
                print(f"  [{ds}] OK")
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"  [{ds}] eval error: {e}")
    model.train()


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical VLP Training")

    # Model (from train.py)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--text_model_name", type=str, default="marcobombieri/surgicberta")
    p.add_argument("--vision_pretrained_weights", type=str, default="lemonfm.pth")
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--max_length", type=int, default=256)

    # Data (from train.py)
    p.add_argument("--main_csv_path", type=str, required=True)
    p.add_argument("--annotations_root", type=str, default=None)
    p.add_argument("--annotations_folder", type=str, default=None)
    p.add_argument("--video_root_folder", type=str, required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)

    # Training (from train.py)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size_action", type=int, default=32)
    p.add_argument("--batch_size_keystep", type=int, default=16)
    p.add_argument("--batch_size_abstract", type=int, default=8)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.02)

    # Schedule (PeskaVLP-style)
    p.add_argument("--action_every", type=int, default=1)
    p.add_argument("--keystep_every", type=int, default=3)
    p.add_argument("--abstract_every", type=int, default=5)

    # Loss
    p.add_argument("--temperature", type=float, default=0.1)

    # Augmentation
    p.add_argument("--no_aug", action="store_true")
    p.add_argument("--aug_scale_min", type=float, default=0.2)

    # Multi-text
    p.add_argument("--multi_text", type=int, default=4)

    # Eval
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_datasets", type=str, default="cholec80_phase")

    # Logging (from train.py)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--output_dir", type=str, default="runs/hierarchical_vlp")
    p.add_argument("--warmup_steps", type=int, default=500,
                        help="Linear warmup steps")
    p.add_argument("--eval_triplet", action="store_true",
                        help="Also run triplet evaluation during training")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--use_compile", action="store_true")

    args = p.parse_args()
    if args.annotations_root is None:
        args.annotations_root = args.annotations_folder
    return args


def train():
    args = parse_args()
    rank = setup_ddp()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)

    # Performance (from train.py)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    args.amp_dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16

    # TensorBoard (from train.py)
    args.writer = None
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

    # ---- Model (from train.py) ----
    if rank == 0:
        print("Building VLP model...")
    model = VLP(embed_dim=args.embed_dim, text_model_name=args.text_model_name,
                vision_pretrained_weights=args.vision_pretrained_weights,
                num_frames=args.num_frames).to(device)

    if args.use_compile:
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception:
            pass

    model = DDP(model, device_ids=[rank], find_unused_parameters=False,
                broadcast_buffers=False, gradient_as_bucket_view=True)

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)

    # ---- Datasets (reuses PretrainDataset) ----
    if rank == 0:
        print("Building datasets...")

    # Shared kwargs for PretrainDataset
    ds_kwargs = dict(
        main_csv_path=args.main_csv_path,
        annotations_folder=args.annotations_folder,
        annotations_root=args.annotations_root,
        tokenizer=tokenizer,
        image_size=args.image_size,
        max_length=args.max_length,
        video_root_folder=args.video_root_folder,
        num_frames=args.num_frames,
        return_level_id=True,
    )

    def make_loader(level, batch_size, use_aug):
        base = PretrainDataset(annotation_levels=[level], **ds_kwargs)
        ds = PeskaAugmentedDataset(
            base, [level],
            main_csv_path=args.main_csv_path,
            annotations_root=args.annotations_root,
            annotations_folder=args.annotations_folder,
            video_root_folder=args.video_root_folder,
            use_augmentation=use_aug,
            augmentation_kwargs={"size": args.image_size,
                                 "scale": (args.aug_scale_min, 1.0)} if use_aug else None,
            multi_text_candidates=args.multi_text,
        )
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True, collate_fn=peska_collate_fn)

    action_loader = make_loader("fine", args.batch_size_action, not args.no_aug)
    keystep_loader = make_loader("mid", args.batch_size_keystep, False)
    abstract_loader = make_loader("coarse", args.batch_size_abstract, False)

    if rank == 0:
        print(f"  Action:   {len(action_loader.dataset)}")
        print(f"  Keystep:  {len(keystep_loader.dataset)}")
        print(f"  Abstract: {len(abstract_loader.dataset)}")

    # ---- Losses ----
    action_loss = HierarchicalLossAction(temperature=args.temperature)
    phase_loss = HierarchicalLossPhase(temperature=args.temperature)

    # ---- Optimizer & Scheduler (from train.py) ----
    optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                      weight_decay=args.weight_decay, betas=(0.9, 0.999))

    acts = math.ceil(len(action_loader) / args.accum_steps)
    keys = math.ceil(len(keystep_loader) / args.accum_steps) * max(0, (args.epochs - 1) // args.keystep_every)
    abss = math.ceil(len(abstract_loader) / args.accum_steps) * max(0, (args.epochs - 1) // args.abstract_every)
    scheduler = get_cosine_warmup_scheduler(
        optimizer, warmup_steps=args.warmup_steps,
        total_steps=acts * args.epochs + keys + abss,
    )
    scaler = GradScaler(enabled=(args.amp_dtype == torch.float16))

    # ---- Resume (from train.py) ----
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        if rank == 0:
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]

    # ---- Training Loop ----
    loaders = [
        (action_loader, action_loss, train_action_batch, "action", args.action_every),
        (keystep_loader, phase_loss, train_phase_batch, "keystep", args.keystep_every),
        (abstract_loader, phase_loss, train_phase_batch, "abstract", args.abstract_every),
    ]

    if rank == 0:
        print(f"\nTraining {args.epochs} epochs; eval every {args.eval_every}\n")

    for epoch in range(start_epoch, args.epochs):
        for loader, loss_fn, train_fn, name, freq in loaders:
            loader.sampler.set_epoch(epoch)
            if epoch % freq == 0 and epoch >= freq:
                train_epoch(loader, model, loss_fn, optimizer, scheduler, scaler,
                            args, rank, world_size, epoch, name, train_fn)

        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0 and epoch > 0:
            if rank == 0:
                print(f"\n  Eval @ epoch {epoch+1}")
            run_eval(model, tokenizer, epoch, args, device)

        # Checkpoint (from train.py pattern)
        if rank == 0:
            torch.save({"epoch": epoch + 1,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()},
                       os.path.join(args.output_dir, f"epoch_{epoch+1}.pt"))
            if args.writer:
                args.writer.flush()

    if rank == 0:
        torch.save({"epoch": args.epochs,
                    "model_state_dict": model.module.state_dict()},
                   os.path.join(args.output_dir, "final.pt"))
        print(f"\nDone: {args.output_dir}")
        if args.writer:
            args.writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    train()
