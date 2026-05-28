"""
Hierarchical Vision-Language Pretraining with PeskaVLP-style Training Flow.

Integrates PeskaVLP's multi-dataloader hierarchical training schedule,
triple video augmentation, multi-text candidates, and in-training zero-shot
evaluation into the CLIP repo's VLP model.

Key features:
  - 3 dataloaders at different annotation levels (fine/mid/coarse)
  - Action level: triple video augmentation + InfoNCE/NTXent/MILNCE/SimCLR losses
  - Keystep/Abstract levels: single video + InfoNCE/NTXent/cross-text losses
  - Hierarchical training schedule (configurable frequency per level)
  - Gradient-preserving AllGather for distributed contrastive learning
  - In-training zero-shot evaluation on surgical datasets

Usage:
    torchrun --nproc_per_node=4 -m train_hierarchical \
        --main_csv_path /path/to/all_videos.csv \
        --annotations_root /path/to/annotations \
        --video_root_folder /path/to/videos \
        --epochs 50
"""

import os
import math
import argparse
import contextlib
import time

import torch
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
from peska_dataset import PeskaAugmentedPretrainDataset, peska_collate_fn
from peska_losses import HierarchicalLossAction, HierarchicalLossPhase
from peska_all_gather import all_gather

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ==============================================================================
# DDP Utilities
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
# Per-Batch Training Functions
# ==============================================================================

def train_action_batch(model, batch, loss_fn, optimizer, scheduler,
                       scaler, args, rank, world_size, accum_step,
                       total_accum_steps):
    """Train one action-level (fine) batch with triple augmentation."""
    video = batch["video"].cuda(rank, non_blocking=True)
    aug1 = batch["video_aug1"].cuda(rank, non_blocking=True)
    aug2 = batch["video_aug2"].cuda(rank, non_blocking=True)

    bs, T, C, H, W = video.shape

    # Flatten and concatenate all 3 views
    video_flat = video.reshape(-1, C, H, W)
    aug1_flat = aug1.reshape(-1, C, H, W)
    aug2_flat = aug2.reshape(-1, C, H, W)
    all_video = torch.cat([video_flat, aug1_flat, aug2_flat], dim=0)

    primary_ids = batch["primary_text"]["input_ids"].cuda(rank, non_blocking=True)
    primary_mask = batch["primary_text"]["attention_mask"].cuda(rank, non_blocking=True)

    cand_ids = batch["candidate_texts"]["input_ids"].cuda(rank, non_blocking=True)
    cand_mask = batch["candidate_texts"]["attention_mask"].cuda(rank, non_blocking=True)

    sync_ctx = (
        model.no_sync()
        if ((accum_step + 1) % total_accum_steps != 0)
        else contextlib.nullcontext()
    )

    with sync_ctx:
        with autocast(dtype=args.amp_dtype):
            # Encode all 3 views
            video_emb_all = model.module.encode_image(all_video)

            bt = bs * T
            video_ori_emb = video_emb_all[:bt]
            aug1_emb = video_emb_all[bt:2 * bt]
            aug2_emb = video_emb_all[2 * bt:3 * bt]

            # Temporal pooling
            if T > 1:
                video_ori_emb = video_ori_emb.view(bs, T, -1).mean(dim=1)
                aug1_emb = aug1_emb.view(bs, T, -1).mean(dim=1)
                aug2_emb = aug2_emb.view(bs, T, -1).mean(dim=1)

            # Encode text
            primary_text_emb = model.module.encode_text(primary_ids, primary_mask)

            num_cand = batch["num_candidates"]
            if num_cand > 0 and cand_ids.shape[0] > 0:
                cand_text_emb = model.module.encode_text(cand_ids, cand_mask)
            else:
                cand_text_emb = torch.zeros(0, args.embed_dim, device=video.device)

            # AllGather
            if world_size > 1:
                video_ori_emb = all_gather(video_ori_emb, world_size, rank)
                aug1_emb = all_gather(aug1_emb, world_size, rank)
                aug2_emb = all_gather(aug2_emb, world_size, rank)
                primary_text_emb = all_gather(primary_text_emb, world_size, rank)
                if cand_text_emb.shape[0] > 0:
                    cand_text_emb = all_gather(cand_text_emb, world_size, rank)

            logit_scale = (
                model.module.logit_scale.exp()
                if hasattr(model.module, 'logit_scale') else None
            )

            loss = loss_fn(
                video_ori_emb, aug1_emb, aug2_emb,
                primary_text_emb, cand_text_emb,
                logit_scale=logit_scale,
            )
            loss = loss / total_accum_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

    return loss


def train_phase_batch(model, batch, loss_fn, optimizer, scheduler,
                      scaler, args, rank, world_size, accum_step,
                      total_accum_steps):
    """Train one keystep/abstract-level (mid/coarse) batch."""
    video = batch["video"].cuda(rank, non_blocking=True)
    bs, T, C, H, W = video.shape
    video_flat = video.reshape(-1, C, H, W)

    primary_ids = batch["primary_text"]["input_ids"].cuda(rank, non_blocking=True)
    primary_mask = batch["primary_text"]["attention_mask"].cuda(rank, non_blocking=True)

    cand_ids = batch["candidate_texts"]["input_ids"].cuda(rank, non_blocking=True)
    cand_mask = batch["candidate_texts"]["attention_mask"].cuda(rank, non_blocking=True)

    sync_ctx = (
        model.no_sync()
        if ((accum_step + 1) % total_accum_steps != 0)
        else contextlib.nullcontext()
    )

    with sync_ctx:
        with autocast(dtype=args.amp_dtype):
            video_emb = model.module.encode_image(video_flat)
            if T > 1:
                video_emb = video_emb.view(bs, T, -1).mean(dim=1)

            primary_text_emb = model.module.encode_text(primary_ids, primary_mask)

            num_cand = batch["num_candidates"]
            if num_cand > 0 and cand_ids.shape[0] > 0:
                cand_text_emb = model.module.encode_text(cand_ids, cand_mask)
            else:
                cand_text_emb = torch.zeros(0, args.embed_dim, device=video.device)

            if world_size > 1:
                video_emb = all_gather(video_emb, world_size, rank)
                primary_text_emb = all_gather(primary_text_emb, world_size, rank)
                if cand_text_emb.shape[0] > 0:
                    cand_text_emb = all_gather(cand_text_emb, world_size, rank)

            # Reshape candidates to (global_bs, n_c, d)
            if num_cand > 0 and cand_text_emb.shape[0] > 0:
                global_bs = primary_text_emb.shape[0]
                cand_text_emb = cand_text_emb.view(global_bs, num_cand, -1)

            logit_scale = (
                model.module.logit_scale.exp()
                if hasattr(model.module, 'logit_scale') else None
            )

            loss = loss_fn(
                video_emb, primary_text_emb, cand_text_emb,
                logit_scale=logit_scale,
            )
            loss = loss / total_accum_steps

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

    return loss


# ==============================================================================
# Training Epoch
# ==============================================================================

def train_epoch_with_loader(loader, model, loss_fn, optimizer, scheduler,
                            scaler, args, rank, world_size, epoch,
                            loader_name, train_fn):
    """Train one full epoch using a specific dataloader."""
    model.train()
    running_loss = 0.0
    start_time = time.time()
    total_accum = args.accum_steps

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(
        loader,
        desc=f"E{epoch + 1:02d} {loader_name:8s}",
        disable=(rank != 0),
        leave=False,
    )

    for step, batch in enumerate(pbar):
        if batch is None:
            continue

        loss = train_fn(
            model=model, batch=batch, loss_fn=loss_fn,
            optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, args=args, rank=rank, world_size=world_size,
            accum_step=step, total_accum_steps=total_accum,
        )

        if (step + 1) % total_accum == 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running_loss += loss.item() * total_accum

        if rank == 0 and (step + 1) % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

            if args.writer is not None:
                gs = epoch * len(loader) + step
                args.writer.add_scalar(f"train/{loader_name}_loss", avg_loss, gs)
                args.writer.add_scalar(f"train/{loader_name}_lr", lr, gs)

            running_loss = 0.0
            start_time = time.time()


# ==============================================================================
# In-Training Evaluation
# ==============================================================================

def run_in_training_eval(model, tokenizer, epoch, args, device):
    """Run zero-shot evaluation using the existing zeroshot_evaluate infrastructure."""
    from zeroshot_evaluate import build_dataloader, evaluation_wrapper

    model.eval()
    datasets = [d.strip() for d in args.eval_datasets.split(",") if d.strip()]

    for ds_name in datasets:
        try:
            loader, _ = build_dataloader(
                dataset_name=ds_name, batch_size=16, num_workers=2,
                num_frames=args.num_frames,
            )
            loader.dataset.name = ds_name

            out_dir = os.path.join(args.output_dir, f"eval_epoch_{epoch + 1}")
            result = evaluation_wrapper(
                model=model.module, data_loader=loader, tokenizer=tokenizer,
                device=device, output_dir=out_dir, prefix=ds_name,
                expected_num_frames=args.num_frames,
            )

            if args.writer is not None and result:
                for k, v in result.items():
                    if isinstance(v, (int, float)):
                        args.writer.add_scalar(f"eval/{ds_name}/{k}", v, epoch)

            print(f"  [{ds_name}] OK")
        except Exception as e:
            print(f"  [{ds_name}] FAILED: {e}")

    model.train()


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Hierarchical VLP (PeskaVLP-style)")

    # Model
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--text_model_name", type=str, default="marcobombieri/surgicberta")
    p.add_argument("--vision_pretrained_weights", type=str, default="lemonfm.pth")
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--max_length", type=int, default=256)

    # Data
    p.add_argument("--main_csv_path", type=str, required=True)
    p.add_argument("--annotations_root", type=str, default=None)
    p.add_argument("--annotations_folder", type=str, default=None)
    p.add_argument("--video_root_folder", type=str, required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size_action", type=int, default=32)
    p.add_argument("--batch_size_keystep", type=int, default=16)
    p.add_argument("--batch_size_abstract", type=int, default=8)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--warmup_steps", type=int, default=500)

    # Schedule
    p.add_argument("--action_every_n_epoch", type=int, default=1)
    p.add_argument("--keystep_every_n_epoch", type=int, default=3)
    p.add_argument("--abstract_every_n_epoch", type=int, default=5)

    # Loss
    p.add_argument("--loss_temperature", type=float, default=0.1)
    p.add_argument("--ntxent_alpha", type=float, default=0.75)

    # Augmentation
    p.add_argument("--no_augmentation", action="store_true")
    p.add_argument("--aug_scale_min", type=float, default=0.2)

    # Multi-text
    p.add_argument("--multi_text_candidates", type=int, default=4)

    # Eval
    p.add_argument("--eval_every_n_epoch", type=int, default=5)
    p.add_argument("--eval_datasets", type=str, default="cholec80_phase")

    # Logging
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--output_dir", type=str, default="runs/hierarchical_vlp")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--use_compile", action="store_true")

    args = p.parse_args()
    if args.annotations_root is None and args.annotations_folder is not None:
        args.annotations_root = args.annotations_folder
    return args


def train():
    args = parse_args()
    rank = setup_ddp()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)

    args.amp_dtype = (
        torch.bfloat16
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        else torch.float16
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    args.writer = None
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

    # ---- Model ----
    if rank == 0:
        print("Building VLP model...")
    model = VLP(
        embed_dim=args.embed_dim,
        text_model_name=args.text_model_name,
        vision_pretrained_weights=args.vision_pretrained_weights,
        num_frames=args.num_frames,
    ).to(device)

    if args.use_compile:
        try:
            model = torch.compile(model, mode="max-autotune")
            if rank == 0:
                print("torch.compile enabled")
        except Exception as e:
            if rank == 0:
                print(f"torch.compile failed: {e}")

    model = DDP(
        model, device_ids=[rank],
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)

    # ---- Datasets ----
    if rank == 0:
        print("Building datasets...")

    ds_kwargs = dict(
        main_csv_path=args.main_csv_path,
        annotations_folder=args.annotations_folder,
        annotations_root=args.annotations_root,
        tokenizer=tokenizer,
        image_size=args.image_size,
        max_length=args.max_length,
        video_root_folder=args.video_root_folder,
        num_frames=args.num_frames,
        multi_text_candidates=args.multi_text_candidates,
        samples_cache_dir=".cache/peska_pretrain",
        use_samples_cache=True,
    )

    action_ds = PeskaAugmentedPretrainDataset(
        annotation_levels=["fine"],
        use_augmentation=not args.no_augmentation,
        augmentation_scale=(args.aug_scale_min, 1.0),
        **ds_kwargs,
    )
    keystep_ds = PeskaAugmentedPretrainDataset(
        annotation_levels=["mid"], use_augmentation=False, **ds_kwargs,
    )
    abstract_ds = PeskaAugmentedPretrainDataset(
        annotation_levels=["coarse"], use_augmentation=False, **ds_kwargs,
    )

    action_sampler = DistributedSampler(action_ds, num_replicas=world_size,
                                        rank=rank, shuffle=True)
    keystep_sampler = DistributedSampler(keystep_ds, num_replicas=world_size,
                                         rank=rank, shuffle=True)
    abstract_sampler = DistributedSampler(abstract_ds, num_replicas=world_size,
                                          rank=rank, shuffle=True)

    loader_kwargs = dict(
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=peska_collate_fn,
    )

    action_loader = DataLoader(action_ds, batch_size=args.batch_size_action,
                               sampler=action_sampler, **loader_kwargs)
    keystep_loader = DataLoader(keystep_ds, batch_size=args.batch_size_keystep,
                                sampler=keystep_sampler, **loader_kwargs)
    abstract_loader = DataLoader(abstract_ds, batch_size=args.batch_size_abstract,
                                 sampler=abstract_sampler, **loader_kwargs)

    if rank == 0:
        print(f"  Action:   {len(action_ds)} samples")
        print(f"  Keystep:  {len(keystep_ds)} samples")
        print(f"  Abstract: {len(abstract_ds)} samples")

    # ---- Losses ----
    action_loss = HierarchicalLossAction(
        temperature=args.loss_temperature, alpha_weight=args.ntxent_alpha,
    )
    phase_loss = HierarchicalLossPhase(
        temperature=args.loss_temperature, alpha_weight=args.ntxent_alpha,
    )

    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )

    act_steps = math.ceil(len(action_loader) / args.accum_steps)
    key_epochs = max(0, (args.epochs - 1) // args.keystep_every_n_epoch)
    abs_epochs = max(0, (args.epochs - 1) // args.abstract_every_n_epoch)
    key_steps = math.ceil(len(keystep_loader) / args.accum_steps)
    abs_steps = math.ceil(len(abstract_loader) / args.accum_steps)

    total_steps = (
        act_steps * args.epochs + key_steps * key_epochs + abs_steps * abs_epochs
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = GradScaler(enabled=(args.amp_dtype == torch.float16))

    # ---- Resume ----
    start_epoch = 0
    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        if rank == 0:
            print(f"Resuming from: {args.resume_from_checkpoint}")
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        if rank == 0:
            print(f"  Resumed from epoch {start_epoch}")

    if rank == 0:
        print(f"\nTraining: {args.epochs} epochs, {total_steps} steps")
        print(f"  Action:   every epoch")
        print(f"  Keystep:  every {args.keystep_every_n_epoch} epochs")
        print(f"  Abstract: every {args.abstract_every_n_epoch} epochs\n")

    # ---- Training Loop ----
    for epoch in range(start_epoch, args.epochs):
        action_sampler.set_epoch(epoch)
        keystep_sampler.set_epoch(epoch)
        abstract_sampler.set_epoch(epoch)

        # Action level: every epoch
        if epoch % args.action_every_n_epoch == 0:
            train_epoch_with_loader(
                action_loader, model, action_loss, optimizer, scheduler,
                scaler, args, rank, world_size, epoch, "action",
                train_action_batch,
            )

        # Keystep level: every N epochs
        if (epoch % args.keystep_every_n_epoch == 0 and
                epoch >= args.keystep_every_n_epoch):
            train_epoch_with_loader(
                keystep_loader, model, phase_loss, optimizer, scheduler,
                scaler, args, rank, world_size, epoch, "keystep",
                train_phase_batch,
            )

        # Abstract level: every N epochs
        if (epoch % args.abstract_every_n_epoch == 0 and
                epoch >= args.abstract_every_n_epoch):
            train_epoch_with_loader(
                abstract_loader, model, phase_loss, optimizer, scheduler,
                scaler, args, rank, world_size, epoch, "abstract",
                train_phase_batch,
            )

        # In-training evaluation
        if (args.eval_every_n_epoch > 0 and
                (epoch + 1) % args.eval_every_n_epoch == 0 and epoch > 0):
            if rank == 0:
                print(f"\n  Eval @ epoch {epoch + 1}...")
            try:
                run_in_training_eval(model, tokenizer, epoch, args, device)
            except Exception as e:
                if rank == 0:
                    print(f"  Eval error: {e}")

        # Checkpoint
        if rank == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            if args.writer is not None:
                args.writer.flush()

    # ---- Final ----
    if rank == 0:
        final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        torch.save({
            "epoch": args.epochs,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
        }, final_path)
        print(f"\nDone. Final: {final_path}")
        if args.writer is not None:
            args.writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    train()
