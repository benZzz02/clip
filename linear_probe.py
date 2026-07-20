import argparse
import hashlib
import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from downstream_datasets import SurgLaViClipDataset, SurgLaViSingleFrameDataset
from eval_report_utils import export_evaluation_reports, flatten_metrics
from external_feature_extractors import (
    EXTERNAL_FEATURE_MODES,
    build_external_feature_model,
    build_external_transform,
    canonical_feature_mode,
    is_external_feature_mode,
)
from model import VLP
from visual_backbones import build_visual_backbone
from zeroshot_evaluate import ToolPresenceEvaluator, WorkflowEvaluator, to_builtin


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ANNO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "anno_downstream"))
DEFAULT_DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))


@dataclass(frozen=True)
class SplitSpec:
    label_file: str
    data_root: str


@dataclass(frozen=True)
class ProbeTaskSpec:
    anno_subdir: str
    frame_list: str
    train: SplitSpec
    test: SplitSpec
    val: Optional[SplitSpec]
    sample_rate: int
    zero_fill: int
    image_type: str
    name: str
    task: str


def _ann(anno_root, dataset_dir, filename):
    return os.path.join(anno_root, dataset_dir, "annotations", filename)


def _frames(anno_root, dataset_dir):
    return os.path.join(anno_root, dataset_dir, "frame_lists", "frames.csv")


def build_task_specs(anno_root, data_root) -> Dict[str, ProbeTaskSpec]:
    return {
        "cholec80_phase": ProbeTaskSpec(
            anno_subdir="cholec80",
            frame_list=_frames(anno_root, "cholec80"),
            train=SplitSpec(_ann(anno_root, "cholec80", "train.json"), os.path.join(data_root, "cholecdata/cholecdata/cholec80/frames")),
            val=SplitSpec(_ann(anno_root, "cholec80", "val.json"), os.path.join(data_root, "cholecdata/cholecdata/cholec80/frames")),
            test=SplitSpec(_ann(anno_root, "cholec80", "test.json"), os.path.join(data_root, "cholecdata/cholecdata/cholec80/frames")),
            sample_rate=1,
            zero_fill=6,
            image_type="png",
            name="cholec80",
            task="phases",
        ),
        "cholec80_instrument": ProbeTaskSpec(
            anno_subdir="cholec80",
            frame_list=_frames(anno_root, "cholec80"),
            train=SplitSpec(_ann(anno_root, "cholec80", "instruments_train.json"), os.path.join(data_root, "cholecdata/cholecdata/cholec80/frames")),
            val=SplitSpec(_ann(anno_root, "cholec80", "instruments_val.json"), os.path.join(data_root, "cholecdata/cholecdata/cholec80/frames")),
            test=SplitSpec(_ann(anno_root, "cholec80", "instruments_test.json"), os.path.join(data_root, "cholecdata/cholecdata/cholec80/frames")),
            sample_rate=1,
            zero_fill=6,
            image_type="png",
            name="cholec80",
            task="instruments",
        ),
        "autolaparo_phase": ProbeTaskSpec(
            anno_subdir="autolaparo",
            frame_list=_frames(anno_root, "autolaparo"),
            train=SplitSpec(_ann(anno_root, "autolaparo", "train.json"), os.path.join(data_root, "DATA-Yui/Dataset/AutoLaparoDataset/AutoLaparo_Task1/frames_cutmargin")),
            val=SplitSpec(_ann(anno_root, "autolaparo", "val.json"), os.path.join(data_root, "DATA-Yui/Dataset/AutoLaparoDataset/AutoLaparo_Task1/frames_cutmargin")),
            test=SplitSpec(_ann(anno_root, "autolaparo", "test.json"), os.path.join(data_root, "DATA-Yui/Dataset/AutoLaparoDataset/AutoLaparo_Task1/frames_cutmargin")),
            sample_rate=1,
            zero_fill=4,
            image_type="jpg",
            name="autolaparo",
            task="phases",
        ),
        "bernbypass70_phase": ProbeTaskSpec(
            anno_subdir="bernbypass70",
            frame_list=_frames(anno_root, "bernbypass70"),
            train=SplitSpec(_ann(anno_root, "bernbypass70", "train.json"), os.path.join(data_root, "MultiBypass140/BernBypass70/frames")),
            val=SplitSpec(_ann(anno_root, "bernbypass70", "val.json"), os.path.join(data_root, "MultiBypass140/BernBypass70/frames")),
            test=SplitSpec(_ann(anno_root, "bernbypass70", "test.json"), os.path.join(data_root, "MultiBypass140/BernBypass70/frames")),
            sample_rate=1,
            zero_fill=8,
            image_type="jpg",
            name="bernbypass70",
            task="phases",
        ),
        "strasbypass70_phase": ProbeTaskSpec(
            anno_subdir="strasbypass70",
            frame_list=_frames(anno_root, "strasbypass70"),
            train=SplitSpec(_ann(anno_root, "strasbypass70", "train.json"), os.path.join(data_root, "MultiBypass140/StrasBypass70/frames")),
            val=SplitSpec(_ann(anno_root, "strasbypass70", "val.json"), os.path.join(data_root, "MultiBypass140/StrasBypass70/frames")),
            test=SplitSpec(_ann(anno_root, "strasbypass70", "test.json"), os.path.join(data_root, "MultiBypass140/StrasBypass70/frames")),
            sample_rate=1,
            zero_fill=8,
            image_type="jpg",
            name="strasbypass70",
            task="phases",
        ),
        "grasp_phase": ProbeTaskSpec(
            anno_subdir="grasp",
            frame_list=_frames(anno_root, "grasp"),
            train=SplitSpec(_ann(anno_root, "grasp", "grasp_long-term_train.json"), os.path.join(data_root, "Grasp/frames")),
            val=None,
            test=SplitSpec(_ann(anno_root, "grasp", "grasp_long-term_test.json"), os.path.join(data_root, "Grasp/frames")),
            sample_rate=1,
            zero_fill=5,
            image_type="jpg",
            name="grasp_phase",
            task="phases",
        ),
        "grasp_step": ProbeTaskSpec(
            anno_subdir="grasp",
            frame_list=_frames(anno_root, "grasp"),
            train=SplitSpec(_ann(anno_root, "grasp", "grasp_long-term_train.json"), os.path.join(data_root, "Grasp/frames")),
            val=None,
            test=SplitSpec(_ann(anno_root, "grasp", "grasp_long-term_test.json"), os.path.join(data_root, "Grasp/frames")),
            sample_rate=1,
            zero_fill=5,
            image_type="jpg",
            name="grasp_step",
            task="steps",
        ),
        "grasp_instrument": ProbeTaskSpec(
            anno_subdir="grasp",
            frame_list=_frames(anno_root, "grasp"),
            train=SplitSpec(_ann(anno_root, "grasp", "grasp_short-term_train.json"), os.path.join(data_root, "Grasp/frames")),
            val=None,
            test=SplitSpec(_ann(anno_root, "grasp", "grasp_short-term_test.json"), os.path.join(data_root, "Grasp/frames")),
            sample_rate=1,
            zero_fill=5,
            image_type="jpg",
            name="grasp_instrument",
            task="instruments",
        ),
        "sarrarp50_phase": ProbeTaskSpec(
            anno_subdir="sarrarp50",
            frame_list=_frames(anno_root, "sarrarp50"),
            train=SplitSpec(_ann(anno_root, "sarrarp50", "train.json"), os.path.join(data_root, "SAR50")),
            val=None,
            test=SplitSpec(_ann(anno_root, "sarrarp50", "test.json"), os.path.join(data_root, "SAR50")),
            sample_rate=1,
            zero_fill=9,
            image_type="png",
            name="sarrarp50",
            task="phases",
        ),
        "heichole_phase": ProbeTaskSpec(
            anno_subdir="heichole",
            frame_list=_frames(anno_root, "heichole"),
            train=SplitSpec(_ann(anno_root, "heichole", "train.json"), os.path.join(data_root, "HeiChole/outputs")),
            val=None,
            test=SplitSpec(_ann(anno_root, "heichole", "test.json"), os.path.join(data_root, "HeiChole/outputs")),
            sample_rate=1,
            zero_fill=5,
            image_type="png",
            name="heichole",
            task="phases",
        ),
        "heichole_instrument": ProbeTaskSpec(
            anno_subdir="heichole",
            frame_list=_frames(anno_root, "heichole"),
            train=SplitSpec(_ann(anno_root, "heichole", "instruments_train.json"), os.path.join(data_root, "HeiChole/outputs")),
            val=None,
            test=SplitSpec(_ann(anno_root, "heichole", "instruments_test.json"), os.path.join(data_root, "HeiChole/outputs")),
            sample_rate=1,
            zero_fill=5,
            image_type="png",
            name="heichole",
            task="instruments",
        ),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_seeds(spec):
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


def build_ann_file(task_spec: ProbeTaskSpec, split_spec: SplitSpec):
    return [
        split_spec.label_file,
        split_spec.data_root,
        "video",
        task_spec.frame_list,
        task_spec.sample_rate,
        task_spec.zero_fill,
        task_spec.image_type,
        task_spec.name,
        task_spec.task,
    ]


def build_dataset(task_spec: ProbeTaskSpec, split_spec: SplitSpec, transform, num_frames, frame_stride):
    dataset_cls = SurgLaViClipDataset if num_frames > 1 else SurgLaViSingleFrameDataset
    kwargs = {
        "ann_file": build_ann_file(task_spec, split_spec),
        "transform": transform,
    }
    if dataset_cls is SurgLaViClipDataset:
        kwargs.update({"num_frames": num_frames, "frame_stride": frame_stride})
    return dataset_cls(**kwargs)


def build_transform(args):
    feature_mode = canonical_feature_mode(args.feature_mode)
    if is_external_feature_mode(feature_mode):
        return build_external_transform(feature_mode, args.image_size)

    return transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def normalize_checkpoint_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod.") :]
        normalized[key] = value
    return normalized


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def file_fingerprint(path):
    if path and os.path.exists(path):
        stat = os.stat(path)
        return f"{path}:{stat.st_size}:{int(stat.st_mtime)}"
    return str(path)


def load_vlp_checkpoint(model, ckpt_path, device, strict=False):
    if not ckpt_path:
        raise ValueError("--ckpt is required when --feature_mode=vlp")

    ckpt = safe_torch_load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    state_dict = normalize_checkpoint_state_dict(state_dict)
    if getattr(model, "frame_pool", None) is None:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("frame_pool.")}

    msg = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)
    if strict and (msg.missing_keys or msg.unexpected_keys):
        raise RuntimeError(
            f"Checkpoint mismatch.\nMissing keys: {msg.missing_keys}\nUnexpected keys: {msg.unexpected_keys}"
        )
    return model


def build_feature_model(args, device):
    feature_mode = canonical_feature_mode(args.feature_mode)
    if feature_mode == "raw_visual":
        model = build_visual_backbone(args.vision_backbone, args.vision_weights).to(device)
        return model.eval(), model.output_dim

    if is_external_feature_mode(feature_mode):
        return build_external_feature_model(args, device)

    model = VLP(
        embed_dim=args.embed_dim,
        text_model_name=args.text_model,
        vision_backbone=args.vision_backbone,
        vision_pretrained_weights=args.vision_weights,
        num_frames=args.num_frames,
        temporal_num_layers=args.temporal_layers,
        temporal_num_heads=args.temporal_heads,
        temporal_dropout=args.temporal_dropout,
    ).to(device)
    model = load_vlp_checkpoint(model, args.ckpt, device, strict=args.strict_load)
    return model.eval(), args.embed_dim


def encode_raw_visual(model, images):
    if images.ndim == 5:
        batch_size, num_frames, channels, height, width = images.shape
        flat = images.reshape(batch_size * num_frames, channels, height, width)
        features = model(flat).reshape(batch_size, num_frames, -1).mean(dim=1)
    elif images.ndim == 4:
        features = model(images)
    else:
        raise ValueError(f"Unexpected image tensor shape: {tuple(images.shape)}")
    return F.normalize(features, dim=-1)


def encode_features(model, images, feature_mode):
    feature_mode = canonical_feature_mode(feature_mode)
    if feature_mode == "raw_visual":
        return encode_raw_visual(model, images)
    if is_external_feature_mode(feature_mode):
        return model.encode_image(images)
    return F.normalize(model.encode_image(images), dim=-1)


def get_context_num_frames(args):
    return int(args.context_num_frames) if args.context_num_frames else int(args.num_frames)


def get_context_stride(args):
    return int(args.context_stride) if args.context_stride else int(args.num_frames)


def cache_key(args, dataset_name, split_name, split_spec: SplitSpec):
    feature_mode = canonical_feature_mode(args.feature_mode)
    ckpt_id = file_fingerprint(args.ckpt) if args.ckpt else ""
    vision_id = file_fingerprint(args.vision_weights)
    context_num_frames = get_context_num_frames(args)
    payload = "|".join(
        [
            dataset_name,
            split_name,
            feature_mode,
            str(ckpt_id),
            str(vision_id),
            str(args.external_config),
            str(args.external_cache_dir),
            str(args.surgclip_model_name),
            str(args.probe_feature),
            str(args.text_model),
            str(args.embed_dim),
            str(args.num_frames),
            str(context_num_frames),
            str(get_context_stride(args)),
            str(args.context_pooling),
            str(args.frame_stride),
            str(args.temporal_layers),
            str(args.temporal_heads),
            str(args.temporal_dropout),
            str(args.image_size),
            split_spec.label_file,
            split_spec.data_root,
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return (
        f"{dataset_name}_{split_name}_{feature_mode}_nf{args.num_frames}"
        f"_ctx{context_num_frames}_s{args.frame_stride}_{digest}.pt"
    )


def context_window_starts(total_frames, encoder_frames, context_stride):
    total_frames = int(total_frames)
    encoder_frames = max(1, int(encoder_frames))
    if total_frames <= encoder_frames:
        return [0]

    stride = int(context_stride) if context_stride else encoder_frames
    stride = max(1, stride)
    last_start = total_frames - encoder_frames
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def encode_vlp_probe_features(model, images, probe_feature):
    probe_feature = str(probe_feature or "projected").lower()
    video_global_hidden, _ = model._encode_image_tokens(images)
    backbone = F.normalize(video_global_hidden, dim=-1)
    projected = model._project_video_global(video_global_hidden)

    if probe_feature == "backbone":
        return backbone
    if probe_feature == "concat":
        return F.normalize(torch.cat([backbone, projected], dim=-1), dim=-1)
    if probe_feature == "projected":
        return projected
    raise ValueError(f"Unsupported probe_feature: {probe_feature}")


def encode_base_features(model, images, args):
    feature_mode = canonical_feature_mode(args.feature_mode)
    probe_feature = str(args.probe_feature or "projected").lower()
    if feature_mode == "raw_visual":
        return encode_raw_visual(model, images)
    if is_external_feature_mode(feature_mode):
        if hasattr(model, "encode_probe_features"):
            return model.encode_probe_features(images, probe_feature=probe_feature)
        return model.encode_image(images)
    return encode_vlp_probe_features(model, images, probe_feature)


def encode_context_features(model, images, args):
    feature_mode = canonical_feature_mode(args.feature_mode)
    if images.ndim != 5:
        return encode_base_features(model, images, args)

    total_frames = images.shape[1]
    encoder_frames = max(1, int(args.num_frames))
    if total_frames <= encoder_frames:
        return encode_base_features(model, images, args)

    if args.context_pooling != "mean":
        raise ValueError(f"Unsupported context_pooling: {args.context_pooling}")

    chunk_features = []
    for start in context_window_starts(total_frames, encoder_frames, get_context_stride(args)):
        chunk = images[:, start : start + encoder_frames]
        chunk_features.append(encode_base_features(model, chunk, args))

    features = torch.stack(chunk_features, dim=0).mean(dim=0)
    return F.normalize(features, dim=-1)


def extract_features(model, data_loader, device, args):
    all_features = []
    all_labels = []
    all_video_idxs = []
    all_frame_idxs = []

    with torch.no_grad():
        for images, labels, video_idxs, frame_idxs in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device, non_blocking=True)
            amp_context = (
                torch.amp.autocast("cuda", enabled=args.amp)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                features = encode_context_features(model, images, args)
            all_features.append(features.float().cpu())
            all_labels.append(labels.cpu())
            all_video_idxs.append(video_idxs.cpu())
            all_frame_idxs.append(frame_idxs.cpu())

    return {
        "features": torch.cat(all_features, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "video_idxs": torch.cat(all_video_idxs, dim=0),
        "frame_idxs": torch.cat(all_frame_idxs, dim=0),
    }


def feature_cache_path(args, dataset_name, split_name, split_spec: SplitSpec, cache_dir):
    return os.path.join(cache_dir, cache_key(args, dataset_name, split_name, split_spec))


def load_cached_features(cache_path, split_name):
    print(f"Loading cached {split_name} features: {cache_path}")
    return safe_torch_load(cache_path, map_location="cpu")


def load_or_extract_features(model, dataset, split_spec, split_name, args, device, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = feature_cache_path(args, args.dataset, split_name, split_spec, cache_dir)

    if not args.no_cache and os.path.exists(cache_path):
        return load_cached_features(cache_path, split_name)

    loader = DataLoader(
        dataset,
        batch_size=args.encode_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    pack = extract_features(model, loader, device, args)
    pack["categories"] = dataset.categories
    pack["task"] = dataset.task

    if not args.no_cache:
        torch.save(pack, cache_path)
        print(f"Saved {split_name} features: {cache_path}")
    return pack


def sample_ratio_indices(labels, task, shot_ratio, seed, num_classes):
    num_samples = labels.shape[0]
    if shot_ratio >= 1.0:
        return torch.arange(num_samples)

    rng = np.random.default_rng(seed)
    if task in {"phases", "steps", "actions"}:
        labels_np = labels.cpu().numpy().astype(np.int64)
        selected = []
        for class_idx in range(num_classes):
            class_indices = np.where(labels_np == class_idx)[0]
            if len(class_indices) == 0:
                continue
            count = max(1, int(round(len(class_indices) * shot_ratio)))
            selected.extend(rng.choice(class_indices, size=min(count, len(class_indices)), replace=False).tolist())
        selected = np.array(sorted(set(selected)), dtype=np.int64)
        rng.shuffle(selected)
        return torch.from_numpy(selected)

    count = max(1, int(round(num_samples * shot_ratio)))
    selected = set(rng.choice(num_samples, size=min(count, num_samples), replace=False).tolist())

    labels_np = labels.cpu().numpy()
    for class_idx in range(num_classes):
        positives = np.where(labels_np[:, class_idx] > 0)[0]
        if len(positives) == 0:
            continue
        if not any(labels_np[list(selected), class_idx] > 0):
            selected.add(int(rng.choice(positives)))

    selected = np.array(sorted(selected), dtype=np.int64)
    rng.shuffle(selected)
    return torch.from_numpy(selected)


def sample_cls_fewshot_indices(labels, task, shots_per_class, seed, num_classes):
    shots_per_class = int(shots_per_class)
    if shots_per_class <= 0:
        raise ValueError(f"shots_per_class must be positive, got {shots_per_class}")

    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    selected = []

    if task in {"phases", "steps", "actions"}:
        labels_np = labels_np.astype(np.int64)
        for class_idx in range(num_classes):
            class_indices = np.where(labels_np == class_idx)[0]
            if len(class_indices) == 0:
                continue
            selected.extend(
                rng.choice(
                    class_indices,
                    size=min(shots_per_class, len(class_indices)),
                    replace=False,
                ).tolist()
            )
    else:
        for class_idx in range(num_classes):
            positives = np.where(labels_np[:, class_idx] > 0)[0]
            if len(positives) == 0:
                continue
            selected.extend(
                rng.choice(
                    positives,
                    size=min(shots_per_class, len(positives)),
                    replace=False,
                ).tolist()
            )

    if not selected:
        raise ValueError("No samples selected for CLS few-shot probing.")

    selected = np.array(sorted(set(selected)), dtype=np.int64)
    rng.shuffle(selected)
    return torch.from_numpy(selected)


def sample_video_fewshot_indices(video_idxs, video_ratio, seed):
    video_ratio = float(video_ratio)
    if video_ratio <= 0:
        raise ValueError(f"video_ratio must be positive, got {video_ratio}")

    video_np = video_idxs.cpu().numpy()
    unique_videos = np.unique(video_np)
    if video_ratio >= 1.0:
        return torch.arange(len(video_np))

    rng = np.random.default_rng(seed)
    num_videos = max(1, int(round(len(unique_videos) * video_ratio)))
    chosen_videos = rng.choice(unique_videos, size=min(num_videos, len(unique_videos)), replace=False)
    selected = np.where(np.isin(video_np, chosen_videos))[0]
    rng.shuffle(selected)
    return torch.from_numpy(selected.astype(np.int64))


def sample_probe_indices(train_pack, task, args, seed, num_classes):
    labels = train_pack["labels"]
    shot_mode = str(args.shot_mode).lower()

    if shot_mode == "ratio":
        return sample_ratio_indices(labels, task, args.shot_ratio, seed, num_classes)

    if shot_mode == "cls":
        shots_per_class = args.shots_per_class
        if shots_per_class is None:
            shots_per_class = int(round(args.shot_ratio))
        return sample_cls_fewshot_indices(labels, task, shots_per_class, seed, num_classes)

    if shot_mode == "video":
        return sample_video_fewshot_indices(train_pack["video_idxs"], args.shot_ratio, seed)

    raise ValueError(f"Unsupported shot_mode: {args.shot_mode}")


class SklearnProbeHead:
    def __init__(self, estimator, scaler, task, num_classes):
        self.estimator = estimator
        self.scaler = scaler
        self.task = task
        self.num_classes = int(num_classes)

    def _transform(self, features):
        x = features.float().cpu().numpy()
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return x

    def predict_logits(self, features):
        x = self._transform(features)

        if self.task == "instruments":
            if hasattr(self.estimator, "decision_function"):
                scores = self.estimator.decision_function(x)
            else:
                scores = self.estimator.predict_proba(x)
                if isinstance(scores, list):
                    scores = np.stack([score[:, 1] for score in scores], axis=1)
            return torch.from_numpy(np.asarray(scores, dtype=np.float32))

        if hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(x)
        else:
            scores = self.estimator.predict_proba(x)

        scores = np.asarray(scores, dtype=np.float32)
        classes = np.asarray(self.estimator.classes_, dtype=np.int64)

        if scores.ndim == 1:
            if len(classes) == 1:
                full_scores = np.full((scores.shape[0], self.num_classes), -1e6, dtype=np.float32)
                full_scores[:, int(classes[0])] = 0.0
                return torch.from_numpy(full_scores)

            binary_scores = np.stack([-scores, scores], axis=1)
            full_scores = np.full((scores.shape[0], self.num_classes), -1e6, dtype=np.float32)
            for col, class_idx in enumerate(classes):
                full_scores[:, int(class_idx)] = binary_scores[:, col]
            return torch.from_numpy(full_scores)

        full_scores = np.full((scores.shape[0], self.num_classes), -1e6, dtype=np.float32)
        for col, class_idx in enumerate(classes):
            full_scores[:, int(class_idx)] = scores[:, col]
        return torch.from_numpy(full_scores)


class ConstantProbeHead:
    def __init__(self, class_idx, num_classes):
        self.class_idx = int(class_idx)
        self.num_classes = int(num_classes)

    def predict_logits(self, features):
        logits = torch.full((features.shape[0], self.num_classes), -1e6, dtype=torch.float32)
        logits[:, self.class_idx] = 0.0
        return logits


def _sklearn_class_weight(value):
    value = str(value or "").strip().lower()
    if not value or value == "none":
        return None
    if value == "balanced":
        return "balanced"
    raise ValueError(f"Unsupported sklearn_class_weight: {value}")


def train_sklearn_linear_head(features, labels, task, num_classes, args, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler

    x_train = features.float().cpu().numpy()
    scaler = None
    if args.probe_standardize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

    class_weight = _sklearn_class_weight(args.sklearn_class_weight)
    logreg_kwargs = {
        "C": float(args.sklearn_c),
        "max_iter": int(args.sklearn_max_iter),
        "solver": args.sklearn_solver,
        "class_weight": class_weight,
        "random_state": int(seed),
        "verbose": int(args.sklearn_verbose),
    }

    if task == "instruments":
        y_train = labels.float().cpu().numpy()
        estimator = OneVsRestClassifier(LogisticRegression(**logreg_kwargs))
    else:
        y_train = labels.long().cpu().numpy()
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            print(
                "Only one class is present in the selected probe samples; "
                "using a constant classifier."
            )
            return ConstantProbeHead(unique_labels[0], num_classes)
        estimator = LogisticRegression(**logreg_kwargs)

    estimator.fit(x_train, y_train)
    print(
        "Trained sklearn LogisticRegression "
        f"(C={args.sklearn_c}, max_iter={args.sklearn_max_iter}, "
        f"solver={args.sklearn_solver}, standardize={args.probe_standardize}, "
        f"class_weight={class_weight})"
    )
    return SklearnProbeHead(estimator, scaler, task, num_classes)


def train_torch_linear_head(features, labels, task, num_classes, args, seed, device):
    set_seed(seed)

    head = nn.Linear(features.shape[-1], num_classes).to(device)
    if args.probe_optimizer == "sgd":
        optimizer = torch.optim.SGD(
            head.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.probe_optimizer == "adamw":
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported probe_optimizer: {args.probe_optimizer}")
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=args.probe_batch_size, shuffle=True, num_workers=0)

    if task == "instruments":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    head.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_seen = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            logits = head(batch_features)
            if task == "instruments":
                loss = criterion(logits, batch_labels.float())
            else:
                loss = criterion(logits, batch_labels.long())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * batch_features.size(0)
            total_seen += batch_features.size(0)

        if (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(f"epoch {epoch + 1}/{args.epochs} loss={total_loss / max(total_seen, 1):.6f}")

    return head.eval()


def train_linear_head(train_pack, task, num_classes, args, seed, device):
    set_seed(seed)

    features = train_pack["features"]
    labels = train_pack["labels"]
    selected_indices = sample_probe_indices(train_pack, task, args, seed, num_classes)
    features = features[selected_indices]
    labels = labels[selected_indices]

    print(
        f"Training linear probe with {len(selected_indices)}/{len(train_pack['features'])} "
        f"samples (shot_mode={args.shot_mode}, solver={args.probe_solver})"
    )

    if args.probe_solver == "sklearn_logreg":
        head = train_sklearn_linear_head(features, labels, task, num_classes, args, seed)
    elif args.probe_solver == "torch":
        head = train_torch_linear_head(features, labels, task, num_classes, args, seed, device)
    else:
        raise ValueError(f"Unsupported probe_solver: {args.probe_solver}")

    return head, selected_indices


def predict_dataframe(head, pack, task, device, batch_size):
    features = pack["features"]
    labels = pack["labels"]
    video_idxs = pack["video_idxs"]
    frame_idxs = pack["frame_idxs"]

    rows = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            if hasattr(head, "predict_logits"):
                logits = head.predict_logits(features[start:end]).float().cpu()
            else:
                logits = head(features[start:end].to(device)).float().cpu()
            for offset, pred in enumerate(logits):
                idx = start + offset
                label = labels[idx]
                if task == "instruments":
                    ground_truth = label.float().tolist()
                else:
                    ground_truth = int(label.item())
                rows.append(
                    {
                        "video_idx": int(video_idxs[idx].item()),
                        "frame_idx": int(frame_idxs[idx].item()),
                        "prediction": pred.tolist(),
                        "ground_truth": ground_truth,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_predictions(predictions_df, categories, task, dataset_name):
    if task in {"phases", "steps", "actions"}:
        evaluator = WorkflowEvaluator(phases=categories, prefix=dataset_name)
    elif task == "instruments":
        evaluator = ToolPresenceEvaluator(tools=categories, prefix=dataset_name)
    else:
        raise ValueError(f"Unsupported probing task: {task}")
    return evaluator.evaluate(predictions_df)


def write_seed_outputs(results, predictions_df, selected_indices, output_dir, args, seed):
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, f"predictions_{args.dataset}.csv")
    predictions_df.to_csv(pred_path, index=False)

    result_path = os.path.join(output_dir, f"results_{args.dataset}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(results), f, ensure_ascii=False, indent=2)

    torch.save(selected_indices, os.path.join(output_dir, f"selected_indices_seed{seed}.pt"))

    metadata = {
        "dataset": args.dataset,
        "model_family": f"linear_probe_{args.feature_mode}",
        "ckpt": args.ckpt if args.ckpt else ("official-auto" if is_external_feature_mode(args.feature_mode) else args.vision_weights),
        "feature_mode": args.feature_mode,
        "shot_mode": args.shot_mode,
        "shot_ratio": args.shot_ratio,
        "shots_per_class": args.shots_per_class,
        "seed": seed,
        "probe_solver": args.probe_solver,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "probe_optimizer": args.probe_optimizer,
        "momentum": args.momentum,
        "probe_standardize": args.probe_standardize,
        "sklearn_c": args.sklearn_c,
        "sklearn_max_iter": args.sklearn_max_iter,
        "sklearn_solver": args.sklearn_solver,
        "sklearn_class_weight": args.sklearn_class_weight,
        "num_frames": args.num_frames,
        "context_num_frames": get_context_num_frames(args),
        "context_stride": get_context_stride(args),
        "context_pooling": args.context_pooling,
        "probe_feature": args.probe_feature,
        "frame_stride": args.frame_stride,
        "embed_dim": args.embed_dim,
        "vision_weights": args.vision_weights,
        "text_model": args.text_model,
        "result_json": result_path,
        "external_config": args.external_config,
        "external_cache_dir": args.external_cache_dir,
        "surgclip_model_name": args.surgclip_model_name,
    }
    report_paths = export_evaluation_reports(
        results=to_builtin(results),
        dataset=args.dataset,
        output_dir=output_dir,
        metadata=metadata,
        sota_file=None,
    )
    return report_paths


def aggregate_seed_summaries(seed_rows, output_dir):
    if not seed_rows:
        return
    df = pd.DataFrame(seed_rows)
    seed_path = os.path.join(output_dir, "summary_seeds.csv")
    df.to_csv(seed_path, index=False)

    numeric_cols = [
        col
        for col in df.columns
        if col not in {"seed"} and pd.api.types.is_numeric_dtype(df[col])
    ]
    rows = []
    for col in numeric_cols:
        rows.append(
            {
                "metric": col,
                "mean": df[col].mean(),
                "std": df[col].std(ddof=0),
            }
        )
    aggregate_path = os.path.join(output_dir, "summary_mean_std.csv")
    pd.DataFrame(rows).to_csv(aggregate_path, index=False)
    print(f"Saved seed summaries: {seed_path}")
    print(f"Saved mean/std summary: {aggregate_path}")


def run(args):
    args.feature_mode = canonical_feature_mode(args.feature_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    task_specs = build_task_specs(args.anno_root, args.data_root)
    if args.dataset not in task_specs:
        raise KeyError(f"Unknown dataset: {args.dataset}. Available: {sorted(task_specs)}")
    task_spec = task_specs[args.dataset]

    transform = build_transform(args)
    context_num_frames = get_context_num_frames(args)
    train_dataset = build_dataset(task_spec, task_spec.train, transform, context_num_frames, args.frame_stride)
    test_dataset = build_dataset(task_spec, task_spec.test, transform, context_num_frames, args.frame_stride)
    train_dataset.name = args.dataset
    test_dataset.name = args.dataset

    print(f"Dataset: {args.dataset}")
    print(f"Task: {train_dataset.task}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print(f"Classes: {len(train_dataset.categories)}")
    print(
        f"Frame context: context_num_frames={context_num_frames} "
        f"encoder_num_frames={args.num_frames} context_stride={get_context_stride(args)} "
        f"context_pooling={args.context_pooling}"
    )

    cache_dir = args.cache_dir or os.path.join(args.output_dir, "cache")
    train_cache_path = feature_cache_path(args, args.dataset, "train", task_spec.train, cache_dir)
    test_cache_path = feature_cache_path(args, args.dataset, "test", task_spec.test, cache_dir)
    if (
        not args.no_cache
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        train_pack = load_cached_features(train_cache_path, "train")
        test_pack = load_cached_features(test_cache_path, "test")
    else:
        model, _ = build_feature_model(args, device)
        for param in model.parameters():
            param.requires_grad = False

        train_pack = load_or_extract_features(
            model, train_dataset, task_spec.train, "train", args, device, cache_dir
        )
        test_pack = load_or_extract_features(
            model, test_dataset, task_spec.test, "test", args, device, cache_dir
        )

    seeds = parse_seeds(args.seeds)
    seed_rows = []
    for seed in seeds:
        seed_output_dir = os.path.join(args.output_dir, f"seed_{seed}")
        head, selected_indices = train_linear_head(
            train_pack=train_pack,
            task=train_dataset.task,
            num_classes=len(train_dataset.categories),
            args=args,
            seed=seed,
            device=device,
        )
        predictions_df = predict_dataframe(
            head=head,
            pack=test_pack,
            task=train_dataset.task,
            device=device,
            batch_size=args.probe_batch_size,
        )
        results = evaluate_predictions(
            predictions_df=predictions_df,
            categories=train_dataset.categories,
            task=train_dataset.task,
            dataset_name=args.dataset,
        )
        write_seed_outputs(results, predictions_df, selected_indices, seed_output_dir, args, seed)

        flat = flatten_metrics(results[args.dataset])
        seed_rows.append({"seed": seed, **flat})
        print(json.dumps(to_builtin(results), ensure_ascii=False, indent=2))

    aggregate_seed_summaries(seed_rows, args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Few/full-shot linear probing for frozen surgical VLP features")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="vlp",
        choices=[
            "vlp",
            "surgalign",
            "raw_visual",
            "surgvlp",
            "hecvl",
            "peskavlp",
            "surgclip_beta",
            "surgclip-beta",
            "surgclip",
        ],
        help=(
            "Feature encoder to freeze for probing. "
            "vlp/surgalign uses the local SurgAlign checkpoint; "
            f"external SOTA modes: {', '.join(sorted(EXTERNAL_FEATURE_MODES))}."
        ),
    )
    parser.add_argument("--text_model", type=str, default="marcobombieri/surgicberta")
    parser.add_argument(
        "--vision_backbone",
        type=str,
        default="convnext_lemonfm",
        choices=["convnext_lemonfm", "gsvit_m5", "vitl", "endossl_vitl", "endossl"],
    )
    parser.add_argument("--vision_weights", type=str, default="lemonfm.pth")
    parser.add_argument("--anno_root", type=str, default=DEFAULT_ANNO_ROOT)
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=str, default="./linear_probe_outputs")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--external_config",
        type=str,
        default="",
        help="Optional config path for SurgVLP/HecVL/PeskaVLP feature extractors.",
    )
    parser.add_argument(
        "--external_cache_dir",
        type=str,
        default="",
        help="Optional cache dir for official external-model weights downloaded by their adapters.",
    )
    parser.add_argument(
        "--surgclip_model_name",
        type=str,
        default="SurgCLIP-B",
        help="SurgCLIP package model name for --feature_mode=surgclip_beta.",
    )
    parser.add_argument(
        "--probe_feature",
        type=str,
        default="projected",
        choices=["projected", "backbone", "concat"],
        help=(
            "Feature representation for the linear head. "
            "SurgLaVi linear probing concatenates backbone and projection embeddings."
        ),
    )
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument(
        "--context_num_frames",
        type=int,
        default=None,
        help=(
            "Temporal context window sampled from the downstream video. "
            "Defaults to --num_frames. If larger than --num_frames, features "
            "are extracted from --num_frames chunks and pooled."
        ),
    )
    parser.add_argument(
        "--context_stride",
        type=int,
        default=None,
        help="Stride between encoder chunks inside a larger context window. Defaults to --num_frames.",
    )
    parser.add_argument(
        "--context_pooling",
        type=str,
        default="mean",
        choices=["mean"],
        help="Pooling used to aggregate chunk embeddings inside --context_num_frames.",
    )
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--temporal_layers", type=int, default=2)
    parser.add_argument("--temporal_heads", type=int, default=12)
    parser.add_argument("--temporal_dropout", type=float, default=0.1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--shot_mode",
        type=str,
        default="ratio",
        choices=["ratio", "cls", "video"],
        help=(
            "ratio: old sample-ratio probing; "
            "cls: SurgLaVi CLS few-shot probing with N samples per class; "
            "video: SurgLaVi video few/full-shot probing with a ratio of training videos."
        ),
    )
    parser.add_argument("--shot_ratio", type=float, default=1.0)
    parser.add_argument(
        "--shots_per_class",
        type=int,
        default=None,
        help="Number of labeled frames per class for --shot_mode=cls. Defaults to round(--shot_ratio).",
    )
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument(
        "--probe_solver",
        type=str,
        default="torch",
        choices=["torch", "sklearn_logreg"],
        help=(
            "Linear probe solver. 'torch' keeps the legacy SGD/AdamW head; "
            "'sklearn_logreg' fits a standard logistic-regression probe on frozen features."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--probe_optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--probe_standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize cached features before sklearn logistic regression.",
    )
    parser.add_argument("--sklearn_c", type=float, default=1.0)
    parser.add_argument("--sklearn_max_iter", type=int, default=5000)
    parser.add_argument("--sklearn_solver", type=str, default="lbfgs")
    parser.add_argument(
        "--sklearn_class_weight",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Class weighting for sklearn LogisticRegression.",
    )
    parser.add_argument("--sklearn_verbose", type=int, default=0)
    parser.add_argument("--encode_batch_size", type=int, default=32)
    parser.add_argument("--probe_batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--strict_load", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
