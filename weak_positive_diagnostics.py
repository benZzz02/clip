import argparse
import csv
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import VLP
from pretrain_dataset import PretrainDataset


LEVEL_TO_ID = {"fine": 0, "mid": 1, "coarse": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Automatic weak-positive diagnostics for SurgAlign. "
            "Exports outside-window discovery, window-size curves, and "
            "uniform-vs-text-reselection statistics without manual labels."
        )
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/data/znh/surgalign_checkpoints/same_video_triplet_xpool_8f_run1/vlp_epoch_50.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/znh/surgalign_diagnostics/same_video_triplet_xpool_8f_run1_epoch50",
    )

    parser.add_argument("--main_csv_path", type=str, default="surglavi_level_csv/all_video.csv")
    parser.add_argument("--video_root_folder", type=str, default="/data/znh/downloaded_video_224_test")
    parser.add_argument("--annotations_folder", type=str, default=None)
    parser.add_argument("--annotations_root", type=str, default="surglavi_level_csv")
    parser.add_argument("--annotation_levels", type=str, default="coarse,mid,fine")
    parser.add_argument("--level_mix", type=str, default="concat")
    parser.add_argument("--samples_cache_dir", type=str, default=".cache/pretrain_samples")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--use_samples_cache", action="store_true", dest="use_samples_cache", default=True)
    cache_group.add_argument("--no_samples_cache", action="store_false", dest="use_samples_cache")
    parser.add_argument("--rebuild_samples_cache", action="store_true")
    parser.add_argument("--samples_cache_version", type=str, default="v1")
    parser.add_argument("--sample_mode", type=str, default="random", choices=["random", "center"])
    parser.add_argument("--ffmpeg_timeout", type=int, default=10)
    parser.add_argument("--max_retry", type=int, default=5)
    parser.add_argument("--video_reader_threads", type=int, default=2)
    parser.add_argument("--video_reader_cache_size", type=int, default=1)
    parser.add_argument("--assume_resized_video", type=str2bool, default=True)

    parser.add_argument("--text_model_name", type=str, default="marcobombieri/surgicberta")
    parser.add_argument("--vision_pretrained_weights", type=str, default="lemonfm.pth")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--temporal_num_layers", type=int, default=2)
    parser.add_argument("--temporal_num_heads", type=int, default=12)
    parser.add_argument("--temporal_dropout", type=float, default=0.1)
    parser.add_argument("--temporal_hidden_dim", type=int, default=768)
    parser.add_argument("--local_temperature", type=float, default=0.15)
    parser.add_argument("--selection_pooling", type=str, default="xpool")
    parser.add_argument(
        "--level_frame_temperatures",
        type=str,
        default="0.35,0.8,1.6",
        help="Comma-separated fine,mid,coarse frame temperatures.",
    )

    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--levels", type=str, default="fine,mid,coarse")
    parser.add_argument(
        "--window_scales",
        type=str,
        default="1.0,1.25,1.5,2.0,3.0",
        help="Expansion-ratio sweep. x1.5 matches TRAIN_WINDOW_EXPAND_RATIO=1.5.",
    )
    parser.add_argument(
        "--window_offsets",
        type=str,
        default="",
        help="Optional fixed-second sweep, e.g. 0,5,10,20. Overrides window_scales when set.",
    )
    parser.add_argument("--primary_window_scale", type=float, default=1.5)
    parser.add_argument("--primary_window_offset", type=float, default=10.0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:1")
    return parser.parse_args()


def parse_float_list(spec: str) -> List[float]:
    values = [float(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not values:
        raise ValueError(f"Expected at least one float in {spec!r}")
    return values


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_float_tuple(spec: str, expected_len: int) -> Tuple[float, ...]:
    values = tuple(parse_float_list(spec))
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} floats, got {len(values)} from {spec!r}")
    return values


def parse_str_list(spec: str) -> List[str]:
    return [x.strip().lower() for x in str(spec).split(",") if x.strip()]


def build_window_specs(args) -> List[Dict]:
    if str(args.window_offsets).strip():
        specs = []
        for offset in parse_float_list(args.window_offsets):
            specs.append(
                {
                    "mode": "offset",
                    "value": float(offset),
                    "label": table_window_offset(offset),
                    "sort_key": float(offset),
                }
            )
        return specs

    specs = []
    for scale in parse_float_list(args.window_scales):
        scale = max(float(scale), 1.0)
        label = "Original" if math.isclose(scale, 1.0, abs_tol=1e-9) else f"x{scale:g}"
        specs.append(
            {
                "mode": "scale",
                "value": scale,
                "label": label,
                "sort_key": scale,
            }
        )
    return specs


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]
        normalized[key] = value
    return normalized


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    msg = model.load_state_dict(normalize_state_dict_keys(state_dict), strict=False)
    if msg.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {msg.unexpected_keys}")

    allowed_missing = {
        "frame_local_projection.weight",
        "selection_text_query_projection.weight",
        "selection_frame_key_projection.weight",
    }
    disallowed_missing = [key for key in msg.missing_keys if key not in allowed_missing]
    if disallowed_missing:
        raise RuntimeError(f"Missing checkpoint keys: {disallowed_missing}")


def build_model(args, device: str) -> VLP:
    model = VLP(
        embed_dim=args.embed_dim,
        text_model_name=args.text_model_name,
        vision_pretrained_weights=args.vision_pretrained_weights,
        num_frames=args.num_frames,
        temporal_num_layers=args.temporal_num_layers,
        temporal_num_heads=args.temporal_num_heads,
        temporal_dropout=args.temporal_dropout,
        temporal_hidden_dim=args.temporal_hidden_dim,
        local_temperature=args.local_temperature,
        selection_pooling=args.selection_pooling,
        level_frame_temperatures=parse_float_tuple(args.level_frame_temperatures, 3),
    ).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()
    return model


def build_pretrain_dataset(args, tokenizer) -> PretrainDataset:
    return PretrainDataset(
        main_csv_path=args.main_csv_path,
        annotations_folder=args.annotations_folder if not args.annotations_root else None,
        annotations_root=args.annotations_root,
        annotation_levels=args.annotation_levels,
        level_mix=args.level_mix,
        tokenizer=tokenizer,
        image_size=args.image_size,
        max_length=args.max_length,
        sample_mode=args.sample_mode,
        ffmpeg_timeout=args.ffmpeg_timeout,
        max_retry=args.max_retry,
        video_root_folder=args.video_root_folder,
        assume_resized_video=args.assume_resized_video,
        num_frames=args.num_frames,
        return_level_id=True,
        return_sample_index=True,
        return_expanded_frames=True,
        expanded_window_ratio=args.primary_window_scale,
        samples_cache_dir=args.samples_cache_dir,
        use_samples_cache=args.use_samples_cache,
        rebuild_samples_cache=args.rebuild_samples_cache,
        samples_cache_version=args.samples_cache_version,
        video_reader_threads=args.video_reader_threads,
        video_reader_cache_size=args.video_reader_cache_size,
    )


def select_samples(samples: Sequence[Dict], levels: Sequence[str], max_samples: int, seed: int) -> List[Tuple[int, Dict]]:
    allowed_levels = set(levels)
    indexed = [
        (idx, sample)
        for idx, sample in enumerate(samples)
        if str(sample.get("level", "")).lower() in allowed_levels
    ]
    if not indexed:
        raise ValueError(f"No samples found for levels: {sorted(allowed_levels)}")

    if max_samples <= 0 or max_samples >= len(indexed):
        rng = random.Random(seed)
        indexed = list(indexed)
        rng.shuffle(indexed)
        return indexed

    by_level = defaultdict(list)
    for item in indexed:
        by_level[str(item[1].get("level", "")).lower()].append(item)

    rng = random.Random(seed)
    non_empty_levels = [level for level in levels if by_level.get(level)]
    per_level = max_samples // len(non_empty_levels)
    remainder = max_samples % len(non_empty_levels)

    chosen = []
    leftovers = []
    for level_pos, level in enumerate(non_empty_levels):
        candidates = list(by_level[level])
        rng.shuffle(candidates)
        take = min(len(candidates), per_level + (1 if level_pos < remainder else 0))
        chosen.extend(candidates[:take])
        leftovers.extend(candidates[take:])

    if len(chosen) < max_samples and leftovers:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: max_samples - len(chosen)])

    rng.shuffle(chosen)
    return chosen[:max_samples]


def sanitize_stem(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text)
    cleaned = cleaned.strip("._")
    return cleaned or "sample"


def distance_to_reference(timestamp: float, reference_start: float, reference_end: float) -> float:
    if timestamp < reference_start:
        return reference_start - timestamp
    if timestamp > reference_end:
        return timestamp - reference_end
    return 0.0


def sample_window_with_dataset(
    dataset: PretrainDataset,
    sample: Dict,
    window_spec: Dict,
    seed: int,
) -> Dict:
    video_path = sample["video_path"]
    vr = dataset._get_video_reader(video_path)
    num_video_frames = len(vr)
    if num_video_frames <= 0:
        raise RuntimeError(f"Empty video: {video_path}")

    fps = float(vr.get_avg_fps())
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    duration = max(num_video_frames - 1, 0) / fps

    reference_start = float(sample["start_time"])
    reference_end = float(sample["end_time"])
    if reference_end < reference_start:
        reference_start, reference_end = reference_end, reference_start

    if window_spec["mode"] == "scale":
        window_start, window_end = dataset._expand_window(
            reference_start,
            reference_end,
            video_duration=duration,
            expand_ratio=window_spec["value"],
        )
    else:
        offset = max(float(window_spec["value"]), 0.0)
        window_start = max(0.0, reference_start - offset)
        window_end = min(duration, reference_end + offset)

    state = random.getstate()
    random.seed(int(seed))
    try:
        timestamps = dataset._sample_timestamps(
            window_start,
            window_end,
            video_duration=duration,
        )
    finally:
        random.setstate(state)

    frame_indices = dataset._timestamps_to_frame_indices(
        timestamps,
        fps=fps,
        num_video_frames=num_video_frames,
        video_duration=duration,
    )
    if frame_indices is None:
        raise RuntimeError(f"Could not sample frames from: {video_path}")

    frames_np = vr.get_batch(frame_indices).asnumpy()
    frames = dataset._postprocess_frames(frames_np)
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)

    in_reference = [
        bool(reference_start <= timestamp <= reference_end)
        for timestamp in timestamps
    ]
    distances = [
        distance_to_reference(timestamp, reference_start, reference_end)
        for timestamp in timestamps
    ]

    return {
        "video_path": video_path,
        "window_mode": window_spec["mode"],
        "window_value": window_spec["value"],
        "window_label": window_spec["label"],
        "window_sort_key": window_spec["sort_key"],
        "reference_start": reference_start,
        "reference_end": reference_end,
        "window_start": window_start,
        "window_end": window_end,
        "duration": duration,
        "fps": fps,
        "times": [float(x) for x in timestamps],
        "frame_indices": [int(x) for x in frame_indices],
        "frames": frames,
        "in_reference": in_reference,
        "distances": distances,
    }


@torch.no_grad()
def compute_window_outputs(
    frames: torch.Tensor,
    text_context: Dict,
    level: str,
    model: VLP,
    device: str,
) -> Dict:
    image = frames.unsqueeze(0).to(device)
    _, frame_tokens = model._encode_image_tokens(image)
    level_id = LEVEL_TO_ID.get(str(level).lower(), 1)
    level_ids = torch.tensor([level_id], dtype=torch.long, device=device)

    frame_weights, confidence = model._compute_frame_selection_weights(
        frame_tokens=frame_tokens,
        token_hidden=text_context["token_hidden"],
        attention_mask=text_context["attention_mask"],
        level_ids=level_ids,
    )
    uniform_weights = torch.full_like(frame_weights, 1.0 / max(frame_weights.size(-1), 1))

    text_features = text_context["text_features"]
    logit_scale = model.logit_scale.exp()
    selected_features = model._project_selected_video(frame_tokens, frame_weights)
    uniform_features = model._project_selected_video(frame_tokens, uniform_weights)
    frame_features = F.normalize(model.video_projection(frame_tokens), dim=-1)

    selected_score = (logit_scale * selected_features @ text_features.t()).squeeze()
    uniform_score = (logit_scale * uniform_features @ text_features.t()).squeeze()
    frame_scores = (logit_scale * frame_features.squeeze(0) @ text_features.t()).squeeze(-1)

    return {
        "frame_scores": frame_scores.detach().cpu().numpy(),
        "frame_weights": frame_weights.squeeze(0).detach().cpu().numpy(),
        "selected_score": float(selected_score.detach().cpu().item()),
        "uniform_score": float(uniform_score.detach().cpu().item()),
        "pair_confidence": float(confidence.squeeze(0).detach().cpu().item()),
        "frame_entropy": float(model._normalized_entropy(frame_weights).squeeze(0).detach().cpu().item()),
        "frame_peak": float(frame_weights.max(dim=-1).values.squeeze(0).detach().cpu().item()),
    }


def topk_mass(weights: Sequence[float], top_k: int) -> float:
    values = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    k = min(max(1, int(top_k)), values.size)
    return float(np.sort(values)[-k:].sum())


def safe_mean(values: Iterable[float]) -> Optional[float]:
    clean = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            clean.append(value)
    if not clean:
        return None
    return float(np.mean(clean))


def safe_std(values: Iterable[float]) -> Optional[float]:
    clean = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            clean.append(value)
    if len(clean) <= 1:
        return 0.0 if clean else None
    return float(np.std(clean, ddof=1))


def weighted_average(scores: np.ndarray, weights: np.ndarray) -> Optional[float]:
    if scores.size == 0 or weights.size == 0:
        return None
    total = weights.sum()
    if total <= 0:
        return None
    weights = weights / total
    return float(np.sum(scores * weights))


def analyze_sampled_window(
    sample: Dict,
    sample_index: int,
    window_data: Dict,
    model_outputs: Dict,
    args,
) -> Dict:
    times = np.asarray(window_data["times"], dtype=np.float64)
    frame_indices = np.asarray(window_data["frame_indices"], dtype=np.int64)
    scores = np.asarray(model_outputs["frame_scores"], dtype=np.float64)
    expanded_weights_np = np.asarray(model_outputs["frame_weights"], dtype=np.float64)
    in_reference = np.asarray(window_data["in_reference"], dtype=bool)
    distances = np.asarray(window_data["distances"], dtype=np.float64)

    inside_idx = np.where(in_reference)[0]
    outside_idx = np.where(~in_reference)[0]
    if inside_idx.size == 0:
        inside_idx = np.arange(scores.size)
        in_reference = np.ones_like(in_reference, dtype=bool)
        outside_idx = np.asarray([], dtype=np.int64)

    best_idx = int(np.argmax(scores))
    best_inside_idx = int(inside_idx[np.argmax(scores[inside_idx])])
    best_outside_idx = None
    if outside_idx.size > 0:
        best_outside_idx = int(outside_idx[np.argmax(scores[outside_idx])])

    best_inside_score = float(scores[best_inside_idx])
    best_outside_score = None if best_outside_idx is None else float(scores[best_outside_idx])
    best_outside_time = None if best_outside_idx is None else float(times[best_outside_idx])
    outside_margin = None
    outside_discovery = False
    if best_outside_score is not None:
        outside_margin = best_outside_score - best_inside_score
        outside_discovery = outside_margin > 0.0

    top_weight_idx = int(np.argmax(expanded_weights_np))
    weighted_distance = weighted_average(distances, expanded_weights_np)
    outside_weight_mass = float(expanded_weights_np[outside_idx].sum()) if outside_idx.size > 0 else 0.0
    top_k_by_score = np.argsort(scores)[-min(int(args.top_k), scores.size):]
    top_k_outside_ratio = float((~in_reference[top_k_by_score]).sum() / max(top_k_by_score.size, 1))
    selection_ranks = np.argsort(np.argsort(-expanded_weights_np)) + 1

    points = []
    for pos, (timestamp, frame_idx, score, weight, is_inside, distance) in enumerate(
        zip(times, frame_indices, scores, expanded_weights_np, in_reference, distances)
    ):
        points.append(
            {
                "time": float(timestamp),
                "frame_index": int(frame_idx),
                "score": float(score),
                "selection_weight": float(weight),
                "selection_rank": int(selection_ranks[pos]),
                "in_reference_window": bool(is_inside),
                "distance_to_reference_sec": float(distance),
            }
        )

    level = str(sample.get("level", ""))
    video_id = Path(sample["video_path"]).stem
    return {
        "sample_index": int(sample_index),
        "sample_id": f"idx_{sample_index:06d}_{sanitize_stem(video_id)}",
        "video_id": video_id,
        "video_path": sample["video_path"],
        "caption": sample["caption"],
        "level": level,
        "window_mode": window_data["window_mode"],
        "window_value": window_data["window_value"],
        "window_label": window_data["window_label"],
        "window_sort_key": window_data["window_sort_key"],
        "window_offset_sec": float(window_data["window_value"]) if window_data["window_mode"] == "offset" else None,
        "window_scale": float(window_data["window_value"]) if window_data["window_mode"] == "scale" else None,
        "reference_start": window_data["reference_start"],
        "reference_end": window_data["reference_end"],
        "reference_duration": window_data["reference_end"] - window_data["reference_start"],
        "window_start": window_data["window_start"],
        "window_end": window_data["window_end"],
        "window_duration": window_data["window_end"] - window_data["window_start"],
        "duration": window_data["duration"],
        "fps": window_data["fps"],
        "num_points": int(scores.size),
        "num_inside_points": int(inside_idx.size),
        "num_outside_points": int(outside_idx.size),
        "best_time": float(times[best_idx]),
        "best_score": float(scores[best_idx]),
        "best_in_reference_window": bool(in_reference[best_idx]),
        "best_inside_time": float(times[best_inside_idx]),
        "best_inside_score": best_inside_score,
        "best_outside_time": best_outside_time,
        "best_outside_score": best_outside_score,
        "outside_discovery": bool(outside_discovery),
        "outside_discovery_margin": outside_margin,
        "score_gain_vs_inside_best": float(scores[best_idx] - best_inside_score),
        "score_gain_vs_inside_mean": float(scores[best_idx] - np.mean(scores[inside_idx])),
        "top_k_score_outside_ratio": top_k_outside_ratio,
        "original_uniform_score": None,
        "original_text_reselect_score": None,
        "original_reselect_gain_over_uniform": None,
        "expanded_uniform_score": model_outputs["uniform_score"],
        "expanded_text_reselect_score": model_outputs["selected_score"],
        "expanded_reselect_gain_over_uniform": model_outputs["selected_score"] - model_outputs["uniform_score"],
        "outside_weight_mass": outside_weight_mass,
        "selection_entropy": model_outputs["frame_entropy"],
        "top_k_weight_mass": topk_mass(expanded_weights_np, args.top_k),
        "weighted_distance_to_reference_sec": weighted_distance,
        "top_weight_time": float(times[top_weight_idx]),
        "top_weight_score": float(scores[top_weight_idx]),
        "top_weight_in_reference_window": bool(in_reference[top_weight_idx]),
        "expanded_pair_confidence": model_outputs["pair_confidence"],
        "original_pair_confidence": None,
        "frame_peak": model_outputs["frame_peak"],
        "score_type": "projected_frame_text_logit",
        "points": points,
    }


def flat_record(record: Dict) -> Dict:
    return {key: value for key, value in record.items() if key != "points"}


def write_csv(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        seen = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_records(records: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for record in records:
        groups[(record["window_label"], "all")].append(record)
        groups[(record["window_label"], record.get("level", ""))].append(record)

    rows = []
    for (window_label, level), group in sorted(
        groups.items(),
        key=lambda item: (float(item[1][0].get("window_sort_key", 0.0)), item[0][1]),
    ):
        first = group[0]
        rows.append(
            {
                "window_label": window_label,
                "window_mode": first.get("window_mode"),
                "window_value": first.get("window_value"),
                "window_offset_sec": first.get("window_offset_sec"),
                "window_scale": first.get("window_scale"),
                "window_sort_key": first.get("window_sort_key"),
                "level": level,
                "num_samples": len(group),
                "reference_duration_mean": safe_mean(
                    r.get("reference_duration") for r in group
                ),
                "window_duration_mean": safe_mean(
                    r.get("window_duration") for r in group
                ),
                "num_points_mean": safe_mean(
                    r.get("num_points") for r in group
                ),
                "outside_best_rate": safe_mean(
                    0.0 if r["best_in_reference_window"] else 1.0 for r in group
                ),
                "outside_discovery_rate": safe_mean(
                    1.0 if r["outside_discovery"] else 0.0 for r in group
                ),
                "outside_discovery_margin_mean": safe_mean(
                    r.get("outside_discovery_margin") for r in group
                ),
                "outside_discovery_margin_std": safe_std(
                    r.get("outside_discovery_margin") for r in group
                ),
                "positive_outside_discovery_margin_mean": safe_mean(
                    r.get("outside_discovery_margin")
                    for r in group
                    if r.get("outside_discovery")
                ),
                "score_gain_vs_inside_best_mean": safe_mean(
                    r.get("score_gain_vs_inside_best") for r in group
                ),
                "score_gain_vs_inside_mean_mean": safe_mean(
                    r.get("score_gain_vs_inside_mean") for r in group
                ),
                "original_uniform_score_mean": safe_mean(
                    r.get("original_uniform_score") for r in group
                ),
                "original_text_reselect_score_mean": safe_mean(
                    r.get("original_text_reselect_score") for r in group
                ),
                "expanded_uniform_score_mean": safe_mean(
                    r.get("expanded_uniform_score") for r in group
                ),
                "expanded_text_reselect_score_mean": safe_mean(
                    r.get("expanded_text_reselect_score") for r in group
                ),
                "expanded_reselect_gain_over_uniform_mean": safe_mean(
                    r.get("expanded_reselect_gain_over_uniform") for r in group
                ),
                "original_reselect_gain_over_uniform_mean": safe_mean(
                    r.get("original_reselect_gain_over_uniform") for r in group
                ),
                "outside_weight_mass_mean": safe_mean(
                    r.get("outside_weight_mass") for r in group
                ),
                "selection_entropy_mean": safe_mean(
                    r.get("selection_entropy") for r in group
                ),
                "top_k_weight_mass_mean": safe_mean(
                    r.get("top_k_weight_mass") for r in group
                ),
                "weighted_distance_to_reference_sec_mean": safe_mean(
                    r.get("weighted_distance_to_reference_sec") for r in group
                ),
            }
        )
    return rows


def closest_primary_window_label(records: List[Dict], args) -> str:
    if not records:
        return ""
    mode = str(records[0].get("window_mode", "scale"))
    target = args.primary_window_scale if mode == "scale" else args.primary_window_offset
    candidates = {}
    for record in records:
        label = str(record["window_label"])
        candidates[label] = float(record["window_value"])
    return min(candidates, key=lambda label: abs(candidates[label] - float(target)))


def write_case_candidates(records: List[Dict], output_dir: Path, primary_window_label: str) -> None:
    primary = [
        record for record in records
        if str(record.get("window_label")) == str(primary_window_label)
    ]
    outside_cases = [
        flat_record(record) for record in primary
        if record.get("outside_discovery") and record.get("outside_discovery_margin") is not None
    ]
    outside_cases.sort(key=lambda row: float(row["outside_discovery_margin"]), reverse=True)
    stable_cases = [
        flat_record(record) for record in primary
        if record.get("best_in_reference_window")
    ]
    stable_cases.sort(key=lambda row: float(row["score_gain_vs_inside_mean"]), reverse=True)
    uncertain_cases = [
        flat_record(record) for record in primary
        if record.get("selection_entropy") is not None
    ]
    uncertain_cases.sort(key=lambda row: float(row["selection_entropy"]), reverse=True)

    preferred = [
        "sample_index",
        "sample_id",
        "video_id",
        "level",
        "window_offset_sec",
        "reference_start",
        "reference_end",
        "window_start",
        "window_end",
        "best_inside_time",
        "best_inside_score",
        "best_outside_time",
        "best_outside_score",
        "outside_discovery_margin",
        "best_time",
        "best_score",
        "top_weight_time",
        "top_weight_score",
        "top_weight_in_reference_window",
        "outside_weight_mass",
        "selection_entropy",
        "expanded_reselect_gain_over_uniform",
        "caption",
        "video_path",
    ]
    write_csv(output_dir / "outside_discovery_cases.csv", outside_cases[:50], preferred)
    write_csv(output_dir / "inside_stable_cases.csv", stable_cases[:50], preferred)
    write_csv(output_dir / "high_entropy_cases.csv", uncertain_cases[:50], preferred)


def get_git_commit(repo_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def pct(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{100.0 * value:.1f}%"


def num(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.3f}"


def table_num(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "--"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def table_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "--"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(value):
        return "--"
    return f"{100.0 * value:.{digits}f}"


def table_window_offset(offset: float) -> str:
    offset = float(offset)
    if math.isclose(offset, 0.0, abs_tol=1e-9):
        return "Original"
    return f"+/-{offset:g}s"


def latex_escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def latex_table(caption: str, label: str, columns: List[str], rows: List[Dict], align: str) -> str:
    header = " & ".join(latex_escape(col) for col in columns) + r" \\"
    body = []
    for row in rows:
        body.append(" & ".join(latex_escape(row.get(col, "")) for col in columns) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{align}}}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def write_paper_tables(records: List[Dict], summary_rows: List[Dict], output_dir: Path, primary_window_label: str, top_k: int) -> None:
    all_summary = [
        row for row in summary_rows
        if row.get("level") == "all"
    ]
    all_summary = sorted(all_summary, key=lambda row: float(row["window_sort_key"]))

    window_rows = []
    for row in all_summary:
        window_rows.append(
            {
                "Window": row["window_label"],
                "N": str(row["num_samples"]),
                "Avg Ref Len (s)": table_num(row.get("reference_duration_mean"), 1),
                "Outside Top (%)": table_pct(row.get("outside_best_rate")),
                "Recoverable Outside (%)": table_pct(row.get("outside_discovery_rate")),
                "Positive Margin": table_num(row.get("positive_outside_discovery_margin_mean")),
                "Score Gain": table_num(row.get("score_gain_vs_inside_mean_mean")),
                "Text > Uniform": table_num(row.get("expanded_reselect_gain_over_uniform_mean")),
                "Outside Weight (%)": table_pct(row.get("outside_weight_mass_mean")),
                f"Top-{top_k} Weight (%)": table_pct(row.get("top_k_weight_mass_mean")),
            }
        )

    primary = [
        record for record in records
        if str(record.get("window_label")) == str(primary_window_label)
    ]
    level_summary = [
        row for row in summary_rows
        if not row.get("level") == "all"
        and str(row.get("window_label")) == str(primary_window_label)
    ]
    level_order = {"fine": 0, "mid": 1, "coarse": 2}
    level_summary = sorted(
        level_summary,
        key=lambda row: level_order.get(str(row.get("level", "")), 99),
    )
    level_rows = []
    for row in level_summary:
        level_rows.append(
            {
                "Level": str(row["level"]),
                "N": str(row["num_samples"]),
                "Avg Ref Len (s)": table_num(row.get("reference_duration_mean"), 1),
                "Recoverable Outside (%)": table_pct(row.get("outside_discovery_rate")),
                "Positive Margin": table_num(row.get("positive_outside_discovery_margin_mean")),
                "Text > Uniform": table_num(row.get("expanded_reselect_gain_over_uniform_mean")),
                "Outside Weight (%)": table_pct(row.get("outside_weight_mass_mean")),
                f"Top-{top_k} Weight (%)": table_pct(row.get("top_k_weight_mass_mean")),
            }
        )

    def mean_primary(key: str) -> Optional[float]:
        return safe_mean(record.get(key) for record in primary)

    original_records = [
        record for record in records
        if str(record.get("window_label")) == "Original"
    ]

    def mean_original(key: str) -> Optional[float]:
        return safe_mean(record.get(key) for record in original_records)

    original_uniform = mean_original("expanded_uniform_score")
    original_text = mean_original("expanded_text_reselect_score")
    expanded_uniform = mean_primary("expanded_uniform_score")
    expanded_text = mean_primary("expanded_text_reselect_score")

    strategy_specs = [
        (
            "Original Uniform",
            "reference interval",
            "uniform",
            original_uniform,
            original_uniform,
            expanded_uniform,
        ),
        (
            "Original Text Reselect",
            "reference interval",
            "text-guided",
            original_text,
            original_uniform,
            expanded_uniform,
        ),
        (
            "Expanded Uniform",
            primary_window_label,
            "uniform",
            expanded_uniform,
            original_uniform,
            expanded_uniform,
        ),
        (
            "Expanded Text Reselect",
            primary_window_label,
            "text-guided",
            expanded_text,
            original_uniform,
            expanded_uniform,
        ),
    ]
    strategy_rows = []
    for name, candidates, rule, score, ref_score, expanded_ref in strategy_specs:
        delta_original = None if score is None or ref_score is None else score - ref_score
        delta_expanded = None if score is None or expanded_ref is None else score - expanded_ref
        strategy_rows.append(
            {
                "Strategy": name,
                "Candidates": candidates,
                "Selection": rule,
                "Mean Score": table_num(score),
                "Delta vs Original": table_num(delta_original),
                "Delta vs Expanded Uniform": table_num(delta_expanded),
            }
        )

    write_csv(output_dir / "paper_table_window_sweep.csv", window_rows)
    write_csv(output_dir / "paper_table_level_breakdown.csv", level_rows)
    write_csv(output_dir / "paper_table_selection_strategies.csv", strategy_rows)

    latex_parts = [
        latex_table(
            caption=(
                "Automatic weak-positive diagnostics under different expansion windows. "
                "Recoverable Outside denotes samples where the best outside-window frame "
                "scores higher than the best frame inside the reference interval."
            ),
            label="tab:weak_positive_window",
            columns=list(window_rows[0].keys()) if window_rows else [],
            rows=window_rows,
            align="lrrrrrrrrr",
        ) if window_rows else "",
        latex_table(
            caption=(
                f"Level-wise weak-positive diagnostics at the primary {primary_window_label} window."
            ),
            label="tab:weak_positive_level",
            columns=list(level_rows[0].keys()) if level_rows else [],
            rows=level_rows,
            align="lrrrrrrr",
        ) if level_rows else "",
        latex_table(
            caption=(
                "Automatic strategy comparison on the same sampled pairs. "
                "Scores are frame-text matching scores averaged over the diagnostic set."
            ),
            label="tab:weak_positive_strategy",
            columns=list(strategy_rows[0].keys()) if strategy_rows else [],
            rows=strategy_rows,
            align="lllrrr",
        ) if strategy_rows else "",
    ]
    (output_dir / "paper_tables.tex").write_text(
        "\n\n".join(part for part in latex_parts if part),
        encoding="utf-8",
    )


def write_paper_summary(records: List[Dict], summary_rows: List[Dict], output_dir: Path, primary_window_label: str) -> None:
    primary = [
        record for record in records
        if str(record.get("window_label")) == str(primary_window_label)
    ]
    primary_all = next(
        row for row in summary_rows
        if str(row.get("window_label")) == str(primary_window_label)
        and row["level"] == "all"
    )
    all_rows = [row for row in summary_rows if row["level"] == "all"]
    best_gain = max(
        all_rows,
        key=lambda row: (
            -1e9
            if row["expanded_reselect_gain_over_uniform_mean"] is None
            else row["expanded_reselect_gain_over_uniform_mean"]
        ),
    )

    lines = [
        "Weak-positive diagnostics without manual labels",
        "",
        f"Primary window: {primary_window_label}",
        f"Valid samples at primary window: {len(primary)}",
        (
            "Outside-reference discovery rate: "
            f"{pct(primary_all['outside_discovery_rate'])}"
        ),
        (
            "Outside best-frame rate: "
            f"{pct(primary_all['outside_best_rate'])}"
        ),
        (
            "Mean positive outside discovery margin: "
            f"{num(primary_all['positive_outside_discovery_margin_mean'])}"
        ),
        (
            "Text-guided reselect gain over expanded uniform: "
            f"{num(primary_all['expanded_reselect_gain_over_uniform_mean'])}"
        ),
        (
            "Mean outside selection weight mass: "
            f"{pct(primary_all['outside_weight_mass_mean'])}"
        ),
        (
            "Best window by text-vs-uniform gain: "
            f"{best_gain['window_label']} "
            f"({num(best_gain['expanded_reselect_gain_over_uniform_mean'])})"
        ),
        "",
        "Recommended paper wording:",
        (
            "We do not use frame-level manual labels. Instead, we treat the "
            "provided intervals as weak reference windows and evaluate weak-positive "
            "construction with automatic diagnostics. The analysis measures whether "
            "high-scoring text-matched frames appear near but outside the reference "
            "interval, whether moderate expansion recovers such candidates, and "
            "whether text-guided reselection improves over uniform use of the "
            "expanded window."
        ),
    ]
    (output_dir / "paper_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def write_run_config(args, output_dir: Path, selected_count: int) -> None:
    repo_dir = Path(__file__).resolve().parent
    payload = {
        "script": str(Path(__file__).resolve()),
        "git_commit": get_git_commit(repo_dir),
        "selected_samples": selected_count,
        "args": vars(args),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{device} was requested, but CUDA is not available.")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    model = build_model(args, device)
    dataset = build_pretrain_dataset(args, tokenizer)
    samples = dataset.samples
    levels = parse_str_list(args.levels)
    window_specs = build_window_specs(args)
    selected = select_samples(samples, levels, args.max_samples, args.seed)

    records = []
    errors = []
    jsonl_path = output_dir / "weak_positive_diagnostics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for ordinal, (sample_index, sample) in enumerate(selected, start=1):
            try:
                input_ids, attention_mask = dataset._build_text(sample["caption"])
                input_ids = input_ids.unsqueeze(0).to(device)
                attention_mask = attention_mask.unsqueeze(0).to(device)
                _, token_hidden, text_global_hidden = model.text(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_hidden=True,
                )
                text_context = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_hidden": token_hidden,
                    "text_features": F.normalize(model.text.project(text_global_hidden), dim=-1),
                }

                for spec_pos, window_spec in enumerate(window_specs):
                    window_seed = int(args.seed + sample_index * 1009 + spec_pos * 9173)
                    window_data = sample_window_with_dataset(
                        dataset=dataset,
                        sample=sample,
                        window_spec=window_spec,
                        seed=window_seed,
                    )
                    model_outputs = compute_window_outputs(
                        frames=window_data["frames"],
                        text_context=text_context,
                        level=str(sample.get("level", "")),
                        model=model,
                        device=device,
                    )
                    record = analyze_sampled_window(
                        sample=sample,
                        sample_index=sample_index,
                        window_data=window_data,
                        model_outputs=model_outputs,
                        args=args,
                    )
                    records.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as exc:
                errors.append(
                    {
                        "sample_index": sample_index,
                        "video_path": sample.get("video_path", ""),
                        "caption": sample.get("caption", ""),
                        "level": sample.get("level", ""),
                        "error": str(exc),
                    }
                )
            if ordinal % 25 == 0:
                print(f"Processed {ordinal}/{len(selected)} selected samples", flush=True)

    flat_rows = [flat_record(record) for record in records]
    summary_rows = aggregate_records(records)
    primary_window_label = closest_primary_window_label(records, args) if records else ""

    write_csv(output_dir / "per_window_metrics.csv", flat_rows)
    write_csv(output_dir / "summary_by_window.csv", summary_rows)
    write_csv(output_dir / "errors.csv", errors)
    if records:
        write_case_candidates(records, output_dir, primary_window_label)
        write_paper_tables(records, summary_rows, output_dir, primary_window_label, args.top_k)
        write_paper_summary(records, summary_rows, output_dir, primary_window_label)
    write_run_config(args, output_dir, len(selected))

    print(f"Saved per-sample JSONL to: {jsonl_path}")
    print(f"Saved per-window CSV to: {output_dir / 'per_window_metrics.csv'}")
    print(f"Saved summary CSV to: {output_dir / 'summary_by_window.csv'}")
    print(f"Saved paper-ready tables to: {output_dir / 'paper_table_window_sweep.csv'}")
    print(f"Saved paper summary to: {output_dir / 'paper_summary.txt'}")
    if errors:
        print(f"Encountered {len(errors)} failed samples; see {output_dir / 'errors.csv'}")


if __name__ == "__main__":
    main()
