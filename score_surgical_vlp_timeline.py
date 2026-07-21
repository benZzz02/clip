#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from decord import VideoReader, cpu

from pretrain_manifest_cache import load_or_build_pretrain_samples
from surgical_vlp_scorers import available_scorers, load_scorer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use frozen surgical VLP scorers to probe timestamp-window quality."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_root", type=str, default="/data/znh/surgical_vlp")
    parser.add_argument(
        "--scorers",
        type=str,
        default="surgclip_beta",
        help="Comma-separated scorer names, or 'all'.",
    )
    parser.add_argument(
        "--clinical_bert_path",
        type=str,
        default=None,
        help="Required for SurgVLP/HecVL/PeskaVLP unless cached locally.",
    )
    parser.add_argument(
        "--strict_scorer_load",
        action="store_true",
        help="Fail immediately if any requested scorer cannot be loaded.",
    )

    parser.add_argument("--main_csv_path", type=str, required=True)
    parser.add_argument("--video_root_folder", type=str, required=True)
    parser.add_argument("--annotations_folder", type=str, default=None)
    parser.add_argument("--annotations_root", type=str, default=None)
    parser.add_argument("--annotation_levels", type=str, default=None)
    parser.add_argument("--level_mix", type=str, default="concat")
    parser.add_argument("--samples_cache_dir", type=str, default=".cache/pretrain_samples")
    parser.add_argument("--use_samples_cache", action="store_true", default=False)
    parser.add_argument("--rebuild_samples_cache", action="store_true")
    parser.add_argument("--samples_cache_version", type=str, default="v1")

    parser.add_argument("--sample_index", type=int, nargs="*", default=None)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--levels", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--window_scale",
        type=float,
        default=1.5,
        help="Search window length as a multiple of the reference duration. Set <=0 to use fixed offsets.",
    )
    parser.add_argument("--start_offset_sec", type=float, default=10.0)
    parser.add_argument("--end_offset_sec", type=float, default=10.0)
    parser.add_argument("--sample_every_sec", type=float, default=1.0)
    parser.add_argument(
        "--num_sample_points",
        type=int,
        default=None,
        help="If set, sample exactly this many frame timestamps uniformly within each window.",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frame_batch_size", type=int, default=64)
    return parser.parse_args()


def parse_list(spec: str) -> List[str]:
    return [x.strip().lower() for x in str(spec).split(",") if x.strip()]


def build_samples(args) -> List[Dict]:
    return load_or_build_pretrain_samples(
        main_csv_path=args.main_csv_path,
        video_root_folder=args.video_root_folder,
        annotations_folder=args.annotations_folder,
        annotations_root=args.annotations_root,
        annotation_levels=args.annotation_levels,
        level_mix=args.level_mix,
        samples_cache_dir=args.samples_cache_dir,
        use_samples_cache=args.use_samples_cache,
        rebuild_samples_cache=args.rebuild_samples_cache,
        samples_cache_version=args.samples_cache_version,
    )


def select_samples(samples: Sequence[Dict], args) -> List[tuple]:
    if args.sample_index:
        selected = []
        for idx in args.sample_index:
            if idx < 0 or idx >= len(samples):
                raise IndexError(f"sample_index {idx} out of range [0, {len(samples)})")
            selected.append((idx, samples[idx]))
        return selected

    allowed_levels = set(parse_list(args.levels))
    indexed = [
        (idx, sample)
        for idx, sample in enumerate(samples)
        if not allowed_levels or str(sample.get("level", "")).lower() in allowed_levels
    ]
    rng = random.Random(args.seed)
    rng.shuffle(indexed)
    if args.max_samples > 0:
        indexed = indexed[: args.max_samples]
    return indexed


def build_search_window(
    reference_start: float,
    reference_end: float,
    video_duration: float,
    window_scale: float,
    start_offset_sec: float,
    end_offset_sec: float,
):
    reference_start = float(reference_start)
    reference_end = float(reference_end)
    if reference_end < reference_start:
        reference_start, reference_end = reference_end, reference_start
    reference_duration = max(reference_end - reference_start, 1e-6)
    if float(window_scale) > 0.0:
        scaled_duration = reference_duration * max(float(window_scale), 1.0)
        extra = max(scaled_duration - reference_duration, 0.0)
        before = extra * 0.5
        after = extra * 0.5
    else:
        before = max(float(start_offset_sec), 0.0)
        after = max(float(end_offset_sec), 0.0)
    window_start = max(0.0, reference_start - before)
    window_end = min(float(video_duration), reference_end + after)
    return window_start, max(window_start, window_end)


def build_sample_times(
    window_start: float,
    window_end: float,
    stride_sec: float,
    num_sample_points: int = None,
) -> List[float]:
    if num_sample_points is not None:
        num_sample_points = int(num_sample_points)
        if num_sample_points <= 0:
            raise ValueError("--num_sample_points must be positive when set.")
        if num_sample_points == 1:
            return [round((float(window_start) + float(window_end)) * 0.5, 3)]
        return [
            round(float(x), 3)
            for x in np.linspace(float(window_start), float(window_end), num_sample_points)
        ]

    stride_sec = max(float(stride_sec), 1e-6)
    times = []
    cur = float(window_start)
    while cur <= float(window_end) + 1e-9:
        times.append(round(cur, 3))
        cur += stride_sec
    return times or [round(float(window_start), 3)]


def frame_index_from_time(timestamp: float, fps: float, num_frames: int) -> int:
    frame_idx = int(round(float(timestamp) * float(fps)))
    return min(max(frame_idx, 0), max(num_frames - 1, 0))


def analyze_scores(sample, sample_index, scorer_name, times, scores, args) -> Dict:
    reference_start = float(sample["start_time"])
    reference_end = float(sample["end_time"])
    if reference_end < reference_start:
        reference_start, reference_end = reference_end, reference_start
    scores = np.asarray(scores, dtype=np.float64)
    inside = np.asarray(
        [reference_start <= float(t) <= reference_end for t in times],
        dtype=bool,
    )
    if not inside.any():
        inside[:] = True
    outside = ~inside

    best_idx = int(np.argmax(scores))
    inside_idx = np.where(inside)[0]
    outside_idx = np.where(outside)[0]
    best_inside_idx = int(inside_idx[np.argmax(scores[inside_idx])])
    best_outside_idx = None
    if outside_idx.size > 0:
        best_outside_idx = int(outside_idx[np.argmax(scores[outside_idx])])

    best_inside_score = float(scores[best_inside_idx])
    best_outside_score = "" if best_outside_idx is None else float(scores[best_outside_idx])
    outside_margin = ""
    normalized_margin = ""
    outside_discovery = False
    if best_outside_idx is not None:
        outside_margin = float(scores[best_outside_idx] - best_inside_score)
        std = float(np.std(scores))
        normalized_margin = outside_margin / max(std, 1e-8)
        outside_discovery = outside_margin > 0.0

    k = min(max(1, int(args.top_k)), scores.size)
    top_idx = np.argsort(scores)[-k:]
    top_k_outside_ratio = float(outside[top_idx].sum() / max(k, 1))

    return {
        "scorer": scorer_name,
        "sample_index": int(sample_index),
        "level": sample.get("level", ""),
        "video_path": sample["video_path"],
        "caption": sample["caption"],
        "reference_start": reference_start,
        "reference_end": reference_end,
        "window_start": float(min(times)) if times else "",
        "window_end": float(max(times)) if times else "",
        "num_points": int(scores.size),
        "best_time": float(times[best_idx]),
        "best_score": float(scores[best_idx]),
        "best_in_reference_window": bool(inside[best_idx]),
        "best_inside_time": float(times[best_inside_idx]),
        "best_inside_score": best_inside_score,
        "best_outside_time": "" if best_outside_idx is None else float(times[best_outside_idx]),
        "best_outside_score": best_outside_score,
        "outside_discovery": bool(outside_discovery),
        "outside_discovery_margin": outside_margin,
        "normalized_outside_margin": normalized_margin,
        "top_k": k,
        "top_k_outside_ratio": top_k_outside_ratio,
        "top_k_times": json.dumps([float(times[i]) for i in top_idx[::-1]]),
        "top_k_scores": json.dumps([float(scores[i]) for i in top_idx[::-1]]),
        "top_k_in_reference": json.dumps([bool(inside[i]) for i in top_idx[::-1]]),
    }


def score_sample(sample, sample_index, scorer, scorer_name, args) -> Dict:
    vr = VideoReader(sample["video_path"], ctx=cpu(0), num_threads=1)
    num_video_frames = len(vr)
    fps = float(vr.get_avg_fps())
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    duration = max(num_video_frames - 1, 0) / fps
    window_start, window_end = build_search_window(
        sample["start_time"],
        sample["end_time"],
        duration,
        args.window_scale,
        args.start_offset_sec,
        args.end_offset_sec,
    )
    times = build_sample_times(
        window_start,
        window_end,
        args.sample_every_sec,
        num_sample_points=args.num_sample_points,
    )
    frame_indices = [frame_index_from_time(t, fps, num_video_frames) for t in times]
    frames_np = vr.get_batch(frame_indices).asnumpy()
    scores = scorer.score_frames(
        frames_np,
        sample["caption"],
        batch_size=args.frame_batch_size,
    )
    return analyze_scores(sample, sample_index, scorer_name, times, scores, args)


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "scorer",
        "level",
        "sample_index",
        "outside_discovery",
        "normalized_outside_margin",
        "outside_discovery_margin",
        "best_in_reference_window",
        "top_k_outside_ratio",
        "best_time",
        "best_inside_time",
        "best_outside_time",
        "best_score",
        "best_inside_score",
        "best_outside_score",
        "reference_start",
        "reference_end",
        "window_start",
        "window_end",
        "num_points",
        "top_k",
        "top_k_times",
        "top_k_scores",
        "top_k_in_reference",
        "video_path",
        "caption",
        "error",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + extras
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def summarize(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        grouped[(row["scorer"], "all")].append(row)
        grouped[(row["scorer"], str(row.get("level", "")))].append(row)

    summary = []
    for (scorer, level), group in sorted(grouped.items()):
        margins = [as_float(row.get("normalized_outside_margin")) for row in group]
        margins = [x for x in margins if x is not None]
        positive_margins = [
            x
            for x, row in zip(
                [as_float(r.get("normalized_outside_margin")) for r in group],
                group,
            )
            if x is not None and bool(row.get("outside_discovery"))
        ]
        summary.append(
            {
                "scorer": scorer,
                "level": level,
                "num_samples": len(group),
                "outside_discovery_rate": sum(bool(r.get("outside_discovery")) for r in group) / len(group),
                "outside_best_rate": sum(not bool(r.get("best_in_reference_window")) for r in group) / len(group),
                "top_k_outside_ratio_mean": float(np.mean([float(r["top_k_outside_ratio"]) for r in group])),
                "normalized_margin_mean": "" if not margins else float(np.mean(margins)),
                "positive_normalized_margin_mean": "" if not positive_margins else float(np.mean(positive_margins)),
            }
        )
    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = parse_list(args.scorers)
    if requested == ["all"]:
        requested = available_scorers()

    samples = select_samples(build_samples(args), args)
    all_rows = []
    for scorer_name in requested:
        print(f"Loading scorer: {scorer_name}", flush=True)
        try:
            scorer = load_scorer(
                scorer_name,
                model_root=args.model_root,
                device=args.device,
                clinical_bert_path=args.clinical_bert_path,
            )
        except Exception as exc:
            if args.strict_scorer_load:
                raise
            print(f"Skipping scorer {scorer_name}: {exc}", flush=True)
            all_rows.append({"scorer": scorer_name, "error": str(exc)})
            continue

        for ordinal, (sample_index, sample) in enumerate(samples, start=1):
            try:
                row = score_sample(sample, sample_index, scorer, scorer_name, args)
            except Exception as exc:
                row = {
                    "scorer": scorer_name,
                    "sample_index": sample_index,
                    "level": sample.get("level", ""),
                    "video_path": sample.get("video_path", ""),
                    "caption": sample.get("caption", ""),
                    "error": str(exc),
                }
            all_rows.append(row)
            if ordinal % 25 == 0:
                print(f"  {scorer_name}: {ordinal}/{len(samples)} samples", flush=True)

        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stats_path = output_dir / "timeline_scorer_stats.csv"
    summary_path = output_dir / "timeline_scorer_summary.csv"
    write_csv(stats_path, all_rows)
    write_csv(summary_path, summarize(all_rows))
    print(f"Saved per-sample stats to: {stats_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
