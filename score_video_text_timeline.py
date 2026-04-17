import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoTokenizer

from model import VLP
from pretrain_manifest_cache import load_or_build_pretrain_samples


PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
PIXEL_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
RESAMPLE_BILINEAR = (
    Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score frame-text alignment over time and render HTML timelines."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to VLP checkpoint.")
    parser.add_argument("--output_dir", type=str, default="visualization", help="Output root.")

    parser.add_argument("--text_model_name", type=str, default="marcobombieri/surgicberta")
    parser.add_argument("--vision_pretrained_weights", type=str, default="lemonfm.pth")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=1, help="Use 1 for per-second single-frame scoring.")
    parser.add_argument("--temporal_num_layers", type=int, default=2)
    parser.add_argument("--temporal_num_heads", type=int, default=12)
    parser.add_argument("--temporal_dropout", type=float, default=0.1)
    parser.add_argument("--temporal_hidden_dim", type=int, default=768)
    parser.add_argument("--local_temperature", type=float, default=0.07)
    parser.add_argument("--selection_pooling", type=str, default="similarity")
    parser.add_argument(
        "--level_frame_temperatures",
        type=str,
        default="0.6,0.9,1.2",
        help="Comma-separated fine,mid,coarse temperatures.",
    )

    parser.add_argument("--main_csv_path", type=str, required=True)
    parser.add_argument("--annotations_folder", type=str, default=None)
    parser.add_argument("--annotations_root", type=str, default=None)
    parser.add_argument("--annotation_levels", type=str, default=None)
    parser.add_argument("--level_mix", type=str, default="concat")
    parser.add_argument("--samples_cache_dir", type=str, default=".cache/pretrain_samples")
    parser.add_argument("--use_samples_cache", action="store_true", default=False)
    parser.add_argument("--rebuild_samples_cache", action="store_true")
    parser.add_argument("--samples_cache_version", type=str, default="v1")
    parser.add_argument("--video_root_folder", type=str, required=True)

    parser.add_argument("--sample_index", type=int, nargs="*", default=None)
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--start_offset_sec", type=float, default=10.0)
    parser.add_argument("--end_offset_sec", type=float, default=10.0)
    parser.add_argument("--sample_every_sec", type=float, default=1.0)
    parser.add_argument("--thumbnail_width", type=int, default=180)
    parser.add_argument("--device", type=str, default="cuda:3")
    return parser.parse_args()


def parse_float_tuple(spec: str, expected_len: int) -> tuple:
    values = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} floats, got {len(values)} from {spec!r}")
    return tuple(values)


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


def sanitize_stem(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text)
    cleaned = cleaned.strip("._")
    return cleaned or "sample"


def preprocess_frame(frame_np: np.ndarray, image_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    tensor = (tensor - PIXEL_MEAN) / PIXEL_STD
    return tensor


def build_model_input(single_frame_tensor: torch.Tensor, model_num_frames: int) -> torch.Tensor:
    if model_num_frames <= 1:
        return single_frame_tensor
    return single_frame_tensor.unsqueeze(1).repeat(1, int(model_num_frames), 1, 1, 1)


def frame_index_from_time(timestamp: float, fps: float, num_frames: int) -> int:
    frame_idx = int(round(float(timestamp) * float(fps)))
    return min(max(frame_idx, 0), max(num_frames - 1, 0))


def build_sample_times(window_start: float, window_end: float, stride_sec: float) -> List[float]:
    stride_sec = max(float(stride_sec), 1e-6)
    times = []
    cur = window_start
    while cur <= window_end + 1e-9:
        times.append(round(cur, 3))
        cur += stride_sec
    if not times:
        times = [round(window_start, 3)]
    return times


@torch.no_grad()
def score_timeline(sample: Dict, model: VLP, tokenizer, args, device: str, sample_dir: Path) -> Dict:
    video_path = sample["video_path"]
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_video_frames = len(vr)
    fps = float(vr.get_avg_fps())
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    duration = max(num_video_frames - 1, 0) / fps

    gt_start = float(sample["start_time"])
    gt_end = float(sample["end_time"])
    window_start = max(0.0, gt_start - float(args.start_offset_sec))
    window_end = min(duration, gt_end + float(args.end_offset_sec))
    if window_end < window_start:
        window_end = window_start

    sampled_times = build_sample_times(window_start, window_end, args.sample_every_sec)
    frame_indices = [frame_index_from_time(t, fps, num_video_frames) for t in sampled_times]
    frames_np = vr.get_batch(frame_indices).asnumpy()

    tokenized = tokenizer(
        sample["caption"],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    text_features = model.encode_text(input_ids=input_ids, attention_mask=attention_mask)
    logit_scale = float(model.logit_scale.exp().detach().cpu().item())

    thumbs_dir = sample_dir / "frames"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    score_points = []
    inside_scores = []
    outside_scores = []

    for idx, (timestamp, frame_idx, frame_np) in enumerate(zip(sampled_times, frame_indices, frames_np)):
        image_tensor = preprocess_frame(frame_np, args.image_size)
        image_tensor = build_model_input(image_tensor, args.num_frames).to(device)
        image_features = model.encode_image(image_tensor)
        score = float((logit_scale * image_features @ text_features.t()).squeeze().detach().cpu().item())

        if gt_start <= timestamp <= gt_end:
            inside_scores.append(score)
        else:
            outside_scores.append(score)

        pil_image = Image.fromarray(frame_np)
        if args.thumbnail_width > 0 and pil_image.width != args.thumbnail_width:
            aspect = pil_image.height / max(pil_image.width, 1)
            pil_image = pil_image.resize(
                (args.thumbnail_width, max(1, int(round(args.thumbnail_width * aspect)))),
                RESAMPLE_BILINEAR,
            )
        frame_filename = f"frame_{idx:03d}_{timestamp:.2f}s.jpg"
        frame_path = thumbs_dir / frame_filename
        pil_image.save(frame_path, quality=90)

        score_points.append(
            {
                "time": timestamp,
                "frame_index": int(frame_idx),
                "score": score,
                "in_reference_window": bool(gt_start <= timestamp <= gt_end),
                "image_relpath": f"frames/{frame_filename}",
            }
        )

    scores = [point["score"] for point in score_points]
    top_idx = int(np.argmax(scores))
    inside_mean = float(np.mean(inside_scores)) if inside_scores else None
    outside_mean = float(np.mean(outside_scores)) if outside_scores else None

    return {
        "video_path": video_path,
        "caption": sample["caption"],
        "level": sample.get("level"),
        "reference_start": gt_start,
        "reference_end": gt_end,
        "window_start": window_start,
        "window_end": window_end,
        "duration": duration,
        "fps": fps,
        "inside_mean_score": inside_mean,
        "outside_mean_score": outside_mean,
        "top_time": score_points[top_idx]["time"],
        "top_score": score_points[top_idx]["score"],
        "top_in_reference_window": score_points[top_idx]["in_reference_window"],
        "points": score_points,
    }


def score_color(score: float, min_score: float, max_score: float) -> str:
    if math.isclose(max_score, min_score):
        alpha = 0.55
    else:
        alpha = 0.25 + 0.75 * ((score - min_score) / (max_score - min_score))
    alpha = min(max(alpha, 0.0), 1.0)
    return f"rgba(22, 101, 52, {alpha:.3f})"


def build_svg(points: List[Dict], reference_start: float, reference_end: float, min_score: float, max_score: float) -> str:
    width = 960
    height = 260
    left = 40
    right = 20
    top = 20
    bottom = 36
    plot_w = width - left - right
    plot_h = height - top - bottom

    times = [point["time"] for point in points]
    start_t = min(times)
    end_t = max(times)
    time_span = max(end_t - start_t, 1e-6)
    score_span = max(max_score - min_score, 1e-6)

    def x_pos(t: float) -> float:
        return left + (t - start_t) / time_span * plot_w

    def y_pos(s: float) -> float:
        return top + (max_score - s) / score_span * plot_h

    ref_left = x_pos(max(min(reference_start, end_t), start_t))
    ref_right = x_pos(max(min(reference_end, end_t), start_t))

    polyline = " ".join(f"{x_pos(p['time']):.1f},{y_pos(p['score']):.1f}" for p in points)
    top_point = max(points, key=lambda point: point["score"])

    ticks = []
    for ratio in [0.0, 0.5, 1.0]:
        tick_score = min_score + ratio * score_span
        y = y_pos(tick_score)
        ticks.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d1d5db" stroke-dasharray="4 4"/>'
            f'<text x="8" y="{y + 4:.1f}" font-size="12" fill="#475569">{tick_score:.2f}</text>'
        )

    time_labels = []
    for point in points:
        x = x_pos(point["time"])
        time_labels.append(
            f'<line x1="{x:.1f}" y1="{height-bottom}" x2="{x:.1f}" y2="{height-bottom+6}" stroke="#64748b"/>'
        )
    for anchor in [points[0], points[len(points) // 2], points[-1]]:
        x = x_pos(anchor["time"])
        time_labels.append(
            f'<text x="{x:.1f}" y="{height-10}" text-anchor="middle" font-size="12" fill="#475569">{anchor["time"]:.1f}s</text>'
        )

    return f"""
  <svg viewBox="0 0 {width} {height}" class="timeline-svg" role="img" aria-label="timeline score plot">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  <rect x="{ref_left:.1f}" y="{top}" width="{max(ref_right - ref_left, 2):.1f}" height="{plot_h}" fill="rgba(34,197,94,0.12)"/>
  {''.join(ticks)}
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#0f172a" stroke-width="1.2"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#0f172a" stroke-width="1.2"/>
  <polyline fill="none" stroke="#2563eb" stroke-width="3" points="{polyline}"/>
  <circle cx="{x_pos(top_point['time']):.1f}" cy="{y_pos(top_point['score']):.1f}" r="5.5" fill="#dc2626"/>
  <text x="{x_pos(top_point['time']):.1f}" y="{top + 12}" text-anchor="middle" font-size="12" fill="#991b1b">best match {top_point['score']:.2f}</text>
  {''.join(time_labels)}
</svg>
""".strip()


def render_sample_html(result: Dict, sample_id: str) -> str:
    points = result["points"]
    scores = [point["score"] for point in points]
    min_score = min(scores)
    max_score = max(scores)
    svg = build_svg(
        points,
        result["reference_start"],
        result["reference_end"],
        min_score,
        max_score,
    )
    top_matches = sorted(points, key=lambda point: point["score"], reverse=True)[: min(3, len(points))]
    low_matches = sorted(points, key=lambda point: point["score"])[: min(3, len(points))]

    cards = []
    for point in points:
        border = "#16a34a" if point["in_reference_window"] else "#cbd5e1"
        if math.isclose(point["time"], result["top_time"], abs_tol=1e-6):
            border = "#dc2626"
        cards.append(
            f"""
            <div class="frame-card" style="border-color:{border};">
              <img src="{html.escape(point['image_relpath'])}" alt="frame at {point['time']:.1f}s"/>
              <div class="frame-meta">
                <div><strong>{point['time']:.1f}s</strong></div>
                <div>score {point['score']:.3f}</div>
                <div>{'reference window' if point['in_reference_window'] else 'outside window'}</div>
              </div>
              <div class="score-bar-wrap">
                <div class="score-bar" style="width:{0 if math.isclose(max_score, min_score) else 100*(point['score']-min_score)/(max_score-min_score):.1f}%; background:{score_color(point['score'], min_score, max_score)};"></div>
              </div>
            </div>
            """.strip()
        )

    def fmt_optional(value):
        return "N/A" if value is None else f"{value:.3f}"

    verdict = (
        "The best-matching moment falls inside the reference window."
        if result["top_in_reference_window"]
        else "The best-matching moment falls outside the reference window."
    )

    def render_focus_cards(title: str, focus_points: List[Dict], accent: str) -> str:
        items = []
        for point in focus_points:
            items.append(
                f"""
                <div class="match-card" style="border-color:{accent};">
                  <img src="{html.escape(point['image_relpath'])}" alt="{html.escape(title)} at {point['time']:.1f}s"/>
                  <div class="frame-meta">
                    <div><strong>{point['time']:.1f}s</strong></div>
                    <div>score {point['score']:.3f}</div>
                    <div>{'reference window' if point['in_reference_window'] else 'outside window'}</div>
                  </div>
                </div>
                """.strip()
            )
        return f"""
        <div class="panel">
          <div class="title" style="font-size:18px;">{html.escape(title)}</div>
          <div class="matches">
            {''.join(items)}
          </div>
        </div>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(sample_id)}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --fg: #0f172a;
      --muted: #475569;
      --card: #ffffff;
      --line: #cbd5e1;
      --blue: #2563eb;
      --green: #16a34a;
      --red: #dc2626;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #eff6ff 0%, var(--bg) 20%, var(--bg) 100%);
      color: var(--fg);
    }}
    .page {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 22px 36px;
    }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
      padding: 20px;
      margin-bottom: 18px;
    }}
    .title {{
      font-size: 22px;
      font-weight: 700;
      margin: 0 0 8px;
    }}
    .caption {{
      font-size: 18px;
      line-height: 1.5;
      margin: 0;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .meta {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #f8fafc;
    }}
    .meta-label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 6px;
    }}
    .meta-value {{
      font-size: 18px;
      font-weight: 700;
    }}
    .timeline-svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .verdict {{
      margin-top: 12px;
      font-size: 16px;
      color: #14532d;
      font-weight: 600;
    }}
    .frames {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 12px;
    }}
    .matches {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .match-card {{
      background: #fff;
      border: 3px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
    }}
    .match-card img {{
      display: block;
      width: 100%;
      height: auto;
      background: #e2e8f0;
    }}
    .frame-card {{
      background: #fff;
      border: 3px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
    }}
    .frame-card img {{
      display: block;
      width: 100%;
      height: auto;
      background: #e2e8f0;
    }}
    .frame-meta {{
      padding: 10px 10px 8px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .score-bar-wrap {{
      padding: 0 10px 12px;
    }}
    .score-bar {{
      height: 10px;
      border-radius: 999px;
      min-width: 8px;
    }}
    .path {{
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
      word-break: break-all;
    }}
    a {{
      color: var(--blue);
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="panel">
      <div class="title">{html.escape(sample_id)}</div>
      <p class="caption">{html.escape(result["caption"])}</p>
      <div class="path">video: {html.escape(result["video_path"])}</div>
      <div class="meta-grid">
        <div class="meta"><div class="meta-label">Reference Window</div><div class="meta-value">{result["reference_start"]:.1f}s - {result["reference_end"]:.1f}s</div></div>
        <div class="meta"><div class="meta-label">Scored Window</div><div class="meta-value">{result["window_start"]:.1f}s - {result["window_end"]:.1f}s</div></div>
        <div class="meta"><div class="meta-label">Best Match Time</div><div class="meta-value">{result["top_time"]:.1f}s</div></div>
        <div class="meta"><div class="meta-label">Best Match Score</div><div class="meta-value">{result["top_score"]:.3f}</div></div>
        <div class="meta"><div class="meta-label">Mean In Window</div><div class="meta-value">{fmt_optional(result["inside_mean_score"])}</div></div>
        <div class="meta"><div class="meta-label">Mean Outside</div><div class="meta-value">{fmt_optional(result["outside_mean_score"])}</div></div>
      </div>
    </div>

    <div class="panel">
      {svg}
      <div class="verdict">{html.escape(verdict)}</div>
      <div class="path" style="margin-top:8px;">Higher scores mean the frame is more semantically consistent with the text. The shaded region is only a source reference window, not strict ground truth.</div>
    </div>

    {render_focus_cards("Top Matched Frames", top_matches, "#dc2626")}
    {render_focus_cards("Lowest Matched Frames", low_matches, "#64748b")}

    <div class="panel">
      <div class="title" style="font-size:18px;">Sampled Frames</div>
      <div class="frames">
        {''.join(cards)}
      </div>
    </div>
  </div>
</body>
</html>
"""


def render_index(entries: List[Dict]) -> str:
    rows = []
    for entry in entries:
        rows.append(
            f"""
            <tr>
              <td><a href="{html.escape(entry['html_relpath'])}">{html.escape(entry['sample_id'])}</a></td>
              <td>{html.escape(entry['caption'])}</td>
              <td>{entry['reference_start']:.1f}s - {entry['reference_end']:.1f}s</td>
              <td>{entry['top_time']:.1f}s</td>
              <td>{entry['top_score']:.3f}</td>
              <td>{'yes' if entry['top_in_reference_window'] else 'no'}</td>
            </tr>
            """.strip()
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Timeline Visualization Index</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f8fafc;
      color: #0f172a;
    }}
    .page {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 22px;
    }}
    h1 {{
      margin: 0 0 18px;
      font-size: 28px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid #e2e8f0;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eff6ff;
    }}
    a {{
      color: #2563eb;
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Timeline Visualization Index</h1>
    <table>
      <thead>
        <tr>
          <th>Sample</th>
          <th>Caption</th>
          <th>Reference Window</th>
          <th>Best Match Time</th>
          <th>Best Match Score</th>
          <th>Best Match In Window</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def select_sample_indices(total: int, explicit_indices: List[int], max_samples: int) -> List[int]:
    if explicit_indices:
        indices = []
        for idx in explicit_indices:
            if idx < 0 or idx >= total:
                raise IndexError(f"sample_index {idx} out of range [0, {total})")
            indices.append(idx)
        return indices
    return list(range(min(total, max_samples)))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but a CUDA device was requested.")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    model = build_model(args, device)
    samples = build_samples(args)
    indices = select_sample_indices(len(samples), args.sample_index, args.max_samples)

    index_entries = []
    results_jsonl_path = output_dir / "results.jsonl"
    with results_jsonl_path.open("w", encoding="utf-8") as jsonl_f:
        for ordinal, sample_idx in enumerate(indices):
            sample = samples[sample_idx]
            sample_id = f"sample_{ordinal:03d}_idx_{sample_idx:06d}_{sanitize_stem(Path(sample['video_path']).stem)}"
            sample_dir = output_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            result = score_timeline(sample, model, tokenizer, args, device, sample_dir)
            html_doc = render_sample_html(result, sample_id)
            html_path = sample_dir / "index.html"
            html_path.write_text(html_doc, encoding="utf-8")

            result_record = {"sample_id": sample_id, "sample_index": sample_idx, **result}
            jsonl_f.write(json.dumps(result_record, ensure_ascii=False) + "\n")

            index_entries.append(
                {
                    "sample_id": sample_id,
                    "caption": sample["caption"],
                    "reference_start": result["reference_start"],
                    "reference_end": result["reference_end"],
                    "top_time": result["top_time"],
                    "top_score": result["top_score"],
                    "top_in_reference_window": result["top_in_reference_window"],
                    "html_relpath": f"{sample_id}/index.html",
                }
            )

    (output_dir / "index.html").write_text(render_index(index_entries), encoding="utf-8")
    print(f"Saved HTML timeline visualization to: {output_dir / 'index.html'}")
    print(f"Saved machine-readable results to: {results_jsonl_path}")


if __name__ == "__main__":
    main()
