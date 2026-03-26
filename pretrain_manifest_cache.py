
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import pandas as pd
import torch.distributed as dist


def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _rank():
    return dist.get_rank() if _is_dist() else 0


def _is_rank0():
    return _rank() == 0


def _log(msg):
    if _is_rank0():
        print(msg, flush=True)


def normalize_annotation_levels(annotation_levels):
    if annotation_levels is None:
        return []
    if isinstance(annotation_levels, str):
        return [x.strip() for x in annotation_levels.split(",") if x.strip()]
    return [str(x).strip() for x in annotation_levels if str(x).strip()]


def resolve_annotation_sources(annotations_folder=None, annotations_root=None, annotation_levels=None):
    levels = normalize_annotation_levels(annotation_levels)

    if annotations_root:
        if not levels:
            levels = ["coarse", "mid", "fine"]

        sources = []
        for level in levels:
            level_dir = os.path.join(annotations_root, level)
            if not os.path.isdir(level_dir):
                raise FileNotFoundError(f"annotation level dir not found: {level_dir}")
            sources.append((level, level_dir))
        return sources

    if not annotations_folder:
        raise ValueError("Either annotations_folder or annotations_root must be provided.")

    level_name = os.path.basename(os.path.normpath(annotations_folder)) or "single"
    return [(level_name, annotations_folder)]


def mix_level_samples(samples_by_level, annotation_sources, level_mix="concat", level_seed=42):
    non_empty = {k: v for k, v in samples_by_level.items() if v}
    if not non_empty:
        return []

    if level_mix == "concat" or len(non_empty) == 1:
        mixed = []
        for level_name, _ in annotation_sources:
            mixed.extend(non_empty.get(level_name, []))
        random.shuffle(mixed)
        return mixed

    if level_mix == "balanced":
        rng = random.Random(int(level_seed))
        target_size = max(len(v) for v in non_empty.values())
        mixed = []

        for level_name, _ in annotation_sources:
            level_samples = non_empty.get(level_name, [])
            if not level_samples:
                continue

            if len(level_samples) > target_size:
                chosen = rng.sample(level_samples, target_size)
            elif len(level_samples) < target_size:
                full_repeats = target_size // len(level_samples)
                remainder = target_size % len(level_samples)
                chosen = level_samples * full_repeats
                if remainder > 0:
                    chosen += rng.sample(level_samples, remainder)
            else:
                chosen = list(level_samples)

            mixed.extend(chosen)

        rng.shuffle(mixed)
        return mixed

    raise ValueError(f"Unsupported level_mix: {level_mix}")


def _file_signature(path):
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}

    st = p.stat()
    return {
        "path": str(p.resolve()),
        "exists": True,
        "size": int(st.st_size),
        "mtime": int(st.st_mtime),
    }


def _dir_signature(path):
    p = Path(path)
    csv_files = sorted(p.glob("*.csv"))
    max_mtime = 0

    for f in csv_files:
        try:
            st = f.stat()
            max_mtime = max(max_mtime, int(st.st_mtime))
        except FileNotFoundError:
            continue

    return {
        "path": str(p.resolve()),
        "num_csv": len(csv_files),
        "max_mtime": max_mtime,
    }


def build_cache_key(
    main_csv_path,
    video_root_folder,
    annotation_sources,
    level_mix="concat",
    level_seed=42,
    samples_cache_version="v1",
):
    payload = {
        "cache_version": str(samples_cache_version),
        "main_csv": _file_signature(main_csv_path),
        "video_root_folder": str(Path(video_root_folder).resolve()),
        "annotation_sources": [
            {
                "level": level_name,
                "dir_sig": _dir_signature(level_dir),
            }
            for level_name, level_dir in annotation_sources
        ],
        "level_mix": str(level_mix),
        "level_seed": int(level_seed),
    }

    payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload_str.encode("utf-8")).hexdigest()[:20]


def get_cache_path(
    samples_cache_dir,
    main_csv_path,
    video_root_folder,
    annotation_sources,
    level_mix="concat",
    level_seed=42,
    samples_cache_version="v1",
):
    cache_dir = Path(samples_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = build_cache_key(
        main_csv_path=main_csv_path,
        video_root_folder=video_root_folder,
        annotation_sources=annotation_sources,
        level_mix=level_mix,
        level_seed=level_seed,
        samples_cache_version=samples_cache_version,
    )
    return cache_dir / f"samples_{cache_key}.pkl"


def save_samples_cache(cache_path, samples, samples_cache_version="v1"):
    payload = {
        "samples": samples,
        "num_samples": len(samples),
        "cache_version": str(samples_cache_version),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_samples_cache(cache_path):
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    samples = payload["samples"]
    _log(f"加载 samples cache: {cache_path} | num_samples={len(samples)}")
    return samples


def build_samples_from_csvs(
    main_csv_path,
    video_root_folder,
    annotation_sources,
    level_mix="concat",
    level_seed=42,
):
    _log("正在准备图文预训练数据...")
    _log(f"标注来源: {[p for _, p in annotation_sources]}")
    _log(f"层级混合方式: {level_mix}")

    main_df = pd.read_csv(main_csv_path)
    samples_by_level = {level_name: [] for level_name, _ in annotation_sources}

    for _, row in main_df.iterrows():
        relative_video_path = row["video_path"]
        video_filename = os.path.basename(relative_video_path)
        video_stem = os.path.splitext(video_filename)[0]
        full_video_path = os.path.join(video_root_folder, video_filename)

        if not os.path.exists(full_video_path):
            continue

        for level_name, annotations_dir in annotation_sources:
            annotation_csv_path = os.path.join(annotations_dir, f"{video_stem}.csv")
            if not os.path.exists(annotation_csv_path):
                continue

            try:
                ann_df = pd.read_csv(annotation_csv_path)

                for _, ann_row in ann_df.iterrows():
                    start_time = ann_row["start"]
                    end_time = ann_row["end"]
                    caption = ann_row["text"]

                    if not isinstance(caption, str):
                        continue
                    if pd.isna(start_time) or pd.isna(end_time):
                        continue

                    start_time = float(start_time)
                    end_time = float(end_time)

                    if end_time <= start_time:
                        continue

                    samples_by_level[level_name].append({
                        "video_path": full_video_path,
                        "caption": caption,
                        "start_time": start_time,
                        "end_time": end_time,
                        "level": level_name,
                    })

            except Exception as e:
                _log(f"处理标注文件 {annotation_csv_path} 时出错: {e}")

    for level_name, level_samples in samples_by_level.items():
        _log(f"level={level_name} | samples={len(level_samples)}")

    samples = mix_level_samples(
        samples_by_level=samples_by_level,
        annotation_sources=annotation_sources,
        level_mix=level_mix,
        level_seed=level_seed,
    )

    _log(f"数据准备完成。共找到 {len(samples)} 个有效的图文样本。")

    if not samples:
        raise ValueError("No valid pretraining samples were found.")

    return samples


def load_or_build_pretrain_samples(
    main_csv_path,
    video_root_folder,
    annotations_folder=None,
    annotations_root=None,
    annotation_levels=None,
    level_mix="concat",
    level_seed=42,
    samples_cache_dir="/mnt/mydisk/CLIP/.cache/pretrain_samples",
    use_samples_cache=True,
    rebuild_samples_cache=False,
    samples_cache_version="v1",
):
    annotation_sources = resolve_annotation_sources(
        annotations_folder=annotations_folder,
        annotations_root=annotations_root,
        annotation_levels=annotation_levels,
    )

    if not use_samples_cache:
        return build_samples_from_csvs(
            main_csv_path=main_csv_path,
            video_root_folder=video_root_folder,
            annotation_sources=annotation_sources,
            level_mix=level_mix,
            level_seed=level_seed,
        )

    cache_path = get_cache_path(
        samples_cache_dir=samples_cache_dir,
        main_csv_path=main_csv_path,
        video_root_folder=video_root_folder,
        annotation_sources=annotation_sources,
        level_mix=level_mix,
        level_seed=level_seed,
        samples_cache_version=samples_cache_version,
    )

    if cache_path.exists() and not rebuild_samples_cache:
        return load_samples_cache(cache_path)

    if _is_dist():
        if _is_rank0():
            _log(f"构建 samples cache: {cache_path}")
            samples = build_samples_from_csvs(
                main_csv_path=main_csv_path,
                video_root_folder=video_root_folder,
                annotation_sources=annotation_sources,
                level_mix=level_mix,
                level_seed=level_seed,
            )
            save_samples_cache(
                cache_path=cache_path,
                samples=samples,
                samples_cache_version=samples_cache_version,
            )
        dist.barrier()
        return load_samples_cache(cache_path)

    _log(f"构建 samples cache: {cache_path}")
    samples = build_samples_from_csvs(
        main_csv_path=main_csv_path,
        video_root_folder=video_root_folder,
        annotation_sources=annotation_sources,
        level_mix=level_mix,
        level_seed=level_seed,
    )
    save_samples_cache(
        cache_path=cache_path,
        samples=samples,
        samples_cache_version=samples_cache_version,
    )
    return load_samples_cache(cache_path)