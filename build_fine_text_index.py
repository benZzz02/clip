# build_fine_text_index.py
"""
Build an index mapping video_id -> list of (start_time, end_time, fine_text).
This index is used by the HTG (Hierarchical Text Grounding) method to find
Fine-level texts that overlap with Mid/Coarse samples.
"""

import os
import pickle
import hashlib
import json
from pathlib import Path

import pandas as pd


def _log(msg):
    print(msg, flush=True)


def build_fine_text_index(
    main_csv_path: str,
    fine_annotations_dir: str,
    video_root_folder: str,
) -> dict:
    """
    Build index: video_stem -> list of {start, end, text}
    """
    _log(f"Building fine text index from {fine_annotations_dir}")
    
    main_df = pd.read_csv(main_csv_path)
    index = {}
    
    for _, row in main_df.iterrows():
        relative_video_path = row["video_path"]
        video_filename = os.path.basename(relative_video_path)
        video_stem = os.path.splitext(video_filename)[0]
        full_video_path = os.path.join(video_root_folder, video_filename)
        
        if not os.path.exists(full_video_path):
            continue
        
        fine_csv_path = os.path.join(fine_annotations_dir, f"{video_stem}.csv")
        if not os.path.exists(fine_csv_path):
            continue
        
        try:
            fine_df = pd.read_csv(fine_csv_path)
            entries = []
            
            for _, ann_row in fine_df.iterrows():
                start_time = ann_row.get("start")
                end_time = ann_row.get("end")
                text = ann_row.get("text")
                
                if not isinstance(text, str) or not text.strip():
                    continue
                if pd.isna(start_time) or pd.isna(end_time):
                    continue
                
                start_time = float(start_time)
                end_time = float(end_time)
                
                if end_time <= start_time:
                    continue
                
                entries.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text.strip(),
                })
            
            if entries:
                entries.sort(key=lambda x: x["start"])
                index[video_stem] = entries
                
        except Exception as e:
            _log(f"Error processing {fine_csv_path}: {e}")
    
    _log(f"Built fine text index for {len(index)} videos")
    return index


def get_cache_key(main_csv_path, fine_annotations_dir, video_root_folder, version="v1"):
    payload = {
        "version": version,
        "main_csv": str(Path(main_csv_path).resolve()),
        "fine_dir": str(Path(fine_annotations_dir).resolve()),
        "video_root": str(Path(video_root_folder).resolve()),
    }
    payload_str = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(payload_str.encode()).hexdigest()[:16]


def load_or_build_fine_text_index(
    main_csv_path: str,
    fine_annotations_dir: str,
    video_root_folder: str,
    cache_dir: str = ".cache/fine_text_index",
    use_cache: bool = True,
    rebuild_cache: bool = False,
    version: str = "v1",
) -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_key = get_cache_key(main_csv_path, fine_annotations_dir, video_root_folder, version)
    cache_path = cache_dir / f"fine_index_{cache_key}.pkl"
    
    if use_cache and cache_path.exists() and not rebuild_cache:
        _log(f"Loading fine text index from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    index = build_fine_text_index(
        main_csv_path=main_csv_path,
        fine_annotations_dir=fine_annotations_dir,
        video_root_folder=video_root_folder,
    )
    
    if use_cache:
        _log(f"Saving fine text index to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return index


def find_overlapping_fines(
    fine_entries: list,
    query_start: float,
    query_end: float,
) -> list:
    """
    Find Fine entries that overlap with the query time range.
    """
    if not fine_entries:
        return []
    
    overlapping = []
    
    for entry in fine_entries:
        fine_start = entry["start"]
        fine_end = entry["end"]
        
        if fine_start >= query_end:
            break
        if fine_end <= query_start:
            continue
        
        overlap_start = max(fine_start, query_start)
        overlap_end = min(fine_end, query_end)
        
        if overlap_end > overlap_start:
            overlapping.append(entry)
    
    return overlapping