"""
Wrapper around PretrainDataset that adds triple video augmentation.

Reuses PretrainDataset (from pretrain_dataset.py) for ALL data loading:
video reading, frame sampling, tokenization, retry logic — nothing reimplemented.

Adds only:
  1. SimCLR video augmentation (aug1, aug2 views)
  2. Multi-text candidates from adjacent hierarchy levels

Multi-text uses pretrain_manifest_cache.load_or_build_pretrain_samples
(the CLIP repo's own function) to query adjacent-level annotations.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from peska_augmentations import SimCLRVideoAugmentation
from pretrain_manifest_cache import load_or_build_pretrain_samples


class PeskaAugmentedDataset(Dataset):
    """
    Wraps PretrainDataset. Calls base[idx] → adds augmentations + multi-text.

    The base dataset handles: decord video reading, frame sampling,
    tokenizer encoding, normalize, retry-on-error. We just post-process.

    Args:
        base: PretrainDataset instance
        annotation_levels: ["fine"] | ["mid"] | ["coarse"]
        use_augmentation: enable SimCLR video augmentations
        multi_text_candidates: max adjacent-level texts per sample
    """

    def __init__(self, base, annotation_levels,
                 main_csv_path, annotations_root, annotations_folder,
                 video_root_folder,
                 use_augmentation=True, augmentation_kwargs=None,
                 multi_text_candidates=4, multi_text_window_overlap=0.5,
                 level_seed=42):
        super().__init__()
        self.base = base
        self._primary_level = annotation_levels[0] if annotation_levels else "mid"
        self.use_augmentation = use_augmentation
        self.multi_text_candidates = multi_text_candidates
        self.multi_text_overlap = multi_text_window_overlap

        # Augmentation (uses torchvision TF functions — see peska_augmentations.py)
        if use_augmentation:
            ak = augmentation_kwargs or {}
            self.aug1 = SimCLRVideoAugmentation(**ak)
            self.aug2 = SimCLRVideoAugmentation(**ak)

        # Multi-text: reuse load_or_build_pretrain_samples from CLIP repo
        self._text_lookup = _build_text_lookup(
            main_csv_path, annotations_root, annotations_folder,
            video_root_folder, level_seed
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # base[idx] returns (images, input_ids, attention_mask, level_id)
        # ALL data loading (decord, tokenizer, normalize, retry) handled by PretrainDataset
        images, input_ids, attention_mask, level_id = self.base[idx][:4]

        # Augmentations (only net-new code beyond PretrainDataset)
        aug1 = self.aug1(images.clone()) if self.use_augmentation else images.clone()
        aug2 = self.aug2(images.clone()) if self.use_augmentation else images.clone()

        # Multi-text: query adjacent levels from pre-built lookup
        item = self.base.samples[idx]
        cand_texts = _query_adjacent(
            self._text_lookup, item.get("video_path", ""),
            float(item.get("start_time", 0)), float(item.get("end_time", 0)),
            self._primary_level, self.multi_text_candidates, self.multi_text_overlap,
            self.base._build_text
        )

        return {
            "video": images,
            "video_aug1": aug1, "video_aug2": aug2,
            "primary_text": {"input_ids": input_ids, "attention_mask": attention_mask},
            "candidate_texts": cand_texts,
            "level_id": level_id.item() if isinstance(level_id, torch.Tensor) else level_id,
        }


# ==============================================================================
# Multi-text helpers: use load_or_build_pretrain_samples (CLIP repo function)
# ==============================================================================

def _build_text_lookup(main_csv_path, annotations_root, annotations_folder,
                       video_root_folder, level_seed):
    """
    Build {video_path: {level: [(start, end, caption), ...]}} using the
    CLIP repo's load_or_build_pretrain_samples. Called once at init.
    """
    lookup = {}
    for level in ["fine", "mid", "coarse"]:
        try:
            samples = load_or_build_pretrain_samples(
                main_csv_path=main_csv_path,
                video_root_folder=video_root_folder,
                annotations_folder=annotations_folder,
                annotations_root=annotations_root,
                annotation_levels=[level],
                level_mix="concat", level_seed=level_seed,
            )
            for s in samples:
                vp = s.get("video_path", "")
                lookup.setdefault(vp, {"fine": [], "mid": [], "coarse": []})
                lookup[vp][level].append({
                    "start": float(s.get("start_time", 0)),
                    "end": float(s.get("end_time", 0)),
                    "caption": str(s.get("caption", "")),
                })
        except Exception:
            pass
    return lookup


def _overlap(s1, e1, s2, e2):
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def _query_adjacent(lookup, video_path, start, end, primary_level,
                    max_candidates, min_overlap, build_text_fn):
    """Query adjacent-level texts with overlapping time windows."""
    if video_path not in lookup or max_candidates <= 0:
        return []
    cands = []
    for level in ["fine", "mid", "coarse"]:
        if level == primary_level:
            continue
        for e in lookup[video_path][level]:
            if _overlap(start, end, e["start"], e["end"]) >= min_overlap:
                ids, mask = build_text_fn(e["caption"])
                cands.append({"input_ids": ids, "attention_mask": mask})
    if len(cands) > max_candidates:
        idxs = np.random.choice(len(cands), max_candidates, replace=False)
        cands = [cands[i] for i in idxs]
    return cands


# ==============================================================================
# Collate (same pattern as train.py's collate_fn_skip_corrupted)
# ==============================================================================

def peska_collate_fn(batch):
    """Collates dict output of PeskaAugmentedDataset."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    v = torch.stack([b["video"] for b in batch])
    a1 = torch.stack([b["video_aug1"] for b in batch])
    a2 = torch.stack([b["video_aug2"] for b in batch])
    p_ids = torch.stack([b["primary_text"]["input_ids"] for b in batch])
    p_mask = torch.stack([b["primary_text"]["attention_mask"] for b in batch])
    lids = torch.tensor([b["level_id"] for b in batch], dtype=torch.long)

    max_c = max(len(b["candidate_texts"]) for b in batch)
    c_ids_list, c_mask_list = [], []
    for b in batch:
        n = len(b["candidate_texts"])
        for i in range(max_c):
            if i < n:
                c_ids_list.append(b["candidate_texts"][i]["input_ids"])
                c_mask_list.append(b["candidate_texts"][i]["attention_mask"])
            else:
                c_ids_list.append(torch.zeros_like(p_ids[0]))
                c_mask_list.append(torch.zeros_like(p_mask[0]))
    c_ids = torch.stack(c_ids_list) if c_ids_list else torch.zeros(0, p_ids.shape[1], dtype=torch.long)
    c_mask = torch.stack(c_mask_list) if c_mask_list else torch.zeros(0, p_mask.shape[1], dtype=torch.long)

    return {
        "video": v, "video_aug1": a1, "video_aug2": a2,
        "primary_text": {"input_ids": p_ids, "attention_mask": p_mask},
        "candidate_texts": {"input_ids": c_ids, "attention_mask": c_mask},
        "level_id": lids, "num_candidates": max_c,
    }
