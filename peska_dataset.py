"""
Wrapper around PretrainDataset adding triple augmentation and multi-text.

Reuses PretrainDataset's robust video reading, frame sampling, tokenization,
and retry logic. Adds:
  1. SimCLR-style video augmentations (aug1, aug2 views)
  2. Multi-text candidates queried from adjacent hierarchy levels
  3. Dict-based return format

Level mapping: fine→action, mid→keystep, coarse→abstract
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from peska_augmentations import SimCLRVideoAugmentation
from pretrain_manifest_cache import load_or_build_pretrain_samples


class PeskaAugmentedDataset(Dataset):
    """
    Wraps a PretrainDataset and adds augmentations + multi-text.

    Args:
        base_dataset: PretrainDataset instance for primary data loading
        annotation_levels: which level this instance filters for
        use_augmentation: enable SimCLR video augmentation
        augmentation_kwargs: passed to SimCLRVideoAugmentation
        multi_text_candidates: max adjacent-level texts per sample
        multi_text_window_overlap: min IoU for adjacent text matching
    """

    def __init__(
        self,
        base_dataset,
        annotation_levels,
        main_csv_path,
        annotations_root,
        annotations_folder,
        video_root_folder,
        level_seed=42,
        samples_cache_dir=".cache/pretrain_samples",
        use_samples_cache=True,
        samples_cache_version="v1",
        use_augmentation=True,
        augmentation_kwargs=None,
        multi_text_candidates=4,
        multi_text_window_overlap=0.5,
    ):
        super().__init__()
        self.base = base_dataset  # PretrainDataset instance
        self.use_augmentation = use_augmentation
        self.multi_text_candidates = multi_text_candidates
        self.multi_text_window_overlap = multi_text_window_overlap
        self.main_csv_path = main_csv_path
        self.annotations_root = annotations_root
        self.annotations_folder = annotations_folder
        self.video_root_folder = video_root_folder
        self.level_seed = level_seed
        self.samples_cache_dir = samples_cache_dir
        self.use_samples_cache = use_samples_cache
        self.samples_cache_version = samples_cache_version
        self._primary_level = annotation_levels[0] if annotation_levels else "mid"

        # Augmentation
        if self.use_augmentation:
            aug_kwargs = augmentation_kwargs or {}
            self.aug1 = SimCLRVideoAugmentation(**aug_kwargs)
            self.aug2 = SimCLRVideoAugmentation(**aug_kwargs)

        # Multi-text lookup: {video_path: {level: [(start, end, caption), ...]}}
        self._text_lookup = self._build_text_lookup()

    def _build_text_lookup(self):
        """Build lookup table from ALL annotation levels for multi-text query."""
        lookup = {}
        for level in ["fine", "mid", "coarse"]:
            try:
                samples = load_or_build_pretrain_samples(
                    main_csv_path=self.main_csv_path,
                    video_root_folder=self.video_root_folder,
                    annotations_folder=self.annotations_folder,
                    annotations_root=self.annotations_root,
                    annotation_levels=[level],
                    level_mix="concat",
                    level_seed=self.level_seed,
                    samples_cache_dir=self.samples_cache_dir,
                    use_samples_cache=self.use_samples_cache,
                    rebuild_samples_cache=False,
                    samples_cache_version=self.samples_cache_version,
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

    @staticmethod
    def _overlap(s1, e1, s2, e2):
        inter = max(0.0, min(e1, e2) - max(s1, s2))
        union = max(e1, e2) - min(s1, s2)
        return inter / union if union > 0 else 0.0

    def _get_adjacent_texts(self, video_path, start, end):
        """Get texts from adjacent levels with overlapping time windows."""
        if video_path not in self._text_lookup:
            return []
        candidates = []
        for level in ["fine", "mid", "coarse"]:
            if level == self._primary_level:
                continue
            for entry in self._text_lookup[video_path][level]:
                if self._overlap(start, end, entry["start"], entry["end"]) >= self.multi_text_window_overlap:
                    tokens = self.base._build_text(entry["caption"])
                    candidates.append(tokens)
        if len(candidates) > self.multi_text_candidates:
            indices = np.random.choice(len(candidates), self.multi_text_candidates, replace=False)
            candidates = [candidates[i] for i in indices]
        return candidates

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # Reuse PretrainDataset's __getitem__: (images, input_ids, attention_mask, level_id)
        result = self.base[idx]
        images, input_ids, attention_mask, level_id = result[:4]

        # Augmentations
        aug1 = self.aug1(images.clone()) if self.use_augmentation else images.clone()
        aug2 = self.aug2(images.clone()) if self.use_augmentation else images.clone()

        # Multi-text: look up sample info
        item = self.base.samples[idx]
        video_path = item.get("video_path", "")
        start = float(item.get("start_time", 0))
        end = float(item.get("end_time", 0))
        cand_texts = [
            {"input_ids": ids, "attention_mask": mask}
            for ids, mask in self._get_adjacent_texts(video_path, start, end)
        ]

        return {
            "video": images,
            "video_aug1": aug1,
            "video_aug2": aug2,
            "primary_text": {"input_ids": input_ids, "attention_mask": attention_mask},
            "candidate_texts": cand_texts,
            "level_id": level_id.item() if isinstance(level_id, torch.Tensor) else level_id,
        }


def peska_collate_fn(batch):
    """Collate for PeskaAugmentedDataset's dict output."""
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
    if max_c > 0:
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
        c_ids = torch.stack(c_ids_list)
        c_mask = torch.stack(c_mask_list)
    else:
        c_ids = torch.zeros(0, p_ids.shape[1], dtype=torch.long)
        c_mask = torch.zeros(0, p_mask.shape[1], dtype=torch.long)

    return {
        "video": v,
        "video_aug1": a1,
        "video_aug2": a2,
        "primary_text": {"input_ids": p_ids, "attention_mask": p_mask},
        "candidate_texts": {"input_ids": c_ids, "attention_mask": c_mask},
        "level_id": lids,
        "num_candidates": max_c,
    }
