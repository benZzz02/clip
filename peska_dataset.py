"""
Extended PretrainDataset with triple video augmentation and multi-text support.

PeskaAugmentedPretrainDataset extends the existing PretrainDataset to add:
1. SimCLR-style triple video augmentation (original + aug1 + aug2)
2. Multi-text candidates from adjacent hierarchy levels
3. Dict-based return format for clarity

Does NOT modify the original PretrainDataset -- pure extension via subclassing.

Level mapping (PeskaVLP -> CLIP repo):
  - action (loader 0)  -> fine
  - keystep (loader 1) -> mid
  - abstract (loader 2) -> coarse
"""

import numpy as np
import torch

from pretrain_dataset import PretrainDataset
from pretrain_manifest_cache import load_or_build_pretrain_samples
from peska_augmentations import SimCLRVideoAugmentation


class PeskaAugmentedPretrainDataset(PretrainDataset):
    """
    Extended pretraining dataset with augmentations and multi-text.

    Args (in addition to PretrainDataset args):
        use_augmentation: Enable SimCLR-style video augmentation
        augmentation_size: Output size for augmented crops
        augmentation_scale: Scale range for RandomResizedCrop
        color_jitter_strength: Color jitter intensity
        multi_text_candidates: Max number of adjacent-level texts to include
        multi_text_window_overlap: Min overlap ratio for adjacent text matching
    """

    def __init__(
        self,
        main_csv_path,
        annotations_folder,
        tokenizer,
        image_size=224,
        max_length=256,
        sample_mode="random",
        video_root_folder="downloaded_video_224_test",
        assume_resized_video=False,
        num_frames=1,
        annotations_root=None,
        annotation_levels=None,
        level_mix="concat",
        level_seed=42,
        samples_cache_dir=".cache/pretrain_samples",
        use_samples_cache=True,
        rebuild_samples_cache=False,
        samples_cache_version="v1",
        video_reader_threads=1,
        video_reader_cache_size=16,
        # --- New PeskaVLP-specific args ---
        use_augmentation=True,
        augmentation_size=224,
        augmentation_scale=(0.2, 1.0),
        color_jitter_strength=0.4,
        multi_text_candidates=4,
        multi_text_window_overlap=0.5,
    ):
        # Initialize parent with all original args
        super().__init__(
            main_csv_path=main_csv_path,
            annotations_folder=annotations_folder,
            tokenizer=tokenizer,
            image_size=image_size,
            max_length=max_length,
            sample_mode=sample_mode,
            video_root_folder=video_root_folder,
            assume_resized_video=assume_resized_video,
            num_frames=num_frames,
            annotations_root=annotations_root,
            annotation_levels=annotation_levels,
            level_mix=level_mix,
            level_seed=level_seed,
            return_level_id=True,
            return_sample_index=False,
            return_expanded_frames=False,
            samples_cache_dir=samples_cache_dir,
            use_samples_cache=use_samples_cache,
            rebuild_samples_cache=rebuild_samples_cache,
            samples_cache_version=samples_cache_version,
            video_reader_threads=video_reader_threads,
            video_reader_cache_size=video_reader_cache_size,
        )

        self.use_augmentation = use_augmentation
        self.multi_text_candidates = multi_text_candidates
        self.multi_text_window_overlap = multi_text_window_overlap

        # Build augmentations if enabled
        if self.use_augmentation:
            self.aug1 = SimCLRVideoAugmentation(
                size=augmentation_size,
                scale=augmentation_scale,
                color_jitter_strength=color_jitter_strength,
            )
            self.aug2 = SimCLRVideoAugmentation(
                size=augmentation_size,
                scale=augmentation_scale,
                color_jitter_strength=color_jitter_strength,
            )

        # Build lookup for multi-text
        self._adjacent_text_lookup = self._build_adjacent_text_lookup()

        # Determine primary level
        if annotation_levels:
            self._primary_level = annotation_levels[0]
        else:
            self._primary_level = "mid"

    def _build_adjacent_text_lookup(self):
        """Build {video_path: {level: [(start, end, caption), ...]}} lookup."""
        lookup = {}
        all_levels = ["fine", "mid", "coarse"]

        for level in all_levels:
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

                for sample in samples:
                    video_path = sample.get("video_path", "")
                    if video_path not in lookup:
                        lookup[video_path] = {"fine": [], "mid": [], "coarse": []}

                    lookup[video_path][level].append({
                        "start_time": float(sample.get("start_time", 0)),
                        "end_time": float(sample.get("end_time", 0)),
                        "caption": str(sample.get("caption", "")),
                    })

            except Exception as e:
                print(f"[PeskaDataset] Warning: Could not load '{level}' "
                      f"annotations: {e}")

        return lookup

    @staticmethod
    def _time_window_overlap(s1, e1, s2, e2):
        """IoU of two time windows."""
        intersection = max(0.0, min(e1, e2) - max(s1, s2))
        union = max(e1, e2) - min(s1, s2)
        if union <= 0:
            return 0.0
        return intersection / union

    def _get_adjacent_texts(self, video_path, start_time, end_time):
        """Get candidate texts from adjacent levels with overlapping time window."""
        if video_path not in self._adjacent_text_lookup:
            return []

        candidates = []
        all_levels = ["fine", "mid", "coarse"]

        for level in all_levels:
            if level == self._primary_level:
                continue

            for entry in self._adjacent_text_lookup[video_path][level]:
                overlap = self._time_window_overlap(
                    start_time, end_time,
                    entry["start_time"], entry["end_time"],
                )
                if overlap >= self.multi_text_window_overlap:
                    tokens = self._build_text(entry["caption"])
                    candidates.append(tokens)

        if len(candidates) > self.multi_text_candidates:
            indices = np.random.choice(
                len(candidates), self.multi_text_candidates, replace=False
            )
            candidates = [candidates[i] for i in indices]

        return candidates

    def __getitem__(self, idx):
        """Returns dict with video, aug1, aug2, primary_text, candidate_texts, level_id."""
        last_error = None

        for _ in range(self.max_retry):
            item = dict(self.samples[idx])
            item["_sample_index"] = idx

            video_path = item.get("video_path", "")
            start_time = float(item.get("start_time", 0))
            end_time = float(item.get("end_time", 0))

            # Get video frames using parent's robust retry logic
            images = self._try_get_images(
                video_path, start_time, end_time, expand_ratio=1.0
            )
            if images is None:
                idx = np.random.randint(0, len(self.samples))
                continue

            # Get primary text
            input_ids, attention_mask = self._build_text(item["caption"])
            primary_text = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            # Get level_id
            level_id = self.LEVEL_TO_ID.get(
                str(item.get("level", "mid")).lower(), 1
            )

            # Apply augmentations
            if self.use_augmentation:
                aug1 = self.aug1(images.clone())
                aug2 = self.aug2(images.clone())
            else:
                aug1 = images.clone()
                aug2 = images.clone()

            # Get candidate texts from adjacent levels
            candidate_texts_raw = self._get_adjacent_texts(
                video_path, start_time, end_time
            )
            candidate_texts = [
                {"input_ids": ids, "attention_mask": mask}
                for ids, mask in candidate_texts_raw
            ]

            return {
                "video": images,
                "video_aug1": aug1,
                "video_aug2": aug2,
                "primary_text": primary_text,
                "candidate_texts": candidate_texts,
                "level_id": level_id,
            }

        raise RuntimeError(
            f"Failed to load valid data after {self.max_retry} retries"
        )


def peska_collate_fn(batch):
    """Collate function that handles variable-length candidate_texts lists."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    videos = torch.stack([b["video"] for b in batch], dim=0)
    aug1s = torch.stack([b["video_aug1"] for b in batch], dim=0)
    aug2s = torch.stack([b["video_aug2"] for b in batch], dim=0)

    primary_input_ids = torch.stack(
        [b["primary_text"]["input_ids"] for b in batch], dim=0
    )
    primary_attention_mask = torch.stack(
        [b["primary_text"]["attention_mask"] for b in batch], dim=0
    )

    level_ids = torch.tensor([b["level_id"] for b in batch], dtype=torch.long)

    max_candidates = max(len(b["candidate_texts"]) for b in batch)
    if max_candidates > 0:
        candidate_input_ids = []
        candidate_attention_masks = []
        for b in batch:
            n_c = len(b["candidate_texts"])
            for i in range(max_candidates):
                if i < n_c:
                    candidate_input_ids.append(
                        b["candidate_texts"][i]["input_ids"]
                    )
                    candidate_attention_masks.append(
                        b["candidate_texts"][i]["attention_mask"]
                    )
                else:
                    candidate_input_ids.append(
                        torch.zeros_like(b["primary_text"]["input_ids"])
                    )
                    candidate_attention_masks.append(
                        torch.zeros_like(b["primary_text"]["attention_mask"])
                    )

        candidate_input_ids = torch.stack(candidate_input_ids, dim=0)
        candidate_attention_masks = torch.stack(candidate_attention_masks, dim=0)
    else:
        candidate_input_ids = torch.zeros(
            0, primary_input_ids.shape[1], dtype=torch.long
        )
        candidate_attention_masks = torch.zeros(
            0, primary_attention_mask.shape[1], dtype=torch.long
        )

    return {
        "video": videos,
        "video_aug1": aug1s,
        "video_aug2": aug2s,
        "primary_text": {
            "input_ids": primary_input_ids,
            "attention_mask": primary_attention_mask,
        },
        "candidate_texts": {
            "input_ids": candidate_input_ids,
            "attention_mask": candidate_attention_masks,
        },
        "level_id": level_ids,
        "num_candidates": max_candidates,
    }
