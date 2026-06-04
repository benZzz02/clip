import random
from collections import defaultdict

import torch
from torch.utils.data.dataloader import default_collate

from pretrain_dataset import PretrainDataset


LEVEL_TO_ID = {
    "fine": 0,
    "mid": 1,
    "coarse": 2,
}


def _level_name(value):
    return str(value or "fine").strip().lower()


def _sample_center(sample):
    return 0.5 * (float(sample.get("start_time", 0.0)) + float(sample.get("end_time", 0.0)))


def _caption(sample):
    value = sample.get("caption", "")
    return value if isinstance(value, str) and value.strip() else ""


class PeskaVLPCompatibleDataset(PretrainDataset):
    """
    A level-specific wrapper around the local SurgLaVi manifest.

    It keeps the existing data source and frame decoder, but exposes batches in a
    PeskaVLP-like shape:
      - fine/action: video, video_aug1, video_aug2, anchor text, candidate texts
      - mid/coarse: video, anchor text, lower-level candidate text sequence, pos_step
    """

    def __init__(
        self,
        target_level,
        max_candidates=8,
        action_view_flip_prob=0.5,
        action_view_noise_prob=0.25,
        action_view_noise_std=0.01,
        **kwargs,
    ):
        super().__init__(
            return_level_id=False,
            return_sample_index=False,
            return_expanded_frames=False,
            **kwargs,
        )
        self.target_level = _level_name(target_level)
        if self.target_level not in LEVEL_TO_ID:
            raise ValueError(f"Unsupported target_level: {target_level}")

        self.max_candidates = max(1, int(max_candidates))
        self.action_view_flip_prob = float(action_view_flip_prob)
        self.action_view_noise_prob = float(action_view_noise_prob)
        self.action_view_noise_std = float(action_view_noise_std)

        self.all_samples = []
        for idx, sample in enumerate(self.samples):
            item = dict(sample)
            item["_all_sample_index"] = idx
            self.all_samples.append(item)

        self.samples_by_video_level = defaultdict(lambda: defaultdict(list))
        for sample in self.all_samples:
            video_path = sample.get("video_path")
            level = _level_name(sample.get("level"))
            if not video_path or level not in LEVEL_TO_ID:
                continue
            self.samples_by_video_level[video_path][level].append(sample)

        for level_map in self.samples_by_video_level.values():
            for level_samples in level_map.values():
                level_samples.sort(
                    key=lambda item: (
                        float(item.get("start_time", 0.0)),
                        float(item.get("end_time", 0.0)),
                    )
                )

        self.samples = [
            sample
            for sample in self.all_samples
            if _level_name(sample.get("level")) == self.target_level
        ]
        if not self.samples:
            raise ValueError(f"No samples found for target_level={self.target_level}")

    def _contained_or_overlapping(self, parent, child):
        parent_start = float(parent.get("start_time", 0.0))
        parent_end = float(parent.get("end_time", 0.0))
        child_start = float(child.get("start_time", 0.0))
        child_end = float(child.get("end_time", 0.0))
        if parent_start <= child_start and child_end <= parent_end:
            return True
        return max(parent_start, child_start) < min(parent_end, child_end)

    def _same_video_candidates(self, sample, level):
        video_path = sample.get("video_path")
        candidates = list(self.samples_by_video_level.get(video_path, {}).get(level, []))
        if not candidates:
            return []
        center = _sample_center(sample)
        candidates.sort(key=lambda item: abs(_sample_center(item) - center))
        return candidates

    def _fine_candidates(self, sample):
        captions = []
        anchor_caption = _caption(sample)
        if anchor_caption:
            captions.append(anchor_caption)
        for candidate in self._same_video_candidates(sample, "fine"):
            text = _caption(candidate)
            if text and text not in captions:
                captions.append(text)
            if len(captions) >= self.max_candidates:
                break
        return self._pad_candidates(captions or [anchor_caption])

    def _hierarchy_candidates(self, sample):
        video_path = sample.get("video_path")
        level_map = self.samples_by_video_level.get(video_path, {})
        lower_levels = ["fine"] if self.target_level == "mid" else ["mid", "fine"]

        matches = []
        for level in lower_levels:
            matches = [
                candidate
                for candidate in level_map.get(level, [])
                if self._contained_or_overlapping(sample, candidate)
            ]
            if matches:
                break

        if not matches:
            for level in lower_levels:
                matches = self._same_video_candidates(sample, level)
                if matches:
                    break

        matches = sorted(matches, key=lambda item: (_sample_center(item), _caption(item)))
        captions = []
        for candidate in matches:
            text = _caption(candidate)
            if text:
                captions.append(text)
            if len(captions) >= self.max_candidates:
                break

        if not captions:
            captions = [_caption(sample)]
        return self._pad_candidates(captions)

    def _pad_candidates(self, captions):
        deduped = []
        for text in captions:
            if text and text not in deduped:
                deduped.append(text)
        if not deduped:
            deduped = [""]

        valid_count = min(len(deduped), self.max_candidates)
        while len(deduped) < self.max_candidates:
            deduped.append(deduped[-1])
        return deduped[: self.max_candidates], valid_count

    def _build_candidate_text(self, captions):
        tokenized = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    def _augment_video_view(self, frames):
        out = frames.clone()
        if random.random() < self.action_view_flip_prob:
            out = torch.flip(out, dims=[-1])
        if self.action_view_noise_std > 0 and random.random() < self.action_view_noise_prob:
            out = out + torch.randn_like(out) * self.action_view_noise_std
        return out

    def _load_video_window(self, sample):
        return self._try_get_images(
            sample["video_path"],
            sample["start_time"],
            sample["end_time"],
            expand_ratio=1.0,
        )

    def _build_action_item(self, sample):
        video = self._load_video_window(sample)
        video_aug1 = self._load_video_window(sample)
        video_aug2 = self._load_video_window(sample)
        if video is None or video_aug1 is None or video_aug2 is None:
            return None

        video_aug1 = self._augment_video_view(video_aug1)
        video_aug2 = self._augment_video_view(video_aug2)
        input_ids, attention_mask = self._build_text(sample["caption"])
        captions, valid_count = self._fine_candidates(sample)
        candidate_input_ids, candidate_attention_mask = self._build_candidate_text(captions)

        return {
            "video": video,
            "video_aug1": video_aug1,
            "video_aug2": video_aug2,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "candidate_input_ids": candidate_input_ids,
            "candidate_attention_mask": candidate_attention_mask,
            "candidate_valid_count": torch.tensor(valid_count, dtype=torch.long),
            "level_id": torch.tensor(LEVEL_TO_ID["fine"], dtype=torch.long),
        }

    def _build_hierarchy_item(self, sample):
        video = self._load_video_window(sample)
        if video is None:
            return None

        input_ids, attention_mask = self._build_text(sample["caption"])
        captions, valid_count = self._hierarchy_candidates(sample)
        candidate_input_ids, candidate_attention_mask = self._build_candidate_text(captions)
        pos_step = torch.full((self.max_candidates,), -1, dtype=torch.long)
        pos_step[:valid_count] = torch.arange(valid_count, dtype=torch.long)

        return {
            "video": video,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "candidate_input_ids": candidate_input_ids,
            "candidate_attention_mask": candidate_attention_mask,
            "candidate_valid_count": torch.tensor(valid_count, dtype=torch.long),
            "pos_step": pos_step,
            "level_id": torch.tensor(LEVEL_TO_ID[self.target_level], dtype=torch.long),
        }

    def _build_item(self, sample):
        if self.target_level == "fine":
            return self._build_action_item(sample)
        return self._build_hierarchy_item(sample)

    def __getitem__(self, idx):
        for _ in range(self.max_retry):
            sample = dict(self.samples[idx])
            item = self._build_item(sample)
            if item is not None:
                return item
            idx = random.randint(0, len(self.samples) - 1)

        retry_count = self.max_retry
        while True:
            sample = dict(self.samples[idx])
            item = self._build_item(sample)
            if item is not None:
                return item
            retry_count += 1
            if retry_count % 100 == 0:
                print(
                    "PeskaVLPCompatibleDataset is still skipping bad samples after "
                    f"{retry_count} retries for level={self.target_level}."
                )
            idx = random.randint(0, len(self.samples) - 1)


def collate_peskavlp_compatible(batch):
    compact = [item for item in batch if item is not None]
    if not compact:
        return None
    return default_collate(compact)
