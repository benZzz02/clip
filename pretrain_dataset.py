# pretrain_dataset.py

import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from pretrain_manifest_cache import load_or_build_pretrain_samples


class PretrainDataset(Dataset):
    LEVEL_TO_ID = {
        "fine": 0,
        "mid": 1,
        "coarse": 2,
    }

    """
    图文预训练数据集：
    - 从视频区间中抽单帧或多帧
    - 返回 image/frames, input_ids, attention_mask
    - 用 decord 在线取帧，避免每帧启动一次 ffmpeg 子进程
    - 支持单层级目录，或 coarse/mid/fine 多层级混合输入
    - 支持将样本清单缓存到磁盘，避免每次启动都重复扫描标注 CSV
    """

    def __init__(
        self,
        main_csv_path,
        annotations_folder,
        tokenizer,
        image_size=224,
        max_length=256,
        sample_mode="random",
        ffmpeg_timeout=20,
        max_retry=20,
        video_root_folder="downloaded_video_224_test",
        assume_resized_video=False,
        num_frames=1,
        annotations_root=None,
        annotation_levels=None,
        level_mix="concat",
        level_seed=42,
        return_level_id=False,
        return_expanded_frames=False,
        expanded_window_ratio=2.0,
        samples_cache_dir=".cache/pretrain_samples",
        use_samples_cache=True,
        rebuild_samples_cache=False,
        samples_cache_version="v1",
        video_reader_threads=1,
        video_reader_cache_size=16,
    ):
        super().__init__()

        self.main_csv_path = main_csv_path
        self.annotations_folder = annotations_folder
        self.annotations_root = annotations_root
        self.annotation_levels = annotation_levels
        self.level_mix = level_mix
        self.level_seed = int(level_seed)
        self.return_level_id = bool(return_level_id)
        self.return_expanded_frames = bool(return_expanded_frames)
        self.expanded_window_ratio = max(1.0, float(expanded_window_ratio))

        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        self.sample_mode = sample_mode
        self.ffmpeg_timeout = ffmpeg_timeout
        self.max_retry = max_retry
        self.video_root_folder = video_root_folder
        self.assume_resized_video = assume_resized_video
        self.num_frames = max(1, int(num_frames))

        self.samples_cache_dir = samples_cache_dir
        self.use_samples_cache = bool(use_samples_cache)
        self.rebuild_samples_cache = bool(rebuild_samples_cache)
        self.samples_cache_version = str(samples_cache_version)
        self.video_reader_threads = max(1, int(video_reader_threads))
        self.video_reader_cache_size = max(0, int(video_reader_cache_size))
        self._video_reader_cache = OrderedDict()

        self.pixel_mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        ).view(1, 3, 1, 1)

        self.samples = load_or_build_pretrain_samples(
            main_csv_path=self.main_csv_path,
            video_root_folder=self.video_root_folder,
            annotations_folder=self.annotations_folder,
            annotations_root=self.annotations_root,
            annotation_levels=self.annotation_levels,
            level_mix=self.level_mix,
            level_seed=self.level_seed,
            samples_cache_dir=self.samples_cache_dir,
            use_samples_cache=self.use_samples_cache,
            rebuild_samples_cache=self.rebuild_samples_cache,
            samples_cache_version=self.samples_cache_version,
        )

    def _get_video_reader(self, video_path):
        if self.video_reader_cache_size > 0:
            cached_reader = self._video_reader_cache.get(video_path)
            if cached_reader is not None:
                self._video_reader_cache.move_to_end(video_path)
                return cached_reader

        reader = VideoReader(
            video_path,
            ctx=cpu(0),
            num_threads=self.video_reader_threads,
        )

        if self.video_reader_cache_size > 0:
            self._video_reader_cache[video_path] = reader
            self._video_reader_cache.move_to_end(video_path)
            while len(self._video_reader_cache) > self.video_reader_cache_size:
                self._video_reader_cache.popitem(last=False)

        return reader

    def __len__(self):
        return len(self.samples)

    def _sample_timestamps(self, start_time, end_time, video_duration):
        if video_duration is not None and video_duration > 0:
            start_time = max(0.0, min(float(start_time), video_duration))
            end_time = max(start_time, min(float(end_time), video_duration))

        safe_end = end_time
        if video_duration is not None and video_duration > 0:
            safe_end = min(end_time, max(0.0, video_duration - 1e-3))

        if safe_end <= start_time:
            return [start_time for _ in range(self.num_frames)]

        if self.num_frames == 1:
            if self.sample_mode == "center":
                return [(start_time + safe_end) / 2.0]
            return [random.uniform(start_time, safe_end)]

        segment_edges = np.linspace(start_time, safe_end, self.num_frames + 1)

        if self.sample_mode == "center":
            return [
                float((segment_edges[i] + segment_edges[i + 1]) / 2.0)
                for i in range(self.num_frames)
            ]

        timestamps = []
        for i in range(self.num_frames):
            left = float(segment_edges[i])
            right = float(segment_edges[i + 1])

            if right <= left:
                timestamps.append(left)
            else:
                timestamps.append(random.uniform(left, right))

        return timestamps

    def _expand_window(self, start_time, end_time, video_duration=None, expand_ratio=None):
        start_time = float(start_time)
        end_time = float(end_time)
        ratio = self.expanded_window_ratio if expand_ratio is None else max(1.0, float(expand_ratio))

        if end_time < start_time:
            start_time, end_time = end_time, start_time

        if video_duration is not None and video_duration > 0:
            video_duration = float(video_duration)
            start_time = min(max(start_time, 0.0), video_duration)
            end_time = min(max(end_time, start_time), video_duration)

        base_duration = max(end_time - start_time, 1e-3)
        target_duration = base_duration * ratio
        center = 0.5 * (start_time + end_time)

        expanded_start = center - 0.5 * target_duration
        expanded_end = center + 0.5 * target_duration

        if video_duration is not None and video_duration > 0:
            if target_duration >= video_duration:
                return 0.0, video_duration

            if expanded_start < 0.0:
                expanded_end -= expanded_start
                expanded_start = 0.0

            if expanded_end > video_duration:
                expanded_start -= (expanded_end - video_duration)
                expanded_end = video_duration

            expanded_start = max(0.0, expanded_start)
            expanded_end = min(video_duration, expanded_end)

        if expanded_end <= expanded_start:
            expanded_end = expanded_start + base_duration
            if video_duration is not None and video_duration > 0:
                expanded_end = min(expanded_end, video_duration)
                expanded_start = max(0.0, expanded_end - base_duration)

        return expanded_start, expanded_end

    def _timestamps_to_frame_indices(self, timestamps, fps, num_video_frames, video_duration):
        if num_video_frames <= 0:
            return None

        max_frame_idx = num_video_frames - 1
        frame_indices = []

        for ts in timestamps:
            ts = max(0.0, float(ts))
            if video_duration is not None and video_duration > 0:
                ts = min(ts, video_duration)

            frame_idx = int(round(ts * fps))
            frame_idx = min(max(frame_idx, 0), max_frame_idx)
            frame_indices.append(frame_idx)

        return frame_indices

    def _postprocess_frames(self, frames_np):
        frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0

        if frames.shape[-2:] != (self.image_size, self.image_size):
            frames = F.interpolate(
                frames,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        frames = (frames - self.pixel_mean) / self.pixel_std

        if self.num_frames == 1:
            return frames[0]
        return frames

    def _try_get_images(self, video_path, text_start_time, text_end_time, expand_ratio=1.0):
        try:
            vr = self._get_video_reader(video_path)
        except Exception as e:
            print(f"[decord open failed] {video_path} | {e}")
            return None

        try:
            num_video_frames = len(vr)
            if num_video_frames <= 0:
                print(f"[decord empty video] {video_path}")
                return None

            fps = float(vr.get_avg_fps())
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0

            video_duration = max(num_video_frames - 1, 0) / fps

            if float(expand_ratio) > 1.0:
                text_start_time, text_end_time = self._expand_window(
                    text_start_time,
                    text_end_time,
                    video_duration=video_duration,
                    expand_ratio=expand_ratio,
                )

            timestamps = self._sample_timestamps(
                text_start_time,
                text_end_time,
                video_duration=video_duration,
            )

            frame_indices = self._timestamps_to_frame_indices(
                timestamps,
                fps=fps,
                num_video_frames=num_video_frames,
                video_duration=video_duration,
            )
            if frame_indices is None:
                return None

            frames_np = vr.get_batch(frame_indices).asnumpy()
            return self._postprocess_frames(frames_np)

        except Exception as e:
            print(f"[decord decode failed] {video_path} | {e}")
            return None

    def _build_text(self, caption):
        tokenized_text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokenized_text["input_ids"].squeeze(0)
        attention_mask = tokenized_text["attention_mask"].squeeze(0)
        return input_ids, attention_mask

    def _build_return_value(self, item, images, expanded_images):
        input_ids, attention_mask = self._build_text(item["caption"])
        level_id = self.LEVEL_TO_ID.get(str(item.get("level", "mid")).lower(), 1)

        if self.return_expanded_frames:
            expanded_images = images if expanded_images is None else expanded_images

        if self.return_level_id:
            if self.return_expanded_frames:
                return (
                    images,
                    expanded_images,
                    input_ids,
                    attention_mask,
                    torch.tensor(level_id, dtype=torch.long),
                )
            return images, input_ids, attention_mask, torch.tensor(level_id, dtype=torch.long)

        if self.return_expanded_frames:
            return images, expanded_images, input_ids, attention_mask
        return images, input_ids, attention_mask

    def __getitem__(self, idx):
        last_error = None

        for _ in range(self.max_retry):
            item = self.samples[idx]

            expanded_images = None
            if self.return_expanded_frames:
                expanded_images = self._try_get_images(
                    item["video_path"],
                    item["start_time"],
                    item["end_time"],
                    expand_ratio=self.expanded_window_ratio,
                )
                if expanded_images is None:
                    images = self._try_get_images(
                        item["video_path"],
                        item["start_time"],
                        item["end_time"],
                        expand_ratio=1.0,
                    )
                    expanded_images = images
                else:
                    images = expanded_images
            else:
                images = self._try_get_images(
                    item["video_path"],
                    item["start_time"],
                    item["end_time"],
                    expand_ratio=1.0,
                )

            if images is not None:
                return self._build_return_value(item, images, expanded_images)

            last_error = (
                f"video={item['video_path']}, "
                f"start={item['start_time']}, end={item['end_time']}"
            )
            idx = random.randint(0, len(self.samples) - 1)

        retry_count = self.max_retry
        while True:
            item = self.samples[idx]

            expanded_images = None
            if self.return_expanded_frames:
                expanded_images = self._try_get_images(
                    item["video_path"],
                    item["start_time"],
                    item["end_time"],
                    expand_ratio=self.expanded_window_ratio,
                )
                if expanded_images is None:
                    images = self._try_get_images(
                        item["video_path"],
                        item["start_time"],
                        item["end_time"],
                        expand_ratio=1.0,
                    )
                    expanded_images = images
                else:
                    images = expanded_images
            else:
                images = self._try_get_images(
                    item["video_path"],
                    item["start_time"],
                    item["end_time"],
                    expand_ratio=1.0,
                )

            if images is not None:
                return self._build_return_value(item, images, expanded_images)

            retry_count += 1
            if retry_count % 100 == 0:
                print(
                    "PretrainDataset is still skipping bad samples after "
                    f"{retry_count} retries. Last sample: {last_error}"
                )

            last_error = (
                f"video={item['video_path']}, "
                f"start={item['start_time']}, end={item['end_time']}"
            )
            idx = random.randint(0, len(self.samples) - 1)
