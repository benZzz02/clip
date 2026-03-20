# pretrain_dataset.py

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """
    图文预训练数据集：
    - 从视频区间中抽单帧或多帧
    - 返回 image/frames, input_ids, attention_mask
    - 用 decord 在线取帧，避免每帧启动一次 ffmpeg 子进程
    """

    def __init__(
        self,
        main_csv_path,
        annotations_folder,
        tokenizer,
        image_size=224,
        max_length=256,
        sample_mode="random",      # "random" or "center"
        ffmpeg_timeout=20,         # 兼容旧接口，当前 decord 版不使用
        max_retry=20,
        video_root_folder="/mnt/mydisk/CLIP/downloaded_video_224_test",
        assume_resized_video=False,  # 兼容旧接口，当前仅作保留
        num_frames=1,
    ):
        super().__init__()
        self.main_df = pd.read_csv(main_csv_path)
        self.annotations_folder = annotations_folder
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        self.sample_mode = sample_mode
        self.ffmpeg_timeout = ffmpeg_timeout
        self.max_retry = max_retry
        self.video_root_folder = video_root_folder
        self.assume_resized_video = assume_resized_video
        self.num_frames = max(1, int(num_frames))

        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        print("正在准备图文预训练数据...")

        samples_list = []

        for _, row in self.main_df.iterrows():
            relative_video_path = row["video_path"]
            video_filename = os.path.basename(relative_video_path)
            full_video_path = os.path.join(self.video_root_folder, video_filename)

            annotation_csv_path = os.path.join(
                self.annotations_folder,
                f"{os.path.splitext(video_filename)[0]}.csv"
            )

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

                    samples_list.append({
                        "video_path": full_video_path,
                        "caption": caption,
                        "start_time": start_time,
                        "end_time": end_time,
                    })

            except Exception as e:
                print(f"处理标注文件 {annotation_csv_path} 时出错: {e}")

        print(f"数据准备完成。共找到 {len(samples_list)} 个有效的图文样本。")
        return samples_list

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
        # frames_np: [T, H, W, 3], RGB uint8
        frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0  # [T, 3, H, W]

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

    def _try_get_images(self, video_path, text_start_time, text_end_time):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
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

    def __getitem__(self, idx):
        for _ in range(self.max_retry):
            item = self.samples[idx]

            images = self._try_get_images(
                item["video_path"],
                item["start_time"],
                item["end_time"],
            )

            if images is not None:
                input_ids, attention_mask = self._build_text(item["caption"])
                return images, input_ids, attention_mask

            idx = random.randint(0, len(self.samples) - 1)

        while True:
            idx = random.randint(0, len(self.samples) - 1)
            item = self.samples[idx]

            images = self._try_get_images(
                item["video_path"],
                item["start_time"],
                item["end_time"],
            )

            if images is not None:
                input_ids, attention_mask = self._build_text(item["caption"])
                return images, input_ids, attention_mask
