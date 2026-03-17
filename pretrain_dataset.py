# pretrain_dataset.py

import os
import random
import subprocess

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PretrainDataset(Dataset):
    """
    图文预训练数据集：
    - 从视频中抽一帧
    - 返回 image, input_ids, attention_mask
    - 遇到坏视频 / ffmpeg 卡住时自动跳过并重采样
    """

    def __init__(
        self,
        main_csv_path,
        annotations_folder,
        tokenizer,
        image_size=224,
        max_length=256,
        sample_mode="random",      # "random" or "center"
        ffmpeg_timeout=20,         # 单个样本 ffmpeg 最长等待秒数
        max_retry=20,              # 单个 __getitem__ 最多重试次数
        video_root_folder="/mnt/mydisk/CLIP/downloaded_video_224_test",
        assume_resized_video=False,
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

        self.transforms = transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

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

    def _sample_timestamp(self, start_time, end_time):
        # 标注的 end_time 有时会略微超过真实视频时长，避开最末尾一小段
        safe_end = max(start_time, end_time - 0.5)

        if self.sample_mode == "center":
            return (start_time + safe_end) / 2.0

        if safe_end <= start_time:
            return start_time

        return random.uniform(start_time, safe_end)


    def _try_get_image(self, video_path, text_start_time, text_end_time):
        """
        成功返回 [3, H, W] tensor
        失败返回 None
        """
        sample_time = self._sample_timestamp(text_start_time, text_end_time)

        cmd = [
            "ffmpeg",
            "-v", "error",
            "-ss", str(sample_time),
            "-i", video_path,
            "-frames:v", "1",
        ]
        if not self.assume_resized_video:
            cmd.extend(["-vf", f"scale={self.image_size}:{self.image_size}"])
        cmd.extend([
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ])

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.ffmpeg_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"[ffmpeg timeout] {video_path}")
            return None
        except Exception as e:
            print(f"[ffmpeg exception] {video_path} | {e}")
            return None

        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="ignore")[:300]
            print(f"[ffmpeg failed] {video_path} | {err}")
            return None

        expected_bytes = self.image_size * self.image_size * 3
        if len(proc.stdout) != expected_bytes:
            print(f"[ffmpeg empty/broken frame] {video_path}")
            return None

        try:
            image = np.frombuffer(proc.stdout, np.uint8).reshape(
                self.image_size, self.image_size, 3
            )
            image = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
            image = self.transforms(image)
            return image
        except Exception as e:
            print(f"[frame decode failed] {video_path} | {e}")
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
        # 失败就随机重采样，保证永远返回合法样本
        for _ in range(self.max_retry):
            item = self.samples[idx]

            image = self._try_get_image(
                item["video_path"],
                item["start_time"],
                item["end_time"],
            )

            if image is not None:
                input_ids, attention_mask = self._build_text(item["caption"])
                return image, input_ids, attention_mask

            idx = random.randint(0, len(self.samples) - 1)

        # 兜底：一直找直到成功
        while True:
            idx = random.randint(0, len(self.samples) - 1)
            item = self.samples[idx]

            image = self._try_get_image(
                item["video_path"],
                item["start_time"],
                item["end_time"],
            )
            if image is not None:
                input_ids, attention_mask = self._build_text(item["caption"])
                return image, input_ids, attention_mask
