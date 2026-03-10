import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import glob
import torch
from collections import defaultdict


class Cholec80FrameDataset(Dataset):
    """
    Cholec80 单帧评测版本：
    每个样本只返回当前帧和当前帧标签，不再构造滑动窗口。
    """
    def __init__(self, annotation_folder, video_folder, transform=None):
        self.video_folder = video_folder
        self.transform = transform
        self.samples = self._create_samples(annotation_folder)

    def _create_samples(self, annotation_folder):
        print("正在构建单帧样本...")

        csv_files = glob.glob(os.path.join(annotation_folder, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"在文件夹 '{annotation_folder}' 中未找到任何 .csv 文件。")

        all_samples = []

        for csv_file in sorted(csv_files):
            df = pd.read_csv(csv_file)

            for _, row in df.iterrows():
                relative_path = row["path"]
                full_path = os.path.join(self.video_folder, relative_path)
                label = int(row["label"]) - 1   # 如果你的标签本来就是0开始，这里把 -1 去掉

                all_samples.append({
                    "path": full_path,
                    "label": label
                })

        print(f"样本构建完成。共生成 {len(all_samples)} 个单帧样本。")
        return all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["path"]
        label = sample["label"]

        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            print(f"警告: 图像文件未找到: {image_path}，将使用空张量代替。")
            image = torch.zeros(3, 224, 224)

        return image, label