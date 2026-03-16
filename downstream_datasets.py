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
    
    class CholecT50TripletFrameDataset(Dataset):
    """
    labels_root 与 frames_root 分离：

    labels_root/
      instrument/VID06.txt
      verb/VID06.txt
      target/VID06.txt
      triplet/VID06.txt

    frames_root/VID06/000560.png  (默认；可用模板改)
    """

    def __init__(
        self,
        labels_root: str,
        frames_root: str,
        video_id: str,
        transform=None,
        strict: bool = True,
        frame_filename_template: str = "{frame_id:06d}.png",
        video_subdir: bool = True,
    ):
        self.labels_root = labels_root
        self.frames_root = frames_root
        self.video_id = video_id
        self.transform = transform
        self.strict = strict
        self.frame_filename_template = frame_filename_template
        self.video_subdir = video_subdir

        self.triplet_labels = self._load_txt("triplet")
        self.tool_labels = self._load_txt("instrument")
        self.verb_labels = self._load_txt("verb")
        self.target_labels = self._load_txt("target")

        self.n = self._infer_len()

    def _load_txt(self, folder):
        path = os.path.join(self.labels_root, folder, f"{self.video_id}.txt")
        if not os.path.exists(path):
            if self.strict:
                raise FileNotFoundError(f"Missing label file: {path}")
            print(f"[WARN] Missing label file: {path}")
            return None
        arr = np.loadtxt(path, dtype=np.int32, delimiter=",")
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def _infer_len(self):
        arrays = [a for a in [self.triplet_labels, self.tool_labels, self.verb_labels, self.target_labels] if a is not None]
        if not arrays:
            return 0
        n = arrays[0].shape[0]
        if self.strict:
            for a in arrays[1:]:
                if a.shape[0] != n:
                    raise ValueError(f"Row mismatch for {self.video_id}: {n} vs {a.shape[0]}")
        return n

    def __len__(self):
        return self.n

    def _img_path(self, frame_id: int) -> str:
        fname = self.frame_filename_template.format(frame_id=frame_id)
        if self.video_subdir:
            return os.path.join(self.frames_root, self.video_id, fname)
        return os.path.join(self.frames_root, fname)

    def __getitem__(self, idx):
        base = self.triplet_labels if self.triplet_labels is not None else self.tool_labels
        frame_id = int(base[idx, 0])
        img_path = self._img_path(frame_id)

        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        else:
            if self.strict:
                raise FileNotFoundError(img_path)
            image = torch.zeros(3, 224, 224, dtype=torch.float32)

        def multihot(arr, fallback_dim):
            if arr is None:
                return torch.zeros(fallback_dim, dtype=torch.float32)
            return torch.from_numpy(arr[idx, 1:].astype(np.float32))

        y_i = multihot(self.tool_labels, 6)
        y_v = multihot(self.verb_labels, 10)
        y_t = multihot(self.target_labels, 15)
        y_ivt = multihot(self.triplet_labels, 100)

        return image, (y_i, y_v, y_t, y_ivt)