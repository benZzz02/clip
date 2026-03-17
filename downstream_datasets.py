import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def resolve_frame_path(data_root, rel_path):
    candidates = []

    candidates.append(os.path.join(data_root, rel_path))

    for split in ["train", "val", "test"]:
        candidates.append(os.path.join(data_root, split, rel_path))

    rel_dir = os.path.dirname(rel_path)
    stem, _ = os.path.splitext(os.path.basename(rel_path))

    if stem.isdigit():
        frame_num = int(stem)

        candidates.append(os.path.join(data_root, rel_dir, f"{frame_num:05d}.png"))
        candidates.append(os.path.join(data_root, rel_dir, f"{frame_num:05d}.jpg"))
        candidates.append(os.path.join(data_root, rel_dir, f"{frame_num:04d}.png"))
        candidates.append(os.path.join(data_root, rel_dir, f"{frame_num:04d}.jpg"))

        for split in ["train", "val", "test"]:
            candidates.append(os.path.join(data_root, split, rel_dir, f"{frame_num:05d}.png"))
            candidates.append(os.path.join(data_root, split, rel_dir, f"{frame_num:05d}.jpg"))
            candidates.append(os.path.join(data_root, split, rel_dir, f"{frame_num:04d}.png"))
            candidates.append(os.path.join(data_root, split, rel_dir, f"{frame_num:04d}.jpg"))

    for path in candidates:
        if os.path.exists(path):
            return path

    return os.path.join(data_root, rel_path)


def load_image_lists(file, data_root, labels):
    frame_paths = defaultdict(dict)
    video_name_to_idx = {}
    video_idx_to_name = []
    split_videos = set(labels.keys())

    is_autolaparo = "autolaparo" in os.path.normpath(file).lower()

    with open(file, "r") as f:
        for line in f:
            row = line.strip().split()
            if not row:
                continue

            assert len(row) == 4, f"Bad frame-list row: {line}"
            video_name = row[0]
            if video_name not in split_videos:
                continue

            if video_name not in video_name_to_idx:
                idx = len(video_name_to_idx)
                video_name_to_idx[video_name] = idx
                video_idx_to_name.append(video_name)

            data_key = video_name_to_idx[video_name]
            frame_num = int(row[2])
            rel_path = row[3]

            if is_autolaparo:
                rel_dir = os.path.dirname(rel_path)
                stem, _ = os.path.splitext(os.path.basename(rel_path))
                if stem.isdigit():
                    mapped_num = max(int(stem) - 1, 0)
                    rel_path = os.path.join(rel_dir, f"{mapped_num:05d}.png")

            frame_paths[data_key][frame_num] = resolve_frame_path(data_root, rel_path)

    frame_paths = [frame_paths[i] for i in range(len(frame_paths))]
    return frame_paths, video_idx_to_name



def _category_name(item):
    if "name" in item:
        return item["name"]
    if "label" in item:
        return item["label"]
    if "id" in item:
        return str(item["id"])
    return str(item)


def _category_prompt(item):
    if "description" in item:
        return item["description"]
    if "name" in item:
        return item["name"]
    return str(item)


def parse_annotations(filename, task):
    count = 0
    annotated_frames_dict = {}
    id2frame = {}

    with open(filename, "r") as f:
        data = json.load(f)

    if task == "triplet":
        categories = {
            "triplet": [_category_name(x) for x in data["triplet_categories"]],
            "instrument": [_category_name(x) for x in data["instrument_categories"]],
            "verb": [_category_name(x) for x in data["verb_categories"]],
            "target": [_category_name(x) for x in data["target_categories"]],
        }
        prompts = {
            "triplet": [_category_prompt(x) for x in data["triplet_categories"]],
            "instrument": [_category_prompt(x) for x in data["instrument_categories"]],
            "verb": [_category_prompt(x) for x in data["verb_categories"]],
            "target": [_category_prompt(x) for x in data["target_categories"]],
        }
        label_key = "triplet"

    elif task == "instruments":
        categories = [_category_name(x) for x in data["categories"]]
        prompts = [_category_prompt(x) for x in data["categories"]]
        label_key = "instruments"

    elif task == "actions":
        category_key = "actions_categories" if "actions_categories" in data else "phases_categories"
        categories = [_category_name(x) for x in data[category_key]]
        prompts = [_category_prompt(x) for x in data[category_key]]

        sample_ann = data["annotations"][0]
        label_key = "actions" if "actions" in sample_ann else "phases"

    else:
        categories = [_category_name(x) for x in data[f"{task}_categories"]]
        prompts = [_category_prompt(x) for x in data[f"{task}_categories"]]
        label_key = task

    for image in data["images"]:
        video_name = image["video_name"]
        frame_num = int(image["frame_num"])

        id2frame[image["id"]] = (
            video_name,
            frame_num,
            image.get("width", 0),
            image.get("height", 0),
        )

        if video_name not in annotated_frames_dict:
            annotated_frames_dict[video_name] = {}
        if frame_num not in annotated_frames_dict[video_name]:
            annotated_frames_dict[video_name][frame_num] = []

    for annotation in data["annotations"]:
        video_name, frame_num, _, _ = id2frame[annotation["image_id"]]
        label = annotation[label_key]

        if isinstance(label, list):
            annotated_frames_dict[video_name][frame_num].extend(label)
        else:
            annotated_frames_dict[video_name][frame_num].append(label)

        count += 1

    return annotated_frames_dict, count, categories, prompts


def load_coco_annotations(filename, task):
    labels, count, categories, prompts = parse_annotations(filename, task)
    return labels, categories, prompts


def get_keyframe_data(labels):
    keyframe_indices = []
    keyframe_labels = []

    for video_idx in range(len(labels)):
        keyframe_labels.append([])
        frame_nums = sorted(labels[video_idx].keys())
        for sec_idx, frame_num in enumerate(frame_nums):
            keyframe_indices.append((video_idx, sec_idx, frame_num, sec_idx))
            keyframe_labels[video_idx].append(labels[video_idx][frame_num])

    return keyframe_indices, keyframe_labels


def _normalize_ids(values, num_classes):
    arr = np.asarray(values).reshape(-1).tolist()
    ids = []

    for x in arr:
        xi = int(x)
        if xi < 0:
            continue
        ids.append(xi)

    if not ids:
        return []

    if min(ids) >= 1 and max(ids) <= num_classes:
        return [x - 1 for x in ids]

    return [x for x in ids if 0 <= x < num_classes]


def _to_multihot(values, num_classes):
    arr = np.asarray(values)

    if arr.ndim == 1 and arr.shape[0] == num_classes:
        uniq = set(np.unique(arr).tolist())
        if uniq.issubset({0, 1, 0.0, 1.0}):
            return torch.tensor(arr.astype(np.float32), dtype=torch.float32)

    out = torch.zeros(num_classes, dtype=torch.float32)
    for idx in _normalize_ids(values, num_classes):
        out[idx] = 1.0
    return out


class SurgLaViSingleFrameDataset(Dataset):
    def __init__(self, ann_file, transform=None):
        super().__init__()

        (
            self.label_file,
            self.data_root,
            self.media_type,
            self.frame_lists,
            self.sample_rate,
            self.zero_fill,
            self.image_type,
            self.name,
            self.task,
        ) = ann_file

        self.transform = transform

        labels, self.categories, self.prompts = load_coco_annotations(
            self.label_file,
            self.task,
        )
        self._image_paths, self._video_idx_to_name = load_image_lists(
            self.frame_lists,
            self.data_root,
            labels,
        )

        self.labels = [
            labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        if self.name == "heichole":
            filtered_labels = []
            for video_idx in range(len(self.labels)):
                kept = {}
                for frame_num, label in self.labels[video_idx].items():
                    image_path = self._image_paths[video_idx].get(frame_num)
                    if image_path is not None and os.path.exists(image_path):
                        kept[frame_num] = label
                filtered_labels.append(kept)
            self.labels = filtered_labels

        self._keyframe_indices, _ = get_keyframe_data(self.labels)
        self.num_examples = len(self._keyframe_indices)

    def __len__(self):
        return self.num_examples

    def _get_video_number(self, video_name, fallback):
        matches = re.findall(r"\d+", video_name)
        if matches:
            return int(matches[0])
        return int(fallback)

    def _convert_single_label(self, raw_label):
        if isinstance(raw_label, list):
            if len(raw_label) != 1:
                raise ValueError(f"Expected single label, got: {raw_label}")
            return int(raw_label[0])
        return int(raw_label)

    def _convert_instrument_label(self, raw_label):
        return _to_multihot(raw_label, len(self.categories))

    def _convert_triplet_label(self, raw_label):
        if isinstance(raw_label, list):
            if len(raw_label) != 1 or not isinstance(raw_label[0], dict):
                raise ValueError(f"Unexpected triplet label format: {raw_label}")
            raw_label = raw_label[0]

        if not isinstance(raw_label, dict):
            raise TypeError(f"Triplet label must be dict, got {type(raw_label)}")

        dims = {"instrument": 6, "verb": 10, "target": 15, "triplet": 100}
        out = {}

        for key, dim in dims.items():
            out[key] = _to_multihot(raw_label.get(key, []), dim)

        return out

    def __getitem__(self, idx):
        video_idx, _, frame_num, _ = self._keyframe_indices[idx]
        video_name = self._video_idx_to_name[video_idx]
        video_num = self._get_video_number(video_name, video_idx)

        if frame_num not in self._image_paths[video_idx]:
            raise KeyError(f"Frame {frame_num} of video {video_name} not found in frame list.")

        image_path = self._image_paths[video_idx][frame_num]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        raw_label = self.labels[video_idx][frame_num]

        if self.task in ["phases", "steps", "actions"]:
            label = self._convert_single_label(raw_label)
        elif self.task == "instruments":
            label = self._convert_instrument_label(raw_label)
        elif self.task == "triplet":
            label = self._convert_triplet_label(raw_label)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        return (
            image,
            label,
            torch.tensor(video_num, dtype=torch.long),
            torch.tensor(int(frame_num), dtype=torch.long),
        )
