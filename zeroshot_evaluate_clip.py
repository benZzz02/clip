# zeroshot_evaluate_clip.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from transformers import CLIPModel, AutoProcessor
from PIL import Image
import torchvision.transforms.functional as TF

from downstream_datasets import Cholec80FrameDataset


def collate_fn(batch):
    images = []
    labels = []

    for img, label in batch:
        # 兼容 dataset 里偶尔返回的零张量占位
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        images.append(img)
        labels.append(label)

    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def evaluate_zero_shot():
    # --- 1. 配置 ---
    MODEL_NAME = "openai/clip-vit-base-patch32"
    CHOLEC80_ANNOTATIONS_FOLDER = "cholec80_peska/"
    CHOLEC80_VIDEOS_FOLDER = "/mnt/mydisk/Data/cholec80/cutMargin/"
    BATCH_SIZE = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 2. 加载 transformers 版 CLIP ---
    print(f"正在加载 transformers CLIP: {MODEL_NAME}")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    # --- 3. 文本提示 ---
    cholec80_phase_prompts = {
        "Preparation": [
            "In preparation phase I insert trocars to patient abdomen cavity"
        ],
        "Calot Triangle Dissection": [
            "In calot triangle dissection phase I use grasper to hold gallbladder and use hook to expose the hepatic triangle area and cystic duct and cystic artery"
        ],
        "Clipping and Cutting": [
            "In clip and cut phase I use clipper to clip the cystic duct and artery then use scissor to cut them"
        ],
        "Gallbladder Dissection": [
            "In dissection phase I use the hook to dissect the connective tissue between gallbladder and liver"
        ],
        "Gallbladder Retraction": [
            "In retraction phase I grasp the specimen bag and remove it from trocar"
        ],
        "Cleaning and Coagulation": [
            "In clean and coagulation phase I use suction and irrigation to clear the surgical field and coagulate bleeding vessels"
        ],
        "Gallbladder Packaging": [
            "In packaging phase I put the gallbladder into the specimen bag"
        ]
    }
    class_names = list(cholec80_phase_prompts.keys())
    class_prompts = [cholec80_phase_prompts[name][0] for name in class_names]

    # --- 4. 先把文本处理好 ---
    text_inputs = processor(
        text=class_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = {
        "input_ids": text_inputs["input_ids"].to(device),
        "attention_mask": text_inputs["attention_mask"].to(device),
    }

    # --- 5. 加载单帧数据集 ---
    dataset = Cholec80FrameDataset(
        annotation_folder=CHOLEC80_ANNOTATIONS_FOLDER,
        video_folder=CHOLEC80_VIDEOS_FOLDER,
        transform=None,   # 这里不要自己做 transform，交给 processor
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # --- 6. 遍历评测 ---
    all_preds = []
    all_labels = []

    print("\n开始在整个数据集上进行 zero-shot 评估...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            labels = labels.to(device, non_blocking=True)

            image_inputs = processor(
                images=images,
                return_tensors="pt",
            )
            pixel_values = image_inputs["pixel_values"].to(device, non_blocking=True)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )

            logits = outputs.logits_per_image   # [B, C]
            probs = logits.softmax(dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 7. 计算指标 ---
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )

    print("\n--- transformers CLIP 零样本测试评估结果 ---")
    print(f"总准确率 (Overall Accuracy): {accuracy:.4f}")
    print("\n详细分类报告:")
    print(report)


if __name__ == "__main__":
    evaluate_zero_shot()