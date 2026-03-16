import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

import ivtmetrics

from model import VLP
from zero_shot_prompts import get_prompt_dict, get_label_names
from downstream_datasets import Cholec80FrameDataset
from downstream_datasets.cholect50_triplet import CholecT50TripletFrameDataset


# =========================
# 1. 数据集配置字典
# =========================
DATASET_CONFIGS = {
    "cholec80_phase": {
        "prompt_name": "cholec80_phase",
        "dataset_class": Cholec80FrameDataset,
        "annotation_folder": "cholec80_peska/",
        "video_folder": "/mnt/mydisk/Data/cholec80/cutMargin/",
    },
    "bern_bypass70_phase": {
        "prompt_name": "bern_bypass70_phase",
        "dataset_class": Cholec80FrameDataset,
        "annotation_folder": "/mnt/mydisk/MultiBypass140/MultiBypass140/labels/bern/labels_csv/individual/",
        "video_folder": "/mnt/mydisk/MultiBypass140/BernBypass70/frames/",
    },
    "stras_bypass70_phase": {
        "prompt_name": "stras_bypass70_phase",
        "dataset_class": Cholec80FrameDataset,
        "annotation_folder": "/mnt/mydisk/MultiBypass140/MultiBypass140/labels/strasbourg/labels_csv/individual/",
        "video_folder": "/mnt/mydisk/MultiBypass140/StrasBypass70/frames/",
    },
    "grasp_phase": {
        "prompt_name": "grasp_phase",
        "dataset_class": Cholec80FrameDataset,
        "annotation_folder": "/mnt/mydisk/Grasp/annotations/labels/phase/",
        "video_folder": "/mnt/mydisk/Grasp/frames/",
    },
    "grasp_step": {
        "prompt_name": "grasp_step",
        "dataset_class": Cholec80FrameDataset,
        "annotation_folder": "/mnt/mydisk/Grasp/annotations/labels/step/",
        "video_folder": "/mnt/mydisk/Grasp/frames/",
    },
    "grasp_instrument": {
        "prompt_name": "grasp_instrument",
        "dataset_class": Cholec80FrameDataset,
        "annotation_folder": "/mnt/mydisk/Grasp/annotations/labels/instrument/",
        "video_folder": "/mnt/mydisk/Grasp/frames/",
    },
    "autolaparo_phase": {
        "prompt_name": "autolaparo_phase",
        "dataset_class": Cholec80FrameDataset,
        "annotation_folder": "/mnt/mydisk/CLIP/auto/",
        "video_folder": "/mnt/mydisk/DATA-Yui/Dataset/AutoLaparoDataset/AutoLaparo_Task1/frames_cutmargin/",
    },

    # =========================
    # CholecT50 Triplet（labels 和 frames 分开）
    # =========================
    "cholect50_triplet": {
        "prompt_name": None,
        "dataset_class": CholecT50TripletFrameDataset,

        # TODO: 改成你的真实路径
        "labels_root": "/path/to/OUT_DIR",            # 包含 instrument/verb/target/triplet 的目录
        "frames_root": "/path/to/frames_root",        # 真实帧目录根

        "video_id": "VID06",

        # TODO: 如果你帧命名不是 000560.png，改这里
        "frame_filename_template": "{frame_id:06d}.png",
        "video_subdir": True,

        # TODO: 改成你的 100 triplet prompt 文件
        "triplet_prompt_path": "/path/to/triplet_prompt.txt",
    },
}


def load_model_checkpoint(model, ckpt_path, device):
    print(f"正在加载模型权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print("检测到 checkpoint 格式: model_state_dict")
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print("检测到 checkpoint 格式: state_dict")
    else:
        state_dict = ckpt
        print("检测到纯 state_dict 格式")

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    msg = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)
    return model


def build_zero_shot_text_features(model, tokenizer, prompt_dataset_name, device):
    prompt_dict = get_prompt_dict(prompt_dataset_name)
    class_names = get_label_names(prompt_dataset_name)

    text_features_list = []
    with torch.no_grad():
        print("正在编码文本提示...")
        for class_name in tqdm(class_names, desc="Encoding prompts"):
            prompts = [prompt_dict[class_name]]

            tokenized = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            class_embeddings = model.encode_text(tokenized["input_ids"], tokenized["attention_mask"])
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            class_embeddings = class_embeddings.mean(dim=0)
            text_features_list.append(class_embeddings)

        text_features = torch.stack(text_features_list, dim=0)  # [C, D]
        text_features = F.normalize(text_features, dim=-1)

    return text_features, class_names


def load_triplet_prompts_from_file(triplet_prompt_path: str):
    with open(triplet_prompt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    prompts = [ln.split(":")[-1].strip().replace(",", " ").replace("_", " ") for ln in lines]
    return prompts


def build_text_features_from_prompts_list(model, tokenizer, prompts, device):
    text_features_list = []
    with torch.no_grad():
        print(f"正在编码 {len(prompts)} 条 prompts ...")
        for p in tqdm(prompts, desc="Encoding prompts"):
            tokenized = tokenizer(
                [p],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            emb = model.encode_text(tokenized["input_ids"], tokenized["attention_mask"])
            emb = F.normalize(emb, dim=-1)
            emb = emb.mean(dim=0)
            text_features_list.append(emb)

        text_features = torch.stack(text_features_list, dim=0)
        text_features = F.normalize(text_features, dim=-1)

    return text_features


def build_dataloader(dataset_name, batch_size, num_workers):
    if dataset_name not in DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset: {dataset_name}")

    cfg = DATASET_CONFIGS[dataset_name]
    dataset_class = cfg["dataset_class"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if dataset_name == "cholect50_triplet":
        dataset = dataset_class(
            labels_root=cfg["labels_root"],
            frames_root=cfg["frames_root"],
            video_id=cfg["video_id"],
            transform=transform,
            strict=True,
            frame_filename_template=cfg.get("frame_filename_template", "{frame_id:06d}.png"),
            video_subdir=cfg.get("video_subdir", True),
        )
    else:
        dataset = dataset_class(
            annotation_folder=cfg["annotation_folder"],
            video_folder=cfg["video_folder"],
            transform=transform,
        )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader, cfg


def evaluate_cholect50_triplet(args, model, tokenizer, data_loader, dataset_cfg, device):
    i_prompts = ["grapser", "bipolar", "hook", "scissor", "clipper", "irrigator"]
    v_prompts = ["grasp", "retract", "dissect", "coagulate", "clip", "cut", "aspirate", "irrigate", "pack", "null verb"]
    t_prompts = [
        "gallbladder", "cystic plate", "cystic duct", "cystic artery", "cystic pedicle",
        "blood vessel", "fluid", "abdominal wall cavity", "liver", "adhesion", "omentum",
        "peritoneum", "gut", "specimen bag", "null target",
    ]
    ivt_prompts = load_triplet_prompts_from_file(dataset_cfg["triplet_prompt_path"])
    if len(ivt_prompts) == 0:
        raise RuntimeError("triplet_prompt_path 读出来是空的，请检查路径/文件内容。")

    print("\n构建 CholecT50 text features ...")
    text_i = build_text_features_from_prompts_list(model, tokenizer, i_prompts, device)
    text_v = build_text_features_from_prompts_list(model, tokenizer, v_prompts, device)
    text_t = build_text_features_from_prompts_list(model, tokenizer, t_prompts, device)
    text_ivt = build_text_features_from_prompts_list(model, tokenizer, ivt_prompts, device)

    mAPi = ivtmetrics.Recognition(len(i_prompts))
    mAPv = ivtmetrics.Recognition(len(v_prompts))
    mAPt = ivtmetrics.Recognition(len(t_prompts))
    mAPivt = ivtmetrics.Recognition(len(ivt_prompts))
    for m in (mAPi, mAPv, mAPt, mAPivt):
        m.reset_global()
        m.reset()

    sigmoid = torch.nn.Sigmoid()

    print("\n开始 CholecT50 triplet 评测（sigmoid + ivtmetrics）...")
    with torch.no_grad():
        for images, (y_i, y_v, y_t, y_ivt) in tqdm(data_loader, desc="Evaluating CholecT50"):
            images = images.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            scale = model.logit_scale.exp()

            logits_i = scale * (image_features @ text_i.t())
            logits_v = scale * (image_features @ text_v.t())
            logits_t = scale * (image_features @ text_t.t())
            logits_ivt = scale * (image_features @ text_ivt.t())

            prob_i = sigmoid(logits_i).detach().cpu()
            prob_v = sigmoid(logits_v).detach().cpu()
            prob_t = sigmoid(logits_t).detach().cpu()
            prob_ivt = sigmoid(logits_ivt).detach().cpu()

            mAPi.update(y_i.float().cpu(), prob_i)
            mAPv.update(y_v.float().cpu(), prob_v)
            mAPt.update(y_t.float().cpu(), prob_t)
            mAPivt.update(y_ivt.float().cpu(), prob_ivt)

    for m in (mAPi, mAPv, mAPt, mAPivt):
        m.video_end()

    print("\n--- CholecT50 triplet mAP 结果 ---")
    print("I  :", mAPi.compute_video_AP())
    print("V  :", mAPv.compute_video_AP())
    print("T  :", mAPt.compute_video_AP())
    print("IVT:", mAPivt.compute_video_AP())


def evaluate_zero_shot(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("正在加载模型结构...")
    model = VLP(
        embed_dim=512,
        text_model_name=args.text_model,
        vision_pretrained_weights=args.vision_weights,
    ).to(device)

    model = load_model_checkpoint(model, args.ckpt, device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    data_loader, dataset_cfg = build_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.dataset == "cholect50_triplet":
        print(f"\n当前评测数据集: {args.dataset}")
        print(f"数据集配置: {dataset_cfg}")
        evaluate_cholect50_triplet(args, model, tokenizer, data_loader, dataset_cfg, device)
        return

    # 单标签评测
    text_features, class_names = build_zero_shot_text_features(
        model=model,
        tokenizer=tokenizer,
        prompt_dataset_name=dataset_cfg["prompt_name"],
        device=device,
    )

    print(f"\n当前评测数据集: {args.dataset}")
    print(f"类别顺序: {class_names}")
    print(f"数据集配置: {dataset_cfg}")

    all_preds = []
    all_labels = []

    print("\n开始在整个数据集上进行零样本评估...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            logits = model.logit_scale.exp() * image_features @ text_features.t()
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    print("\n--- 零样本测试评估结果 ---")
    print(f"总准确率 (Overall Accuracy): {accuracy:.4f}")
    print("\n详细分类报告:")
    print(report)


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation for VLP")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()), help="选择测试数据集")
    parser.add_argument("--ckpt", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--text_model", type=str, default="marcobombieri/surgicberta", help="文本编码器/分词器路径或模型名")
    parser.add_argument("--vision_weights", type=str, default="lemonfm.pth", help="视觉 backbone 初始化权重路径")
    parser.add_argument("--batch_size", type=int, default=64, help="测试 batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_zero_shot(args)