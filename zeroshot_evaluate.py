import argparse
import ast
import json
import os

import ivtmetrics
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer

from model import VLP
from downstream_datasets import SurgLaViSingleFrameDataset


DATASET_CONFIGS = {
    "cholec80_phase": {
        "dataset_class": SurgLaViSingleFrameDataset,
        "ann_file": [
            "/mnt/mydisk/CLIP/anno_downstream/cholec80/annotations/test.json",
            "/mnt/mydisk/cholecdata/cholecdata/cholec80/frames",
            "video",
            "/mnt/mydisk/CLIP/anno_downstream/cholec80/frame_lists/frames.csv",
            1,
            6,
            "png",
            "cholec80",
            "phases",
        ],
    },
    "cholec80_instrument": {
        "dataset_class": SurgLaViSingleFrameDataset,
        "ann_file": [
            "/mnt/mydisk/CLIP/anno_downstream/cholec80/annotations/instruments_test.json",
            "/mnt/mydisk/cholecdata/cholecdata/cholec80/frames",
            "video",
            "/mnt/mydisk/CLIP/anno_downstream/cholec80/frame_lists/frames.csv",
            1,
            6,
            "png",
            "cholec80",
            "instruments",
        ],
    },
    "bern_bypass70_phase": {
        "dataset_class": SurgLaViSingleFrameDataset,
        "ann_file": [
            "/mnt/mydisk/CLIP/anno_downstream/bernbypass70/annotations/test.json",
            "/mnt/mydisk/MultiBypass140/BernBypass70/frames",
            "video",
            "/mnt/mydisk/CLIP/anno_downstream/bernbypass70/frame_lists/frames.csv",
            1,
            8,
            "jpg",
            "bernbypass_test",
            "phases",
        ],
    },
    "stras_bypass70_phase": {
        "dataset_class": SurgLaViSingleFrameDataset,
        "ann_file": [
            "/mnt/mydisk/CLIP/anno_downstream/strasbypass70/annotations/test.json",
            "/mnt/mydisk/MultiBypass140/StrasBypass70/frames",
            "video",
            "/mnt/mydisk/CLIP/anno_downstream/strasbypass70/frame_lists/frames.csv",
            1,
            8,
            "jpg",
            "strasbypass_test",
            "phases",
        ],
    },
    "grasp_phase": {
        "dataset_class": SurgLaViSingleFrameDataset,
        "ann_file": [
            "/mnt/mydisk/CLIP/anno_downstream/grasp/annotations/grasp_long-term_test.json",
            "/mnt/mydisk/Grasp/frames",
            "video",
            "/mnt/mydisk/CLIP/anno_downstream/grasp/frame_lists/frames.csv",
            1,
            5,
            "jpg",
            "grasp_test_phases",
            "phases",
        ],
    },
    "grasp_step": {
        "dataset_class": SurgLaViSingleFrameDataset,
        "ann_file": [
            "/mnt/mydisk/CLIP/anno_downstream/grasp/annotations/grasp_long-term_test.json",
            "/mnt/mydisk/Grasp/frames",
            "video",
            "/mnt/mydisk/CLIP/anno_downstream/grasp/frame_lists/frames.csv",
            1,
            5,
            "jpg",
            "grasp_test_steps",
            "steps",
        ],
    },
    "grasp_instrument": {
        "dataset_class": SurgLaViSingleFrameDataset,
        "ann_file": [
            "/mnt/mydisk/CLIP/anno_downstream/grasp/annotations/grasp_short-term_test.json",
            "/mnt/mydisk/Grasp/frames",
            "video",
            "/mnt/mydisk/CLIP/anno_downstream/grasp/frame_lists/frames.csv",
            1,
            5,
            "jpg",
            "grasp_test_instruments",
            "instruments",
        ],
    },
    "autolaparo_phase": {
        "dataset_class": SurgLaViSingleFrameDataset,
        "ann_file": [
            "/mnt/mydisk/CLIP/anno_downstream/autolaparo/annotations/test.json",
            "/mnt/mydisk/DATA-Yui/Dataset/AutoLaparoDataset/AutoLaparo_Task1/frames_cutmargin",
            "video",
            "/mnt/mydisk/CLIP/anno_downstream/autolaparo/frame_lists/frames.csv",
            1,
            4,
            "jpg",
            "autolaparo",
            "phases",
        ],
    },
    "cholect50_triplet": {
        "dataset_class": SurgLaViSingleFrameDataset,
        "ann_file": [
            "/mnt/mydisk/CLIP/anno_downstream/cholect50/annotations/test.json",
            "/mnt/mydisk/cholecdata/cholecdata/cholect50/videos",
            "video",
            "/mnt/mydisk/CLIP/anno_downstream/cholect50/frame_lists/frames.csv",
            1,
            6,
            "png",
            "cholect50",
            "triplet",
        ],
    },
    "sarrarp50_action": {
        "dataset_class": SurgLaViSingleFrameDataset,
        "ann_file": [
            "/mnt/mydisk/CLIP/anno_downstream/sarrarp50/annotations/test.json",
            "/mnt/mydisk/SAR50/test",
            "video",
            "/mnt/mydisk/CLIP/anno_downstream/sarrarp50/frame_lists/frames.csv",
            1,
            9,
            "png",
            "sarrarp50",
            "actions",
        ],
    },
    "sarrarp50": {
        "dataset_class": SurgLaViSingleFrameDataset,
        "ann_file": [
            "/mnt/mydisk/CLIP/anno_downstream/sarrarp50/annotations/test.json",
            "/mnt/mydisk/SAR50/test",
            "video",
            "/mnt/mydisk/CLIP/anno_downstream/sarrarp50/frame_lists/frames.csv",
            1,
            9,
            "png",
            "sarrarp50",
            "actions",
        ],
    },
        "heichole_action": {
        "dataset_class": SurgLaViSingleFrameDataset,
        "ann_file": [
            "/mnt/mydisk/CLIP/anno_downstream/heichole/annotations/test.json",
            "/mnt/mydisk/HeiChole/outputs",
            "video",
            "/mnt/mydisk/CLIP/anno_downstream/heichole/frame_lists/frames.csv",
            1,
            5,
            "png",
            "heichole",
            "actions",
        ],
    },
    "heichole_instrument": {
        "dataset_class": SurgLaViSingleFrameDataset,
        "ann_file": [
            "/mnt/mydisk/CLIP/anno_downstream/heichole/annotations/instruments_test.json",
            "/mnt/mydisk/HeiChole/outputs",
            "video",
            "/mnt/mydisk/CLIP/anno_downstream/heichole/frame_lists/frames.csv",
            1,
            5,
            "png",
            "heichole",
            "instruments",
        ],
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


def convert_to_2d_array(preds):
    preds = np.asarray(preds, dtype=object)
    if len(preds) > 0 and isinstance(preds[0], str) and preds[0].strip().startswith("["):
        return np.array([ast.literal_eval(x) for x in preds], dtype=np.float32)
    return np.stack(preds)


def to_builtin(obj):
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def build_dataloader(dataset_name, batch_size, num_workers):
    if dataset_name not in DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset: {dataset_name}")

    cfg = DATASET_CONFIGS[dataset_name]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = cfg["dataset_class"](
        ann_file=cfg["ann_file"],
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


def extract_text_feats(texts, model, tokenizer, device, max_length=256, text_bs=256):
    text_feats = []

    with torch.no_grad():
        for i in range(0, len(texts), text_bs):
            batch = texts[i:i + text_bs]
            tokenized = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            feats = model.encode_text(tokenized["input_ids"], tokenized["attention_mask"])
            feats = F.normalize(feats, dim=-1)
            text_feats.append(feats)

    return F.normalize(torch.cat(text_feats, dim=0), dim=-1)


def format_results(results, preds, labels, video_idxs, frame_idxs):
    preds = preds.detach().cpu()

    if torch.is_tensor(labels):
        labels = labels.detach().cpu().tolist()
    if torch.is_tensor(video_idxs):
        video_idxs = video_idxs.detach().cpu().tolist()
    if torch.is_tensor(frame_idxs):
        frame_idxs = frame_idxs.detach().cpu().tolist()

    for pred, label, video_idx, frame_idx in zip(preds, labels, video_idxs, frame_idxs):
        results.append(
            {
                "video_idx": int(video_idx),
                "frame_idx": int(frame_idx),
                "prediction": pred.tolist(),
                "ground_truth": label,
            }
        )
    return results


def inference(data_loader, model, device, text_feats, output_csv):
    results = []

    with torch.no_grad():
        for images, labels, video_idxs, frame_idxs in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            logits = model.logit_scale.exp() * (image_features @ text_feats.t())
            results = format_results(results, logits, labels, video_idxs, frame_idxs)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    return results_df


def inference_triplet(data_loader, model, device, task_text_feats, output_dir, dataset_name):
    results = {task: [] for task in task_text_feats}
    results_df = {}

    with torch.no_grad():
        for images, labels, video_idxs, frame_idxs in tqdm(data_loader, desc="Evaluating triplet"):
            images = images.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            for task, text_feats in task_text_feats.items():
                logits = model.logit_scale.exp() * (image_features @ text_feats.t())
                results[task] = format_results(results[task], logits, labels[task], video_idxs, frame_idxs)

    for task, rows in results.items():
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, f"predictions_{task}_{dataset_name}.csv"), index=False)
        results_df[task] = df

    return results_df


class WorkflowEvaluator:
    def __init__(self, phases, prefix=""):
        self.phases = phases
        self.prefix = prefix
        self.num_phases = len(self.phases)

    def calc_accuracy(self, preds, labels):
        correct = (np.array(preds) == np.array(labels)).astype(int).sum().item()
        total = len(labels)
        return correct / total

    def calc_f1(self, preds, labels):
        fixed_labels = list(range(self.num_phases))
        f1_class = f1_score(labels, preds, average=None, labels=fixed_labels, zero_division=0)
        f1_average = f1_score(labels, preds, average="macro", labels=fixed_labels, zero_division=0)
        return f1_class, f1_average

    def evaluate(self, predictions_df, group_by_video=True):
        results = {
            self.prefix: {
                "accuracy": [],
                "f1_average": [],
                "f1_per_class": [],
                "video_ids": [],
            }
        }

        if group_by_video:
            for video_idx, video_df in predictions_df.groupby("video_idx"):
                video_df = video_df.sort_values("frame_idx")
                preds = convert_to_2d_array(video_df["prediction"].values).argmax(axis=1)
                labels = video_df["ground_truth"].values

                acc = self.calc_accuracy(preds, labels)
                f1_class, f1_average = self.calc_f1(preds, labels)

                results[self.prefix]["accuracy"].append(acc)
                results[self.prefix]["f1_average"].append(f1_average)
                results[self.prefix]["f1_per_class"].append(f1_class)
                results[self.prefix]["video_ids"].append(video_idx)

        all_preds = convert_to_2d_array(predictions_df["prediction"].values).argmax(axis=1)
        all_labels = predictions_df["ground_truth"].values

        overall_acc = self.calc_accuracy(all_preds, all_labels)
        overall_f1_class, overall_f1_average = self.calc_f1(all_preds, all_labels)

        results[self.prefix]["overall_accuracy"] = overall_acc
        results[self.prefix]["overall_f1_average"] = overall_f1_average
        results[self.prefix]["overall_f1_per_class"] = overall_f1_class
        results[self.prefix]["video_avg_accuracy"] = np.mean(results[self.prefix]["accuracy"])
        results[self.prefix]["video_avg_f1"] = np.mean(results[self.prefix]["f1_average"])
        results[self.prefix]["video_avg_f1_class"] = np.array(results[self.prefix]["f1_per_class"]).mean(axis=0)

        return results


class ToolPresenceEvaluator:
    def __init__(self, tools, prefix=""):
        self.tools = tools
        self.prefix = prefix
        self.num_classes = len(self.tools)

    def calc_mAP(self, preds, labels):
        ap = {}
        for c in range(self.num_classes):
            ap[c] = average_precision_score(labels[:, c], preds[:, c])

        mAP = np.nanmean(list(ap.values()))
        ap = list(ap.values())
        return ap, mAP

    def evaluate(self, predictions_df, group_by_video=True):
        results = {
            self.prefix: {
                "mAP": [],
                "AP_per_class": [],
            }
        }

        if group_by_video:
            for video_idx, video_df in predictions_df.groupby("video_idx"):
                video_df = video_df.sort_values("frame_idx")
                preds = convert_to_2d_array(video_df["prediction"].values)
                labels = convert_to_2d_array(video_df["ground_truth"].values)

                ap, mAP = self.calc_mAP(preds, labels)
                results[self.prefix]["mAP"].append(mAP)
                results[self.prefix]["AP_per_class"].append(ap)

        all_preds = convert_to_2d_array(predictions_df["prediction"].values)
        all_labels = convert_to_2d_array(predictions_df["ground_truth"].values)

        overall_ap, overall_mAP = self.calc_mAP(all_preds, all_labels)
        results[self.prefix]["overall_mAP"] = overall_mAP
        results[self.prefix]["overall_AP_per_class"] = overall_ap
        results[self.prefix]["video_avg_mAP"] = np.mean(results[self.prefix]["mAP"])
        results[self.prefix]["video_avg_AP_class"] = np.array(results[self.prefix]["AP_per_class"]).mean(axis=0)

        return results


class TripletEvaluator:
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.metrics = {
            "triplet": ivtmetrics.Recognition(100),
            "instrument": ivtmetrics.Recognition(6),
            "verb": ivtmetrics.Recognition(10),
            "target": ivtmetrics.Recognition(15),
        }

        for metric in self.metrics.values():
            metric.reset_global()
            metric.reset()

    def evaluate(self, predictions_df, group_by_video=True):
        results = {
            self.prefix: {
                "triplet": {},
                "instrument": {},
                "verb": {},
                "target": {},
            }
        }

        for task, df in predictions_df.items():
            for _, video_df in df.groupby("video_idx"):
                video_df = video_df.sort_values("frame_idx")
                preds = convert_to_2d_array(video_df["prediction"].values)
                labels = convert_to_2d_array(video_df["ground_truth"].values)
                self.metrics[task].update(labels, preds)
                self.metrics[task].video_end()

        for task, metric in self.metrics.items():
            results[self.prefix][task] = metric.compute_video_AP(ignore_null=False)

        results[self.prefix]["instrument-target"] = self.metrics["triplet"].compute_video_AP(
            "it",
            ignore_null=False,
        )
        results[self.prefix]["instrument-verb"] = self.metrics["triplet"].compute_video_AP(
            "iv",
            ignore_null=False,
        )

        return results


def evaluation(model, data_loader, tokenizer, device, output_dir):
    prompts = data_loader.dataset.prompts
    text_feats = extract_text_feats(prompts, model, tokenizer, device)
    output_csv = os.path.join(output_dir, f"predictions_{data_loader.dataset.name}.csv")
    return inference(data_loader, model, device, text_feats, output_csv)


def evaluation_triplet(model, data_loader, tokenizer, device, output_dir):
    task_text_feats = {
        task_name: extract_text_feats(prompts, model, tokenizer, device)
        for task_name, prompts in data_loader.dataset.prompts.items()
    }
    return inference_triplet(
        data_loader=data_loader,
        model=model,
        device=device,
        task_text_feats=task_text_feats,
        output_dir=output_dir,
        dataset_name=data_loader.dataset.name,
    )


def evaluation_wrapper(model, data_loader, tokenizer, device, output_dir, prefix=""):
    task = data_loader.dataset.task

    if task == "triplet":
        results_df = evaluation_triplet(
            model=model,
            data_loader=data_loader,
            tokenizer=tokenizer,
            device=device,
            output_dir=output_dir,
        )
        evaluator = TripletEvaluator(prefix=prefix)
    else:
        results_df = evaluation(
            model=model,
            data_loader=data_loader,
            tokenizer=tokenizer,
            device=device,
            output_dir=output_dir,
        )

        if task in ["phases", "steps", "actions"]:
            evaluator = WorkflowEvaluator(phases=data_loader.dataset.categories, prefix=prefix)
        elif task == "instruments":
            evaluator = ToolPresenceEvaluator(tools=data_loader.dataset.categories, prefix=prefix)
        else:
            raise ValueError(f"Unsupported task: {task}")

    return evaluator.evaluate(results_df)


def evaluate_zero_shot(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("正在加载模型结构...")
    model = VLP(
        embed_dim=512,
        text_model_name=args.text_model,
        vision_pretrained_weights=args.vision_weights,
    ).to(device)

    model = load_model_checkpoint(model, args.ckpt, device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    data_loader, _ = build_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_loader.dataset.name = args.dataset

    results = evaluation_wrapper(
        model=model,
        data_loader=data_loader,
        tokenizer=tokenizer,
        device=device,
        output_dir=args.output_dir,
        prefix=args.dataset,
    )

    result_path = os.path.join(args.output_dir, f"results_{args.dataset}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(results), f, ensure_ascii=False, indent=2)

    print(json.dumps(to_builtin(results), ensure_ascii=False, indent=2))
    print(f"\n结果已保存到: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation for VLP")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--text_model", type=str, default="marcobombieri/surgicberta")
    parser.add_argument("--vision_weights", type=str, default="lemonfm.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_zero_shot(args)
