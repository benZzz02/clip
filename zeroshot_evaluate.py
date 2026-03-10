import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from model import VLP
from downstream_datasets import Cholec80FrameDataset


def evaluate_zero_shot():
    # --- 1. 配置 ---
    MODEL_WEIGHTS_PATH = "/mnt/mydisk/CLIP/vlp_epoch_18.pt"   # 先用你保存的初始权重也能跑通
    CHOLEC80_ANNOTATIONS_FOLDER = "cholec80_peska/"
    CHOLEC80_VIDEOS_FOLDER = "/mnt/mydisk/Data/cholec80/cutMargin/"
    BATCH_SIZE = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 2. 加载模型和 tokenizer ---
    print(f"正在加载模型: {MODEL_WEIGHTS_PATH}")
    model = VLP(
        embed_dim=512,
        text_model_name="marcobombieri/surgicberta",
        vision_pretrained_weights="lemonfm.pth",
    ).to(device)

    ckpt = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("marcobombieri/surgicberta")

    # --- 3. 文本提示 ---
    cholec80_phase_prompts = {
        "Preparation": ["In preparation phase I insert trocars to patient abdomen cavity"],
        "Calot Triangle Dissection": ["In calot triangle dissection phase I use grasper to hold gallbladder and use hook to expose the hepatic triangle area and cystic duct and cystic artery"],
        "Clipping and Cutting": ["In clip and cut phase I use clipper to clip the cystic duct and artery then use scissor to cut them"],
        "Gallbladder Dissection": ["In dissection phase I use the hook to dissect the connective tissue between gallbladder and liver"],
        "Gallbladder Retraction": ["In retraction phase I grasp the specimen bag and remove it from trocar"],
        "Cleaning and Coagulation": ["In clean and coagulation phase I use suction and irrigation to clear the surgical field and coagulate bleeding vessels"],
        "Gallbladder Packaging": ["In packaging phase I put the gallbladder into the specimen bag"]
    }
    class_names = list(cholec80_phase_prompts.keys())

    # --- 4. 编码文本 ---
    text_features_list = []
    with torch.no_grad():
        print("正在编码文本提示...")
        for class_name in tqdm(class_names, desc="Encoding prompts"):
            prompts = cholec80_phase_prompts[class_name]

            tokenized = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            class_embeddings = model.encode_text(
                tokenized["input_ids"],
                tokenized["attention_mask"]
            )
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            class_embeddings = class_embeddings.mean(dim=0)
            text_features_list.append(class_embeddings)

        text_features = torch.stack(text_features_list, dim=0)   # [num_classes, D]
        text_features = F.normalize(text_features, dim=-1)

    # --- 5. 加载单帧数据集 ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = Cholec80FrameDataset(
        annotation_folder=CHOLEC80_ANNOTATIONS_FOLDER,
        video_folder=CHOLEC80_VIDEOS_FOLDER,
        transform=transform
    )
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- 6. 遍历评测 ---
    all_preds = []
    all_labels = []

    print("\n开始在整个数据集上进行零样本评估...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)          # [B, D]
            image_features = F.normalize(image_features, dim=-1)

            logits = model.logit_scale.exp() * image_features @ text_features.t()   # [B, C]
            probs = logits.softmax(dim=-1)
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

    print("\n--- 零样本测试评估结果 ---")
    print(f"总准确率 (Overall Accuracy): {accuracy:.4f}")
    print("\n详细分类报告:")
    print(report)


if __name__ == "__main__":
    evaluate_zero_shot()