import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence, Union

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_ROOT = Path("/data/znh/surgical_vlp")
CAMMA_MODEL_FILES = {
    "surgvlp": "SurgVLP.pth",
    "hecvl": "HecVL.pth",
    "peskavlp": "PeskaVLP.pth",
}
SURGCLIP_BETA_FILE = "surgclip_beta.pth"


def _as_pil(frame) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu()
        if frame.ndim == 3 and frame.shape[0] in (1, 3):
            frame = frame.permute(1, 2, 0)
        frame = frame.numpy()
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _pil_list(frames: Union[np.ndarray, torch.Tensor, Sequence]) -> List[Image.Image]:
    if isinstance(frames, np.ndarray):
        if frames.ndim == 3:
            return [_as_pil(frames)]
        if frames.ndim == 4:
            return [_as_pil(frames[i]) for i in range(frames.shape[0])]
    if isinstance(frames, torch.Tensor):
        if frames.ndim == 3:
            return [_as_pil(frames)]
        if frames.ndim == 4:
            return [_as_pil(frames[i]) for i in range(frames.shape[0])]
    return [_as_pil(frame) for frame in frames]


def _normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod.") :]
        normalized[key] = value
    return normalized


def _load_state_dict_file(path: Union[str, Path]):
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "module"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
    return ckpt


def _resolve_local_hf_model(model_name: str) -> str:
    model_name = str(model_name)
    path = Path(model_name).expanduser()
    if path.exists():
        return str(path)

    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception:
        pass

    cache_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_name.replace('/', '--')}"
        / "snapshots"
    )
    if cache_dir.exists():
        snapshots = sorted(
            [item for item in cache_dir.iterdir() if item.is_dir()],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0])
    return model_name


class SurgicalVLPScorer:
    name: str

    def encode_image(self, frames, batch_size: int = 64) -> torch.Tensor:
        raise NotImplementedError

    def encode_text(self, captions: Union[str, Sequence[str]]) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def score_frames(self, frames, caption: str, batch_size: int = 64) -> np.ndarray:
        image_features = self.encode_image(frames, batch_size=batch_size)
        text_features = self.encode_text([caption])
        scores = self.logit_scale * (image_features @ text_features.t()).squeeze(-1)
        return scores.detach().cpu().float().reshape(-1).numpy()


class SurgCLIPBetaScorer(SurgicalVLPScorer):
    def __init__(
        self,
        ckpt_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
        image_size: int = 224,
        text_model_name: str = "bert-base-uncased",
    ):
        from surgclip.surgclip.config import get_config
        from surgclip.surgclip.model import SurgCLIP

        self.name = "surgclip_beta"
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        text_model_path = _resolve_local_hf_model(text_model_name)
        cfg = get_config(
            "SurgCLIP-B",
            overrides={
                "device": str(self.device),
                "num_frames": 1,
                "inputs": {
                    "image_res": self.image_size,
                    "video_input": {"num_frames_test": 1},
                },
                "model": {
                    "text_encoder": {"pretrained": text_model_path},
                    "temporal_modeling": {"enabled": False},
                },
            },
        )
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_path, local_files_only=True)
        self.model = SurgCLIP(config=cfg, tokenizer=self.tokenizer, is_pretrain=False)
        state_dict = _normalize_state_dict_keys(_load_state_dict_file(ckpt_path))
        msg = self.model.load_state_dict(state_dict, strict=False)
        bad_missing = [
            key
            for key in msg.missing_keys
            if not key.startswith("vision_encoder.model.time_embed")
        ]
        if bad_missing:
            raise RuntimeError(f"SurgCLIP-beta missing keys: {bad_missing[:20]}")
        self.model = self.model.to(self.device).eval()

    @property
    def logit_scale(self) -> float:
        temp = float(self.model.temp.detach().cpu().item())
        return 1.0 / max(temp, 1e-6)

    @torch.no_grad()
    def encode_image(self, frames, batch_size: int = 64) -> torch.Tensor:
        pil_frames = _pil_list(frames)
        outputs = []
        for start in range(0, len(pil_frames), batch_size):
            batch = torch.stack(
                [self.transform(frame) for frame in pil_frames[start : start + batch_size]],
                dim=0,
            ).to(self.device)
            video = batch.unsqueeze(1)
            _, pooled = self.model.encode_vision(video)
            feats = F.normalize(self.model.vision_proj(pooled), dim=-1)
            if feats.ndim == 3 and feats.shape[1] == 1:
                feats = feats[:, 0, :]
            outputs.append(feats)
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def encode_text(self, captions: Union[str, Sequence[str]]) -> torch.Tensor:
        if isinstance(captions, str):
            captions = [captions]
        tokenized = self.tokenizer(
            list(captions),
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        text = SimpleNamespace(
            input_ids=tokenized["input_ids"].to(self.device),
            attention_mask=tokenized["attention_mask"].to(self.device),
        )
        _, pooled = self.model.encode_text(text)
        feats = F.normalize(self.model.text_proj(pooled), dim=-1)
        if feats.ndim == 3 and feats.shape[1] == 1:
            feats = feats[:, 0, :]
        return feats


class Identity(nn.Module):
    def forward(self, x):
        return x


class CammaImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=None)
        feature_dim = self.model.fc.in_features
        self.model.fc = Identity()
        self.global_embedder = nn.Linear(feature_dim, 768)

    def forward(self, x):
        return self.global_embedder(self.model(x))


class CammaBertEncoder(nn.Module):
    def __init__(self, text_model_path: str, tokenizer=None):
        super().__init__()
        self.last_n_layers = 4
        self.aggregate_method = "sum"
        self.embedding_dim = 768
        self.agg_tokens = True
        self.model = AutoModel.from_pretrained(
            text_model_path,
            output_hidden_states=True,
            local_files_only=True,
        )
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            text_model_path,
            local_files_only=True,
        )
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def aggregate_tokens(self, embeddings, caption_ids):
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        for embs, caption_id in zip(embeddings, caption_ids):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []
            for word_emb, word_id in zip(embs, caption_id):
                word = self.idxtoword.get(word_id.item(), "[UNK]")
                if word == "[SEP]":
                    if token_bank:
                        agg_embs.append(torch.stack(token_bank).sum(axis=0))
                        words.append("".join(word_bank))
                    agg_embs.append(word_emb)
                    words.append(word)
                    break
                if not word.startswith("##"):
                    if not word_bank:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        agg_embs.append(torch.stack(token_bank).sum(axis=0))
                        words.append("".join(word_bank))
                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])
            if not agg_embs and token_bank:
                agg_embs.append(torch.stack(token_bank).sum(axis=0))
                words.append("".join(word_bank))
            if not agg_embs:
                agg_embs.append(torch.zeros(num_layers, dim, device=embs.device, dtype=embs.dtype))
                words.append("[PAD]")
            agg_embs = torch.stack(agg_embs)
            padding_size = max(num_words - len(agg_embs), 0)
            if padding_size:
                padding = torch.zeros(
                    padding_size,
                    num_layers,
                    dim,
                    device=agg_embs.device,
                    dtype=agg_embs.dtype,
                )
                agg_embs = torch.cat([agg_embs, padding], dim=0)
                words = words + ["[PAD]"] * padding_size
            agg_embs_batch.append(agg_embs[:num_words])
            sentences.append(words[:num_words])
        agg_embs_batch = torch.stack(agg_embs_batch)
        return agg_embs_batch.permute(0, 2, 1, 3), sentences

    def forward(self, ids=None, attn_mask=None, token_type=None):
        outputs = self.model(ids, attention_mask=attn_mask, token_type_ids=token_type)
        all_embeddings = outputs.hidden_states
        embeddings = torch.stack(all_embeddings[-self.last_n_layers :]).permute(1, 0, 2, 3)
        embeddings, sentences = self.aggregate_tokens(embeddings, ids)
        word_embeddings = embeddings.sum(axis=1)
        sent_embeddings = embeddings.mean(axis=2).sum(axis=1)
        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim, num_words, self.embedding_dim).permute(0, 2, 1)
        return word_embeddings, sent_embeddings, sentences


class CammaMVNet(nn.Module):
    def __init__(self, text_model_path: str, tokenizer=None):
        super().__init__()
        self.backbone_img = CammaImageEncoder()
        self.backbone_text = CammaBertEncoder(text_model_path, tokenizer=tokenizer)

    def forward(self, inputs_img=None, inputs_text=None, mode="all"):
        out = {}
        if mode in ("video", "all", "action"):
            out["img_emb"] = self.backbone_img(inputs_img)
        if mode in ("text", "all", "action"):
            _, text_global, _ = self.backbone_text(
                ids=inputs_text["input_ids"],
                attn_mask=inputs_text["attention_mask"],
                token_type=inputs_text["token_type_ids"],
            )
            out["text_emb"] = text_global
        return out


def _resolve_clinical_bert_path(clinical_bert_path: Optional[str], model_root: Path) -> str:
    candidates = []
    if clinical_bert_path:
        candidates.append(str(clinical_bert_path))
    env_path = os.environ.get("CLINICAL_BERT_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        [
            str(model_root / "Bio_ClinicalBERT"),
            str(model_root / "emilyalsentzer_Bio_ClinicalBERT"),
            "emilyalsentzer/Bio_ClinicalBERT",
            "milyalsentzer/Bio_ClinicalBERT",
        ]
    )
    for candidate in candidates:
        resolved = _resolve_local_hf_model(candidate)
        try:
            AutoTokenizer.from_pretrained(resolved, local_files_only=True)
            AutoModel.from_pretrained(resolved, local_files_only=True)
            return resolved
        except Exception:
            pass
    raise FileNotFoundError(
        "BioClinicalBERT tokenizer/config is required for SurgVLP/HecVL/PeskaVLP. "
        "Put it under /data/znh/surgical_vlp/Bio_ClinicalBERT or pass --clinical_bert_path."
    )


class CammaVLPScorer(SurgicalVLPScorer):
    def __init__(
        self,
        name: str,
        ckpt_path: Union[str, Path],
        clinical_bert_path: Optional[str] = None,
        model_root: Union[str, Path] = DEFAULT_MODEL_ROOT,
        device: Union[str, torch.device] = "cuda",
        image_size: int = 336,
        text_max_length: int = 77,
    ):
        self.name = name.lower()
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.text_max_length = int(text_max_length)
        self.model_root = Path(model_root)
        text_model_path = _resolve_clinical_bert_path(clinical_bert_path, self.model_root)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_path, local_files_only=True)
        self.model = CammaMVNet(text_model_path, tokenizer=self.tokenizer)
        state_dict = _normalize_state_dict_keys(_load_state_dict_file(ckpt_path))
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if "position_ids" not in k]
        unexpected = [k for k in unexpected if "position_ids" not in k]
        if missing or unexpected:
            raise RuntimeError(
                f"{self.name} load mismatch: missing={missing[:20]}, "
                f"unexpected={unexpected[:20]}"
            )
        self.model = self.model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    @property
    def logit_scale(self) -> float:
        return 100.0

    @torch.no_grad()
    def encode_image(self, frames, batch_size: int = 64) -> torch.Tensor:
        pil_frames = _pil_list(frames)
        outputs = []
        for start in range(0, len(pil_frames), batch_size):
            batch = torch.stack(
                [self.transform(frame) for frame in pil_frames[start : start + batch_size]],
                dim=0,
            ).to(self.device)
            feats = self.model(batch, None, mode="video")["img_emb"]
            outputs.append(F.normalize(feats, dim=-1))
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def encode_text(self, captions: Union[str, Sequence[str]]) -> torch.Tensor:
        if isinstance(captions, str):
            captions = [captions]
        tokenized = self.tokenizer(
            list(captions),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.text_max_length,
        )
        token_type_ids = tokenized.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(tokenized["input_ids"])
        text = {
            "input_ids": tokenized["input_ids"].to(self.device),
            "attention_mask": tokenized["attention_mask"].to(self.device),
            "token_type_ids": token_type_ids.to(self.device),
        }
        feats = self.model(None, text, mode="text")["text_emb"]
        return F.normalize(feats, dim=-1)


def available_scorers() -> List[str]:
    return ["surgclip_beta", "surgvlp", "hecvl", "peskavlp"]


def load_scorer(
    name: str,
    model_root: Union[str, Path] = DEFAULT_MODEL_ROOT,
    device: Union[str, torch.device] = "cuda",
    clinical_bert_path: Optional[str] = None,
):
    model_root = Path(model_root)
    key = name.lower().replace("-", "_")
    if key in ("surgclip", "surgclip_beta", "surgclip-b", "surgclip_b"):
        return SurgCLIPBetaScorer(model_root / SURGCLIP_BETA_FILE, device=device)
    if key in CAMMA_MODEL_FILES:
        return CammaVLPScorer(
            key,
            model_root / CAMMA_MODEL_FILES[key],
            clinical_bert_path=clinical_bert_path,
            model_root=model_root,
            device=device,
        )
    raise ValueError(f"Unknown scorer {name!r}. Available: {available_scorers()}")
