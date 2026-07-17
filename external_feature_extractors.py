import importlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parent
SURGVLP_ROOT = PROJECT_ROOT / "third_party" / "SurgVLP"

EXTERNAL_FEATURE_MODES = {"surgvlp", "hecvl", "peskavlp", "surgclip_beta"}


def canonical_feature_mode(feature_mode: str) -> str:
    mode = str(feature_mode or "vlp").strip().lower()
    aliases = {
        "surgalign": "vlp",
        "surgclip": "surgclip_beta",
        "surgclip-b": "surgclip_beta",
        "surgclip_beta": "surgclip_beta",
        "surgclip-beta": "surgclip_beta",
        "peska": "peskavlp",
        "peska_vlp": "peskavlp",
        "peska-vlp": "peskavlp",
    }
    return aliases.get(mode, mode)


def is_external_feature_mode(feature_mode: str) -> bool:
    return canonical_feature_mode(feature_mode) in EXTERNAL_FEATURE_MODES


def build_external_transform(feature_mode: str, image_size: int):
    mode = canonical_feature_mode(feature_mode)
    if mode == "surgclip_beta":
        return _build_surgclip_transform(image_size)
    if mode in {"surgvlp", "hecvl", "peskavlp"}:
        return _build_surgvlp_transform(image_size)
    raise ValueError(f"Unsupported external feature mode: {feature_mode}")


def build_external_feature_model(args, device):
    mode = canonical_feature_mode(args.feature_mode)
    if mode in {"surgvlp", "hecvl", "peskavlp"}:
        model = SurgVLPVisionFeatureExtractor(
            model_name=mode,
            ckpt_path=args.ckpt or None,
            config_path=args.external_config or None,
            cache_dir=args.external_cache_dir or None,
            device=device,
        )
        return model.eval(), model.output_dim
    if mode == "surgclip_beta":
        model = SurgCLIPBetaVisionFeatureExtractor(
            ckpt_path=args.ckpt or None,
            model_name=args.surgclip_model_name,
            num_frames=args.num_frames,
            image_size=args.image_size,
            device=device,
        )
        return model.eval(), model.output_dim
    raise ValueError(f"Unsupported external feature mode: {args.feature_mode}")


def _build_surgvlp_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((360, 640)),
            transforms.CenterCrop(int(image_size)),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _build_surgclip_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(
                (int(image_size), int(image_size)),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "network"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _strip_common_prefixes(key: str) -> str:
    while key.startswith("module.") or key.startswith("_orig_mod."):
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
    return key


def _add_surgvlp_to_path():
    if not SURGVLP_ROOT.exists():
        raise FileNotFoundError(
            f"SurgVLP source tree not found at {SURGVLP_ROOT}. "
            "Expected the official CAMMA-public/SurgVLP repo under third_party/SurgVLP."
        )
    path = str(SURGVLP_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _default_surgvlp_config(model_name: str) -> Path:
    filename = {
        "surgvlp": "config_surgvlp.py",
        "hecvl": "config_hecvl.py",
        "peskavlp": "config_peskavlp.py",
    }[model_name]
    return SURGVLP_ROOT / "tests" / filename


class SurgVLPVisionFeatureExtractor(nn.Module):
    """Vision-only adapter for CAMMA SurgVLP/HecVL/PeskaVLP checkpoints."""

    def __init__(
        self,
        model_name: str,
        ckpt_path: Optional[str],
        config_path: Optional[str],
        cache_dir: Optional[str],
        device,
    ):
        super().__init__()
        _add_surgvlp_to_path()

        from mmengine.config import Config
        from surgvlp.codes.models import build_backbone

        surgvlp_api = importlib.import_module("surgvlp.surgvlp")

        self.model_name = canonical_feature_mode(model_name)
        cfg_path = Path(config_path) if config_path else _default_surgvlp_config(self.model_name)
        configs = Config.fromfile(str(cfg_path))["config"]

        model_config = dict(configs.model_config)
        backbone_cfg = dict(model_config["backbone_img"])
        # The released checkpoint supplies these weights; avoid an unnecessary
        # torchvision ImageNet download when constructing the backbone shell.
        backbone_cfg["pretrained"] = "random"

        self.backbone = build_backbone(backbone_cfg).to(device)
        self.output_dim = int(backbone_cfg.get("num_classes", 768))

        checkpoint_path = ckpt_path
        if checkpoint_path is None:
            official_name = model_config["type"]
            checkpoint_path = surgvlp_api._download(
                surgvlp_api._MODELS,
                official_name,
                cache_dir or os.path.expanduser("~/.cache/surgvlp"),
            )

        self._load_backbone_checkpoint(checkpoint_path, device)

    def _load_backbone_checkpoint(self, checkpoint_path, device):
        checkpoint = _safe_torch_load(checkpoint_path, map_location=device)
        state_dict = _extract_state_dict(checkpoint)
        target_state = self.backbone.state_dict()
        visual_state = {}

        for raw_key, value in state_dict.items():
            key = _strip_common_prefixes(str(raw_key))
            if key.startswith("backbone_img."):
                key = key[len("backbone_img.") :]
            elif key not in target_state:
                continue

            if key in target_state and tuple(value.shape) == tuple(target_state[key].shape):
                visual_state[key] = value

        if not visual_state:
            raise RuntimeError(
                f"No SurgVLP visual weights matched {checkpoint_path}. "
                "Expected keys such as backbone_img.model.conv1.weight."
            )

        msg = self.backbone.load_state_dict(visual_state, strict=False)
        print(
            f"Loaded {self.model_name} visual encoder from {checkpoint_path}: "
            f"matched={len(visual_state)} missing={len(msg.missing_keys)} "
            f"unexpected={len(msg.unexpected_keys)}"
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 5:
            batch_size, num_frames, channels, height, width = images.shape
            flat = images.reshape(batch_size * num_frames, channels, height, width)
            features = self.backbone(flat).reshape(batch_size, num_frames, -1).mean(dim=1)
        elif images.ndim == 4:
            features = self.backbone(images)
        else:
            raise ValueError(f"Unexpected image tensor shape: {tuple(images.shape)}")
        return F.normalize(features, dim=-1)

    def encode_probe_features(self, images: torch.Tensor, probe_feature: str = "projected") -> torch.Tensor:
        # SurgVLP/HecVL expose a single visual embedding from the official
        # backbone. Treat it as the probe feature for all representation modes.
        return self.encode_image(images)


class SurgCLIPBetaVisionFeatureExtractor(nn.Module):
    """Vision-only adapter for SurgLaVi/SurgCLIP-beta checkpoints."""

    def __init__(self, ckpt_path: Optional[str], model_name: str, num_frames: int, image_size: int, device):
        super().__init__()
        from surgclip.surgclip.config import get_config
        from surgclip.surgclip.download import download_weights
        from surgclip.surgclip.models.timesformer.timesformer import TimeSformer

        self.model_name = model_name
        self.config = get_config(
            model_name,
            overrides={
                "device": str(device),
                "num_frames": int(num_frames),
                "inputs": {"image_res": int(image_size)},
                "model": {"temporal_modeling": {"enabled": int(num_frames) > 1}},
            },
        )

        vit_cfg = json.load(open(self.config.model.vision_encoder.config))
        if self.config.model.temporal_modeling.enabled:
            vit_cfg.update({"num_frames": int(num_frames), "attention_type": "divided_space_time"})
        else:
            vit_cfg.update({"num_frames": 1, "attention_type": "space_only"})
        vit_cfg.update({"num_classes": 0, "gradient_checkpointing": self.config.gradient_checkpointing})

        self.vision_encoder = TimeSformer(**vit_cfg)
        self.vision_layernorm = (
            nn.LayerNorm(self.config.model.vision_encoder.d_model, eps=1e-12)
            if self.config.model.vit_add_ln
            else nn.Identity()
        )
        self.vision_proj = nn.Linear(self.config.model.vision_encoder.d_model, self.config.model.embed_dim)
        self.output_dim = int(self.config.model.embed_dim)

        checkpoint_path = ckpt_path or str(download_weights(model_name))
        self._load_vision_checkpoint(checkpoint_path, device)
        self.to(device)

    def _load_vision_checkpoint(self, checkpoint_path, device):
        checkpoint = _safe_torch_load(checkpoint_path, map_location=device)
        state_dict = _extract_state_dict(checkpoint)
        target_state = self.state_dict()
        visual_state = {}

        for raw_key, value in state_dict.items():
            key = _strip_common_prefixes(str(raw_key))
            if key.startswith("surgclip."):
                key = key[len("surgclip.") :]
            if key in {"logit_scale", "temp"}:
                continue
            if key in target_state and tuple(value.shape) == tuple(target_state[key].shape):
                visual_state[key] = value

        if not visual_state:
            raise RuntimeError(
                f"No SurgCLIP-beta visual weights matched {checkpoint_path}. "
                "Expected keys such as vision_encoder.patch_embed.proj.weight."
            )

        msg = self.load_state_dict(visual_state, strict=False)
        print(
            f"Loaded {self.model_name} visual encoder from {checkpoint_path}: "
            f"matched={len(visual_state)} missing={len(msg.missing_keys)} "
            f"unexpected={len(msg.unexpected_keys)}"
        )

    def _encode_pooled_vision(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(f"Expected [B, C, H, W] or [B, T, C, H, W], got {tuple(images.shape)}")

        images = images.permute(0, 2, 1, 3, 4)
        _, pooled = self.vision_encoder(images)

        if pooled.ndim == 3:
            pooled = pooled.mean(dim=1)
        elif pooled.ndim != 2:
            raise ValueError(f"Expected pooled vision features to be 2D or 3D, got {tuple(pooled.shape)}")

        return pooled

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        pooled = self._encode_pooled_vision(images)
        return F.normalize(self.vision_proj(pooled), dim=-1)

    def encode_probe_features(self, images: torch.Tensor, probe_feature: str = "projected") -> torch.Tensor:
        pooled = self._encode_pooled_vision(images)
        backbone = F.normalize(pooled, dim=-1)
        projected = F.normalize(self.vision_proj(pooled), dim=-1)

        probe_feature = str(probe_feature or "projected").lower()
        if probe_feature == "backbone":
            return backbone
        if probe_feature == "concat":
            return F.normalize(torch.cat([backbone, projected], dim=-1), dim=-1)
        if probe_feature == "projected":
            return projected
        raise ValueError(f"Unsupported probe_feature: {probe_feature}")
