"""Training entry point for per-frame EndoSSL ViT with uniform pooling.

This keeps frames independent in the visual encoder:
1. Each frame is encoded by the same 2D EndoSSL ViT.
2. The base video feature is the uniform mean of frame features.
3. Text-conditioned frame selection still uses the existing similarity/xpool
   logic in model.VLP.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import timm

os.environ.setdefault("DATA_NORMALIZATION", "none")
os.environ.setdefault("ENDOSSL_VIT_SIZE", os.environ.get("TIMESFORMER_SIZE", "vitb"))

import model as model_module
from model import VLP, ResidualFeatureAdapter


ENDOSSL_VIT_CONFIGS = {
    "vits": {
        "timm_name": "vit_small_patch16_224",
        "output_dim": 384,
        "ckpt_name": "endossl_vits.pth",
    },
    "vitb": {
        "timm_name": "vit_base_patch16_224",
        "output_dim": 768,
        "ckpt_name": "endossl_vitb.pth",
    },
    "vitl": {
        "timm_name": "vit_large_patch16_224",
        "output_dim": 1024,
        "ckpt_name": "endossl_vitl.pth",
    },
}

ENDOSSL_SIZE_ALIASES = {
    "s": "vits",
    "small": "vits",
    "vits": "vits",
    "b": "vitb",
    "base": "vitb",
    "vitb": "vitb",
    "l": "vitl",
    "large": "vitl",
    "vitl": "vitl",
    "endossl": "vitl",
    "endossl_vitl": "vitl",
}


class _DummyBackbone(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, x):
        return x


class UniformFramePool(nn.Module):
    """No-parameter temporal head: mean pool for global, raw frame tokens for selection."""

    def forward(self, frame_features, return_tokens=False):
        if frame_features.ndim != 3:
            raise ValueError(
                f"Expected frame_features shape [B, T, D], got {tuple(frame_features.shape)}"
            )
        pooled = frame_features.mean(dim=1)
        if return_tokens:
            return pooled, frame_features
        return pooled


class EndoSSLFrameViTBackbone(nn.Module):
    """Standard 2D ViT loaded with EndoSSL weights and applied per frame."""

    def __init__(self, size, pretrained_weights=""):
        super().__init__()
        size = ENDOSSL_SIZE_ALIASES.get(str(size or "vitb").strip().lower(), str(size).lower())
        if size not in ENDOSSL_VIT_CONFIGS:
            raise ValueError(f"Unknown EndoSSL ViT size: {size}. Expected vits, vitb, or vitl.")

        cfg = ENDOSSL_VIT_CONFIGS[size]
        self.size = size
        self.name = f"endossl_{size}_uniform_pool"
        self.output_dim = int(cfg["output_dim"])
        self.model = timm.create_model(
            cfg["timm_name"],
            pretrained=False,
            num_classes=0,
            global_pool="token",
        )
        self._load_endossl_weights(pretrained_weights, cfg["ckpt_name"])

    def _resolve_ckpt_path(self, pretrained_weights, default_name):
        requested_path = str(pretrained_weights or "").strip()
        if requested_path:
            ckpt_path = Path(requested_path).expanduser()
            if not ckpt_path.is_absolute() and not ckpt_path.is_file():
                ckpt_path = Path(__file__).parent / requested_path
            return ckpt_path
        return Path(__file__).parent / default_name

    def _load_endossl_weights(self, pretrained_weights, default_name):
        ckpt_path = self._resolve_ckpt_path(pretrained_weights, default_name)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"EndoSSL checkpoint not found: {ckpt_path}")

        try:
            state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(str(ckpt_path), map_location="cpu")

        target_state = self.model.state_dict()
        matched_keys = [
            key for key, value in state_dict.items()
            if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
        ]
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(
            f"[EndoSSLFrameViTBackbone] Loaded {ckpt_path.name}: "
            f"{len(matched_keys)}/{len(state_dict)} keys matched, "
            f"{len(msg.missing_keys)} missing keys, "
            f"{len(msg.unexpected_keys)} unexpected keys"
        )

    def forward(self, x):
        # EndoSSL's released TF model consumes resized raw pixel values.
        x = x.float()
        if x.min() < 0 or x.max() > 1.1:
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x * std + mean) * 255.0
        else:
            x = x * 255.0
        return self.model(x)

    def unfreeze_last_n_blocks(self, n):
        num_blocks = len(self.model.blocks)
        n = max(1, int(n))
        n = min(n, num_blocks)
        for p in self.model.blocks[-n:].parameters():
            p.requires_grad = True

    def set_last_n_blocks_train(self, n):
        num_blocks = len(self.model.blocks)
        n = max(1, int(n))
        n = min(n, num_blocks)
        for blk in self.model.blocks[-n:]:
            blk.train()


class VLPWithEndoSSLUniformPool(VLP):
    """VLP with independent per-frame EndoSSL ViT and uniform global pooling."""

    def __init__(self, **kwargs):
        size = os.environ.get("ENDOSSL_VIT_SIZE", "").strip().lower()
        if not size:
            size = str(kwargs.get("vision_backbone", "vitb")).strip().lower()
        size = ENDOSSL_SIZE_ALIASES.get(size, size)
        if size not in ENDOSSL_VIT_CONFIGS:
            size = "vitb"

        output_dim = ENDOSSL_VIT_CONFIGS[size]["output_dim"]

        actual_num_frames = max(1, int(kwargs.get("num_frames", 1)))
        parent_kwargs = dict(kwargs)
        parent_kwargs["num_frames"] = 1

        original_build = model_module.build_visual_backbone
        model_module.build_visual_backbone = lambda name, weights: _DummyBackbone(output_dim)
        try:
            super().__init__(**parent_kwargs)
        finally:
            model_module.build_visual_backbone = original_build

        self.num_frames = actual_num_frames
        self.visual = EndoSSLFrameViTBackbone(
            size=size,
            pretrained_weights=kwargs.get("vision_pretrained_weights", ""),
        )
        self.frame_pool = UniformFramePool()
        self.visual_dim = output_dim
        self.frame_token_dim = output_dim
        self.temporal_hidden_dim = output_dim

        embed_dim = self.embed_dim
        self.video_adapter = ResidualFeatureAdapter(output_dim)
        self.video_projection = nn.Linear(output_dim, embed_dim, bias=False)
        self.frame_local_projection = nn.Linear(output_dim, embed_dim, bias=False)
        self.selection_frame_key_projection = nn.Linear(output_dim, embed_dim, bias=False)

        nn.init.normal_(self.video_projection.weight, std=output_dim ** -0.5)
        nn.init.normal_(self.frame_local_projection.weight, std=embed_dim ** -0.5)
        nn.init.normal_(self.selection_frame_key_projection.weight, std=embed_dim ** -0.5)

        print(
            "[VLPWithEndoSSLUniformPool] "
            f"size={size}, frame_pool=uniform, frame_interaction=disabled"
        )


model_module.VLP = VLPWithEndoSSLUniformPool

from train_frozen_vis import train


if __name__ == "__main__":
    train()
