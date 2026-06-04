import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _build_convnext_large_backbone(weights=None):
    net = torchvision.models.convnext_large(weights=weights)
    in_dim = net.classifier[2].in_features
    net.classifier[2] = nn.Identity()
    net.output_dim = in_dim
    net.name = "convnext_lemonfm"
    return net


def build_LemonFM(pretrained_weights="lemonfm.pth"):
    if pretrained_weights is None:
        raise ValueError("pretrained_weights is None")

    mode = pretrained_weights.strip().lower() if isinstance(pretrained_weights, str) else pretrained_weights

    if mode in {"random", "none", "scratch"}:
        print("Initializing ConvNeXt-Large with random weights.")
        return _build_convnext_large_backbone(weights=None)

    if mode in {"imagenet", "imagenet1k", "torchvision", "default"}:
        print("Loading ConvNeXt-Large torchvision ImageNet-1K weights.")
        return _build_convnext_large_backbone(
            weights=torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        )

    net = _build_convnext_large_backbone(weights=None)

    if not os.path.isfile(pretrained_weights):
        raise FileNotFoundError(f"Local checkpoint not found: {pretrained_weights}")

    print(f"Loading LemonFM weights from local file: {os.path.abspath(pretrained_weights)}")
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    state_dict = state_dict["teacher"]
    state_dict = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }

    msg = net.load_state_dict(state_dict, strict=False)
    print(msg)

    first_key = next(iter(state_dict))
    assert torch.equal(
        net.state_dict()[first_key].cpu(),
        state_dict[first_key].cpu(),
    ), f"Local checkpoint not actually loaded for key: {first_key}"

    print(f"Verified local checkpoint loaded into model for key: {first_key}")
    return net


class ConvNeXtLemonFMBackbone(nn.Module):
    name = "convnext_lemonfm"

    def __init__(self, pretrained_weights="lemonfm.pth"):
        super().__init__()
        self.model = self._build_convnext_large_backbone(weights=None)
        self.output_dim = self.model.output_dim
        self._load_weights(pretrained_weights)

    @staticmethod
    def _build_convnext_large_backbone(weights=None):
        net = torchvision.models.convnext_large(weights=weights)
        in_dim = net.classifier[2].in_features
        net.classifier[2] = nn.Identity()
        net.output_dim = in_dim
        return net

    def _load_weights(self, pretrained_weights):
        if pretrained_weights is None:
            raise ValueError("pretrained_weights is None")

        mode = pretrained_weights.strip().lower() if isinstance(pretrained_weights, str) else pretrained_weights

        if mode in {"random", "none", "scratch"}:
            print("Initializing ConvNeXt-Large with random weights.")
            return

        if mode in {"imagenet", "imagenet1k", "torchvision", "default"}:
            print("Loading ConvNeXt-Large torchvision ImageNet-1K weights.")
            imagenet_model = self._build_convnext_large_backbone(
                weights=torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
            )
            self.model.load_state_dict(imagenet_model.state_dict())
            return

        if not os.path.isfile(pretrained_weights):
            raise FileNotFoundError(f"Local checkpoint not found: {pretrained_weights}")

        print(f"Loading LemonFM weights from local file: {os.path.abspath(pretrained_weights)}")
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        state_dict = state_dict["teacher"]
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }

        msg = self.model.load_state_dict(state_dict, strict=False)
        print(msg)

        first_key = next(iter(state_dict))
        assert torch.equal(
            self.model.state_dict()[first_key].cpu(),
            state_dict[first_key].cpu(),
        ), f"Local checkpoint not actually loaded for key: {first_key}"

        print(f"Verified local checkpoint loaded into model for key: {first_key}")

    def forward(self, x):
        return self.model(x)

    def unfreeze_last_stage(self):
        for p in self.model.features[7].parameters():
            p.requires_grad = True
        for p in self.model.classifier[0].parameters():
            p.requires_grad = True

    def set_last_stage_train(self):
        self.model.features[7].train()
        self.model.classifier[0].train()


class GSViTM5Backbone(nn.Module):
    name = "gsvit_m5"
    output_dim = 384

    def __init__(self, pretrained_weights):
        super().__init__()
        if not pretrained_weights or not os.path.isfile(pretrained_weights):
            raise FileNotFoundError(f"GSViT checkpoint not found: {pretrained_weights}")

        repo_root = Path(__file__).resolve().parent / "third_party" / "GSViT"
        sys.path.insert(0, str(repo_root))
        try:
            from EfficientViT.classification.model.build import EfficientViT_M5
        finally:
            if sys.path[0] == str(repo_root):
                sys.path.pop(0)

        base = EfficientViT_M5(pretrained=False)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        checkpoint = torch.load(pretrained_weights, map_location="cpu")
        self._load_official_gsvit_checkpoint(checkpoint, pretrained_weights)

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "net", "network"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
        return checkpoint

    @staticmethod
    def _normalize_official_encoder_key(key):
        while key.startswith("module."):
            key = key[len("module."):]
        for prefix in ("model.", "visual.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]

        # Official GSViT load_gsvit.py loads into a wrapper whose encoder is a
        # Sequential EfficientViT_M5 without the classification head.
        for prefix in ("evit.", "gsvit."):
            if key.startswith(prefix):
                key = key[len(prefix):]

        full_to_sequential = {
            "patch_embed": "0",
            "blocks1": "1",
            "blocks2": "2",
            "blocks3": "3",
        }
        first, sep, rest = key.partition(".")
        if sep and first in full_to_sequential:
            key = f"{full_to_sequential[first]}.{rest}"
        return key

    def _load_official_gsvit_checkpoint(self, checkpoint, pretrained_weights):
        state_dict = self._extract_state_dict(checkpoint)
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"GSViT checkpoint must contain a state_dict-like object: {pretrained_weights}"
            )

        target_state = self.encoder.state_dict()
        cleaned = {}
        ignored = []

        for raw_key, value in state_dict.items():
            key = self._normalize_official_encoder_key(str(raw_key))
            if key.startswith(("head.", "head_dist.", "decoder.")):
                ignored.append(raw_key)
                continue

            if key in target_state and tuple(value.shape) == tuple(target_state[key].shape):
                cleaned[key] = value
            else:
                ignored.append(raw_key)

        missing = sorted(set(target_state) - set(cleaned))
        if missing:
            raise RuntimeError(
                "GSViT checkpoint does not fully match the official encoder. "
                f"matched={len(cleaned)}/{len(target_state)} "
                f"missing_sample={missing[:20]} ignored_sample={ignored[:20]}"
            )

        self.encoder.load_state_dict(cleaned, strict=True)
        first_key = next(iter(cleaned))
        assert torch.equal(
            self.encoder.state_dict()[first_key].cpu(),
            cleaned[first_key].cpu(),
        ), f"GSViT checkpoint not actually loaded for key: {first_key}"
        print(
            "Loaded GSViT checkpoint with official encoder path: "
            f"{os.path.abspath(pretrained_weights)} | matched={len(cleaned)} "
            f"ignored={len(ignored)}"
        )

    @staticmethod
    def _flip_rgb_to_bgr(x):
        return x[:, [2, 1, 0], :, :]

    def forward(self, x):
        x = self._flip_rgb_to_bgr(x)
        x = self.encoder(x)
        if x.ndim == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x

    def _last_encoder_block(self):
        blocks3 = self.encoder[3]
        if isinstance(blocks3, nn.Sequential) and len(blocks3) > 0:
            return blocks3[-1]
        return blocks3

    def unfreeze_last_stage(self):
        for p in self._last_encoder_block().parameters():
            p.requires_grad = True

    def set_last_stage_train(self):
        self._last_encoder_block().train()


def build_visual_backbone(name="convnext_lemonfm", weights="lemonfm.pth"):
    name = str(name or "convnext_lemonfm").strip().lower()
    aliases = {
        "lemonfm": "convnext_lemonfm",
        "convnext": "convnext_lemonfm",
        "convnext_large": "convnext_lemonfm",
        "gsvit": "gsvit_m5",
    }
    name = aliases.get(name, name)

    if name == "convnext_lemonfm":
        return build_LemonFM(weights)
    if name == "gsvit_m5":
        return GSViTM5Backbone(weights)

    raise ValueError(
        f"Unknown vision backbone: {name}. "
        "Expected one of: convnext_lemonfm, gsvit_m5."
    )
