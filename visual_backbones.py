import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
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

        model = EfficientViT_M5(pretrained=False)
        model.head = nn.Identity()
        if hasattr(model, "head_dist"):
            model.head_dist = nn.Identity()
        model.distillation = False

        checkpoint = torch.load(pretrained_weights, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]

        cleaned = {}
        for key, value in checkpoint.items():
            while key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("gsvit."):
                key = key[len("gsvit."):]
            if key.startswith("evit."):
                key = key[len("evit."):]
            cleaned[key] = value

        msg = model.load_state_dict(cleaned, strict=False)
        print(f"Loaded GSViT visual backbone: {msg}")
        self.model = model

    @staticmethod
    def _flip_rgb_to_bgr(x):
        return x[:, [2, 1, 0], :, :]

    def forward(self, x):
        x = self._flip_rgb_to_bgr(x)
        return self.model(x)


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
