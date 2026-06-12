import os
import sys
from pathlib import Path

import timm
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

    def unfreeze_stages(self, stage_indices):
        num_features = len(self.model.features)
        for idx in stage_indices:
            if 0 <= idx < num_features:
                for p in self.model.features[idx].parameters():
                    p.requires_grad = True
        if num_features - 1 in stage_indices and len(self.model.classifier) > 0:
            for p in self.model.classifier[0].parameters():
                p.requires_grad = True

    def set_stages_train(self, stage_indices):
        num_features = len(self.model.features)
        for idx in stage_indices:
            if 0 <= idx < num_features:
                self.model.features[idx].train()
        if num_features - 1 in stage_indices and len(self.model.classifier) > 0:
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

        model = EfficientViT_M5(pretrained="efficientvit_m5")
        checkpoint = torch.load(pretrained_weights, map_location="cpu")
        self._load_gsvit_checkpoint(
            model=model,
            checkpoint=checkpoint,
            pretrained_weights=pretrained_weights,
        )
        self._drop_classification_linear(model)
        self.model = model

    @staticmethod
    def _drop_classification_linear(model):
        head = getattr(model, "head", None)
        if not hasattr(head, "l"):
            raise RuntimeError("Expected EfficientViT head to expose final classifier as head.l")
        out_dim = head.l.in_features
        head.l = nn.Identity()
        model.output_dim = out_dim

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "net", "network"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
        return checkpoint

    @staticmethod
    def _normalize_gsvit_key(key):
        while key.startswith("module."):
            key = key[len("module."):]
        for prefix in ("model.", "visual.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]

        if key.startswith("gsvit."):
            key = key[len("gsvit."):]
        if key.startswith("encoder."):
            key = key[len("encoder."):]
        if key.startswith("evit."):
            key = key[len("evit."):]

        sequential_to_full = {
            "0": "patch_embed",
            "1": "blocks1",
            "2": "blocks2",
            "3": "blocks3",
        }
        first, sep, rest = key.partition(".")
        if sep and first in sequential_to_full:
            key = f"{sequential_to_full[first]}.{rest}"
        return key

    def _load_gsvit_checkpoint(self, model, checkpoint, pretrained_weights):
        state_dict = self._extract_state_dict(checkpoint)
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"GSViT checkpoint must contain a state_dict-like object: {pretrained_weights}"
            )

        target_state = model.state_dict()
        encoder_keys = {
            key
            for key in target_state
            if key.startswith(("patch_embed.", "blocks1.", "blocks2.", "blocks3."))
        }
        cleaned = {}
        ignored = []
        loaded_head_bn = 0

        for raw_key, value in state_dict.items():
            key = self._normalize_gsvit_key(str(raw_key))
            if key.startswith(("head.l.", "head_dist.", "decoder.")):
                ignored.append(raw_key)
                continue

            if key in target_state and tuple(value.shape) == tuple(target_state[key].shape):
                cleaned[key] = value
                if key.startswith("head.bn."):
                    loaded_head_bn += 1
            else:
                ignored.append(raw_key)

        missing = sorted(encoder_keys - set(cleaned))
        if missing:
            raise RuntimeError(
                "GSViT checkpoint does not fully match the EfficientViT-M5 encoder. "
                f"matched_encoder_keys={len(encoder_keys) - len(missing)}/{len(encoder_keys)} "
                f"missing_sample={missing[:20]} ignored_sample={ignored[:20]}"
            )

        model.load_state_dict(cleaned, strict=False)
        first_key = next(key for key in cleaned if key in encoder_keys)
        assert torch.equal(
            model.state_dict()[first_key].cpu(),
            cleaned[first_key].cpu(),
        ), f"GSViT checkpoint not actually loaded for key: {first_key}"
        print(
            "Loaded GSViT checkpoint into EfficientViT-M5 encoder: "
            f"{os.path.abspath(pretrained_weights)} | "
            f"encoder_matched={len(encoder_keys)} "
            f"head_bn_matched={loaded_head_bn} "
            f"ignored={len(ignored)}"
        )

    @staticmethod
    def _flip_rgb_to_bgr(x):
        return x[:, [2, 1, 0], :, :]

    def forward(self, x):
        # PretrainDataset applies RGB ImageNet normalization, but GSViT was
        # autoencoder-fine-tuned on BGR frames in [0,1] range without normalization.
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = x * std + mean
        x = self._flip_rgb_to_bgr(x)
        return self.model(x)

    def unfreeze_last_stage(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def set_last_stage_train(self):
        self.model.train()


class EndoSSLViTBackbone(nn.Module):
    """ViT-L/16 backbone pretrained with MSN on private laparoscopic videos.

    EndoSSL (MICCAI 2023) uses a standard ViT-L/16 architecture with Masked
    Siamese Networks pretraining on 23.3M frames of private clinical data.
    This class loads the converted PyTorch weights and follows the same
    interface as other visual backbones in this module.

    Weights must first be converted from the TF SavedModel format:
        python convert_endossl_to_torch.py \\
            --tf_model_dir <google_drive_download>/saved_model_inference \\
            --output checkpoints/endossl_vitl.pth
    """

    name = "endossl_vitl"
    output_dim = 1024  # ViT-L hidden dimension

    def __init__(self, pretrained_weights):
        super().__init__()
        if not pretrained_weights or not os.path.isfile(pretrained_weights):
            raise FileNotFoundError(
                f"EndoSSL ViT-L checkpoint not found: {pretrained_weights}\n"
                "Download the ViT-L laparoscopy weights from Google Drive, then run:\n"
                "  python convert_endossl_to_torch.py --tf_model_dir <dir> --output <out>.pth"
            )

        self.model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="token",
        )

        state_dict = torch.load(pretrained_weights, map_location="cpu")
        msg = self.model.load_state_dict(state_dict, strict=True)
        print(f"EndoSSL ViT-L loaded: {msg}")

    def forward(self, x):
        # PretrainDataset applies ImageNet normalization, but EndoSSL
        # Flax pretraining expects [0,1] RGB input (same as TF pipeline).
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = x * std + mean
        return self.model(x)

    def unfreeze_last_stage(self):
        for p in self.model.blocks[-2:].parameters():
            p.requires_grad = True

    def set_last_stage_train(self):
        for blk in self.model.blocks[-2:]:
            blk.train()

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


def build_visual_backbone(name="convnext_lemonfm", weights="lemonfm.pth"):
    name = str(name or "convnext_lemonfm").strip().lower()
    aliases = {
        "lemonfm": "convnext_lemonfm",
        "convnext": "convnext_lemonfm",
        "convnext_large": "convnext_lemonfm",
        "gsvit": "gsvit_m5",
        "endossl": "endossl_vitl",
        "endossl_vitl": "endossl_vitl",
        "vitl": "endossl_vitl",
    }
    name = aliases.get(name, name)

    if name == "convnext_lemonfm":
        return build_LemonFM(weights)
    if name == "gsvit_m5":
        return GSViTM5Backbone(weights)
    if name == "endossl_vitl":
        return EndoSSLViTBackbone(weights)

    raise ValueError(
        f"Unknown vision backbone: {name}. "
        "Expected one of: convnext_lemonfm, gsvit_m5, endossl_vitl."
    )
