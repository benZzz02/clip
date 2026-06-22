"""Training entry point for per-frame LemonFM with uniform pooling.

This keeps frames independent in the visual encoder:
1. Each frame is encoded by the same 2D LemonFM/ConvNeXt encoder.
2. The base video feature is the uniform mean of frame features.
3. Text-conditioned frame selection still uses the existing similarity/xpool
   logic in model.VLP.
"""

import os

import torch
import torch.nn as nn

os.environ.setdefault("DATA_NORMALIZATION", "imagenet")
os.environ.setdefault("VISION_BACKBONE", "convnext_lemonfm")
os.environ.setdefault("VISION_PRETRAINED_WEIGHTS", "lemonfm.pth")

import model as model_module
from model import VLP


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


class VLPWithLemonFMUniformPool(VLP):
    """VLP with independent per-frame LemonFM and uniform global pooling."""

    def __init__(self, **kwargs):
        actual_num_frames = max(1, int(kwargs.get("num_frames", 1)))
        parent_kwargs = dict(kwargs)
        parent_kwargs["num_frames"] = 1
        parent_kwargs["vision_backbone"] = "convnext_lemonfm"
        if not str(parent_kwargs.get("vision_pretrained_weights", "")).strip():
            parent_kwargs["vision_pretrained_weights"] = "lemonfm.pth"

        super().__init__(**parent_kwargs)

        self.num_frames = actual_num_frames
        self.frame_pool = UniformFramePool()
        self.temporal_hidden_dim = self.visual_dim

        print(
            "[VLPWithLemonFMUniformPool] "
            "backbone=convnext_lemonfm, frame_pool=uniform, frame_interaction=disabled"
        )


model_module.VLP = VLPWithLemonFMUniformPool

from train_frozen_vis import train


if __name__ == "__main__":
    train()
