"""Training entry point for VLPWithTimeSformer (EndoSSL native pipeline).

Usage:
    torchrun --standalone --nproc_per_node=2 train_timesformer_vitl.py <args>

This wraps train_frozen_vis.py but:
1. Replaces VLP with VLPWithTimeSformer (TimeSformer ViT-L backbone)
2. Sets DATA_NORMALIZATION=none so data stays in [0,1] range
   (matching EndoSSL's native pipeline: raw pixel input, no ImageNet normalization)
"""

import os
os.environ.setdefault("DATA_NORMALIZATION", "none")
os.environ.setdefault("TIMESFORMER_SIZE", "vitb")

import model as model_module
from timesformer_vitl_model import VLPWithTimeSformer

model_module.VLP = VLPWithTimeSformer

from train_frozen_vis import train

if __name__ == "__main__":
    train()
