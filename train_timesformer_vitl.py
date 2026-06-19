"""Training entry point for VLPWithTimeSformer.

Usage:
    torchrun --standalone --nproc_per_node=2 train_timesformer_vitl.py <args>

This wraps train_frozen_vis.py but replaces VLP with VLPWithTimeSformer
so that the TimeSformer ViT-L backbone is used instead of EndoSSL ViT-L +
separate temporal pooling.
"""

import model as model_module
from timesformer_vitl_model import VLPWithTimeSformer

# Replace VLP in the model module so all downstream imports resolve to
# VLPWithTimeSformer without modifying any existing files.
model_module.VLP = VLPWithTimeSformer

from train_frozen_vis import train

if __name__ == "__main__":
    train()
