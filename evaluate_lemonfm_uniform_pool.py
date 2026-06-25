"""Evaluation entry point for VLPWithLemonFMUniformPool checkpoints."""

import os

os.environ.setdefault("DATA_NORMALIZATION", "imagenet")
os.environ.setdefault("VISION_BACKBONE", "convnext_lemonfm")
os.environ.setdefault("VISION_PRETRAINED_WEIGHTS", "lemonfm.pth")

# Importing this module patches model.VLP to VLPWithLemonFMUniformPool before
# zeroshot_evaluate imports `from model import VLP`.
import train_lemonfm_uniform_pool  # noqa: F401

from zeroshot_evaluate import evaluate_zero_shot, parse_args


if __name__ == "__main__":
    args = parse_args()
    evaluate_zero_shot(args)
