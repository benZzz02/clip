"""Evaluation entry point for VLPWithTimeSformer.

Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluate_timesformer_vitl.py \\
        --dataset cholec80_phase \\
        --ckpt outputs/xxx/vlp_epoch_50.pt
"""

import model as model_module
from timesformer_vitl_model import VLPWithTimeSformer

model_module.VLP = VLPWithTimeSformer

from zeroshot_evaluate import evaluate_zero_shot, parse_args

if __name__ == "__main__":
    args = parse_args()
    evaluate_zero_shot(args)
