"""Evaluation entry point for VLPWithTimeSformer / EndoSSL ViT-L.

Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluate_timesformer_vitl.py \
        --dataset cholec80_phase \
        --ckpt outputs/xxx/vlp_epoch_50.pt
"""

import os
os.environ.setdefault("DATA_NORMALIZATION", "none")
os.environ.setdefault("TIMESFORMER_SIZE", "vitb")

import model as model_module
from timesformer_vitl_model import VLPWithTimeSformer

model_module.VLP = VLPWithTimeSformer

# Override eval transform to remove Normalize for EndoSSL ViT
import zeroshot_evaluate
from torchvision import transforms

_original_build = zeroshot_evaluate.build_dataloader

def _endossl_build_dataloader(dataset_name, batch_size, num_workers, num_frames=1, frame_stride=1):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    cfg = zeroshot_evaluate.DATASET_CONFIGS[dataset_name]
    from torch.utils.data import DataLoader
    from zeroshot_evaluate import SurgLaViClipDataset
    if num_frames > 1:
        dataset = SurgLaViClipDataset(
            ann_file=cfg["ann_file"],
            transform=transform,
            num_frames=num_frames,
            frame_stride=frame_stride,
        )
    else:
        dataset = cfg["dataset_class"](
            ann_file=cfg["ann_file"],
            transform=transform,
        )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader, cfg

zeroshot_evaluate.build_dataloader = _endossl_build_dataloader

from zeroshot_evaluate import evaluate_zero_shot, parse_args

if __name__ == "__main__":
    args = parse_args()
    evaluate_zero_shot(args)
