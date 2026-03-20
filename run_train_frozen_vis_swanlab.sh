#!/usr/bin/env bash
set -euo pipefail

# 激活 conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export PER_GPU_BATCH_SIZE=64
export PRETRAIN_VIDEO_ALREADY_RESIZED=1
export ACCUM_STEPS=1
export NUM_WORKERS=10
export NUM_FRAMES=16
export SWANLAB_EXPERIMENT_NAME=16frame
torchrun --standalone --nproc_per_node=2 train_frozen_vis.py