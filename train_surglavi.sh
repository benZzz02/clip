#!/usr/bin/env bash

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/nfs_data/CLIP}"

cd "$PROJECT_ROOT"
mkdir -p "${SAVE_DIR:-${DATA_ROOT}/surglavi_checkpoint/${EXP_NAME:-surglavi_lora_single128}}" "${TB_LOGDIR:-runs/${EXP_NAME:-surglavi_lora_single128}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}" \
TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}" \
SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-${EXP_NAME:-surglavi_lora_single128}}" \
TB_LOGDIR="${TB_LOGDIR:-runs/${EXP_NAME:-surglavi_lora_single128}}" \
PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT}" \
torchrun --standalone --nproc_per_node="${NPROC:-1}" train_surglavi_ddp.py \
  --epochs "${EPOCHS:-50}" \
  --learning_rate "${LEARNING_RATE:-5e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.02}" \
  --adam_beta1 "${ADAM_BETA1:-0.9}" \
  --adam_beta2 "${ADAM_BETA2:-0.999}" \
  --per_gpu_batch_size "${PER_GPU_BATCH_SIZE:-128}" \
  --accum_steps "${ACCUM_STEPS:-1}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --image_size "${IMAGE_SIZE:-224}" \
  --max_length "${MAX_LENGTH:-256}" \
  --num_frames "${NUM_FRAMES:-8}" \
  --tokenizer_name "${TOKENIZER_NAME:-bert-base-uncased}" \
  --surgclip_model_name "${SURGCLIP_MODEL_NAME:-SurgCLIP-B}" \
  --video_root_folder "${VIDEO_ROOT_FOLDER:-${DATA_ROOT}/downloaded_video_224_test}" \
  --assume_resized_video "${ASSUME_RESIZED_VIDEO:-1}" \
  --main_csv_path "${MAIN_CSV_PATH:-${DATA_ROOT}/surglavi_level_csv/all_video.csv}" \
  --annotations_root "${ANNOTATIONS_ROOT:-${DATA_ROOT}/surglavi_level_csv}" \
  --annotation_levels "${ANNOTATION_LEVELS:-coarse,mid,fine}" \
  --level_mix "${LEVEL_MIX:-concat}" \
  --sample_mode "${SAMPLE_MODE:-center}" \
  --save_dir "${SAVE_DIR:-${DATA_ROOT}/surglavi_checkpoint/${EXP_NAME:-surglavi_lora_single128}}" \
  --save_every "${SAVE_EVERY:-5}" \
  --save_name "${SAVE_NAME:-final.pt}" \
  --seed "${SEED:-42}" \
  --finetune_mode "${FINETUNE_MODE:-lora}" \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-1}" \
  --lora_targets "${LORA_TARGETS:-text_encoder.encoder.layer.,vision_encoder.model.blocks.}" \
  ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"}
