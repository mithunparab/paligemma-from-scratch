#!/bin/bash
set -e

# Usage: ./train_launcher.sh <dataset_type> <config_json_string> <nproc>
DATASET_TYPE=$1
CONFIG_JSON=$2
NPROC=${3:-1}  # Default to single GPU

if [ -z "$DATASET_TYPE" ] || [ -z "$CONFIG_JSON" ]; then
  echo "Usage: $0 <dataset_type> <config_json> [nproc]"
  exit 1
fi

# Optional: model params as env vars or defaults
MODEL_PATH=${MODEL_PATH:-"google/paligemma-3b-pt-224"}
OUTPUT_DIR=${OUTPUT_DIR:-"./finetuned_${DATASET_TYPE}"}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-512}
USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-True}
USE_BFLOAT16=${USE_BFLOAT16:-True}

CMD="python train.py"
if [ "$NPROC" -gt 1 ]; then
  CMD="torchrun --nproc_per_node=$NPROC train.py"
fi

$CMD \
  --model_path "$MODEL_PATH" \
  --dataset_type "$DATASET_TYPE" \
  --dataset_config "$CONFIG_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --use_gradient_checkpointing "$USE_GRADIENT_CHECKPOINTING" \
  --use_bfloat16 "$USE_BFLOAT16"
