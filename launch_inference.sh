#!/bin/bash

# --- Configuration ---
MODEL_PATH="google/paligemma-3b-pt-224"
LORA_WEIGHTS_PATH="./paligemma_finetuned_hub/lora_weights_final.pt"
PROMPT="extract data in JSON format"
IMAGE_FILE_PATH="image_0.PNG"
SCRIPT="inference.py"

# --- Check if required files exist ---
if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: $SCRIPT not found in current directory."
    exit 1
fi

if [[ ! -f "$IMAGE_FILE_PATH" ]]; then
    echo "Error: Image file '$IMAGE_FILE_PATH' not found."
    exit 1
fi

if [[ ! -f "$LORA_WEIGHTS_PATH" ]]; then
    echo "Warning: LoRA weights file '$LORA_WEIGHTS_PATH' not found. Continuing..."
fi

# --- Run inference ---
python "$SCRIPT" \
    --model_path "$MODEL_PATH" \
    --lora_weights_path "$LORA_WEIGHTS_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH"
