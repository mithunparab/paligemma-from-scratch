# paligemma-from-scratch

A minimal, from-scratch implementation of key components from the PaLiGemma project, including custom versions of the Gemma and SigLIP models. This repository provides scripts for training, inference, and data processing, making it easy to experiment with and extend these models.

## Datasets

- [Roboflow (pallet load manifest json)](https://universe.roboflow.com/roboflow-jvuqo/pallet-load-manifest-json)

- [Hugging Face (cord-v2)](https://huggingface.co/datasets/naver-clova-ix/cord-v2)

## Installation

1. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

2. **Set up environment variables:**

    ```sh
    export ROBOFLOW_API_KEY="your_real_api_key_here"
    export HUGGINGFACE_API_KEY="your_real_api_key_here"
    ```

    - **Login to Hugging Face:**

    ```sh
    huggingface-cli login --token $HUGGINGFACE_API_KEY --add-to-git-credential
    ```

### Make the all scripts executable:

```bash
chmod +x *.sh
```

3. **Train a model:**


- **Roboflow (pallet load manifest json) dataset is used for training.**
- Ensure you have a valid Roboflow API key set in your environment variables.

    ```bash
    CONFIG=$(jq -c -n \
    --arg key "$ROBOFLOW_API_KEY" \
    --arg workspace "roboflow-jvuqo" \
    --arg project "pallet-load-manifest-json" \
    --argjson version 1 \
    '{
        roboflow_api_key: $key,
        roboflow_workspace: $workspace,
        roboflow_project: $project,
        roboflow_version: $version
    }')

    ./train_launcher.sh roboflow_jsonl "$CONFIG" 2
    ```

- **Hugging Face (cord-v2) dataset is used for training.**
- Ensure you have a valid Hugging Face API key set in your environment variables.

    ```bash
    CONFIG=$(jq -c -n \
    --arg name "naver-clova-ix/cord-v2" \
    --arg train "train" \
    --arg val "validation" \
    '{
        dataset_name: $name,
        train_split: $train,
        val_split: $val
    }')

    ./train_launcher.sh huggingface_cord "$CONFIG" 2
    ```

- **Flikr8k dataset is used for training.**

    ```bash
    # Define the path to your Flickr8k dataset root
    FLICKR8K_ROOT="./flickr8k"

    # Construct the JSON configuration for Flickr8k
    # This configuration includes:
    # - image_directory_path: The absolute or relative path to the 'Images' folder.
    # - caption_files: A dictionary where keys are language codes (e.g., "en", "mr")
    #                  and values are the paths to the respective caption files.
    #                  Ensure these paths are correct relative to your script or absolute.
    # - split_ratio: Optional, used by Flickr8kDataset to internally split train/validation.
    #                Here, 0.1 means 10% of images/captions will be reserved for validation.
    # - random_seed: Optional, used by Flickr8kDataset for reproducible splitting.
    CONFIG=$(jq -c -n \
    --arg img_dir "$FLICKR8K_ROOT/Images" \
    --arg en_cap "$FLICKR8K_ROOT/captions.txt" \
    --arg mr_cap "$FLICKR8K_ROOT/captions_mr.txt" \
    --argjson split_ratio 0.1 \
    --argjson seed 42 \
    '{
        image_directory_path: $img_dir,
        caption_files: {
            en: $en_cap,
            mr: $mr_cap
        },
        split_ratio: $split_ratio,
        random_seed: $seed
    }')

    # Launch the training script for Flickr8k on 2 GPUs
    ./train_launcher.sh flickr8k "$CONFIG" 2
    ```

4. **Run inference:**

    ```sh
    bash launch_inference.sh
    ```

## Features

- **DDP Training**: Supports distributed data parallel training for efficient model training across multiple GPUs.
- **LoRA Support**: Implements LoRA (Low-Rank Adaptation) for efficient fine-tuning of large models.
- **Custom DataLoader**: Includes a custom DataLoader for efficient data loading and preprocessing.

### TODO

- [ ] Implement QLoRA (Quantized LoRA) for further model efficiency.
- [x] Add multilingual datasets for training and evaluation.

## References

[1] [PaLiGemma](https://huggingface.co/google/paligemma-3b-pt-224)

[2] [pytorch-paligemma (inference only)](https://github.com/hkproj/pytorch-paligemma)
