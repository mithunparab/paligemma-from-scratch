"""
Architectural notes:
- This code supports both single-GPU and multi-GPU (DDP) training.
- Dataset loading is modularized via `dataset.py`.
- TODO: Consider a dataset factory or registry pattern for extensibility across formats.
- TODO: Abstract distributed setup/teardown into context managers for robustness.
- TODO: Refactor training/validation loop for extensibility (callbacks, hooks, etc.).
- TODO: LoRA state_dict extraction assumes naming conventions; revisit if model structure changes.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import json
import os
import fire
import time
import torch.multiprocessing as mp
import random # For set_seed
import numpy as np # For set_seed

import torch.distributed as dist
import torch.nn.parallel as ddp
from torch.utils.data.distributed import DistributedSampler

from modeling_gemma import (
    LoraConfig, add_lora_adapters, mark_only_lora_as_trainable, print_trainable_parameters
)
from processing_paligemma import PaliGemmaProcessor
from utils import load_hf_model
from typing import Optional, List, Dict, Any, Tuple

from dataset import get_dataset_metadata, create_pytorch_dataset


def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across different libraries.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior for CUDA (might impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed} for reproducibility.")


class PaliGemmaCollator:
    """
    Batches and tokenizes image-text data for PaliGemma.

    Parameters
    ----------
    processor : PaliGemmaProcessor
        Preprocessing pipeline for images and text.
    max_seq_length : int
        Maximum sequence length for tokenization.

    Returns
    -------
    Dict[str, torch.Tensor]
        Batched and tokenized inputs for the model, on CPU.
    """
    def __init__(self, processor: PaliGemmaProcessor, max_seq_length: int):
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[Tuple[Image.Image, Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        """
        Collates a batch of (image, label_json) pairs into model-ready tensors.

        Parameters
        ----------
        batch : List[Tuple[Image.Image, Dict[str, Any]]]
            List of image and label dict pairs. The dict may contain 'suffix' and 'language'.

        Returns
        -------
        Dict[str, torch.Tensor]
            Tokenized and batched input for the model.
        """
        # Filter out any placeholder entries (e.g., from failed image loads)
        # Note: This means DataLoader's batch_size might not always reflect effective batch size.
        # For simplicity, we just filter and let PyTorch handle smaller batches or raise error if empty.
        filtered_batch = [item for item in batch if item[0] is not None and "suffix" in item[1] and "error" not in item[1]["suffix"]]
        
        if not filtered_batch:
            # Return empty tensors that can be skipped in the training loop
            return {
                "input_ids": torch.empty(0, dtype=torch.long),
                "attention_mask": torch.empty(0, dtype=torch.long),
                "pixel_values": torch.empty(0, dtype=torch.float),
                "labels": torch.empty(0, dtype=torch.long)
            }

        images, labels_meta = zip(*filtered_batch)
        images = list(images)
        
        # Prompts can be fixed or derived from language/task
        # For this setup, a fixed prompt "extract data in JSON format" is used.
        # If language-specific prompts are needed, modify this.
        prompts = ["extract data in JSON format" for _ in labels_meta]
        
        target_suffixes = []
        for entry in labels_meta:
            suffix_content = entry["suffix"]
            # Suffix could be a string or a dict (if original dataset had it as JSON)
            if isinstance(suffix_content, dict):
                target_suffixes.append(json.dumps(suffix_content))
            else:
                target_suffixes.append(str(suffix_content))
        
        inputs = self.processor(
            text=prompts,
            images=images,
            suffix=target_suffixes,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        return inputs

def main(
    model_path: str,
    dataset_type: str, 
    dataset_config: dict,
    output_dir: str = "paligemma_finetuned_lora",
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    max_seq_length: int = 512,
    save_steps: int = 1000,
    gradient_accumulation_steps: int = 4,
    use_bfloat16: bool = True,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    num_dataloader_workers: int = 2,
    hf_token: Optional[str] = None,
    use_gradient_checkpointing: bool = False,
    seed: int = 42, # Added seed argument
):
    """
    Main training loop for PaliGemma with LoRA adapters.

    Parameters
    ----------
    model_path : str
        Path to the base model.
    dataset_type : str
        Identifier for dataset type. Supported: "roboflow_jsonl", "huggingface_cord", "flickr8k".
    dataset_config : dict
        Configuration dictionary for the dataset.
        - For "roboflow_jsonl": Requires "roboflow_api_key", "roboflow_workspace", "roboflow_project", "roboflow_version".
        - For "huggingface_cord": Requires "dataset_name", "train_split", "val_split".
        - For "flickr8k": Requires "image_directory_path", "caption_files" (dict mapping lang to path).
                          Optional: "split_ratio" (for internal train/val split), "random_seed" (for splitting).
    output_dir : str, optional
        Directory to save outputs.
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Batch size per device.
    learning_rate : float, optional
        Learning rate for optimizer.
    lora_r : int, optional
        LoRA rank.
    lora_alpha : int, optional
        LoRA alpha scaling.
    lora_dropout : float, optional
        LoRA dropout probability.
    lora_target_modules : List[str], optional
        Target modules for LoRA adaptation.
    max_seq_length : int, optional
        Maximum sequence length for tokenization.
    save_steps : int, optional
        Steps between saving LoRA weights.
    gradient_accumulation_steps : int, optional
        Number of steps to accumulate gradients.
    use_bfloat16 : bool, optional
        Whether to use bfloat16 precision.
    max_train_samples : Optional[int], optional
        Maximum number of training samples.
    max_val_samples : Optional[int], optional
        Maximum number of validation samples.
    num_dataloader_workers : int, optional
        Number of DataLoader worker processes.
    hf_token : Optional[str], optional
        HuggingFace token for model loading.
    use_gradient_checkpointing : bool, optional
        Whether to use gradient checkpointing.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    None
    """
    set_seed(seed) # Set the global seed first

    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.set_device(0)

    if rank == 0:
        print(f"[{rank}/{world_size}] Using device: {device}")
        print(f"[{rank}/{world_size}] Is distributed: {is_distributed}")

    if use_bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        if rank == 0:
            print("bfloat16 is not supported on this GPU. Using float32.")
        use_bfloat16 = False

    model_dtype = torch.bfloat16 if use_bfloat16 and device == "cuda" else torch.float32
    if rank == 0:
        print(f"[{rank}/{world_size}] Using dtype: {model_dtype}")

    if rank == 0:
        print(f"Preparing dataset of type '{dataset_type}'...")

    train_data_path, train_image_dir, val_data_path, val_image_dir, dataset_format_id, dataset_specific_kwargs = get_dataset_metadata(
        dataset_type=dataset_type,
        dataset_config=dataset_config,
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        print(f"Dataset metadata acquired:")
        print(f"  Train data path: {train_data_path}")
        print(f"  Train image dir: {train_image_dir if train_image_dir else 'N/A'}")
        print(f"  Val data path  : {val_data_path}")
        print(f"  Val image dir  : {val_image_dir if val_image_dir else 'N/A'}")
        print(f"  Dataset format : {dataset_format_id}")
        print(f"  Dataset specific kwargs: {dataset_specific_kwargs}")


    if rank == 0:
        print(f"Loading base model from {model_path}...")

    model, tokenizer = load_hf_model(
        model_path, device="cpu", model_dtype=model_dtype, token=hf_token,
        gradient_checkpointing=use_gradient_checkpointing,
        use_cache=not use_gradient_checkpointing # Cache usually disabled with GC during training
    )

    if tokenizer.pad_token_id is None:
        if rank == 0:
            print("Tokenizer pad_token_id not set. Setting to 0 (<pad> for Gemma).")
        tokenizer.pad_token = "<pad>"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            if tokenizer.pad_token_id is None: # Fallback if convert_tokens_to_ids returns None
                tokenizer.pad_token_id = 0

    lora_conf = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules
    )

    if rank == 0:
        print("Adding LoRA adapters...")
    add_lora_adapters(model.language_model, lora_conf)
    mark_only_lora_as_trainable(model, bias='lora_only')

    model = model.to(device=device, dtype=model_dtype)

    if is_distributed:
        model = ddp.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print_trainable_parameters(model)

    current_model_config = model.module.config if is_distributed else model.config
    num_image_tokens = current_model_config.text_config.num_image_tokens
    image_size = current_model_config.vision_config.image_size

    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
    if tokenizer.pad_token_id is not None and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = tokenizer.pad_token_id
        processor.tokenizer.pad_token = tokenizer.pad_token

    collator_instance = PaliGemmaCollator(
        processor=processor,
        max_seq_length=max_seq_length
    )

    if rank == 0:
        print("Creating PyTorch datasets...")

    train_split_name = dataset_config.get("train_split", "train") # Default for HF datasets, used by Flickr8k too
    val_split_name = dataset_config.get("val_split", "validation") # Default for HF datasets, used by Flickr8k too
    
    # Create training dataset
    train_dataset_obj = create_pytorch_dataset(
        dataset_format_identifier=dataset_format_id,
        data_path=train_data_path,
        image_dir_path=train_image_dir,
        max_samples=max_train_samples,
        split=train_split_name, # Pass split name to dataset
        **dataset_specific_kwargs # Pass Flickr8k specific kwargs like caption_files_dict, split_ratio, random_seed
    )
    # Create validation dataset
    val_dataset_obj = create_pytorch_dataset(
        dataset_format_identifier=dataset_format_id,
        data_path=val_data_path,
        image_dir_path=val_image_dir,
        max_samples=max_val_samples,
        split=val_split_name, # Pass split name to dataset
        # For Flickr8k, ensure `is_validation_split=True` is passed when creating the validation split
        is_validation_split=True if dataset_format_id == "flickr8k" else False, # Explicitly mark val split for Flickr8k
        **dataset_specific_kwargs # Pass Flickr8k specific kwargs
    )

    if is_distributed:
        # Samplers ensure each GPU gets a unique subset of data
        train_sampler = DistributedSampler(train_dataset_obj, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset_obj, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
        
        train_loader = DataLoader(
            train_dataset_obj,
            batch_size=batch_size,
            sampler=train_sampler, # Use sampler for DDP
            collate_fn=collator_instance,
            num_workers=num_dataloader_workers,
            pin_memory=True,
            drop_last=True # Required when using DistributedSampler without handling remainder
        )
        val_loader = DataLoader(
            val_dataset_obj,
            batch_size=batch_size,
            sampler=val_sampler, # Use sampler for DDP
            collate_fn=collator_instance,
            num_workers=num_dataloader_workers,
            pin_memory=True,
            drop_last=True # Required when using DistributedSampler without handling remainder
        )
    else:
        train_loader = DataLoader(train_dataset_obj, batch_size=batch_size, shuffle=True, collate_fn=collator_instance, num_workers=num_dataloader_workers, drop_last=True)
        val_loader = DataLoader(val_dataset_obj, batch_size=batch_size, shuffle=False, collate_fn=collator_instance, num_workers=num_dataloader_workers, drop_last=True)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print("Starting training...")

    global_step = 0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss_accum = 0.0

        if is_distributed:
            train_sampler.set_epoch(epoch) # Crucial for proper shuffling in DDP

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # Skip batches that resulted in empty tensors due to loading errors
            if batch["input_ids"].numel() == 0:
                if rank == 0:
                    print(f"Warning: Skipping empty batch at epoch {epoch+1}, step {step}.")
                continue

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            pixel_values_batch = batch["pixel_values"].to(device, dtype=model_dtype, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True) if batch["labels"] is not None else None

            model_to_call = model.module if is_distributed else model
            
            outputs = model_to_call(
                input_ids=input_ids,
                pixel_values=pixel_values_batch,
                attention_mask=attention_mask,
                labels=labels,
                kv_cache=None 
            )
            loss = outputs["loss"]

            if loss is None:
                if rank == 0:
                    print(f"Warning: Loss is None at epoch {epoch+1}, step {step}. Skipping batch.")
                continue
            
            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    print(f"Warning: NaN or Inf loss detected at epoch {epoch+1}, step {step}. Loss: {loss.item()}. Skipping batch update.")
                optimizer.zero_grad()
                continue

            loss = loss / gradient_accumulation_steps
            current_batch_loss = loss.detach().item() * gradient_accumulation_steps
            train_loss_accum += current_batch_loss

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and global_step > 0 and global_step % 10 == 0:
                    avg_loss_interval = train_loss_accum / (10 * gradient_accumulation_steps)
                    print(f"Epoch {epoch+1}, Global Step {global_step}, Train Loss: {avg_loss_interval:.4f}")
                    train_loss_accum = 0.0

                if rank == 0 and global_step > 0 and global_step % save_steps == 0:
                    print(f"Saving LoRA adapter weights at global step {global_step}...")
                    lora_weights_path = os.path.join(output_dir, f"lora_weights_step_{global_step}.pt")
                    model_state_dict_to_save = model.module.state_dict() if is_distributed else model.state_dict()
                    lora_state_dict = {k: v for k, v in model_state_dict_to_save.items() if "lora_" in k}
                    torch.save(lora_state_dict, lora_weights_path)
                    print(f"LoRA weights saved to {lora_weights_path}")
        
        # Apply any remaining accumulated gradients at epoch end
        if (step + 1) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        epoch_duration = time.time() - epoch_start_time
        if rank == 0:
            print(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s.")

        model.eval()
        val_loss_accum = 0.0
        val_steps = 0
        if rank == 0:
            print("Running validation...")
        with torch.no_grad():
            for batch_val in val_loader:
                # Skip batches that resulted in empty tensors due to loading errors
                if batch_val["input_ids"].numel() == 0:
                    if rank == 0:
                        print(f"Warning: Skipping empty validation batch at epoch {epoch+1}.")
                    continue

                input_ids_val = batch_val["input_ids"].to(device, non_blocking=True)
                pixel_values_val_batch = batch_val["pixel_values"].to(device, dtype=model_dtype, non_blocking=True)
                attention_mask_val = batch_val["attention_mask"].to(device, non_blocking=True)
                labels_val = batch_val["labels"].to(device, non_blocking=True) if batch_val["labels"] is not None else None
                
                model_to_call_eval = model.module if is_distributed else model
                outputs_val = model_to_call_eval(
                    input_ids=input_ids_val,
                    pixel_values=pixel_values_val_batch,
                    attention_mask=attention_mask_val,
                    labels=labels_val,
                    kv_cache=None
                )
                loss_val = outputs_val["loss"]
                if loss_val is not None:
                    if not (torch.isnan(loss_val) or torch.isinf(loss_val)):
                        val_loss_accum += loss_val.item()
                        val_steps += 1
                    elif rank == 0:
                        print(f"Warning: NaN/Inf validation loss encountered. Value: {loss_val.item()}")

        if is_distributed:
            val_loss_tensor = torch.tensor(val_loss_accum, device=device)
            val_steps_tensor = torch.tensor(val_steps, device=device)
            dist.reduce(val_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(val_steps_tensor, dst=0, op=dist.ReduceOp.SUM)
            dist.barrier() # Ensure all processes have completed the reduction
            val_loss_accum = val_loss_tensor.item()
            val_steps = val_steps_tensor.item()

        if rank == 0:
            if val_steps > 0:
                avg_val_loss = val_loss_accum / val_steps
                print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}: No valid validation steps executed or val_loader empty. Unable to calculate validation loss.")

    if rank == 0:
        print("Training finished. Saving final LoRA adapter weights...")
        final_lora_weights_path = os.path.join(output_dir, "lora_weights_final.pt")
        model_state_dict_to_save = model.module.state_dict() if is_distributed else model.state_dict()
        lora_state_dict = {k: v for k, v in model_state_dict_to_save.items() if "lora_" in k}
        torch.save(lora_state_dict, final_lora_weights_path)
        print(f"Final LoRA weights saved to {final_lora_weights_path}")

    # Cleanup distributed environment and release resources
    if is_distributed:
        dist.barrier() # Ensure all processes are done before cleanup

    del model, optimizer # Explicitly delete large objects
    del train_dataset_obj, val_dataset_obj, train_loader, val_loader, collator_instance, processor # Delete datasets and loaders
    if device != "cpu":
        torch.cuda.empty_cache() # Clear CUDA cache

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    try:
        # Ensure 'spawn' is set for multiprocessing. This is required for CUDA DataLoader workers.
        # It must be called before any CUDA operations or other multiprocessing starts.
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
            print("Multiprocessing start method set to 'spawn'.")
        else:
            print("Multiprocessing start method already 'spawn' or previously set.")
    except RuntimeError as e:
        current_method = mp.get_start_method(allow_none=True)
        print(f"Warning: Could not set multiprocessing start method to 'spawn' (currently {current_method}): {e}. "
              "If using CUDA with num_dataloader_workers > 0, this might lead to issues if not already 'spawn'.")
    except Exception as e: 
        print(f"An unexpected error occurred while setting multiprocessing start method: {e}")

    fire.Fire(main)