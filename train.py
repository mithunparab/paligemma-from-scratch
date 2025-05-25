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
            List of image and label dict pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Tokenized and batched input for the model.
        """
        images, labels_json = zip(*batch)
        images = list(images)
        prompts = ["extract data in JSON format" for _ in labels_json]
        target_suffixes = []
        for entry in labels_json:
            suffix_content = entry["suffix"]
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
):
    """
    Main training loop for PaliGemma with LoRA adapters.

    Parameters
    ----------
    model_path : str
        Path to the base model.
    dataset_type : str
        Identifier for dataset type.
    dataset_config : dict
        Dataset configuration parameters.
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

    Returns
    -------
    None

    Notes
    -----
    - Handles both single and distributed (DDP) training.
    - Only LoRA parameters are trainable.
    - Saves LoRA weights periodically and at the end.
    - Handles edge cases for pad_token_id and dtype support.
    """
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

    train_data_path, train_image_dir, val_data_path, val_image_dir, dataset_format_id = get_dataset_metadata(
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

    if rank == 0:
        print(f"Loading base model from {model_path}...")

    model, tokenizer = load_hf_model(
        model_path, device="cpu", model_dtype=model_dtype, token=hf_token,
        gradient_checkpointing=use_gradient_checkpointing,
        use_cache=not use_gradient_checkpointing
    )

    if tokenizer.pad_token_id is None:
        if rank == 0:
            print("Tokenizer pad_token_id not set. Setting to 0 (<pad> for Gemma).")
        tokenizer.pad_token = "<pad>"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            if tokenizer.pad_token_id is None:
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
        # I’m keeping find_unused_parameters=False here; with current LoRA setup and no gradient checkpointing, all params should be used. Set to True if DDP errors arise.
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

    train_split_name = dataset_config.get("train_split", "train")
    val_split_name = dataset_config.get("val_split", "validation")
    
    dataset_creation_kwargs = {}
    if "sort_json_key" in dataset_config:
        dataset_creation_kwargs["sort_json_key"] = dataset_config["sort_json_key"]

    train_dataset_obj = create_pytorch_dataset(
        dataset_format_identifier=dataset_format_id,
        data_path=train_data_path,
        image_dir_path=train_image_dir,
        max_samples=max_train_samples,
        split=train_split_name if dataset_format_id == "hf_cord" else None,
        **dataset_creation_kwargs
    )
    val_dataset_obj = create_pytorch_dataset(
        dataset_format_identifier=dataset_format_id,
        data_path=val_data_path,
        image_dir_path=val_image_dir,
        max_samples=max_val_samples,
        split=val_split_name if dataset_format_id == "hf_cord" else None,
        **dataset_creation_kwargs
    )

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset_obj, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset_obj, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
        train_loader = DataLoader(
            train_dataset_obj,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collator_instance,
            num_workers=num_dataloader_workers,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset_obj,
            batch_size=batch_size,
            sampler=val_sampler,
            collate_fn=collator_instance,
            num_workers=num_dataloader_workers,
            pin_memory=True,
            drop_last=True
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
            train_sampler.set_epoch(epoch)

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            pixel_values_batch = batch["pixel_values"].to(device, dtype=model_dtype, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True) if batch["labels"] is not None else None

            # I’m keeping this explicit model_to_call logic to ensure DDP-wrapped models are handled correctly.
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

            # DDP: loss.backward() synchronizes gradients across processes when find_unused_parameters=False.
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
        
        # I’m keeping this block to ensure any remaining accumulated gradients are applied at epoch end.
        if (step + 1) % gradient_accumulation_steps != 0:
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
            # I’m keeping this reduction/barrier to ensure validation loss is aggregated across all processes.
            val_loss_tensor = torch.tensor(val_loss_accum, device=device)
            val_steps_tensor = torch.tensor(val_steps, device=device)
            dist.reduce(val_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(val_steps_tensor, dst=0, op=dist.ReduceOp.SUM)
            dist.barrier()
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

    if is_distributed:
        dist.barrier()

    del model, optimizer
    del train_dataset_obj, val_dataset_obj, train_loader, val_loader, collator_instance, processor
    if device != "cpu":
        torch.cuda.empty_cache()

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    try:
        # I’m keeping this block to ensure 'spawn' is set for multiprocessing, which is required for CUDA DataLoader workers.
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