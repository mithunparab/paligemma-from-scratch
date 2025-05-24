"""
Architectural notes:
- The code is structured for both single-GPU and multi-GPU (DDP) training, with dataset download synchronization.
- Data loading and model preparation are modularized for clarity.
- TODO: Add support for other dataset formats (e.g., CSV, XML) if needed.
- TODO: Consider abstracting distributed setup/teardown and dataset download into context managers for robustness.
- TODO: The training/validation loop could be refactored for extensibility (callbacks, hooks, etc.).
- TODO: Current LoRA state_dict extraction assumes naming conventions; revisit if model structure changes.
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import fire
import time
import torch.multiprocessing as mp

import torch.distributed as dist
import torch.nn.parallel as ddp
from torch.utils.data.distributed import DistributedSampler

from roboflow import Roboflow
from modeling_gemma import (
    LoraConfig, add_lora_adapters, mark_only_lora_as_trainable, print_trainable_parameters
)
from processing_paligemma import PaliGemmaProcessor
from utils import load_hf_model
from typing import Optional, List, Dict, Any, Tuple

class JSONLDataset(Dataset):
    """
    Dataset for loading image-text pairs from a JSONL file and corresponding image directory.

    Parameters
    ----------
    jsonl_file_path : str
        Path to the JSONL file containing metadata.
    image_directory_path : str
        Directory containing image files referenced in the JSONL.
    max_samples : Optional[int]
        If set, limits the dataset to the first `max_samples` entries.

    Notes
    -----
    Skips malformed JSON lines and entries missing required keys.
    Raises FileNotFoundError if an image is missing.
    """
    def __init__(self, jsonl_file_path: str, image_directory_path: str, max_samples: Optional[int] = None):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()
        if max_samples is not None and max_samples < len(self.entries):
            self.entries = self.entries[:max_samples]

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    if 'image' in data and 'suffix' in data:
                        entries.append(data)
                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line: {line.strip()}")
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            raise
        return image, entry

class PaliGemmaCollator:
    """
    Collator for batching image-text data for PaliGemma.

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
    roboflow_api_key: str,
    roboflow_workspace: str,
    roboflow_project: str,
    roboflow_version: int,
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
    Main training loop for PaliGemma with LoRA adapters and Roboflow dataset.

    Parameters
    ----------
    model_path : str
        Path or HuggingFace identifier for the base model.
    roboflow_api_key : str
        API key for Roboflow dataset download.
    roboflow_workspace : str
        Roboflow workspace name.
    roboflow_project : str
        Roboflow project name.
    roboflow_version : int
        Roboflow dataset version.
    output_dir : str
        Directory to save LoRA weights.
    epochs : int
        Number of training epochs.
    batch_size : int
        Per-GPU batch size.
    learning_rate : float
        Learning rate for AdamW.
    lora_r, lora_alpha, lora_dropout, lora_target_modules
        LoRA adapter configuration.
    max_seq_length : int
        Max sequence length for tokenization.
    save_steps : int
        Save LoRA weights every `save_steps`.
    gradient_accumulation_steps : int
        Number of steps to accumulate gradients.
    use_bfloat16 : bool
        Use bfloat16 if available.
    max_train_samples, max_val_samples : Optional[int]
        Limit number of samples for train/val.
    num_dataloader_workers : int
        Number of DataLoader workers.
    hf_token : Optional[str]
        HuggingFace token for private models.
    use_gradient_checkpointing : bool
        Enable gradient checkpointing for memory efficiency.

    Notes
    -----
    Handles distributed setup/teardown and dataset download synchronization.
    Cleans up CUDA memory and DDP group at the end.
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

    # Only rank 0 downloads the dataset; others wait and then re-derive the path.
    dataset_location = None
    if rank == 0:
        print("Downloading dataset from Roboflow...")
        rf = Roboflow(api_key=roboflow_api_key)
        project = rf.workspace(roboflow_workspace).project(roboflow_project)
        version_obj = project.version(roboflow_version)
        dataset_download = version_obj.download("jsonl")
        dataset_location = dataset_download.location
        print(f"Dataset downloaded to: {dataset_location}")

    if is_distributed:
        dist.barrier()
        if rank != 0:
            rf_temp = Roboflow(api_key=roboflow_api_key)
            proj_temp = rf_temp.workspace(roboflow_workspace).project(roboflow_project)
            ver_temp = proj_temp.version(roboflow_version)
            dataset_location = ver_temp.download("jsonl").location

    train_jsonl_path = os.path.join(dataset_location, "train", "annotations.jsonl")
    train_img_dir_path = os.path.join(dataset_location, "train")
    val_jsonl_path = os.path.join(dataset_location, "valid", "annotations.jsonl")
    val_img_dir_path = os.path.join(dataset_location, "valid")

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

    # Access config via .module if DDP-wrapped
    if is_distributed:
        num_image_tokens = model.module.config.text_config.num_image_tokens
        image_size = model.module.config.vision_config.image_size
    else:
        num_image_tokens = model.config.text_config.num_image_tokens
        image_size = model.config.vision_config.image_size

    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
    if tokenizer.pad_token_id is not None and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = tokenizer.pad_token_id
        processor.tokenizer.pad_token = tokenizer.pad_token

    collator_instance = PaliGemmaCollator(
        processor=processor,
        max_seq_length=max_seq_length
    )

    if rank == 0:
        print("Loading datasets from downloaded Roboflow paths...")
    train_dataset_obj = JSONLDataset(train_jsonl_path, train_img_dir_path, max_samples=max_train_samples)
    val_dataset_obj = JSONLDataset(val_jsonl_path, val_img_dir_path, max_samples=max_val_samples)

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
        train_loader = DataLoader(train_dataset_obj, batch_size=batch_size, shuffle=True, collate_fn=collator_instance, num_workers=num_dataloader_workers)
        val_loader = DataLoader(val_dataset_obj, batch_size=batch_size, shuffle=False, collate_fn=collator_instance, num_workers=num_dataloader_workers)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print("Starting training...")

    global_step = 0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss_accum = 0.0
        optimizer.zero_grad()

        if is_distributed:
            train_sampler.set_epoch(epoch)  # Ensures shuffling is different each epoch

        for step, batch in enumerate(train_loader):
            # All tensors in 'batch' are on CPU; move to device as needed.
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(device, dtype=model_dtype, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True) if batch["labels"] is not None else None

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
                kv_cache=None
            )
            loss = outputs["loss"]

            if loss is None:
                if rank == 0:
                    print("Warning: Loss is None.")
                continue

            loss = loss / gradient_accumulation_steps
            train_loss_accum += loss.detach().item() * gradient_accumulation_steps

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and global_step > 0 and global_step % 10 == 0:
                    avg_loss = train_loss_accum / (10 * gradient_accumulation_steps)
                    print(f"Epoch {epoch+1}, Global Step {global_step}, Train Loss: {avg_loss:.4f}")
                    train_loss_accum = 0.0

                if rank == 0 and global_step > 0 and global_step % save_steps == 0:
                    print(f"Saving LoRA adapter weights at global step {global_step}...")
                    lora_weights_path = os.path.join(output_dir, f"lora_weights_step_{global_step}.pt")
                    # I’m keeping this block: DDP-wrapped model requires .module for state_dict access.
                    lora_state_dict = {k: v for k, v in model.module.state_dict().items() if "lora_" in k}
                    torch.save(lora_state_dict, lora_weights_path)
                    print(f"LoRA weights saved to {lora_weights_path}")

        epoch_duration = time.time() - epoch_start_time
        if rank == 0:
            print(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s.")

        # Validation loop: runs on all ranks, but only rank 0 aggregates and prints.
        model.eval()
        val_loss_accum = 0.0
        val_steps = 0
        if rank == 0:
            print("Running validation...")
        with torch.no_grad():
            for batch_val in val_loader:
                input_ids_val = batch_val["input_ids"].to(device, non_blocking=True)
                pixel_values_val = batch_val["pixel_values"].to(device, dtype=model_dtype, non_blocking=True)
                attention_mask_val = batch_val["attention_mask"].to(device, non_blocking=True)
                labels_val = batch_val["labels"].to(device, non_blocking=True) if batch_val["labels"] is not None else None

                outputs_val = model(
                    input_ids=input_ids_val,
                    pixel_values=pixel_values_val,
                    attention_mask=attention_mask_val,
                    labels=labels_val,
                    kv_cache=None
                )
                loss_val = outputs_val["loss"]
                if loss_val is not None:
                    val_loss_accum += loss_val.item()
                    val_steps += 1

        if is_distributed:
            # I’m keeping this: DDP requires explicit reduction for metrics.
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
                print("No validation steps executed for epoch or val_loader empty.")

    if rank == 0:
        print("Training finished. Saving final LoRA adapter weights...")
        final_lora_weights_path = os.path.join(output_dir, "lora_weights_final.pt")
        if is_distributed:
            lora_state_dict = {k: v for k, v in model.module.state_dict().items() if "lora_" in k}
        else:
            lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
        torch.save(lora_state_dict, final_lora_weights_path)
        print(f"Final LoRA weights saved to {final_lora_weights_path}")

    # I’m keeping this: Ensures all ranks finish before cleanup to avoid deadlocks.
    if is_distributed:
        dist.barrier()

    # Explicit cleanup to avoid CUDA OOM in multi-run scenarios.
    del model
    del optimizer
    if device != "cpu":
        torch.cuda.empty_cache()

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    try:
        # I’m keeping this: PyTorch requires 'spawn' for CUDA DataLoader workers.
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'. This is required for CUDA with num_dataloader_workers > 0.")
    except RuntimeError as e:
        current_method = mp.get_start_method(allow_none=True)
        if current_method != 'spawn':
            print(f"Warning: Could not set multiprocessing start method to 'spawn' (currently {current_method}): {e}. If using CUDA with num_dataloader_workers > 0, this may cause issues.")
        else:
            print(f"Multiprocessing start method already set to 'spawn'.")
        pass

    fire.Fire(main)