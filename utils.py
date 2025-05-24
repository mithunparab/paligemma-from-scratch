from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig, LoraConfig, add_lora_adapters
from transformers import AutoTokenizer 
import json
import glob
from safetensors import safe_open
from typing import Tuple, Optional 
import os
import torch
from huggingface_hub import hf_hub_download, HfFileSystem

# TODO: Consider abstracting model loading (local vs. hub) into a strategy pattern if more backends are added.

def load_hf_model(
    model_id_or_path: str, 
    device: str, 
    model_dtype: Optional[torch.dtype] = None, 
    token: Optional[str]=None,
    gradient_checkpointing: bool = False,
    use_cache: bool = True,
) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    Loads a PaliGemma model and tokenizer from a local path or Hugging Face Hub, supporting both .safetensors and .bin formats.

    Parameters
    ----------
    model_id_or_path : str
        Local directory or Hugging Face Hub repo ID.
    device : str
        Device identifier for model placement (e.g., 'cpu', 'cuda').
    model_dtype : Optional[torch.dtype], default=None
        If specified, casts model to this dtype.
    token : Optional[str], default=None
        Hugging Face authentication token, if required.
    gradient_checkpointing : bool, default=False
        If True, enables gradient checkpointing in config.
    use_cache : bool, default=True
        If False, disables cache in config (useful for training with checkpointing).

    Returns
    -------
    Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]
        The loaded model and tokenizer.

    Notes
    -----
    - Handles missing pad_token in tokenizer.
    - Loads weights from both .safetensors and .bin files.
    - Raises FileNotFoundError or ValueError on missing assets.
    - Modifies config dict in-place to inject checkpointing/cache flags.
    """
    is_local = os.path.isdir(model_id_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, padding_side="right", token=token)
    if tokenizer.pad_token is None:
        # I’m setting pad_token to '<pad>' to avoid downstream errors in generation.
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0

    tensors = {}
    config_file_path = ""
    model_config_file = {}

    if is_local:
        config_file_path = os.path.join(model_id_or_path, "config.json")
        safetensors_files = glob.glob(os.path.join(model_id_or_path, "*.safetensors"))
        bin_files = glob.glob(os.path.join(model_id_or_path, "*.bin"))
        weight_files = safetensors_files + bin_files
        if not weight_files:
            raise FileNotFoundError(f"No model weights (.safetensors or .bin) found in local path {model_id_or_path}")
    else: 
        try:
            config_file_path = hf_hub_download(repo_id=model_id_or_path, filename="config.json", token=token)
        except Exception as e:
            # I’m surfacing config download errors explicitly for easier debugging.
            raise FileNotFoundError(f"Could not download config.json for {model_id_or_path} from Hub. Error: {e}")

        fs = HfFileSystem(token=token)
        try:
            repo_files_details = fs.ls(model_id_or_path, detail=True) 
        except Exception as e:
            # Handles rare case where repo listing fails (e.g., permissions, network).
            raise FileNotFoundError(f"Could not list files for repo {model_id_or_path} on Hub. Error: {e}")

        safetensors_filenames_on_hub = [
            os.path.relpath(f_info["name"], model_id_or_path).lstrip('/') 
            for f_info in repo_files_details if f_info["name"].endswith(".safetensors")
        ]
        bin_filenames_on_hub = [
            os.path.relpath(f_info["name"], model_id_or_path).lstrip('/')
            for f_info in repo_files_details if f_info["name"].endswith(".bin")
        ]
        
        weight_filenames_to_download = []
        if safetensors_filenames_on_hub:
            weight_filenames_to_download.extend(safetensors_filenames_on_hub)
        elif bin_filenames_on_hub:
            weight_filenames_to_download.extend(bin_filenames_on_hub)
        else:
            raise FileNotFoundError(f"No .safetensors or .bin weight files found in Hub repo {model_id_or_path}")

        weight_files = []
        for filename_in_repo in weight_filenames_to_download:
            try:
                downloaded_path = hf_hub_download(repo_id=model_id_or_path, filename=filename_in_repo, token=token)
                weight_files.append(downloaded_path)
            except Exception as e:
                # I’m logging but not failing on partial download errors; model load will fail if no weights are usable.
                print(f"Warning: Could not download {filename_in_repo} from {model_id_or_path}. Error: {e}")
        
        if not weight_files:
            raise FileNotFoundError(f"Failed to download any weight files for {model_id_or_path} from Hub.")

    for weight_file in weight_files:
        if weight_file.endswith(".safetensors"):
            # I’m using safe_open for .safetensors to avoid partial loads and ensure framework compatibility.
            with safe_open(weight_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
        elif weight_file.endswith(".bin"):
            state_dict = torch.load(weight_file, map_location="cpu")
            tensors.update(state_dict)

    if not tensors:
        raise ValueError(f"No weights loaded for {model_id_or_path}.")

    with open(config_file_path, "r") as f:
        model_config_file = json.load(f)
    
    # I’m injecting checkpointing/cache flags here to ensure downstream config objects are aware.
    model_config_file['gradient_checkpointing'] = gradient_checkpointing
    model_config_file['use_cache'] = use_cache

    config = PaliGemmaConfig(**model_config_file)
    model = PaliGemmaForConditionalGeneration(config)
    
    load_output = model.load_state_dict(tensors, strict=False)
    # Filtering missing keys to avoid false positives from tied weights.
    missing_keys_filtered = [
        k for k in load_output.missing_keys
        if not (k == "model.embed_tokens.weight" and hasattr(model.language_model, 'lm_head') and model.language_model.lm_head.weight is model.language_model.model.embed_tokens.weight)
    ]
    if missing_keys_filtered:
        print("Warning: Missing keys during state_dict load:", missing_keys_filtered)
    if load_output.unexpected_keys:
        print("Warning: Unexpected keys during state_dict load:", load_output.unexpected_keys)

    model.tie_weights()

    if model_dtype is not None:
        model = model.to(device=device, dtype=model_dtype)
    else:
        model = model.to(device=device)
        
    return (model, tokenizer)

def load_lora_weights_into_model(
    model: PaliGemmaForConditionalGeneration,
    lora_weights_path: str,
    lora_config: LoraConfig,
    device: str
) -> PaliGemmaForConditionalGeneration:
    """
    Loads LoRA adapter weights into a PaliGemma model, with diagnostics for missing/unexpected keys.

    Parameters
    ----------
    model : PaliGemmaForConditionalGeneration
        The base model to receive LoRA weights.
    lora_weights_path : str
        Path to the LoRA weights file (.bin or .pt).
    lora_config : LoraConfig
        Configuration for the LoRA adapters (not used directly here, but may be required for future extensions).
    device : str
        Device identifier for loading weights.

    Returns
    -------
    PaliGemmaForConditionalGeneration
        The model with LoRA weights loaded.

    Notes
    -----
    - Warns if any expected LoRA keys are missing or if unexpected non-LoRA keys are present.
    - Model is set to eval() after loading.
    """
    lora_state_dict = torch.load(lora_weights_path, map_location=device)
    load_status = model.load_state_dict(lora_state_dict, strict=False)
    
    # I’m surfacing missing LoRA keys for debugging adapter mismatches.
    missing_lora_keys = [k for k in load_status.missing_keys if "lora_" in k]
    if missing_lora_keys:
        print(f"Warning: Missing LoRA keys when loading from {lora_weights_path}: {missing_lora_keys}")
    
    # I’m flagging unexpected non-LoRA keys to catch accidental weight pollution.
    unexpected_non_lora_keys = [k for k in load_status.unexpected_keys if "lora_" not in k]
    if unexpected_non_lora_keys:
        print(f"Warning: Unexpected non-LoRA keys from {lora_weights_path}: {unexpected_non_lora_keys}")
    
    print(f"LoRA weights loaded from {lora_weights_path}")
    model.eval()
    return model