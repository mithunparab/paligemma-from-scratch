import torch
import fire
from typing import Optional, List, Dict
from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration, LoraConfig, add_lora_adapters
from utils import load_hf_model, load_lora_weights_into_model
from PIL import Image

# TODO: Consider abstracting device/dtype logic into a utility module if reused elsewhere.

def move_inputs_to_device(model_inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """
    Move all tensors in model_inputs to the specified device, handling dtype for pixel_values.

    Parameters
    ----------
    model_inputs : dict[str, torch.Tensor]
        Dictionary of model input tensors.
    device : str
        Target device ('cpu', 'cuda', 'mps').

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with tensors moved to the target device.
    """
    # I’m keeping the explicit dtype handling for pixel_values, as some models are sensitive to float precision here.
    processed_inputs = {}
    for k, v in model_inputs.items():
        if k == "pixel_values":
            processed_inputs[k] = v.to(device, dtype=torch.get_default_dtype())
        else:
            processed_inputs[k] = v.to(device)
    return processed_inputs


def get_model_inputs_for_inference(
    processor: PaliGemmaProcessor,
    prompt: str,
    image_file_path: str,
    device: str,
    model_dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    """
    Prepare and move model inputs for inference, including text and image.

    Parameters
    ----------
    processor : PaliGemmaProcessor
        Preprocessing pipeline for text and images.
    prompt : str
        Input text prompt.
    image_file_path : str
        Path to the input image file.
    device : str
        Target device.
    model_dtype : torch.dtype
        Desired dtype for pixel_values.

    Returns
    -------
    dict[str, torch.Tensor]
        Model-ready input tensors on the correct device and dtype.
    """
    image = Image.open(image_file_path).convert("RGB")
    model_inputs = processor(text=[prompt], images=[image], padding=False, return_tensors="pt")
    # I’m retaining explicit dtype/device moves here, as processor output may not match model expectations.
    final_model_inputs = {
        "input_ids": model_inputs["input_ids"].to(device),
        "attention_mask": model_inputs["attention_mask"].to(device),
        "pixel_values": model_inputs["pixel_values"].to(device, dtype=model_dtype)
    }
    return final_model_inputs


def generate_text(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    model_dtype: torch.dtype,
) -> None:
    """
    Autoregressively generate text conditioned on a prompt and image.

    Parameters
    ----------
    model : PaliGemmaForConditionalGeneration
        The multimodal generative model.
    processor : PaliGemmaProcessor
        Preprocessing pipeline.
    device : str
        Target device.
    prompt : str
        Input text prompt.
    image_file_path : str
        Path to the input image.
    max_tokens_to_generate : int
        Maximum number of tokens to generate.
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus sampling probability.
    do_sample : bool
        Whether to sample or use greedy decoding.
    model_dtype : torch.dtype
        Model computation dtype.

    Returns
    -------
    None
    """
    model_inputs = get_model_inputs_for_inference(processor, prompt, image_file_path, device, model_dtype)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()
    generated_token_ids = []

    stop_token_id = processor.tokenizer.eos_token_id
    current_input_ids = input_ids

    # I’m keeping the pixel_values=None logic after the first step, as some models will re-encode the image if not gated.
    for i in range(max_tokens_to_generate):
        current_attention_mask_for_model = torch.ones_like(current_input_ids, device=device)
        outputs = model(
            input_ids=current_input_ids,
            pixel_values=pixel_values if i == 0 else None,
            attention_mask=current_attention_mask_for_model,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]

        if do_sample:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = _sample_top_p(probs, top_p)
        else:
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        next_token_id_item = next_token_id.item()
        generated_token_ids.append(next_token_id_item)

        if next_token_id_item == stop_token_id:
            break

        current_input_ids = next_token_id

    decoded_text = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {decoded_text}")


def _sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Parameters
    ----------
    probs : torch.Tensor
        Probability tensor of shape (batch, vocab_size).
    p : float
        Cumulative probability threshold.

    Returns
    -------
    torch.Tensor
        Sampled token indices of shape (batch, 1).
    """
    # I’m keeping this custom implementation for transparency, though torch.multinomial could suffice for simple top-k.
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def main(
    model_path: str,
    prompt: str,
    image_file_path: str,
    lora_weights_path: Optional[str] = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_target_modules: List[str] = ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    max_tokens_to_generate: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    use_bfloat16: bool = True,
    only_cpu: bool = False,
) -> None:
    """
    Entrypoint for running multimodal inference with optional LoRA adaptation.

    Parameters
    ----------
    model_path : str
        Path to the base HuggingFace model.
    prompt : str
        Text prompt.
    image_file_path : str
        Path to the input image.
    lora_weights_path : Optional[str], optional
        Path to LoRA weights, by default None.
    lora_r : int, optional
        LoRA rank, by default 8.
    lora_alpha : int, optional
        LoRA alpha, by default 16.
    lora_target_modules : list[str], optional
        Target modules for LoRA, by default common projection layers.
    max_tokens_to_generate : int, optional
        Maximum tokens to generate, by default 100.
    temperature : float, optional
        Sampling temperature, by default 0.7.
    top_p : float, optional
        Nucleus sampling probability, by default 0.9.
    do_sample : bool, optional
        Whether to sample or use greedy decoding, by default True.
    use_bfloat16 : bool, optional
        Whether to use bfloat16 if available, by default True.
    only_cpu : bool, optional
        Force CPU execution, by default False.

    Returns
    -------
    None
    """
    # I’m keeping the device selection logic explicit, as MPS/CPU fallback is non-trivial in some environments.
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
    print(f"Device in use: {device}")

    model_dtype = torch.bfloat16 if use_bfloat16 and device != "cpu" and (device != "cuda" or torch.cuda.is_bf16_supported()) else torch.float32
    if use_bfloat16 and model_dtype == torch.float32:
        print("bfloat16 specified but not available or on CPU. Using float32.")
    print(f"Using dtype: {model_dtype}")

    print(f"Loading base model from {model_path}...")
    model, tokenizer = load_hf_model(model_path, device, model_dtype=model_dtype)

    if lora_weights_path:
        print(f"Applying LoRA weights from {lora_weights_path}...")
        lora_conf = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules)
        add_lora_adapters(model.language_model, lora_conf)
        # I’m keeping this explicit device/dtype move after LoRA injection, as some frameworks misplace LoRA params otherwise.
        model = model.to(device=device, dtype=model_dtype)
        model = load_lora_weights_into_model(model, lora_weights_path, lora_conf, device)

    model.eval()

    num_image_tokens = model.config.text_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
    # I’m keeping this pad_token sync, as tokenizer/processor mismatches can cause silent failures in generation.
    if tokenizer.pad_token_id is not None and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = tokenizer.pad_token_id
        processor.tokenizer.pad_token = tokenizer.pad_token

    print("Running generation...")
    with torch.no_grad():
        generate_text(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
            model_dtype=model_dtype,
        )

if __name__ == "__main__":
    fire.Fire(main)