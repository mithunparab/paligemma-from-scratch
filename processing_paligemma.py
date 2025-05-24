"""
Image and text preprocessing utilities for PaliGemma-style multimodal models.
Focuses on deterministic, explicit handling of image normalization, resizing, and prompt construction.
Tokenization and batching logic is designed for both training (with suffix/labels) and inference.

TODO: 
- Consider extracting image processing into a strategy pattern if supporting multiple vision backbones.
- The tokenizer augmentation logic is idempotent but could be factored into a utility for reuse.
- If batch sizes or sequence lengths grow, revisit padding/truncation logic for efficiency.
"""

from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(
    prefix_prompt: str, 
    bos_token: str, 
    image_seq_len: int, 
    image_token: str
) -> str:
    """
    Construct a prompt string with image tokens and BOS token prepended.

    Args:
        prefix_prompt (str): The text prompt to be prefixed.
        bos_token (str): Beginning-of-sequence token.
        image_seq_len (int): Number of image tokens to prepend.
        image_token (str): The image token string.

    Returns:
        str: The constructed prompt string.
    """
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: np.ndarray, 
    scale: float, 
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Rescale image pixel values by a given factor.

    Args:
        image (np.ndarray): Input image array.
        scale (float): Scaling factor.
        dtype (np.dtype, optional): Output data type.

    Returns:
        np.ndarray: Rescaled image.
    """
    return (image * scale).astype(dtype)


def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Optional[Image.Resampling] = None,
    reducing_gap: Optional[int] = None,
) -> Image.Image:
    """
    Resize a PIL image to the specified size.

    Args:
        image (Image.Image): Input PIL image.
        size (Tuple[int, int]): (height, width) target size.
        resample (Optional[Image.Resampling]): Resampling filter.
        reducing_gap (Optional[int]): Passed to PIL for antialiasing.

    Returns:
        Image.Image: Resized PIL image.
    """
    height, width = size
    return image.resize((width, height), resample=resample, reducing_gap=reducing_gap)


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """
    Normalize image by subtracting mean and dividing by std.

    Args:
        image (np.ndarray): Input image array.
        mean (float or Iterable[float]): Mean for normalization.
        std (float or Iterable[float]): Std for normalization.

    Returns:
        np.ndarray: Normalized image.
    """
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    return (image - mean) / std


def process_images_for_paligemma(
    images: List[Image.Image],
    size: Tuple[int, int],
    resample: Optional[Image.Resampling] = None,
    rescale_factor: Optional[float] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    Preprocess a batch of images for PaliGemma: resize, convert to array, rescale, normalize, and transpose.

    Args:
        images (List[Image.Image]): List of PIL images.
        size (Tuple[int, int]): Target (height, width).
        resample (Optional[Image.Resampling]): Resampling filter.
        rescale_factor (Optional[float]): Factor to rescale pixel values.
        image_mean (Optional[float or List[float]]): Mean for normalization.
        image_std (Optional[float or List[float]]): Std for normalization.

    Returns:
        List[np.ndarray]: List of processed images as CHW float32 arrays.
    """
    height, width = size
    processed_images = [
        resize(image=img, size=(height, width), resample=resample) for img in images
    ]
    processed_images = [np.array(img) for img in processed_images]
    processed_images = [rescale(img, scale=rescale_factor) for img in processed_images]
    processed_images = [normalize(img, mean=image_mean, std=image_std) for img in processed_images]
    processed_images = [img.transpose(2, 0, 1) for img in processed_images]  # HWC to CHW
    return processed_images


class PaliGemmaProcessor:
    """
    Preprocessing pipeline for PaliGemma multimodal models.

    Handles tokenizer augmentation, image preprocessing, and batch collation for both training and inference.
    Designed for explicit control over BOS/EOS handling and label masking.

    Args:
        tokenizer: Tokenizer object supporting HuggingFace-style API.
        num_image_tokens (int): Number of image tokens to prepend.
        image_size (int): Target image size (assumes square images).

    Attributes:
        IMAGE_TOKEN (str): Special token for images.
        IGNORE_INDEX (int): Label mask value for non-target tokens.
    """

    IMAGE_TOKEN = "<image>"
    IGNORE_INDEX = -100

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        """
        Initialize processor and augment tokenizer with required special tokens.

        Notes:
            - Tokenizer is mutated in-place to add special tokens if missing.
            - Ensures pad_token_id is set for downstream batching.
        """
        super().__init__()
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # I’m keeping this block: ensures all required tokens are present for multimodal and downstream tasks.
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]
        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # I’m keeping this: ensures tokenizer is robust to missing special tokens (common in custom checkpoints).
        if tokenizer.bos_token is None:
            tokenizer.bos_token = "<s>"
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "</s>"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<pad>"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        # I’m keeping this: disables automatic BOS/EOS to allow explicit control in prompt construction.
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: Union[str, List[str]],
        images: Union[Image.Image, List[Image.Image]],
        suffix: Optional[Union[str, List[str]]] = None,
        padding: Union[str, bool] = "longest",
        truncation: Union[str, bool] = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess and collate a batch of text-image pairs for model input.

        Args:
            text (str or List[str]): Input prompts.
            images (Image.Image or List[Image.Image]): Corresponding images.
            suffix (Optional[str or List[str]]): Target text for training (labels).
            padding (str or bool): Padding strategy.
            truncation (str or bool): Truncation strategy.
            max_length (Optional[int]): Max sequence length.
            return_tensors (str): Output tensor type ("pt" supported).

        Returns:
            Dict[str, torch.Tensor]: Batch dict with keys: pixel_values, input_ids, attention_mask, (labels if suffix).
        
        Raises:
            ValueError: If images are missing or batch sizes mismatch.
            NotImplementedError: If unsupported tensor type requested.

        Notes:
            - I’m keeping explicit checks for batch size mismatches; these are common sources of silent bugs.
            - Padding/truncation logic is explicit to avoid accidental sequence length drift.
        """
        if images is None or (isinstance(images, list) and not images):
            raise ValueError("Images must be provided.")
        if isinstance(text, str):
            text = [text]
        if isinstance(images, Image.Image):
            images = [images]
        if suffix is not None and isinstance(suffix, str):
            suffix = [suffix]

        if len(text) != len(images):
            raise ValueError(f"Number of text prompts ({len(text)}) and images ({len(images)}) must match.")
        if suffix is not None and len(text) != len(suffix):
            raise ValueError(f"Number of text prompts ({len(text)}) and suffixes ({len(suffix)}) must match.")

        pixel_values_list = process_images_for_paligemma(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        input_ids_batch = []
        labels_batch = [] if suffix is not None else None

        for i in range(len(text)):
            current_text = text[i]
            current_suffix = suffix[i] if suffix is not None else None

            prompt_str_for_tokenizer = add_image_tokens_to_prompt(
                prefix_prompt=current_text,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            prompt_token_ids = self.tokenizer(prompt_str_for_tokenizer, add_special_tokens=False)["input_ids"]

            if current_suffix is not None:
                suffix_token_ids = self.tokenizer(current_suffix, add_special_tokens=False)["input_ids"]
                # I’m keeping this: EOS is appended to suffix to mark generation boundary for training.
                suffix_token_ids_with_eos = suffix_token_ids + [self.tokenizer.eos_token_id]
                current_input_ids = prompt_token_ids + suffix_token_ids_with_eos
                current_labels = [self.IGNORE_INDEX] * len(prompt_token_ids) + suffix_token_ids_with_eos
            else:
                current_input_ids = prompt_token_ids

            input_ids_batch.append(current_input_ids)
            if labels_batch is not None:
                labels_batch.append(current_labels)

        # I’m keeping this: max_length logic is explicit to avoid silent truncation or over-padding.
        if padding and truncation and max_length is None:
            effective_max_length = (
                self.tokenizer.model_max_length
                if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length
                else None
            )
            max_len_in_batch = max(len(ids) for ids in input_ids_batch)
            actual_max_len = min(max_len_in_batch, effective_max_length) if effective_max_length else max_len_in_batch
        elif max_length is not None:
            actual_max_len = max_length
        elif padding:
            actual_max_len = max(len(ids) for ids in input_ids_batch)
        else:
            if any(len(ids) != len(input_ids_batch[0]) for ids in input_ids_batch):
                if len(input_ids_batch) > 1:
                    raise ValueError("Sequences must have same length for no padding with batch_size > 1")
            actual_max_len = len(input_ids_batch[0]) if input_ids_batch else 0

        final_input_ids = []
        final_attention_mask = []
        final_labels = [] if labels_batch is not None else None

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("pad_token_id must be set in tokenizer for padding.")

        for i in range(len(input_ids_batch)):
            ids = input_ids_batch[i]
            if truncation and len(ids) > actual_max_len:
                ids = ids[:actual_max_len]
            current_length = len(ids)
            padding_length = actual_max_len - current_length if padding else 0
            attention_mask = [1] * current_length + [0] * padding_length
            ids_padded = ids + [pad_token_id] * padding_length
            final_input_ids.append(ids_padded)
            final_attention_mask.append(attention_mask)

            if final_labels is not None:
                lbls = labels_batch[i]
                if truncation and len(lbls) > actual_max_len:
                    lbls = lbls[:actual_max_len]
                # I’m keeping this: IGNORE_INDEX padding ensures loss is masked for non-target tokens.
                lbls_padded = lbls + [self.IGNORE_INDEX] * (actual_max_len - len(lbls))
                final_labels.append(lbls_padded)

        batch_outputs = {}
        if return_tensors == "pt":
            batch_outputs["pixel_values"] = torch.tensor(np.array(pixel_values_list), dtype=torch.float32)
            batch_outputs["input_ids"] = torch.tensor(final_input_ids, dtype=torch.long)
            batch_outputs["attention_mask"] = torch.tensor(final_attention_mask, dtype=torch.long)
            if final_labels is not None:
                batch_outputs["labels"] = torch.tensor(final_labels, dtype=torch.long)
        else:
            raise NotImplementedError(f"return_tensors='{return_tensors}' not implemented.")

        return batch_outputs