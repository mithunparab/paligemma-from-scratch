"""
Dataset utilities for loading and preparing datasets in various formats for PyTorch training.
Supports Roboflow JSONL, HuggingFace CORD, and Flickr8k datasets.

Classes:
    JSONLDataset: PyTorch Dataset for Roboflow JSONL-format datasets.
    HuggingFaceCORDDataset: PyTorch Dataset for HuggingFace CORD datasets.
    Flickr8kDataset: PyTorch Dataset for Flickr8k image-caption datasets.

Functions:
    _prepare_roboflow_dataset_paths: Download and synchronize Roboflow datasets across distributed ranks.
    get_dataset_metadata: Extracts dataset paths and configuration for supported dataset types.
    create_pytorch_dataset: Factory for instantiating a PyTorch Dataset based on format and configuration.
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import torch.distributed as dist
from roboflow import Roboflow
from datasets import load_dataset as hf_load_dataset
import random
import pandas as pd
from sklearn.model_selection import GroupKFold


class JSONLDataset(Dataset):
    """
    PyTorch Dataset for Roboflow JSONL-format datasets.

    Args:
        jsonl_file_path (str): Path to the JSONL annotation file.
        image_directory_path (str): Directory containing the images.
        max_samples (Optional[int]): Maximum number of samples to load.

    Raises:
        FileNotFoundError: If an image file is missing.
        RuntimeError: If an image cannot be opened or decoded.
        IndexError: If an index is out of range.
    """
    def __init__(
        self,
        jsonl_file_path: str,
        image_directory_path: str,
        max_samples: Optional[int] = None
    ) -> None:
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()
        if max_samples is not None and max_samples < len(self.entries):
            self.entries = self.entries[:max_samples]

    def _load_entries(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    if 'image' in data and 'suffix' in data:
                        entries.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in {self.jsonl_file_path}: {line.strip()}")
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image, {"suffix": entry["suffix"]}
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found for JSONLDataset: {image_path}")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Error opening/decoding image {image_path}: {e}")


class HuggingFaceCORDDataset(Dataset):
    """
    PyTorch Dataset for HuggingFace datasets (e.g., 'naver-clova-ix/cord-v2').

    Args:
        dataset_name_or_path (str): HuggingFace dataset name or local path.
        split (str): Dataset split to load (e.g., "train", "validation").
        max_samples (Optional[int]): Maximum number of samples to load.

    Raises:
        IndexError: If an index is out of range.
    """
    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset_name_or_path = dataset_name_or_path
        self.split = split

        print(f"Initializing HuggingFaceCORDDataset for: {dataset_name_or_path}, split: {self.split}")
        raw_hf_dataset = hf_load_dataset(dataset_name_or_path, split=self.split, streaming=False)

        if max_samples is not None and 0 < max_samples < len(raw_hf_dataset):
            print(f"Selecting first {max_samples} samples for {split} split.")
            self.hf_dataset = raw_hf_dataset.select(range(max_samples))
        elif max_samples is not None and max_samples <= 0:
            print(f"Warning: max_samples is {max_samples}, which is invalid. Using full dataset for {split} split.")
            self.hf_dataset = raw_hf_dataset
        else:
            self.hf_dataset = raw_hf_dataset

        self.dataset_length = len(self.hf_dataset)
        if self.dataset_length == 0:
            print(f"Warning: Dataset {dataset_name_or_path} split {split} is empty after applying max_samples (or originally).")
        print(f"Initialized HuggingFaceCORDDataset. Final length for {split} split: {self.dataset_length}")

    def _get_processed_suffix(self, ground_truth_json_str: str) -> str:
        """
        Processes the ground truth JSON string to extract a suffix.
        Returns a JSON string of the selected ground truth parse.
        """
        try:
            ground_truth_dict = json.loads(ground_truth_json_str)
            gt_jsons_to_choose_from = []
            if "gt_parses" in ground_truth_dict:
                if isinstance(ground_truth_dict["gt_parses"], list) and ground_truth_dict["gt_parses"]:
                    gt_jsons_to_choose_from = ground_truth_dict["gt_parses"]
            elif "gt_parse" in ground_truth_dict:
                if isinstance(ground_truth_dict["gt_parse"], dict):
                    gt_jsons_to_choose_from = [ground_truth_dict["gt_parse"]]
            if not gt_jsons_to_choose_from:
                chosen_gt_for_suffix = {}
            else:
                chosen_gt_for_suffix = random.choice(gt_jsons_to_choose_from)
            return json.dumps(chosen_gt_for_suffix)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode ground_truth JSON: {ground_truth_json_str[:100]}... Using empty JSON.")
            return json.dumps({})
        except Exception as e:
            print(f"Warning: Error processing ground_truth: {e}. Using empty JSON. GT: {ground_truth_json_str[:100]}...")
            return json.dumps({})

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= self.dataset_length:
            raise IndexError(f"Index {idx} out of range for HuggingFaceCORDDataset with length {self.dataset_length}")
        try:
            sample = self.hf_dataset[idx]
            image = sample["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            target_suffix_str = self._get_processed_suffix(sample["ground_truth"])
            return image, {"suffix": target_suffix_str}
        except Exception as e:
            print(f"Error loading or processing sample at index {idx} for split {self.split}: {e}. Returning placeholder.")
            dummy_image = Image.new('RGB', (224, 224), color='red')
            dummy_suffix = json.dumps({"error": f"failed_to_load_sample_at_idx_{idx}"})
            return dummy_image, {"suffix": dummy_suffix}


class Flickr8kDataset(Dataset):
    """
    PyTorch Dataset for Flickr8k image-caption datasets.

    Args:
        image_directory_path (str): Directory containing images.
        caption_files_dict (Dict[str, str]): Mapping from language to caption CSV file path.
        max_samples (Optional[int]): Maximum number of samples to load.
        split_ratio (Optional[float]): Ratio for validation split (between 0.0 and 1.0).
        random_seed (Optional[int]): Seed for reproducible splitting.
        is_validation_split (bool): Whether to use the validation split.

    Raises:
        ValueError: If no valid caption files are loaded or split_ratio is invalid.
        IndexError: If an index is out of range.
    """
    def __init__(
        self,
        image_directory_path: str,
        caption_files_dict: Dict[str, str],
        max_samples: Optional[int] = None,
        split_ratio: Optional[float] = None,
        random_seed: Optional[int] = None,
        is_validation_split: bool = False,
    ) -> None:
        super().__init__()
        self.image_directory_path = image_directory_path
        self.caption_files_dict = caption_files_dict
        self.random_seed = random_seed
        self.is_validation_split = is_validation_split

        all_caption_dfs: List[pd.DataFrame] = []
        for lang, caption_file_path in self.caption_files_dict.items():
            if not os.path.isfile(caption_file_path):
                print(f"Warning: Caption file not found for language '{lang}': {caption_file_path}. Skipping.")
                continue
            try:
                df_lang = pd.read_csv(caption_file_path, sep=',')
                if 'image' not in df_lang.columns or 'caption' not in df_lang.columns:
                    print(f"Warning: Missing 'image' or 'caption' column in {caption_file_path} (using sep=','). Columns found: {df_lang.columns.tolist()}. Skipping.")
                    continue
                df_lang["image_path"] = df_lang["image"].apply(lambda x: os.path.join(self.image_directory_path, x))
                df_lang["language"] = lang
                all_caption_dfs.append(df_lang)
            except Exception as e:
                print(f"Error loading {caption_file_path}: {e}. Skipping.")
                continue

        if not all_caption_dfs:
            raise ValueError(f"No valid caption files loaded from {caption_files_dict}. Please check paths and formats (expected comma-separated with 'image' and 'caption' columns).")

        self.full_df = pd.concat(all_caption_dfs, ignore_index=True)

        if max_samples is not None and max_samples > 0 and max_samples < len(self.full_df):
            self.full_df = self.full_df.head(max_samples)
            print(f"Limited dataset to {len(self.full_df)} samples due to max_samples={max_samples}")

        if split_ratio is not None:
            if not (0.0 < split_ratio < 1.0):
                raise ValueError("split_ratio must be between 0.0 and 1.0 (exclusive).")
            # TODO: Consider using numpy random state for reproducibility with GroupKFold if needed.
            gkf = GroupKFold(n_splits=int(round(1 / split_ratio)))
            unique_images = self.full_df["image"].unique()
            folds = list(gkf.split(unique_images, groups=unique_images))
            val_fold_image_indices = folds[0][1]
            train_fold_image_indices = folds[0][0]
            val_images = unique_images[val_fold_image_indices]
            train_images = unique_images[train_fold_image_indices]

            if self.is_validation_split:
                self.data_df = self.full_df[self.full_df['image'].isin(val_images)].reset_index(drop=True)
                print(f"Flickr8kDataset: Using validation split. Length: {len(self.data_df)}")
            else:
                self.data_df = self.full_df[self.full_df['image'].isin(train_images)].reset_index(drop=True)
                print(f"Flickr8kDataset: Using training split. Length: {len(self.data_df)}")
        else:
            self.data_df = self.full_df
            print(f"Flickr8kDataset: Using full dataset (no split_ratio provided). Length: {len(self.data_df)}")

        self.dataset_length = len(self.data_df)
        if self.dataset_length == 0:
            print(f"Warning: Flickr8kDataset is empty after loading and splitting. Check paths and split parameters.")

        print(f"Initialized Flickr8kDataset. Final length: {self.dataset_length}")

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.data_df):
            raise IndexError("Index out of range for Flickr8kDataset")

        entry = self.data_df.iloc[idx]
        image_path = entry["image_path"]
        caption = entry["caption"]
        language = entry["language"]

        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image, {"suffix": caption, "language": language}
        except FileNotFoundError:
            print(f"Warning: Image not found for Flickr8kDataset: {image_path}. Returning placeholder.")
            dummy_image = Image.new('RGB', (224, 224), color='red')
            dummy_suffix = json.dumps({"error": f"image_not_found_{os.path.basename(image_path)}"})
            return dummy_image, {"suffix": dummy_suffix, "language": language}
        except (UnidentifiedImageError, OSError) as e:
            print(f"Warning: Error opening/decoding image {image_path}: {e}. Returning placeholder.")
            dummy_image = Image.new('RGB', (224, 224), color='red')
            dummy_suffix = json.dumps({"error": f"image_decode_error_{os.path.basename(image_path)}"})
            return dummy_image, {"suffix": dummy_suffix, "language": language}


def _prepare_roboflow_dataset_paths(
    roboflow_api_key: str,
    roboflow_workspace: str,
    roboflow_project: str,
    roboflow_version: int,
    rank: int,
    world_size: int
) -> str:
    """
    Downloads and synchronizes the Roboflow dataset across distributed ranks.

    Args:
        roboflow_api_key (str): Roboflow API key.
        roboflow_workspace (str): Roboflow workspace name.
        roboflow_project (str): Roboflow project name.
        roboflow_version (int): Roboflow project version.
        rank (int): Distributed rank of the current process.
        world_size (int): Total number of distributed processes.

    Returns:
        str: Path to the root of the downloaded dataset.

    Raises:
        RuntimeError: If the dataset root location cannot be determined.
    """
    dataset_root_location: Optional[str] = None
    if rank == 0:
        print(f"[{rank}/{world_size}] Rank 0: Downloading Roboflow dataset (format: jsonl)...")
        rf = Roboflow(api_key=roboflow_api_key)
        project_obj = rf.workspace(roboflow_workspace).project(roboflow_project)
        version_obj = project_obj.version(roboflow_version)
        dataset_download = version_obj.download("jsonl")
        dataset_root_location = dataset_download.location
        print(f"[{rank}/{world_size}] Rank 0: Dataset downloaded to: {dataset_root_location}")

    if world_size > 1:
        dist.barrier()
        path_list = [dataset_root_location] if rank == 0 else [None]
        dist.broadcast_object_list(path_list, src=0)
        dataset_root_location = path_list[0]
        if rank != 0:
            print(f"[{rank}/{world_size}] Rank {rank}: Received dataset location: {dataset_root_location}")

    if dataset_root_location is None:
        raise RuntimeError(f"[{rank}/{world_size}] Dataset root location could not be determined.")
    return dataset_root_location


def get_dataset_metadata(
    dataset_type: str,
    dataset_config: Dict[str, Any],
    rank: int = 0,
    world_size: int = 1
) -> Tuple[str, Optional[str], str, Optional[str], str, Dict[str, Any]]:
    """
    Extracts dataset paths and configuration for supported dataset types.

    Args:
        dataset_type (str): Type of dataset ("roboflow_jsonl", "huggingface_cord", "flickr8k").
        dataset_config (Dict[str, Any]): Dataset configuration dictionary.
        rank (int): Distributed rank of the current process.
        world_size (int): Total number of distributed processes.

    Returns:
        Tuple containing:
            - train_data_path (str)
            - train_img_dir (Optional[str])
            - val_data_path (str)
            - val_img_dir (Optional[str])
            - dataset_format_id (str)
            - dataset_kwargs (Dict[str, Any])

    Raises:
        ValueError: If required configuration keys are missing or dataset_type is unsupported.
    """
    dataset_kwargs: Dict[str, Any] = {}

    if dataset_type == "roboflow_jsonl":
        required_keys = ["roboflow_api_key", "roboflow_workspace", "roboflow_project", "roboflow_version"]
        if not all(k in dataset_config for k in required_keys):
            missing_keys = [k for k in required_keys if k not in dataset_config]
            raise ValueError(f"Missing Roboflow configuration keys in dataset_config: {missing_keys}")
        api_key_to_use = dataset_config["roboflow_api_key"]
        dataset_root = _prepare_roboflow_dataset_paths(
            roboflow_api_key=api_key_to_use,
            roboflow_workspace=dataset_config["roboflow_workspace"],
            roboflow_project=dataset_config["roboflow_project"],
            roboflow_version=dataset_config["roboflow_version"],
            rank=rank,
            world_size=world_size
        )
        train_jsonl_path = os.path.join(dataset_root, "train", "annotations.jsonl")
        train_img_dir = os.path.join(dataset_root, "train")
        val_jsonl_path = os.path.join(dataset_root, "valid", "annotations.jsonl")
        val_img_dir = os.path.join(dataset_root, "valid")
        dataset_format_id = "jsonl"
        return train_jsonl_path, train_img_dir, val_jsonl_path, val_img_dir, dataset_format_id, dataset_kwargs

    elif dataset_type == "huggingface_cord":
        required_keys_hf = ["dataset_name", "train_split", "val_split"]
        if not all(k in dataset_config for k in required_keys_hf):
            missing_keys = [k for k in required_keys_hf if k not in dataset_config]
            raise ValueError(f"Missing HuggingFace CORD configuration keys in dataset_config: {missing_keys}")
        train_data_path = dataset_config["dataset_name"]
        val_data_path = dataset_config["dataset_name"]
        dataset_format_id = "hf_cord"
        dataset_kwargs["train_split_name"] = dataset_config["train_split"]
        dataset_kwargs["val_split_name"] = dataset_config["val_split"]
        return train_data_path, None, val_data_path, None, dataset_format_id, dataset_kwargs

    elif dataset_type == "flickr8k":
        required_keys_f8k = ["image_directory_path", "caption_files"]
        if not all(k in dataset_config for k in required_keys_f8k):
            missing_keys = [k for k in required_keys_f8k if k not in dataset_config]
            raise ValueError(f"Missing Flickr8k configuration keys in dataset_config: {missing_keys}")

        train_data_path = dataset_config["image_directory_path"]
        val_data_path = dataset_config["image_directory_path"]
        dataset_format_id = "flickr8k"

        dataset_kwargs["caption_files_dict"] = dataset_config["caption_files"]
        dataset_kwargs["split_ratio"] = dataset_config.get("split_ratio")
        dataset_kwargs["random_seed"] = dataset_config.get("random_seed")
        dataset_kwargs["train_split_name"] = dataset_config.get("train_split", "train")
        dataset_kwargs["val_split_name"] = dataset_config.get("val_split", "validation")

        return train_data_path, None, val_data_path, None, dataset_format_id, dataset_kwargs

    else:
        raise ValueError(f"Unsupported dataset_type: '{dataset_type}'")


def create_pytorch_dataset(
    dataset_format_identifier: str,
    data_path: str,
    image_dir_path: Optional[str],
    max_samples: Optional[int] = None,
    split: Optional[str] = None,
    **kwargs: Any
) -> Dataset:
    """
    Factory for instantiating a PyTorch Dataset.

    Args:
        dataset_format_identifier (str): Identifier for the dataset format ("jsonl", "hf_cord", "flickr8k").
        data_path (str): Path to the data or dataset name.
        image_dir_path (Optional[str]): Path to the image directory (if applicable).
        max_samples (Optional[int]): Maximum number of samples to load.
        split (Optional[str]): Dataset split ("train" or "validation").
        **kwargs: Additional dataset-specific keyword arguments.

    Returns:
        Dataset: Instantiated PyTorch Dataset.

    Raises:
        ValueError: If required arguments are missing or unsupported format is specified.
    """
    if dataset_format_identifier == "jsonl":
        if image_dir_path is None:
            raise ValueError("image_dir_path is required for 'jsonl' dataset format.")
        return JSONLDataset(
            jsonl_file_path=data_path,
            image_directory_path=image_dir_path,
            max_samples=max_samples
        )
    elif dataset_format_identifier == "hf_cord":
        if split is None:
            hf_split_name = kwargs.get("train_split_name") if split == "train" else kwargs.get("val_split_name")
            if not hf_split_name:
                raise ValueError("HuggingFace CORD dataset requires 'train_split_name' or 'val_split_name' in kwargs, or a 'split' argument for this factory function.")
        else:
            hf_split_name = split

        return HuggingFaceCORDDataset(
            dataset_name_or_path=data_path,
            split=hf_split_name,
            max_samples=max_samples
        )
    elif dataset_format_identifier == "flickr8k":
        if split not in ["train", "validation"]:
            raise ValueError(f"Flickr8k dataset requires 'split' argument to be 'train' or 'validation', got '{split}'.")
        is_validation_split_for_flickr = (split == "validation")
        if "caption_files_dict" not in kwargs:
            raise ValueError("caption_files_dict is required in kwargs for Flickr8kDataset.")

        return Flickr8kDataset(
            image_directory_path=data_path,
            max_samples=max_samples,
            is_validation_split=is_validation_split_for_flickr,
            caption_files_dict=kwargs["caption_files_dict"],
            split_ratio=kwargs.get("split_ratio"),
            random_seed=kwargs.get("random_seed")
        )
    else:
        raise ValueError(f"Unsupported dataset_format_identifier: '{dataset_format_identifier}'")