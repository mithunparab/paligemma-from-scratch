import os
import json
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from torch.utils.data import Dataset
import torch.distributed as dist
from roboflow import Roboflow
from datasets import load_dataset as hf_load_dataset
import random

# -----------------------------------------------------------------------------
# NOTE: If more dataset types are added, consider a registry/factory pattern
# for dataset instantiation and metadata extraction.
# TODO: Dataset error handling is ad hoc; consider a more robust error/skip mechanism.
# -----------------------------------------------------------------------------

class JSONLDataset(Dataset):
    """
    PyTorch Dataset for Roboflow JSONL-format datasets.

    Parameters
    ----------
    jsonl_file_path : str
        Path to the .jsonl annotation file.
    image_directory_path : str
        Directory containing the images referenced in the JSONL.
    max_samples : Optional[int], default=None
        If set, limits the dataset to the first `max_samples` entries.

    Returns
    -------
    Tuple[Image.Image, Dict[str, Any]]
        A tuple of the loaded image and a dict with at least a "suffix" key.

    Notes
    -----
    Skips malformed JSON lines and missing images, raising exceptions for the latter.
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
                    # Retained: Malformed JSON lines are skipped, but I log for traceability.
                    print(f"Skipping malformed JSON line: {line.strip()}")
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Returns the image and its associated target dict for the given index.

        Raises
        ------
        IndexError
            If idx is out of bounds.
        FileNotFoundError
            If the referenced image file does not exist.
        """
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            # Retained: Missing images are a critical error; I raise to avoid silent data corruption.
            print(f"Image not found: {image_path}")
            raise
        return image, {"suffix": entry["suffix"]}


class HuggingFaceCORDDataset(Dataset):
    """
    PyTorch Dataset for HuggingFace datasets (e.g., 'naver-clova-ix/cord-v2').

    Parameters
    ----------
    dataset_name_or_path : str
        HuggingFace dataset identifier or path.
    split : str, default="train"
        Dataset split to use.
    max_samples : Optional[int], default=None
        If set, limits the dataset to the first `max_samples` entries.

    Returns
    -------
    Tuple[Image.Image, Dict[str, Any]]
        A tuple of the loaded image and a dict with a "suffix" key.

    Notes
    -----
    Loads and processes data lazily in __getitem__ to conserve memory.
    Handles multiple ground truth parses by random selection.
    If a sample fails to load, returns a placeholder image and error marker in the suffix.
    """
    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_name_or_path = dataset_name_or_path
        self.split = split

        print(f"Initializing HuggingFaceCORDDataset for: {dataset_name_or_path}, split: {split}")
        raw_hf_dataset = hf_load_dataset(dataset_name_or_path, split=self.split, streaming=False)
        if max_samples is not None and max_samples > 0 and max_samples < len(raw_hf_dataset):
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
        Processes the ground truth JSON string, selecting one parse if multiple are present.

        Parameters
        ----------
        ground_truth_json_str : str
            JSON string containing ground truth parses.

        Returns
        -------
        str
            JSON string of the selected parse, or empty dict if none found.

        Notes
        -----
        Handles both 'gt_parses' (list) and 'gt_parse' (dict) keys.
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
            # Retained: If ground_truth is malformed, I fallback to empty JSON for robustness.
            print(f"Warning: Could not decode ground_truth JSON: {ground_truth_json_str[:100]}... Using empty JSON.")
            return json.dumps({})
        except Exception as e:
            # Retained: Catch-all for unexpected structure; fallback to empty JSON.
            print(f"Warning: Error processing ground_truth: {e}. Using empty JSON. GT: {ground_truth_json_str[:100]}...")
            return json.dumps({})

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Returns the image and processed suffix dict for the given index.

        Raises
        ------
        IndexError
            If idx is out of bounds.

        Notes
        -----
        If loading or processing fails, returns a placeholder image and error marker.
        """
        if idx < 0 or idx >= self.dataset_length:
            raise IndexError(f"Index {idx} out of range for HuggingFaceCORDDataset with length {self.dataset_length}")
        try:
            sample = self.hf_dataset[idx]
            image = sample["image"].convert("RGB")
            target_suffix_str = self._get_processed_suffix(sample["ground_truth"])
            return image, {"suffix": target_suffix_str}
        except Exception as e:
            # Retained: If a sample fails, I return a placeholder to avoid crashing the DataLoader.
            print(f"Error loading or processing sample at index {idx} for split {self.split}: {e}")
            print(f"Attempting to return a placeholder for index {idx} due to error. THIS IS NOT IDEAL.")
            dummy_image = Image.new('RGB', (224, 224), color='red')
            dummy_suffix = json.dumps({"error": "failed_to_load_sample"})
            return dummy_image, {"suffix": dummy_suffix}


def _prepare_roboflow_dataset_paths(
    roboflow_api_key: str,
    roboflow_workspace: str,
    roboflow_project: str,
    roboflow_version: int,
    rank: int,
    world_size: int
) -> str:
    """
    Downloads and synchronizes the Roboflow dataset location across distributed ranks.

    Parameters
    ----------
    roboflow_api_key : str
        API key for Roboflow.
    roboflow_workspace : str
        Roboflow workspace name.
    roboflow_project : str
        Roboflow project name.
    roboflow_version : int
        Version number of the project.
    rank : int
        Current process rank (for DDP).
    world_size : int
        Total number of distributed processes.

    Returns
    -------
    str
        Path to the root of the downloaded dataset.

    Raises
    ------
    RuntimeError
        If the dataset root location cannot be determined.

    Notes
    -----
    Only rank 0 downloads; others synchronize via broadcast.
    """
    dataset_root_location = None
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
        # Retained: If synchronization fails, I raise to avoid silent errors.
        raise RuntimeError(f"[{rank}/{world_size}] Dataset root location could not be determined.")

    return dataset_root_location

def get_dataset_metadata(
    dataset_type: str,
    dataset_config: Dict[str, Any],
    rank: int = 0,
    world_size: int = 1
) -> Tuple[str, Optional[str], str, Optional[str], str]:
    """
    Returns dataset paths and format identifier for the given dataset type and config.

    Parameters
    ----------
    dataset_type : str
        Type of dataset ("roboflow_jsonl" or "huggingface_cord").
    dataset_config : Dict[str, Any]
        Configuration dictionary for the dataset.
    rank : int, default=0
        DDP process rank.
    world_size : int, default=1
        DDP world size.

    Returns
    -------
    Tuple[str, Optional[str], str, Optional[str], str]
        (train_data_path, train_image_dir, val_data_path, val_image_dir, dataset_format_id)

    Raises
    ------
    ValueError
        If required config keys are missing or dataset_type is unsupported.

    Notes
    -----
    For HuggingFace datasets, image directories are None.
    """
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
        return train_jsonl_path, train_img_dir, val_jsonl_path, val_img_dir, dataset_format_id

    elif dataset_type == "huggingface_cord":
        required_keys_hf = ["dataset_name", "train_split", "val_split"]
        if not all(k in dataset_config for k in required_keys_hf):
            missing_keys = [k for k in required_keys_hf if k not in dataset_config]
            raise ValueError(f"Missing HuggingFace CORD configuration keys in dataset_config: {missing_keys}")
        train_data_path = dataset_config["dataset_name"]
        val_data_path = dataset_config["dataset_name"]
        dataset_format_id = "hf_cord"
        return train_data_path, None, val_data_path, None, dataset_format_id

    else:
        raise ValueError(f"Unsupported dataset_type: '{dataset_type}'")

def create_pytorch_dataset(
    dataset_format_identifier: str,
    data_path: str,
    image_dir_path: Optional[str],
    max_samples: Optional[int] = None,
    split: Optional[str] = None,
    **kwargs
) -> Dataset:
    """
    Factory for instantiating a PyTorch Dataset based on format identifier.

    Parameters
    ----------
    dataset_format_identifier : str
        Identifier for the dataset format ("jsonl" or "hf_cord").
    data_path : str
        Path to data (JSONL file or HuggingFace dataset name).
    image_dir_path : Optional[str]
        Path to image directory (for JSONL datasets).
    max_samples : Optional[int], default=None
        Maximum number of samples to load.
    split : Optional[str], default=None
        Dataset split (for HuggingFace datasets).
    **kwargs
        Additional arguments for dataset constructors.

    Returns
    -------
    Dataset
        Instantiated PyTorch Dataset.

    Raises
    ------
    ValueError
        If required arguments are missing or format is unsupported.

    Notes
    -----
    For new dataset types, extend this function or refactor to a registry.
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
            raise ValueError("`split` argument is required for 'hf_cord' dataset format (e.g., 'train', 'validation').")
        cord_specific_kwargs = {k: v for k, v in kwargs.items() if k in ["sort_json_key"]}
        return HuggingFaceCORDDataset(
            dataset_name_or_path=data_path,
            split=split,
            max_samples=max_samples,
            **cord_specific_kwargs
        )
    else:
        raise ValueError(f"Unsupported dataset_format_identifier: '{dataset_format_identifier}'")