"""Data loading utilities for the Outfit Completer."""

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config import (
    AAT_IMAGE_DIR,
    LAT_IMAGE_DIR,
    AAT_LABEL_PATH,
    LAT_LABEL_PATH,
    MODEL_CONFIG,
)

logger = logging.getLogger(__name__)


class TripletDataset(Dataset):
    """Dataset class for triplet learning with outfit completion."""

    def __init__(self, triplets: List[Tuple], transform=None):
        """
        Initialize the dataset.

        Args:
            triplets: List of (anchor_paths, positive_path, negative_path) tuples
            transform: Optional image transforms
        """
        self.triplets = triplets
        self.transform = transform or self._get_default_transform()

    def _get_default_transform(self):
        """Get default image transforms."""
        return transforms.Compose(
            [
                transforms.Resize(MODEL_CONFIG["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ]
        )

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a triplet of images."""
        anchor_paths, pos_path, neg_path = self.triplets[idx]

        # Load anchor images (multiple items in question)
        anchor_imgs = []
        for path in anchor_paths:
            try:
                img = Image.open(path).convert("RGB")
                img = self.transform(img)
                anchor_imgs.append(img)
            except Exception as e:
                logger.warning(f"Failed to load anchor image {path}: {e}")
                # Create a black image as fallback
                img = torch.zeros(3, *MODEL_CONFIG["image_size"])
                anchor_imgs.append(img)

        anchor = torch.stack(anchor_imgs)

        # Load positive and negative images
        try:
            positive = self.transform(Image.open(pos_path).convert("RGB"))
        except Exception as e:
            logger.warning(f"Failed to load positive image {pos_path}: {e}")
            positive = torch.zeros(3, *MODEL_CONFIG["image_size"])

        try:
            negative = self.transform(Image.open(neg_path).convert("RGB"))
        except Exception as e:
            logger.warning(f"Failed to load negative image {neg_path}: {e}")
            negative = torch.zeros(3, *MODEL_CONFIG["image_size"])

        return anchor, positive, negative


def load_triplets_from_json(json_path: Path, image_root: Path) -> List[Tuple]:
    """
    Load triplets from JSON file.

    Args:
        json_path: Path to JSON label file
        image_root: Path to image directory

    Returns:
        List of triplets (anchor_paths, positive_path, negative_path)
    """
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        return []

    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON file {json_path}: {e}")
        return []

    triplets = []
    for entry in data:
        try:
            question_imgs = entry["question"]
            answers = entry["answers"]
            gt_idx = entry["gt"]

            if gt_idx >= len(answers):
                continue  # Skip malformed entries

            gt_ans = answers[gt_idx]

            for i, ans in enumerate(answers):
                if i == gt_idx:
                    continue

                # Convert IDs to image filenames (strip category prefix)
                anchor_paths = [
                    image_root / f"{q.split('_')[-1]}.jpg" for q in question_imgs
                ]
                pos_path = image_root / f"{gt_ans.split('_')[-1]}.jpg"
                neg_path = image_root / f"{ans.split('_')[-1]}.jpg"

                # Check if all files exist
                if (
                    all(p.exists() for p in anchor_paths)
                    and pos_path.exists()
                    and neg_path.exists()
                ):
                    triplets.append((anchor_paths, pos_path, neg_path))

        except Exception as e:
            logger.warning(f"Failed to process entry: {e}")
            continue

    logger.info(f"Loaded {len(triplets)} triplets from {json_path}")
    return triplets


def load_all_triplets() -> List[Tuple]:
    """Load all triplets from AAT and LAT datasets."""
    aat_triplets = load_triplets_from_json(AAT_LABEL_PATH, AAT_IMAGE_DIR)
    lat_triplets = load_triplets_from_json(LAT_LABEL_PATH, LAT_IMAGE_DIR)

    all_triplets = aat_triplets + lat_triplets
    logger.info(f"Total triplets loaded: {len(all_triplets)}")
    return all_triplets


def pad_anchor_collate(batch):
    """Custom collate function to handle variable-length anchor sequences."""
    anchors, positives, negatives = zip(*batch)

    # Find max sequence length
    max_len = max(a.shape[0] for a in anchors)

    # Pad anchors to max length
    padded_anchors = []
    for anchor in anchors:
        pad_size = max_len - anchor.shape[0]
        if pad_size > 0:
            pad_tensor = torch.zeros((pad_size, 3, *MODEL_CONFIG["image_size"]))
            anchor = torch.cat([anchor, pad_tensor], dim=0)
        padded_anchors.append(anchor)

    anchor_batch = torch.stack(padded_anchors)
    positive_batch = torch.stack(positives)
    negative_batch = torch.stack(negatives)

    return anchor_batch, positive_batch, negative_batch


def extract_item_id(path_or_label: str) -> str:
    """Extract item ID from path or label."""
    if isinstance(path_or_label, Path):
        path_or_label = str(path_or_label)

    if "/" in path_or_label:
        # Extract from file path
        return Path(path_or_label).stem
    else:
        # Extract from label (remove category prefix)
        return path_or_label.split("_")[-1]
