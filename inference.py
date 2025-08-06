"""Inference utilities for outfit completion."""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pickle
import json

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from config import MODEL_CONFIG, AAT_LABEL_PATH, AAT_IMAGE_DIR, CACHE_DIR
from models import create_model
from data_utils import extract_item_id

logger = logging.getLogger(__name__)


class OutfitCompleter:
    """Main class for outfit completion inference."""

    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the outfit completer.

        Args:
            model_path: Path to trained model
            device: Device to use for inference
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.item_embeddings = {}
        self.item_to_outfit = {}
        self.outfit_lookup = []

        # Load model
        self.load_model(model_path)
        self.setup_transform()

        # Load or create embeddings
        self.load_or_create_embeddings()

    def load_model(self, model_path: str) -> None:
        """Load the trained model."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Create model
        self.model = create_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded model from {model_path}")

    def setup_transform(self) -> None:
        """Setup image transformation pipeline."""
        self.transform = transforms.Compose(
            [
                transforms.Resize(MODEL_CONFIG["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_embedding(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Get embedding for a single image.

        Args:
            image_path: Path to image

        Returns:
            Image embedding tensor or None if failed
        """
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model.embedding_net(img_tensor)

            return embedding.squeeze(0).cpu()
        except Exception as e:
            logger.warning(f"Failed to get embedding for {image_path}: {e}")
            return None

    def load_or_create_embeddings(self) -> None:
        """Load cached embeddings or create new ones."""
        cache_path = CACHE_DIR / "embeddings.pkl"
        metadata_path = CACHE_DIR / "outfit_metadata.json"

        if cache_path.exists() and metadata_path.exists():
            try:
                # Load cached embeddings
                with open(cache_path, "rb") as f:
                    self.item_embeddings = pickle.load(f)

                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.item_to_outfit = metadata["item_to_outfit"]
                    self.outfit_lookup = metadata["outfit_lookup"]

                logger.info(f"Loaded {len(self.item_embeddings)} cached embeddings")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")

        # Create new embeddings
        self.create_embeddings()
        self.save_embeddings()

    def create_embeddings(self) -> None:
        """Create embeddings for all outfit items using the fixed approach."""
        logger.info("Creating embeddings for all items...")

        # Load outfit data
        try:
            with open(AAT_LABEL_PATH) as f:
                outfit_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load outfit data: {e}")
            return

        # Build all triplets first (like the working original code)
        all_triplets = []
        valid_entries = 0

        for entry in outfit_data:
            question_imgs = entry["question"]
            answers = entry["answers"]
            gt_idx = entry["gt"]

            # Skip malformed entries (same as original code)
            if gt_idx >= len(answers):
                continue

            valid_entries += 1
            gt_ans = answers[gt_idx]

            for i, ans in enumerate(answers):
                if i == gt_idx:
                    continue

                # Convert IDs to image filenames (strip category) - same as original
                anchor_paths = [
                    AAT_IMAGE_DIR / f"{q.split('_')[-1]}.jpg" for q in question_imgs
                ]
                pos_path = AAT_IMAGE_DIR / f"{gt_ans.split('_')[-1]}.jpg"
                neg_path = AAT_IMAGE_DIR / f"{ans.split('_')[-1]}.jpg"

                # Check if all files exist
                if (
                    all(p.exists() for p in anchor_paths)
                    and pos_path.exists()
                    and neg_path.exists()
                ):
                    all_triplets.append((anchor_paths, pos_path, neg_path))

        logger.info(
            f"Found {valid_entries} valid entries from {len(outfit_data)} total"
        )
        logger.info(f"Created {len(all_triplets)} triplets")

        # Now embed all items (like the original embed_all_items function)
        item_embs = []
        item_paths = []
        item_to_outfit = []

        for triplet_idx, (anchor_paths, pos_path, neg_path) in enumerate(all_triplets):
            # Full outfit includes anchor items + positive answer
            full_outfit = list(anchor_paths) + [pos_path]

            for item_path in full_outfit:
                try:
                    embedding = self.get_embedding(str(item_path))
                    if embedding is not None:
                        item_embs.append(embedding)
                        item_paths.append(str(item_path))
                        item_to_outfit.append([str(p) for p in full_outfit])
                except Exception as e:
                    logger.warning(f"Failed to process {item_path}: {e}")
                    continue

        if item_embs:
            # Convert to proper format for the completer
            for i, (item_path, outfit_paths) in enumerate(
                zip(item_paths, item_to_outfit)
            ):
                item_id = Path(item_path).stem
                self.item_embeddings[item_id] = item_embs[i]
                self.item_to_outfit[item_id] = outfit_paths

            # Create outfit lookup (unique outfits)
            unique_outfits = set()
            for outfit_paths in item_to_outfit:
                outfit_tuple = tuple(sorted([Path(p).stem for p in outfit_paths]))
                unique_outfits.add(outfit_tuple)

            self.outfit_lookup = [list(outfit) for outfit in unique_outfits]

        logger.info(f"Created embeddings for {len(self.item_embeddings)} items")
        logger.info(f"Processed {len(self.outfit_lookup)} outfits")

    def save_embeddings(self) -> None:
        """Save embeddings to cache."""
        cache_path = CACHE_DIR / "embeddings.pkl"
        metadata_path = CACHE_DIR / "outfit_metadata.json"

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.item_embeddings, f)

            metadata = {
                "item_to_outfit": self.item_to_outfit,
                "outfit_lookup": self.outfit_lookup,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            logger.info("Saved embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")

    def find_similar_items(
        self, query_embedding: torch.Tensor, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find most similar items to query embedding.

        Args:
            query_embedding: Query image embedding
            top_k: Number of top matches to return

        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self.item_embeddings:
            logger.warning("No embeddings available")
            return []

        # Stack all embeddings
        item_ids = list(self.item_embeddings.keys())
        embeddings = torch.stack(
            [self.item_embeddings[item_id] for item_id in item_ids]
        )

        # Compute similarities using cosine similarity (like original code)
        query_embedding = query_embedding.unsqueeze(0)
        similarities = cosine_similarity(query_embedding.numpy(), embeddings.numpy())[0]

        # Get top matches
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            item_id = item_ids[idx]
            similarity = similarities[idx]
            results.append((item_id, float(similarity)))

        return results

    def complete_outfit(
        self, input_image_path: str, top_k: int = 1
    ) -> Optional[List[str]]:
        """
        Complete an outfit given an input image.

        Args:
            input_image_path: Path to input fashion item image
            top_k: Number of outfit suggestions to generate

        Returns:
            List of item IDs for completed outfit (excluding input item)
        """
        # Get embedding for input image
        query_embedding = self.get_embedding(input_image_path)
        if query_embedding is None:
            logger.error("Failed to get embedding for input image")
            return None

        # Find similar items
        similar_items = self.find_similar_items(query_embedding, top_k=5)
        if not similar_items:
            logger.error("No similar items found")
            return None

        # Get the outfit for the most similar item
        best_item_id, similarity = similar_items[0]
        logger.info(f"Most similar item: {best_item_id} (similarity: {similarity:.3f})")

        if best_item_id in self.item_to_outfit:
            outfit_paths = self.item_to_outfit[best_item_id]
            # Filter out the matched item (like original code)
            input_item_id = Path(input_image_path).stem
            filtered_outfit = [
                p
                for p in outfit_paths
                if Path(p).stem != input_item_id and Path(p).stem != best_item_id
            ]
            return [Path(p).stem for p in filtered_outfit]

        return None

    def get_outfit_images(self, item_ids: List[str]) -> List[str]:
        """
        Get image paths for a list of item IDs.

        Args:
            item_ids: List of item IDs

        Returns:
            List of image paths
        """
        image_paths = []
        for item_id in item_ids:
            image_path = AAT_IMAGE_DIR / f"{item_id}.jpg"
            if image_path.exists():
                image_paths.append(str(image_path))
            else:
                logger.warning(f"Image not found for item {item_id}")

        return image_paths
