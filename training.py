"""Training utilities for the outfit completion model."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR
from data_utils import TripletDataset, load_all_triplets, pad_anchor_collate
from models import create_model, triplet_loss

logger = logging.getLogger(__name__)


class OutfitTrainer:
    """Trainer class for outfit completion model."""

    def __init__(self, model_name: str = "outfit_completer", device: str = None):
        """
        Initialize the trainer.

        Args:
            model_name: Name for saving the model
            device: Device to use for training
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.history = {"train_loss": [], "val_loss": [], "epochs": []}

        logger.info(f"Using device: {self.device}")

    def prepare_data(self) -> None:
        """Load and prepare training data."""
        logger.info("Loading triplets...")
        all_triplets = load_all_triplets()

        if not all_triplets:
            raise ValueError("No triplets loaded. Check your data paths.")

        # Split data
        train_triplets, val_triplets = train_test_split(
            all_triplets,
            test_size=TRAIN_CONFIG["test_size"],
            random_state=TRAIN_CONFIG["random_state"],
        )

        logger.info(f"Train: {len(train_triplets)} | Val: {len(val_triplets)}")

        # Create datasets
        train_dataset = TripletDataset(train_triplets)
        val_dataset = TripletDataset(val_triplets)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=MODEL_CONFIG["batch_size"],
            shuffle=True,
            collate_fn=pad_anchor_collate,
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=MODEL_CONFIG["batch_size"],
            shuffle=False,
            collate_fn=pad_anchor_collate,
            num_workers=4,
            pin_memory=True,
        )

    def create_model(self) -> None:
        """Create and initialize the model."""
        self.model = create_model().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=MODEL_CONFIG["learning_rate"], weight_decay=1e-4
        )

        logger.info(
            f"Created model with {sum(p.numel() for p in self.model.parameters())} parameters"
        )

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (anchor, positive, negative) in enumerate(pbar):
            anchor = anchor.to(self.device, non_blocking=True)
            positive = positive.to(self.device, non_blocking=True)
            negative = negative.to(self.device, non_blocking=True)

            # Forward pass
            anchor_emb = self.model.embed_anchor_stack(anchor)
            pos_emb = self.model.embedding_net(positive)
            neg_emb = self.model.embedding_net(negative)

            # Compute loss
            loss = triplet_loss(anchor_emb, pos_emb, neg_emb, MODEL_CONFIG["margin"])

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            # Log periodically
            if (batch_idx + 1) % TRAIN_CONFIG["validation_freq"] == 0:
                logger.info(
                    f"Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for anchor, positive, negative in tqdm(self.val_loader, desc="Validating"):
                anchor = anchor.to(self.device, non_blocking=True)
                positive = positive.to(self.device, non_blocking=True)
                negative = negative.to(self.device, non_blocking=True)

                # Forward pass
                anchor_emb = self.model.embed_anchor_stack(anchor)
                pos_emb = self.model.embedding_net(positive)
                neg_emb = self.model.embedding_net(negative)

                # Compute loss
                loss = triplet_loss(
                    anchor_emb, pos_emb, neg_emb, MODEL_CONFIG["margin"]
                )
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, num_epochs: int = None) -> None:
        """Train the model."""
        num_epochs = num_epochs or MODEL_CONFIG["num_epochs"]

        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["epochs"].append(epoch + 1)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Save checkpoint
            self.save_checkpoint(epoch + 1)

        logger.info("Training completed!")

    def save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": MODEL_CONFIG,
        }

        checkpoint_path = MODEL_DIR / f"{self.model_name}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Also save as latest
        latest_path = MODEL_DIR / f"{self.model_name}_latest.pth"
        torch.save(checkpoint, latest_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if self.model is None:
            self.create_model()

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.history = checkpoint.get("history", self.history)

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def save_training_history(self) -> None:
        """Save training history as JSON."""
        history_path = MODEL_DIR / f"{self.model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Saved training history: {history_path}")


def train_model(
    model_name: str = "outfit_completer", resume_from: str = None
) -> OutfitTrainer:
    """
    Train the outfit completion model.

    Args:
        model_name: Name for the model
        resume_from: Path to checkpoint to resume from

    Returns:
        Trained model trainer
    """
    trainer = OutfitTrainer(model_name)

    # Prepare data
    trainer.prepare_data()

    # Create or load model
    if resume_from:
        trainer.load_checkpoint(resume_from)
    else:
        trainer.create_model()

    # Train
    trainer.train()

    # Save history
    trainer.save_training_history()

    return trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = train_model()
