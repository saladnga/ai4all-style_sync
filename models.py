"""Neural network models for outfit completion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple

from config import MODEL_CONFIG


class EmbeddingNet(nn.Module):
    """Neural network for embedding fashion images."""

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 128,
        pretrained: bool = True,
    ):
        """
        Initialize the embedding network.

        Args:
            backbone: Backbone architecture ('resnet18', 'resnet50', 'efficientnet_b0')
            embedding_dim: Dimension of output embeddings
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        if backbone == "resnet18":
            self.base = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == "resnet50":
            self.base = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "efficientnet_b0":
            self.base = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace final layer with embedding layer
        if hasattr(self.base, "fc"):
            self.base.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(feature_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
        elif hasattr(self.base, "classifier"):
            self.base.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(feature_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return F.normalize(self.base(x), p=2, dim=1)  # L2 normalization


class TripletNet(nn.Module):
    """Triplet network for learning outfit embeddings."""

    def __init__(self, embedding_net: EmbeddingNet):
        """
        Initialize the triplet network.

        Args:
            embedding_net: The embedding network
        """
        super().__init__()
        self.embedding_net = embedding_net

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for triplet learning.

        Args:
            anchor: Anchor images
            positive: Positive images
            negative: Negative images

        Returns:
            Tuple of embeddings (anchor, positive, negative)
        """
        return (
            self.embedding_net(anchor),
            self.embedding_net(positive),
            self.embedding_net(negative),
        )

    def embed_anchor_stack(self, anchor_batch: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of anchor stacks and return averaged embeddings.

        Args:
            anchor_batch: Batch of padded anchor stacks (B, N_items, C, H, W)

        Returns:
            Averaged embeddings (B, embedding_dim)
        """
        B, N, C, H, W = anchor_batch.shape
        anchor_batch = anchor_batch.view(B * N, C, H, W)
        emb = self.embedding_net(anchor_batch)
        emb = emb.view(B, N, -1)
        return emb.mean(dim=1)


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Compute triplet loss.

    Args:
        anchor: Anchor embeddings
        positive: Positive embeddings
        negative: Negative embeddings
        margin: Margin for triplet loss

    Returns:
        Triplet loss value
    """
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def create_model(backbone: str = None, embedding_dim: int = None) -> TripletNet:
    """
    Create a triplet network model.

    Args:
        backbone: Model backbone (defaults to config)
        embedding_dim: Embedding dimension (defaults to config)

    Returns:
        TripletNet model
    """
    backbone = backbone or MODEL_CONFIG["backbone"]
    embedding_dim = embedding_dim or MODEL_CONFIG["embedding_dim"]

    embedding_net = EmbeddingNet(
        backbone=backbone, embedding_dim=embedding_dim, pretrained=True
    )

    return TripletNet(embedding_net)
