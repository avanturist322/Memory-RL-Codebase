from __future__ import annotations
import torch.nn as nn
from typing import Optional


class EmbeddingRepresentation(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, obs):
        return self.embedding(obs)

    @staticmethod
    def make_discrete_representation(
        vocab_sizes: int, obs_dim: int, embed_per_obs_dim: int, outer_embed_size: int
    ) -> EmbeddingRepresentation:
        """
        For use in discrete observation environments.

        Args:
            vocab_sizes: The number of different values your observation could include.
            obs_dim: The length of the observation vector (assuming 1d).
            embed_per_obs_dim: The number of features you want to give to each observation
                dimension.
            embed_size: The length of the resulting embedding vector.
        """

        assert (
            vocab_sizes is not None
        ), "Discrete environments need to have a vocab size for the token embeddings"
        assert (
            embed_per_obs_dim > 1
        ), "Each observation feature needs at least 1 embed dim"

        embedding = nn.Sequential(
            nn.Embedding(vocab_sizes, embed_per_obs_dim),
            nn.Flatten(start_dim=-2),
            nn.Linear(embed_per_obs_dim * obs_dim, outer_embed_size),
        )
        return EmbeddingRepresentation(embedding=embedding)

    @staticmethod
    def make_continuous_representation(obs_dim: int, outer_embed_size: int):
        """
        For use in continuous observation environments. Projects the observation to the
            specified dimensionality for use in the network.

        Args:
            obs_dim: the length of the observation vector (assuming 1d)
            embed_size: The length of the resulting embedding vector
        """
        embedding = nn.Linear(obs_dim, outer_embed_size)
        return EmbeddingRepresentation(embedding=embedding)

    @staticmethod
    def make_image_representation(obs_dim: int, outer_embed_size: int):
        """
        For use in continuous observation environments. Projects the observation to the
            specified dimensionality for use in the network.

        Args:
            obs_dim: the length of the observation vector (assuming 1d)
            embed_size: The length of the resulting embedding vector
        """
        #embedding = nn.Linear(obs_dim, outer_embed_size)
        
        # for rl bench input is (128, 128, 3) same as for gtxl and dt 
        # nn.Conv2d(4, 32, 8, stride=4, padding=0) for (3, 84, 84)

        # for rlbench, images size: (128, 128, 3)
        # embedding = nn.Sequential(nn.Conv2d(3, 32, 8, stride=6, padding=0), nn.ReLU(),
        #     nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
        #     nn.Flatten(), 
        #     nn.Linear(3136, 512), nn.ReLU(),
        #     nn.Linear(in_features=512, out_features=outer_embed_size),
        #     nn.Tanh())

        # for atari like, images size: (3, 84, 84)
        embedding = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4, padding=0), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(), 
                nn.Flatten(), nn.Linear(3136, outer_embed_size))
    
        return EmbeddingRepresentation(embedding=embedding)
