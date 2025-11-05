"""
Neural network architectures for Deep RL agents.

This module defines CNN-based feature extractors for processing
stacked frame observations in the pursuit-evasion task.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from gymnasium import spaces


class CNNFeatureExtractor(nn.Module):
    """
    Convolutional Neural Network for extracting features from stacked frames.

    Architecture:
        - Multiple Conv2D layers with ReLU activation
        - Flattening layer
        - Output feature vector

    Input: (batch, channels, height, width) where channels = num_stacked_frames
    Output: (batch, features_dim) feature vector
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        cnn_features: List[int] = [32, 64, 64],
        cnn_kernels: List[int] = [8, 4, 3],
        cnn_strides: List[int] = [4, 2, 1],
    ):
        """
        Initialize CNN feature extractor.

        Args:
            observation_space: Observation space from environment
            features_dim: Dimension of output feature vector
            cnn_features: List of output channels for each conv layer
            cnn_kernels: List of kernel sizes for each conv layer
            cnn_strides: List of strides for each conv layer
        """
        super().__init__()

        # Assume observation_space is either (C, H, W) or (H, W, C)
        # Gymnasium frame stacking typically gives (num_frames, H, W)
        obs_shape = observation_space.shape

        # Handle different frame stacking formats
        if len(obs_shape) == 3:
            # Could be (num_frames, H, W) or (H, W, num_frames)
            if obs_shape[0] <= 4:  # Likely (num_frames, H, W)
                n_input_channels = obs_shape[0]
            else:  # Likely (H, W, num_frames)
                n_input_channels = obs_shape[2]
        else:
            n_input_channels = 1

        # Build convolutional layers
        layers = []
        in_channels = n_input_channels

        for out_channels, kernel_size, stride in zip(
            cnn_features, cnn_kernels, cnn_strides
        ):
            layers.append(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=stride
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        # Compute output shape by doing a forward pass
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, obs_shape[-2], obs_shape[-1])
            n_flatten = self.cnn(sample_input).view(1, -1).shape[1]

        # Linear layer to project to desired feature dimension
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        """Return the dimension of the output feature vector."""
        return self._features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            observations: Batch of observations (batch, channels, height, width)

        Returns:
            Feature vectors (batch, features_dim)
        """
        # Ensure input is float and normalized
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0

        x = self.cnn(observations)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Actor(nn.Module):
    """
    Actor network for SAC (outputs mean and log_std of action distribution).

    Takes feature vectors and outputs parameters of a Gaussian policy.
    """

    def __init__(
        self,
        feature_extractor: CNNFeatureExtractor,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        """
        Initialize actor network.

        Args:
            feature_extractor: CNN feature extractor
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions for MLP
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Build MLP
        layers = []
        in_dim = feature_extractor.features_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(in_dim, action_dim)
        self.log_std_layer = nn.Linear(in_dim, action_dim)

    def forward(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor network.

        Args:
            observations: Batch of observations

        Returns:
            Tuple of (mean, log_std) for action distribution
        """
        features = self.feature_extractor(observations)
        hidden = self.mlp(features)

        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden)

        # Clamp log_std to reasonable range
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


class Critic(nn.Module):
    """
    Critic network for SAC (Q-function).

    Takes feature vectors and actions, outputs Q-value.
    SAC typically uses two critics for stability.
    """

    def __init__(
        self,
        feature_extractor: CNNFeatureExtractor,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
    ):
        """
        Initialize critic network.

        Args:
            feature_extractor: CNN feature extractor
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions for MLP
        """
        super().__init__()

        self.feature_extractor = feature_extractor

        # Build MLP (input is features + actions)
        layers = []
        in_dim = feature_extractor.features_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Output Q-value
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through critic network.

        Args:
            observations: Batch of observations
            actions: Batch of actions

        Returns:
            Q-values (batch, 1)
        """
        features = self.feature_extractor(observations)
        x = torch.cat([features, actions], dim=1)
        q_value = self.mlp(x)
        return q_value


# Custom feature extractor policy for Stable-Baselines3
class CustomCNN(nn.Module):
    """
    Custom CNN feature extractor compatible with Stable-Baselines3.

    This is a wrapper to make our CNN compatible with SB3's policy API.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__()
        self.cnn = CNNFeatureExtractor(
            observation_space=observation_space,
            features_dim=features_dim,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)
