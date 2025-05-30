from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from einops import rearrange


class MLP(nn.Sequential):
    """
    A flexible multi-layer perceptron (MLP) module with:
    - Optional normalization (layer or batch norm)
    - Custom activation functions
    - Dropout
    - Optional output layer with bias and activation
    """

    _out_features: int

    def __init__(
        self,
        input_size: int,
        hidden_size: Sequence[int],
        output_size: Optional[int] = None,
        activation: str = "relu",
        norm: Optional[str] = None,
        dropout_rate: float = 0.0,
        output_bias: bool = True,
        output_activation: bool = False,
        pre_norm: bool = False,
        norm_mode: str = "before",  # "before" or "after" linear layers
    ):
        super().__init__()
        layers: List[nn.Module] = []
        size = input_size  # Current feature size as we build layers

        # Build hidden layers
        for next_size in hidden_size:
            # Apply normalization before the linear layer if configured
            if norm and norm_mode == "before" and (len(layers) > 0 or pre_norm):
                layers.append(NORM_FACTORY[norm](size))

            # Add linear (fully connected) layer
            layers.append(nn.Linear(size, next_size))
            size = next_size  # Update feature size

            # Apply normalization after the linear layer if configured
            if norm and norm_mode == "after":
                layers.append(NORM_FACTORY[norm](size))

            # Apply activation function
            layers.append(ACTIVATION_FACTORY[activation]())

            # Apply dropout if needed
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))

        # Optional output layer
        if output_size is not None:
            # Optionally apply normalization before output
            if norm and norm_mode == "before" and (len(layers) > 0 or pre_norm):
                layers.append(NORM_FACTORY[norm](size))

            # Add output linear layer
            layers.append(nn.Linear(size, output_size, bias=output_bias))
            size = output_size

            # Optionally apply activation at the output
            if output_activation:
                layers.append(ACTIVATION_FACTORY[activation]())

        # Initialize the model with layers
        super().__init__(*layers)
        self._out_features = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that:
        - Flattens all dimensions except the last before feeding through MLP
        - Restores original shape after processing
        """
        # Flatten all dimensions except the last (feature dimension)
        y = x.flatten(0, -2)

        # Feed through the sequential model
        y = super().forward(y)

        # Restore original shape with updated feature size
        return y.view(x.shape[:-1] + (self._out_features,))

    @property
    def out_features(self) -> int:
        """Returns the final output feature size."""
        return self._out_features


class Sine(nn.Module):
    """
    Sine activation function (used in SIREN models).
    Reference: https://arxiv.org/abs/2006.09661
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class BLCBatchNorm(nn.BatchNorm1d):
    """
    BatchNorm1d that handles both:
    - (B, C): standard
    - (B, L, C): sequence inputs (reordered internally)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return super().forward(x)
        elif x.dim() == 3:
            x = rearrange(x, "B L C -> B C L")
            x = super().forward(x)
            return rearrange(x, "B C L -> B L C")
        else:
            raise ValueError("Only 2D or 3D tensors are supported")


# Mapping strings to actual activation/norm classes
ACTIVATION_FACTORY = {
    "relu": lambda: nn.ReLU(inplace=True),
    "sine": Sine,
    "gelu": nn.GELU,
}

NORM_FACTORY = {
    "layer_norm": nn.LayerNorm,
    "batch_norm": BLCBatchNorm,
}
