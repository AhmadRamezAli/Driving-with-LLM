import torch
import torch.nn as nn

from models.vector_encoder import VectorEncoder, VectorEncoderConfig
from models.transformer import Transformer
from utils.vector_utils.randomize_utils import VectorObservation, VectorObservationConfig


class VectorFormer(nn.Module):
    """Encode a VectorObservation and process it with a Transformer."""

    def __init__(
        self,
        encoder_config: VectorEncoderConfig = VectorEncoderConfig(),
        obs_config: VectorObservationConfig = VectorObservationConfig(),
        num_vector_tokens: int = 64,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__()
        self.vector_encoder = VectorEncoder(
            encoder_config, obs_config, num_vector_tokens
        )
        self.transformer = Transformer(
            model_dim=encoder_config.model_dim,
            depth=transformer_depth,
            heads=transformer_heads,
        )
        self.out_features = encoder_config.model_dim

    def forward(self, obs: VectorObservation) -> torch.Tensor:
        tokens = self.vector_encoder(obs)
        transformed, _, _ = self.transformer(tokens)
        return transformed