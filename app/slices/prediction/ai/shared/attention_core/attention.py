from typing import Optional, Protocol, Tuple

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW

class Attention(Protocol):
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Implementation of attention. Should return tuple of (feature, attention_map).
        """
