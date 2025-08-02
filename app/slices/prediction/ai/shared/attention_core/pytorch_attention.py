from typing import Optional, Protocol, Tuple

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW

class PyTorchAttention(nn.Module):
    """
    Attention module using the PyTorch MultiheadAttention module.
    Currently slower and less flexible than PlainAttention, but
    this hopefully improves as we upgrade PyTorch.
    """

    def __init__(
        self,
        model_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        context_dim = model_dim if context_dim is None else context_dim
        self.mha = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
            kdim=context_dim,
            vdim=context_dim,
        )

    def forward(
        self, x, context=None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        :param x: [batch, seq_len, model_dim]
        :param context: [batch, context_len, context_dim]
        :param mask: [batch, seq_len, context_len]
        """
        context = x if context is None else context
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            # We invert the mask here, because for torch True means *empty*,
            mask = repeat(
                # pylint: disable=invalid-unary-operand-type
                ~mask,
                "batch query key -> (batch head) query key",
                head=self.mha.num_heads,
            )
        y, attn = self.mha(
            query=x,
            key=context,
            value=context,
            attn_mask=mask,
            need_weights=True,
            average_attn_weights=False,
        )
        return y, attn
