from typing import Optional, Protocol, Tuple

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW





def plain_attention(
    query: torch.Tensor,
    key_value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    attention_dropout: float = 0.0,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        query: (batch, out_seq_len, dim)
        key_value: (batch, in_seq_len, 2, dim)
        mask: (batch, out_seq_len, in_seq_len)
        attention_dropout: dropout probability
        training: whether in training mode

    Returns:
        Tuple[Tensor, Tensor]: (feature, attention_map)
        where:
            feature: (batch, out_seq_len, dim)
            attention_map: (batch, heads, out_seq_len, in_seq_len)
    """
    key, value = key_value.unbind(2)

    keyT = key.permute(0, 2, 3, 1)  # transpose to (batch, heads, dim, in_seq_len)
    value = value.transpose(1, 2)  # transpose to (batch, heads, in_seq_len, dim)
    query = query.transpose(1, 2)  # transpose to (batch, heads, out_seq_len, dim)

    softmax_scale = query.shape[-1] ** (-0.5)
    dots = torch.matmul(query * softmax_scale, keyT)
    if mask is not None:
        assert (
            mask.shape[-2:] == dots.shape[-2:]
        ), f"Mask shape {mask.shape} does not match attention shape {dots.shape}"
        inv_mask = (
            (~mask).unsqueeze(-3).expand_as(dots)
        )  # pylint: disable=invalid-unary-operand-type
        dots.masked_fill_(inv_mask, float("-inf"))

    attn = dots.softmax(dim=-1, dtype=torch.float).to(
        value.dtype
    )  # (batch, heads, out_seq_len, in_seq_len)
    if attention_dropout > 0:
        attn = F.dropout(attn, p=attention_dropout, training=training)

    y = torch.matmul(attn, value).transpose(
        1, 2
    )  # transpose to (batch, seq_len, heads, dim)
    return y, attn


class PlainAttention(nn.Module):
    """
    Attention module from original Transformer paper.
    """

    def __init__(
        self,
        model_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        context_dim = model_dim if context_dim is None else context_dim
        if head_dim is None:
            assert (
                model_dim % num_heads == 0
            ), f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
            head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.to_q = nn.Linear(model_dim, head_dim * num_heads, bias=False)
        self.to_kv = nn.Linear(context_dim, head_dim * num_heads * 2, bias=False)
        self.to_out = nn.Linear(head_dim * num_heads, model_dim)







    def forward(
        self, x, context=None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        :param x: [batch, seq_len, model_dim]
        :param context: [batch, context_len, context_dim]
        :param mask: [batch, seq_len, context_len]
        """

        context = x if context is None else context
        query = rearrange(
            self.to_q(x),
            "batch seq (head feature) -> batch seq head feature",
            head=self.num_heads,
        )
        key_value = rearrange(
            self.to_kv(context),
            "batch seq (n head feature) -> batch seq n head feature",
            head=self.num_heads,
            n=2,
        )
        y, attn = plain_attention(
            query=query,
            key_value=key_value,
            mask=mask,
            attention_dropout=self.attention_dropout,
            training=self.training,
        )
        y = self.to_out(y.flatten(-2))
        return y, attn
