from typing import Optional, Tuple

import torch
from torch import Tensor, nn

# Assuming attention_core.py is in the same directory or accessible via python path
from shared.attention_core.attention import Attention


def make_ffn(dim, mult=4):
    return nn.Sequential(
        nn.Linear(dim, dim * mult, bias=False),
        nn.GELU(),
        nn.Linear(dim * mult, dim, bias=False),
    )


def generate_square_subsequent_mask(
    num_queries: int, num_keys: int, device: torch.device = torch.device("cpu"), diagonal: int = 0
) -> torch.Tensor:
    """
    Generate the attention mask for causal decoding.
    Returns a mask where True means "attend" and False means "ignore".
    """
    mask = torch.ones(num_queries, num_keys, dtype=torch.bool, device=device)
    effective_diagonal = diagonal + max(0, num_keys - num_queries)
    return torch.tril(mask, diagonal=effective_diagonal)


class TransformerBlock(nn.Module):
    """
    A transformer block with pre-normalization.
    """

    def __init__(
        self,
        model_dim: int,
        attention: Attention,
        context_dim: Optional[int] = None, # This is the dim of the 'context' input to attention
        extra_context_norm: bool = False,
    ):
        super().__init__()
        # If context_dim is not given, assume it's same as model_dim (for self-attention or when context has same dim as x)
        actual_context_dim = context_dim if context_dim is not None else model_dim
        
        self.attention = attention
        self.ff = make_ffn(model_dim)
        self.pre_norm1 = nn.LayerNorm(model_dim) # For x (query)
        # For context (key/value), if extra_context_norm is True
        self.pre_norm2 = nn.LayerNorm(actual_context_dim) if extra_context_norm else None
        self.pre_norm3 = nn.LayerNorm(model_dim) # For the output of FFN path

    def forward(self, x: Tensor, context: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        is_self_attention = context is None or torch.equal(x, context) # More robust check for self-attention
        context_to_use = x if is_self_attention else context
        
        normed_x = self.pre_norm1(x)

        if self.pre_norm2 is not None: # extra_context_norm is True
            normed_context = self.pre_norm2(context_to_use)
            y, attn = self.attention.forward(normed_x, context=normed_context, mask=mask)
        elif not is_self_attention: # Cross-attention, extra_context_norm is False
            if context_to_use.shape[-1] == x.shape[-1]: # Context dim same as x dim
                normed_context = self.pre_norm1(context_to_use)
            else: # Context dim is different, and no pre_norm2 specified.
                normed_context = context_to_use
            y, attn = self.attention.forward(normed_x, context=normed_context, mask=mask)
        else: # Self-attention, extra_context_norm is False
            y, attn = self.attention.forward(normed_x, context=normed_x, mask=mask)
            
        x = x + y
        x = x + self.ff(self.pre_norm3(x))
        return x, attn