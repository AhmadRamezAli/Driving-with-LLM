from typing import Optional, Tuple

import torch
from torch import Tensor, nn

# Assuming other .py files are in the same directory or accessible via python path
from shared.attention_core.plain_attention import PlainAttention
from shared.transformer_components import TransformerBlock, generate_square_subsequent_mask, make_ffn




class CrossPerceiver(nn.Module):
    """
    A residual MLP interleaved with cross attention.
    Query 'x' is a single vector [B,D], context is [B,C_len,C_dim].
    """

    def __init__(self, model_dim: int, context_dim: int, num_blocks=5, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim=model_dim, # Query dim
                    attention=PlainAttention(
                        model_dim=model_dim,      # Query dim for to_q
                        context_dim=context_dim,  # Context dim for to_kv
                        num_heads=num_heads,
                    ),
                    context_dim=context_dim, # Pass context_dim for pre_norm2
                    extra_context_norm=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, context: Tensor, context_mask: Optional[Tensor]):
        
        query_input = x.unsqueeze(1)  # [b, 1, model_dim] (queries)
        
        # Mask for attention: [b, num_queries=1, context_len]
        attn_mask = context_mask.unsqueeze(1) if context_mask is not None else None

        aggregated_attention_maps = torch.zeros(
            (context.shape[0], context.shape[1]), device=context.device, dtype=context.dtype
        )
        
        current_query = query_input
        for block in self.blocks:
            current_query, attn_map = block(current_query, context=context, mask=attn_mask) 
            # attn_map: (B, H, num_queries=1, context_len)
            if attn_map is not None:
                # Sum over heads (dim 1) and query dimension (dim 2, which is 1)
                aggregated_attention_maps += attn_map.sum(dim=(1, 2))

        output_query = self.output_norm(current_query).squeeze(1)  # [b, model_dim]
        return output_query, aggregated_attention_maps

