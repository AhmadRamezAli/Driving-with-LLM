from typing import Optional, Tuple

import torch
from torch import Tensor, nn

# Assuming other .py files are in the same directory or accessible via python path
from shared.attention_core.plain_attention import PlainAttention
from shared.transformer_components import TransformerBlock, generate_square_subsequent_mask, make_ffn


class Perceiver(nn.Module):
    """
    PERCEIVER IO: A GENERAL ARCHITECTURE FOR STRUCTURED INPUTS & OUTPUTS
    https://arxiv.org/abs/2107.14795
    """

    def __init__(
        self,
        model_dim: int, # Dimension of latents and output queries
        context_dim: int, # Dimension of input context data
        num_latents: int,
        num_blocks: int = 5,    # Number of self-attention blocks for latents
        num_heads: int = 8,     # Heads for latent self-attention
        num_queries: int = 1,   # Number of output queries
    ):
        super().__init__()
        self.num_latents = num_latents
        self.num_queries = num_queries
        
        self.input_latents_emb = nn.Parameter(torch.empty((num_latents, model_dim)))
        self.output_queries_emb = nn.Parameter(torch.empty((num_queries, model_dim)))
        
        # Optional projection for 'x' (e.g., ego_vehicle_feat) if provided to forward
        self.input_x_proj = nn.Linear(model_dim, model_dim, bias=False) # Assumes x has model_dim
        self.input_x_ff = nn.Sequential(nn.LayerNorm(model_dim), make_ffn(model_dim))

        # Cross-attention: input_latents (Q) attend to input context (K,V)
        self.input_cross_attn_block = TransformerBlock(
            model_dim=model_dim,        # Latent dim (for Q)
            attention=PlainAttention(
                model_dim=model_dim,    # Latent dim (for Q's projection)
                context_dim=context_dim,# Input context dim (for K,V's projection)
                num_heads=1             # Paper often uses 1 head for cross-attention
            ),
            context_dim=context_dim,    # For pre_norm2 on input context
            extra_context_norm=True,
        )
        
        # Latent self-attention tower
        self.latent_self_attn_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim, # Latent dim
                    PlainAttention(model_dim=model_dim, context_dim=model_dim, num_heads=num_heads),
                    context_dim=model_dim, # For self-attention, context is also latents
                    extra_context_norm=False # Standard self-attention block
                )
                for _ in range(num_blocks)
            ]
        )

        # Cross-attention: output_queries (Q) attend to processed latents (K,V)
        self.output_cross_attn_block = TransformerBlock(
            model_dim=model_dim,        # Output query dim (for Q)
            attention=PlainAttention(
                model_dim=model_dim,    # Output query dim (for Q's projection)
                context_dim=model_dim,  # Processed latent dim (for K,V's projection)
                num_heads=1
            ),
            context_dim=model_dim,      # For pre_norm2 on processed latents
            extra_context_norm=True,
        )
        
        self.output_norm = nn.LayerNorm(model_dim)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.input_latents_emb.data, std=0.02)
        nn.init.normal_(self.output_queries_emb.data, std=0.02)

    def forward(self, x: Optional[Tensor], context: Tensor, context_mask: Optional[Tensor] = None):
        # x: Optional initial features [B, D_model] or [B, N_x, D_model] to condition latents
        # context: Main input data [B, context_len, context_dim]
        # context_mask: Mask for context [B, context_len] (True means attend)

        batch_size = context.shape[0]
        
        # Initialize latents by expanding the learned embedding
        current_latents = self.input_latents_emb.unsqueeze(0).expand(batch_size, -1, -1) # B, num_latents, D_model

        if x is not None:
            projected_x = self.input_x_proj(x) # x is [B,D] or [B, N_x, D]
            if projected_x.dim() == 2: # [B, D] -> [B, 1, D]
                projected_x = projected_x.unsqueeze(1)
            
            if projected_x.shape[1] == 1: # Broadcast add if x is a single vector per batch item
                current_latents = current_latents + projected_x
            elif projected_x.shape[1] == self.num_latents: # Element-wise add if x matches num_latents
                current_latents = current_latents + projected_x
            else:
                raise ValueError(f"Shape of x projection {projected_x.shape} not compatible with latents {current_latents.shape} for addition.")
            current_latents = current_latents + self.input_x_ff(current_latents)

        # Input cross-attention: latents (Q) attend to context (K,V)
        # context_mask: (B, context_len) -> needs to be (B, num_latents, context_len) for block
        input_attn_mask_for_block = None
        if context_mask is not None:
            input_attn_mask_for_block = context_mask.unsqueeze(1).expand(-1, self.num_latents, -1)
        
        current_latents, input_cross_attn_map = self.input_cross_attn_block(
            current_latents, context=context, mask=input_attn_mask_for_block
        )
        # input_cross_attn_map: (B, H_cross_in=1, num_latents, context_len)

        # Latent self-attention tower
        # The original code's attention_maps logic was:
        # attention_maps += (attn @ input_attn).sum(dim=(-2, -3))
        # where 'attn' is from the *current* self-attention block.
        # This means it accumulates contributions from each self-attention block.
        # (B,H,L,L) @ (B,1,L,C) -> (B,H,L,C). Sum(-2,-3) -> (B,H). Added to (B,C) is error.
        # A more likely intent for (B,C) from (B,H,L,C) is .sum(dim=(1,2)) (sum over H and L_query_dim).
        
        # Let's trace attention from final output queries back to input context for a meaningful map.
        # For simplicity as in original, let's try to replicate the sum logic for (B,C)
        # Assuming the sum over (-2,-3) in original means summing over query_latent and key_latent/context_idx.
        # If attention_maps is (B,C), then (attn @ input_attn) should result in (B, H, C) or similar.
        # (B,H,L_q,L_k) @ (B,1,L_k,C) -> (B,H,L_q,C). Sum over H and L_q.
        
        # Replicating the original loop's accumulation idea:
        aggregated_attention_to_context = torch.zeros(
            (batch_size, context.shape[1]), device=context.device, dtype=context.dtype, requires_grad=False
        )

        # Effective map from current latents to original context. Initialize with input_cross_attn_map.
        # Squeeze head dim: (B, num_latents, context_len)
        effective_latent_to_context_map = input_cross_attn_map.squeeze(1)

        for block in self.latent_self_attn_blocks:
            current_latents, self_attn_map = block(current_latents) # self_attn_map: (B, H_self, L, L)
            
            # Original logic: attention_maps += (attn @ input_attn).sum(dim=(-2, -3))
            # attn = self_attn_map (B,H,L,L)
            # input_attn = input_cross_attn_map (B,1,L,C)
            # We need to make dimensions compatible for matmul and sum.
            # (B,H,L,L) @ (B,1,L,C) needs L_k of first to match L_q of second.
            # Let's assume input_attn is (B, L_k_of_self_attn, C) i.e. effective_latent_to_context_map
            
            # Composition: New_Latent_to_Context = Self_Attn @ Old_Latent_to_Context
            # self_attn_map (B, H, L, L) . Sum over H -> (B, L, L)
            # effective_latent_to_context_map (B, L, C)
            # (B,L,L) @ (B,L,C) -> (B,L,C)
            summed_self_attn_map_over_heads = self_attn_map.sum(dim=1) # B, L, L
            effective_latent_to_context_map = torch.matmul(summed_self_attn_map_over_heads, effective_latent_to_context_map)
            
            # The original sum `(attn @ input_attn).sum(dim=(-2, -3))`
            # If attn is self_attn_map (B,H,L,L) and input_attn is input_cross_attn_map (B,1,L,C)
            # Composite map: composed_map = torch.matmul(self_attn_map, input_cross_attn_map) -> (B,H,L,C)
            composed_map = torch.matmul(self_attn_map, input_cross_attn_map) # B,H,L,C
            # Original sum was over last two dims. If applied to (B,H,L,C), it sums over L and C. -> (B,H)
            # This was added to `attention_maps` (B,C). This is a dimensional mismatch.
            # Let's assume the sum should be over heads and the latent_query dimension to get (B,C)
            aggregated_attention_to_context += composed_map.sum(dim=(1,2)) # Sum over H and L (query latents)

        # Output cross-attention: output_queries (Q) attend to final processed latents (K,V)
        output_q_expanded = self.output_queries_emb.unsqueeze(0).expand(batch_size, -1, -1) # B, num_queries, D_model
        
        # Mask for output block (output_queries attend to all latents, so no mask usually needed for latents as context)
        y, output_cross_attn_map = self.output_cross_attn_block(output_q_expanded, context=current_latents)
        # output_cross_attn_map: (B, H_cross_out=1, num_queries, num_latents)
        
        y = self.output_norm(y)
        
        # If num_queries is 1, and input x was a single vector, output y should be [B,D_model]
        if self.num_queries == 1 and (x is None or x.dim() == 2) and y.shape[1] == 1:
             y = y.squeeze(1)

        return y, aggregated_attention_to_context