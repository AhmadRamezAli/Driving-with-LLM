from typing import Optional, Tuple

import torch
from torch import Tensor, nn

# Assuming other .py files are in the same directory or accessible via python path
from shared.attention_core.plain_attention import PlainAttention
from shared.transformer_components import TransformerBlock, generate_square_subsequent_mask, make_ffn



class Transformer(nn.Module):
    """
    A self attention transformer like GPT or BERT
    """

    def __init__(
        self,
        model_dim: int,
        depth: int,
        heads: int = 8,
        attention_dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.causal = causal
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim,
                    PlainAttention(model_dim=model_dim, context_dim=model_dim, num_heads=heads, attention_dropout=attention_dropout),
                    context_dim=model_dim, # For self-attention, context_dim is model_dim
                    extra_context_norm=False 
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self, token: Tensor, state: Optional[Tensor] = None # state: [B, prev_S, D, D_model]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            token: [batch, model_dim] or [batch, current_seq_len, model_dim]
            state: [batch, prev_seq_len, depth, model_dim] (KV cache from previous steps for each layer)
        Returns:
            x_output: [batch, model_dim] or [batch, current_seq_len, model_dim]
            new_combined_kv_state: [batch, total_seq_len, depth, model_dim]
            aggregated_attention_maps: [batch, total_seq_len] (sum of attention weights received by each key token)
        """
        assert token.shape[-1] == self.model_dim
        x_current = token.unsqueeze(1) if token.dim() == 2 else token # Ensure [B, S_cur, D]
        
        current_seq_len = x_current.shape[1]
        prev_seq_len = state.shape[1] if state is not None else 0
        total_seq_len = prev_seq_len + current_seq_len

        effective_mask = None
        if self.causal:
            mask_slice = generate_square_subsequent_mask(
                num_queries=current_seq_len, 
                num_keys=total_seq_len, 
                device=x_current.device,
                diagonal=prev_seq_len # Ensures causality with KV cache
            ) # Shape (S_cur, S_total)
            effective_mask = mask_slice.unsqueeze(0).expand(x_current.shape[0], -1, -1) # (B, S_cur, S_total)


        # Store KVs for the new state. Each element is [B, total_S, D_model]
        new_kv_state_layers = [] 
        
        aggregated_attention_maps = torch.zeros(
            (x_current.shape[0], total_seq_len), device=x_current.device, dtype=x_current.dtype
        )
        
        # Input to the first block is x_current
        current_x_for_block = x_current 

        for i, block in enumerate(self.blocks):
            
            
            
            
            keys_values_source = current_x_for_block # K,V from current tokens if no state
            if state is not None:
                past_kv_material_layer_i = state[:, :, i, :] # (B, prev_S, D)
                # K,V will be computed from the concatenation of past KVs and current token features
                keys_values_source = torch.cat((past_kv_material_layer_i, current_x_for_block), dim=1)
            
            current_x_for_block, attn_map = block(current_x_for_block, context=keys_values_source, mask=effective_mask)
            # attn_map: (B, H, S_cur, S_total)
            
            new_kv_state_layers.append(keys_values_source) 
            
            if attn_map is not None:
                 aggregated_attention_maps += attn_map.sum(dim=(1, 2))


        x_output = self.output_norm(current_x_for_block)
        x_output = x_output.squeeze(1) if token.dim() == 2 and x_output.shape[1] == 1 else x_output
        
        new_combined_kv_state = torch.stack(new_kv_state_layers, dim=2) if new_kv_state_layers else None
        
        return x_output, new_combined_kv_state, aggregated_attention_maps
