import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.vector_encoder import VectorEncoder, VectorEncoderConfig
from shared.transformer_models.Transformer import Transformer
from utils.vector_utils.randomize_utils import VectorObservation, VectorObservationConfig


class VectorFormer(nn.Module):
    def __init__(self, model_dim=256, depth=4, heads=8):
        super().__init__()
        self.transformer = Transformer(model_dim, depth=depth, heads=heads, causal=False)

    def forward(self, x):
        return self.transformer(x)[0]


class DrivingWithLLM(nn.Module):
    def __init__(self, base_model="deepseek-ai/deepseek-coder-1.3b-base", lora_r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        # 1. Token encoder
        self.encoder_config = VectorEncoderConfig()
        self.vector_encoder = VectorEncoder(
            encoder_config=self.encoder_config,
            observation_config=VectorObservationConfig(),
            num_queries=64,
        )

        # 2. VectorFormer
        self.vector_former = VectorFormer(
            model_dim=self.encoder_config.model_dim,
            depth=4,
            heads=self.encoder_config.num_heads,
        )

        # 3. Load pretrained LLM + LoRA
        self.llm = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float16, device_map="auto"
        )

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, self.lora_config)

        # 4. Projection to LLM embedding space
        self.proj = nn.Linear(self.encoder_config.model_dim, self.llm.config.hidden_size)

    def forward(self, input_ids, attention_mask, route_descriptors, vehicle_descriptors, pedestrian_descriptors, ego_vehicle_descriptor, labels=None):
        # Encode vector observation
        vector_obs = VectorObservation(
            route_descriptors=route_descriptors,
            vehicle_descriptors=vehicle_descriptors,
            pedestrian_descriptors=pedestrian_descriptors,
            ego_vehicle_descriptor=ego_vehicle_descriptor,
        )

        vec_embeds = self.vector_encoder(vector_obs)
        vec_embeds = self.vector_former(vec_embeds)
        vec_embeds = self.proj(vec_embeds)

        # Token embeddings
        tok_embeds = self.llm.model.embed_tokens(input_ids)

        # Concatenate vector + token embeddings
        inputs_embeds = torch.cat([vec_embeds, tok_embeds], dim=1)

        # Adjust attention mask
        extended_attention_mask = torch.cat(
            [torch.ones(vec_embeds.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device), attention_mask],
            dim=1,
        )

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=labels,
        )
