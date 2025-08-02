import os
from typing import Tuple
from peft import TaskType

import torch
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict, PeftModel # Added PeftModel
from transformers import GenerationConfig, LlamaTokenizer

from app.slices.prediction.ai.models.vector_lm import LlamaForCausalLMVectorInput, VectorLMWithLoRA

import os
from typing import Tuple, Optional # Added Optional

import torch
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict, PeftModel # Added PeftModel
from transformers import GenerationConfig, LlamaTokenizer

from app.slices.prediction.ai.models.vector_lm import LlamaForCausalLMVectorInput, VectorLMWithLoRA

from transformers import AutoTokenizer

def load_tokenizer(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is defined
    return tokenizer



def default_generation_config(**kwargs):
    return GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        use_cache=False, # Important for training with LoRA and vector inputs
        do_sample=True, # Often True for generation, but can be False for eval if deterministic needed
        max_length=150, # Or a more appropriate default
        pad_token_id=0, # Match tokenizer
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    )


def load_model(
    base_model: str = "deepseek-ai/deepseek-coder-1.3b-base",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
    resume_from_checkpoint: Optional[str] = None, # Explicitly Optional
    load_in_8bit: bool = False, # Changed default to False to match your provided code
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    if torch.cuda.is_available() and not ddp:
        target_device = "cuda:0"
        device_map = "auto"
    elif ddp:
        target_device = f"cuda:{local_rank}"
        device_map = {"": local_rank}
    else:
        target_device = "cpu"
        device_map = {"": target_device}
        print("WARNING: No CUDA GPU found. Loading model on CPU. This will be very slow.")

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    print(f"DEBUG: Attempting to load base model '{base_model}' with device_map: {device_map}")
    llama_model = LlamaForCausalLMVectorInput.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit, # This should be kbit for prepare_model_for_kbit_training
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, # Use float16 on GPU
        device_map=device_map,
    )
    print(f"DEBUG: Base llama_model loaded. Main device: {llama_model.device}")

    # No need to explicitly move with device_map="auto" usually, but good for verification
    if not load_in_8bit and device_map != "auto" and llama_model.device.type != target_device.split(':')[0]:
        try:
            print(f"DEBUG: Explicitly moving base model to {target_device}.")
            llama_model.to(target_device)
            print(f"DEBUG: Base llama_model moved. New main device: {llama_model.device}")
        except Exception as e:
            print(f"ERROR moving base model to {target_device}: {e}")


    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM", # Keep as CAUSAL_LM for Llama
    )

    if load_in_8bit:
        print("DEBUG: Preparing model for k-bit training (e.g., 8-bit, 4-bit).")
        llama_model = prepare_model_for_kbit_training(llama_model, use_gradient_checkpointing=True) # Or int8 specific
        # Then wrap with PeftModel (which VectorLMWithLoRA inherits from)
        model = VectorLMWithLoRA(llama_model, lora_config)

        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            print(f"DEBUG: Resuming from 8-bit checkpoint: {resume_from_checkpoint}")
            adapter_weights_path = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            if os.path.exists(adapter_weights_path):
                adapters_weights = torch.load(adapter_weights_path, map_location="cpu")
                set_peft_model_state_dict(model, adapters_weights)
                print(f"DEBUG: Loaded 8-bit adapter weights from {adapter_weights_path}")
            else:
                print(f"WARNING: adapter_model.bin not found in 8-bit checkpoint {resume_from_checkpoint}")
        else:
            print("DEBUG: Initializing new LoRA layers for 8-bit model.")
    else: # Not loading in k-bit (e.g. float16 or float32)
        print("DEBUG: Loading model in non-k-bit mode.")
        model = VectorLMWithLoRA(llama_model, lora_config) # Initialize with base model first

        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint) and os.path.isdir(resume_from_checkpoint):
            print(f"DEBUG: Attempting to load PEFT adapter and custom weights from directory: {resume_from_checkpoint}")
            
            try:
                # The model object holds the PeftConfig, so we can load weights into it
                model.load_adapter(resume_from_checkpoint, adapter_name="default")
                print(f"DEBUG: Successfully loaded PEFT adapter from {resume_from_checkpoint}")
                # If you have multiple adapters, you might need to manage them (e.g., set the active adapter)
                model.set_adapter("default")
            except Exception as e:
                print(f"ERROR: Could not load PEFT adapter from {resume_from_checkpoint}: {e}")
                print("DEBUG: Proceeding with base model weights for LoRA layers.")

            # 2. Load the custom module weights (vector_encoder and llm_proj)
            vector_encoder_path = os.path.join(resume_from_checkpoint, "vector_encoder.pth")
            llm_proj_path = os.path.join(resume_from_checkpoint, "llm_proj.pth")

            if os.path.exists(vector_encoder_path):
                print(f"DEBUG: Loading vector_encoder weights from {vector_encoder_path}")
                vec_weights = torch.load(vector_encoder_path, map_location=target_device)
                model.vector_encoder.load_state_dict(vec_weights)
            else:
                print(f"WARNING: vector_encoder.pth not found in checkpoint {resume_from_checkpoint}. It will be randomly initialized.")

            if os.path.exists(llm_proj_path):
                print(f"DEBUG: Loading llm_proj weights from {llm_proj_path}")
                proj_weights = torch.load(llm_proj_path, map_location=target_device)
                model.llm_proj.load_state_dict(proj_weights)
            else:
                print(f"WARNING: llm_proj.pth not found in checkpoint {resume_from_checkpoint}. It will be randomly initialized.")

        else:
            if resume_from_checkpoint:
                 print(f"DEBUG: resume_from_checkpoint '{resume_from_checkpoint}' not found or not a directory.")
            print("DEBUG: Initializing VectorLMWithLoRA with new LoRA layers and random custom weights (no checkpoint).")


    model.config.use_cache = False # Important for training

    if not ddp and torch.cuda.device_count() > 1:
        pass # Usually not needed with Trainer and device_map

    # Set generation config on the model instance
    model.generation_config = default_generation_config()
    
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'): # Standard PEFT structure
        underlying_hf_model = model.base_model.model
    elif hasattr(model, 'model'): # If VectorLMWithLoRA directly wraps LlamaForCausalLMVectorInput
         underlying_hf_model = model.model
    else:
        underlying_hf_model = model # Fallback, might not be correct

    if hasattr(underlying_hf_model, 'generation_config'):
        underlying_hf_model.generation_config = model.generation_config
    if hasattr(underlying_hf_model, 'config'):
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        underlying_hf_model.config.pad_token_id = tokenizer.pad_token_id
        underlying_hf_model.config.eos_token_id = 2
    else:
        print("WARNING: Could not set generation_config pad/bos/eos tokens on the underlying model.")


    print(f"DEBUG: Final model device: {next(model.parameters()).device}")
    return model