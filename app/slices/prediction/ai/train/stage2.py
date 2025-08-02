##### pylint: skip-file
import datetime
import os
import tempfile
import json
from pathlib import Path
from typing import List, Optional, Tuple
from transformers import Seq2SeqTrainingArguments

import fire
import numpy as np
import transformers
from peft import get_peft_model_state_dict  # noqa: E402
from transformers import logging  # noqa: F402
import glob
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers import AutoTokenizer
import torch

class StopAfterStepsCallback(TrainerCallback):
    def __init__(self, stop_after_steps: int, output_dir: str, model, vector_encoder, llm_proj):
        self.stop_after_steps = stop_after_steps
        self.output_dir = output_dir
        self.model = model
        self.vector_encoder = vector_encoder
        self.llm_proj = llm_proj

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step >= self.stop_after_steps:
            print(f"💾 Saving model at step {state.global_step} before stopping.")
            self.model.save_pretrained(self.output_dir)
            torch.save(self.vector_encoder.state_dict(), os.path.join(self.output_dir, "vector_encoder.pth"))
            torch.save(self.llm_proj.state_dict(), os.path.join(self.output_dir, "llm_proj.pth"))
            print(f"⛔ Stopping training at step {state.global_step}")
            control.should_training_stop = True
        return control

KAGGLE_OUTPUT_DIR = "/kaggle/working/models/weights/stage2_finetuned/"

def load_tokenizer(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_latest_checkpoint(folder_path: str) -> Optional[str]:
    if not os.path.isdir(folder_path):
        return None
    checkpoints = glob.glob(os.path.join(folder_path, "checkpoint-*"))
    if not checkpoints:
        return None
    try:
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.split("-")[-1]))
        return latest_checkpoint
    except ValueError:
        print(f"Warning: Could not parse step number from checkpoint names in {folder_path}")
        return None

class TrainerWithGeneration(transformers.Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.vqa = kwargs.pop("vqa", False)
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs["data_collator"].tokenizer

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        prediction_loss_only = False
        eval_output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        all_pred_tokens = [tokens[int(torch.sum(torch.tensor(self.eval_dataset[i]["user_attention_mask"]))) :] for i, tokens in enumerate(eval_output.predictions)]
        all_pred = [
            self.tokenizer.decode([t for t in pred if t not in [-100, self.tokenizer.pad_token_id]], skip_special_tokens=True).strip()
            for pred in all_pred_tokens
        ]
        all_label = decode_generation_seqeunces(self.tokenizer, eval_output.label_ids)
        qa_results = []
        for i in range(len(all_pred)):
            # Rebuild input prompt
            prompt = make_observation_prompt(
                {
                    "route_descriptors": self.eval_dataset[i]["route_descriptors"],
                    "vehicle_descriptors": self.eval_dataset[i]["vehicle_descriptors"],
                    "pedestrian_descriptors": self.eval_dataset[i]["pedestrian_descriptors"],
                    "ego_vehicle_descriptor": self.eval_dataset[i]["ego_vehicle_descriptor"]
                }
            )
            instruction = self.tokenizer.decode(self.eval_dataset[i]["user_input_ids"], skip_special_tokens=True)
            full_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{prompt}
### Response:"""

            qa_results.append({
                "id": i,
                "prompt": full_prompt,
                "prediction": all_pred[i],
                "ground_truth": all_label[i],
            })

        output_path = os.path.join(self.args.output_dir, f"{metric_key_prefix}_qa_predictions.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(qa_results, f, indent=4, ensure_ascii=False)
        print(f"📝 Saved evaluation predictions to {output_path}")

        return eval_output

def eval_distance(all_pred, all_label, label_name, pattern):
    distance_errors = get_eval_distance_errors(all_pred, all_label, pattern)
    if distance_errors:
        mean_error = np.mean(distance_errors)
        print(f"{label_name}: Mean Absolute Error (MAE): {mean_error}, Total num: {len(distance_errors)}")

def train(
    base_model: str = "deepseek-ai/deepseek-coder-1.3b-base",
    data_path: str = "/kaggle/input/training-dataset/vqa_train_10k.pkl",
    val_data_path: str = "/kaggle/input/validation-dataset/vqa_valid_500.pkl",
    output_dir: Optional[str] = KAGGLE_OUTPUT_DIR,
    batch_size: int = 32,
    micro_batch_size: int = 2,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    val_set_size: int = 32,
    resume_from_checkpoint: Optional[str] = "/kaggle/input/stage2finetune/stage2 finetune/checkpoint-2000",
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    lora_target_modules: Tuple = ("q_proj", "v_proj"),
    vqa: bool = True,
    eval_items: List[str] = ["vqa", "caption", "action"],
    mode: str = "train",
    generation_max_length: int = 128,
):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    model = load_model(
        base_model=base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    if resume_from_checkpoint:
        base_ckpt_dir = os.path.dirname(resume_from_checkpoint)
        vector_encoder_path = os.path.join(base_ckpt_dir, "vector_encoder.pth")
        llm_proj_path = os.path.join(base_ckpt_dir, "llm_proj.pth")

        if os.path.exists(vector_encoder_path):
            print(f"✅ Loading vector_encoder weights from {vector_encoder_path}")
            model.vector_encoder.load_state_dict(torch.load(vector_encoder_path, map_location="cpu"))
        else:
            print(f"❌ vector_encoder weights not found at {vector_encoder_path}")

        if os.path.exists(llm_proj_path):
            print(f"✅ Loading llm_proj weights from {llm_proj_path}")
            model.llm_proj.load_state_dict(torch.load(llm_proj_path, map_location="cpu"))
        else:
            print(f"❌ llm_proj weights not found at {llm_proj_path}")

    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    tokenizer = load_tokenizer(base_model)
    train_data, val_data = get_train_val_data(
        data_path,
        tokenizer,
        val_data_path=val_data_path,
        val_set_size=val_set_size,
        vqa=vqa,
        eval_only=mode == "eval",
        eval_items=eval_items,
    )

    stop_step = 1500  # ⏹️ Stop after 3000 steps

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=batch_size // micro_batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_strategy="steps",
        logging_steps=500,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=stop_step,
        save_strategy="steps",
        save_steps=stop_step,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=True,
        label_names=[
            "route_descriptors", "vehicle_descriptors", "pedestrian_descriptors",
            "ego_vehicle_descriptor", "user_input_ids", "user_attention_mask"
        ],
        prediction_loss_only=False,
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        generation_config=model.generation_config,
    )

    callbacks = [StopAfterStepsCallback(stop_after_steps=stop_step, output_dir=output_dir, model=model, vector_encoder=model.vector_encoder, llm_proj=model.llm_proj)]

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    trainer = TrainerWithGeneration(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        vqa=vqa,
        callbacks=callbacks,
    )


    logging.set_verbosity_info()
    if mode == "train":
        trainer.train()
        model.save_pretrained(output_dir)
        torch.save(model.vector_encoder.state_dict(), os.path.join(output_dir, "vector_encoder.pth"))
        torch.save(model.llm_proj.state_dict(), os.path.join(output_dir, "llm_proj.pth"))

    elif mode == "eval":
        print("📦 Saving model before evaluation...")

        outputs = trainer.evaluate()
        print(outputs)

if __name__ == "__main__":
    import time
    st = time.time()
    fire.Fire(train)
    print("Total time:", time.time() - st)
