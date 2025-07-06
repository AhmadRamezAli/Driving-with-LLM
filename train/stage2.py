# Final merged training script using DrivingWithLLM and TrainerWithGeneration
import os
import datetime
import tempfile
from typing import List, Optional, Tuple

import fire
import torch
import transformers
import numpy as np
import wandb
from transformers import logging
from peft import get_peft_model_state_dict

from model import DrivingWithLLM
from utils.training_utils import (
    DEFAULT_EVAL_ITEMS,
    decode_generation_seqeunces,
    eval_action,
    eval_tl,
    get_eval_distance_errors,
    get_train_val_data,
    log_txt_as_img,
)


def freeze_llm_weights(model, percent_to_train=0.1):
    total_params = sum(p.numel() for p in model.llm.parameters())
    trainable_params = int(total_params * percent_to_train)

    params = list(model.llm.named_parameters())
    count = 0
    for name, param in reversed(params):
        if count >= trainable_params:
            param.requires_grad = False
        else:
            param.requires_grad = True
            count += param.numel()


class TrainerWithGeneration(transformers.Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.vqa = kwargs.pop("vqa", False)
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs["data_collator"].tokenizer

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        prediction_loss_only = False
        eval_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        all_pred_tokens = eval_output.predictions if self.vqa else eval_output.predictions[:, 77:]
        all_pred = decode_generation_seqeunces(self.tokenizer, all_pred_tokens)
        all_label = decode_generation_seqeunces(self.tokenizer, eval_output.label_ids)

        if self.args.process_index != 0:
            return eval_output

        if wandb.run is None:
            self.log({"i": None})
        images = log_txt_as_img((512, 512), [all_pred[0], all_label[0]])
        wandb.log({"val_logits": wandb.Image(np.concatenate(images, axis=1))})
        wandb.log({"val_results": wandb.Table(columns=["pred", "label"], data=[list(pair) for pair in zip(all_pred, all_label)])})

        tl_accuracy = eval_tl(all_pred, all_label)
        if tl_accuracy is not None:
            print(f"TL accuracy: {tl_accuracy}")
        wandb.log({"tl_accuracy": tl_accuracy})

        eval_distance(all_pred, all_label, "tl_distance", r"It is (\d+(?:\.\d+)?)m ahead")
        eval_distance(all_pred, all_label, "car_error", r"observing (\d+(?:\.\d+)?) cars")
        eval_distance(all_pred, all_label, "ped_error", r"and (\d+(?:\.\d+)?) pedestrians")

        lon_err, lat_err = eval_action(all_pred, all_label)
        if lon_err is not None and lat_err is not None:
            print(f"Control error: {lon_err}, {lat_err}")
            wandb.log({"control_error_lon": lon_err, "control_error_lat": lat_err})

        return eval_output


def eval_distance(all_pred, all_label, label_name, pattern):
    errors = get_eval_distance_errors(all_pred, all_label, pattern)
    if errors:
        mean_error = np.mean(errors)
        print(f"{label_name} MAE: {mean_error:.2f} (N={len(errors)})")
        wandb.log({label_name: mean_error})


def train(
    base_model: str = "deepseek-ai/deepseek-coder-1.3b-base",
    data_path: str = "data/train.pkl",
    val_data_path: str = "data/val.pkl",
    output_dir: Optional[str] = None,
    batch_size: int = 8,
    micro_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    val_set_size: int = 500,
    eval_steps: int = 20,
    freeze_percent: float = 0.9,
    wandb_project: str = "llm-driver",
    wandb_run_name: str = "stage2_final",
    resume_from_checkpoint: str = "checkpoints/stage1/model_epoch3.pt",
    mode: str = "train",
    vqa: bool = False,
    eval_items: List[str] = DEFAULT_EVAL_ITEMS,
):
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="llmdriver_")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    if local_rank == 0:
        print("Running training with:")
        print(f"Base model: {base_model}, Output: {output_dir}")

    gradient_accumulation_steps = batch_size // micro_batch_size

    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_WATCH"] = "false"
        os.environ["WANDB_LOG_MODEL"] = "true"

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = DrivingWithLLM(base_model=base_model).to("cuda")
    model.load_state_dict(torch.load(resume_from_checkpoint, map_location="cuda"), strict=False)
    freeze_llm_weights(model, percent_to_train=(1.0 - freeze_percent))

    model.print_trainable_parameters = lambda: print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.print_trainable_parameters()

    train_data, val_data = get_train_val_data(
        data_path,
        tokenizer,
        val_data_path=val_data_path,
        val_set_size=val_set_size,
        add_input_prompt=True,
        eval_only=mode == "eval",
        eval_items=eval_items,
        vqa=vqa,
    )

    trainer = TrainerWithGeneration(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            logging_steps=5,
            save_steps=999_999_999,  # effectively disables mid-training checkpoint saving
            eval_steps=999_999_999,  # disables mid-training evaluation
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            output_dir=output_dir,
            save_total_limit=2,
            evaluation_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            fp16=True,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb",
            run_name=wandb_run_name,
            label_names=[
                "route_descriptors",
                "vehicle_descriptors",
                "pedestrian_descriptors",
                "ego_vehicle_descriptor",
                "user_input_ids",
                "user_attention_mask",
            ],
            predict_with_generate=True,
            generation_max_length=384,
            generation_config=model.generation_config,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        vqa=vqa,
    )

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))
    logging.set_verbosity_info()

    if mode == "train":
        trainer.train()
        if local_rank == 0:
            print("Model saved to:", output_dir)
            model.save_pretrained(output_dir)
            print("Running final evaluation...")
            outputs = trainer.evaluate()
            print(outputs)
    elif mode == "eval":
        outputs = trainer.evaluate()
        print(outputs)


if __name__ == "__main__":
    fire.Fire(train)
