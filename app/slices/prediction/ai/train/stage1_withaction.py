

# pylint: skip-file
import datetime
import os
import tempfile
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import torch.nn.functional as F
import re
import fire
import numpy as np
import torch
import torch.nn as nn
import transformers
from peft import get_peft_model_state_dict,PeftModel  # noqa: E402
from transformers import logging  # noqa: F402
import glob # For finding latest checkpoint
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers import AutoTokenizer



def _build_combined_dataset(pkl_path: str, max_size: Optional[int] = None) -> Dataset:
    """Read a *.pkl file and return a Hugging-Face Dataset with caption+action merged."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    records = []
    for d in raw:
        obs_dict = d["observation"]

        # Caption prompt for the observation
        caption = make_observation_prompt(obs_dict)
        # Last 4 lines of the original `input_prompt` are the control/action lines
        action = "\n".join(d["input_prompt"].split("\n")[-4:])
        combined_output = caption + action

        records.append(
            {
                "frame_num": d.get("frame_num", -1),
                "input": "",  # no additional input – matches original format
                "instruction": INSTRUCTION,
                "output": combined_output,
                # descriptors required by the model
                "route_descriptors": obs_dict["route_descriptors"],
                "vehicle_descriptors": obs_dict["vehicle_descriptors"],
                "pedestrian_descriptors": obs_dict["pedestrian_descriptors"],
                "ego_vehicle_descriptor": obs_dict["ego_vehicle_descriptor"],
            }
        )

        if max_size is not None and len(records) >= max_size:
            break

    return Dataset.from_list(records)


def get_train_val_data(
    data_path,
    tokenizer,
    val_data_path=None,
    val_set_size: int = 16,
    vqa: bool = False,
    eval_only: bool = False,
    eval_items=None,
    **_,
):
    """Return (train_ds, val_ds) with combined caption+action samples.

    This function mimics utils.training_utils.get_train_val_data but guarantees that each
    frame appears *once* with a fully-combined ground-truth string.
    """

    # Evaluation-only shortcut: just build val dataset and return None for training
    if eval_only:
        if val_data_path is None:
            raise ValueError("eval_only=True requires val_data_path")
        val_ds = _build_combined_dataset(val_data_path, max_size=val_set_size)
        val_ds = val_ds.map(
            partial(generate_and_tokenize_prompt, tokenizer, user_input_ids=True), remove_columns=[], num_proc=1
        )
        return None, val_ds

    # ---------------- Load training data ----------------
    train_raw = _build_combined_dataset(data_path)

    # ---------------- Load / build validation data ----------------
    if val_set_size > 0:
        if val_data_path is not None:
            val_raw = _build_combined_dataset(val_data_path, max_size=val_set_size)
        else:
            split = train_raw.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            train_raw, val_raw = split["train"], split["test"]

        # Tokenise datasets
        train_ds = train_raw.shuffle(seed=42).map(
            partial(generate_and_tokenize_prompt, tokenizer), remove_columns=[], num_proc=1
        )
        val_ds = val_raw.map(
            partial(generate_and_tokenize_prompt, tokenizer, user_input_ids=True), remove_columns=[], num_proc=1
        )
    else:
        train_ds = train_raw.shuffle(seed=42).map(
            partial(generate_and_tokenize_prompt, tokenizer), remove_columns=[], num_proc=1
        )
        val_ds = None

    return train_ds, val_ds


def eval_distance(all_pred, all_label, label_name, pattern):
    distance_errors = get_eval_distance_errors(all_pred, all_label, pattern)
    if distance_errors:
        mean_error = np.mean(distance_errors)
        print(f"{label_name}: Mean Absolute Error (MAE): {mean_error}, Total num: {len(distance_errors)}")









def load_tokenizer(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is defined
    return tokenizer
KAGGLE_OUTPUT_DIR = "/kaggle/working/models/weights/stage1_full_model/"


def get_latest_checkpoint(folder_path: str) -> Optional[str]:
    """Finds the latest checkpoint directory in a given folder."""
    if not os.path.isdir(folder_path):
        return None
    
    checkpoints = glob.glob(os.path.join(folder_path, "checkpoint-*"))
    if not checkpoints:
        return None
    
    # Sort by step number (extracted from directory name)
    try:
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.split("-")[-1]))
        return latest_checkpoint
    except ValueError: # Handle cases where checkpoint name is not as expected
        print(f"Warning: Could not parse step number from checkpoint names in {folder_path}")
        return None


class StopAfterFirstCheckpointCallback(TrainerCallback):
    """
    A TrainerCallback that stops training after the first checkpoint is saved
    during the current training run.
    """
    def __init__(self):
        super().__init__()
        self.first_checkpoint_saved_this_run = False
        self.initial_global_step_this_run = -1 # To track if we've moved past resumed state

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Record the global step when this training run actually starts a new step
        if self.initial_global_step_this_run == -1 and state.global_step > 0 : # state.global_step > 0 indicates actual training started
            self.initial_global_step_this_run = state.global_step

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        if not self.first_checkpoint_saved_this_run and \
           (self.initial_global_step_this_run != -1 and state.global_step >= self.initial_global_step_this_run):


            print(f"DEBUG: on_save called. global_step: {state.global_step}, initial_global_step_this_run: {self.initial_global_step_this_run}")

            if state.global_step > 0 and (state.global_step % args.save_steps == 0 or args.save_steps == 1):  # Check if it's a regular save step
                print(f"First checkpoint saved at step {state.global_step} during this training run. Stopping training.")
                self.first_checkpoint_saved_this_run = True
                control.should_training_stop = True


class TrainerWithGeneration(transformers.Seq2SeqTrainer):
    """
    Custom Trainer that computes and logs detailed perception and action metrics
    during the evaluation loop.
    """






    #     return loss


    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Overrides the default evaluation loop to add custom metric calculations.
        """
        # 1. Run the default evaluation loop to get predictions and base metrics (like eval_loss)
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=False,  # We need predictions
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # The rest of the logic should only run on the main process
        if self.args.process_index != 0:
            return eval_output

        # 2. Decode Predictions, Labels, and Prompts
        all_pred_tokens = []
        all_prompts = []  # To store decoded prompts
        for i, tokens in enumerate(eval_output.predictions):
            try:
                # Use the user_attention_mask to separate prompt from generation
                user_mask = torch.tensor(self.eval_dataset[i]["user_attention_mask"])
                prompt_len = int(torch.sum(user_mask))

                # Get prediction tokens (after prompt)
                pred_only_tokens = tokens[prompt_len:]
                all_pred_tokens.append(pred_only_tokens)

                prompt_tokens = self.eval_dataset[i]["input_ids"][:prompt_len]
                decoded_prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                all_prompts.append(decoded_prompt)

            except (KeyError, IndexError):
                # Fallback if the mask isn't available, might include prompt remnants
                all_pred_tokens.append(tokens)
                all_prompts.append("[Error: Could not decode prompt due to missing user_attention_mask]")

        # Decode the cleaned predictions and the ground truth labels
        all_pred = decode_generation_seqeunces(self.tokenizer, all_pred_tokens)
        all_label = decode_generation_seqeunces(self.tokenizer, eval_output.label_ids)

        if len(all_prompts) != len(all_pred):
            all_prompts.extend(["[Error: Mismatched length]"] * (len(all_pred) - len(all_prompts)))

        qa_results = [{"id": i, "prompt": prompt, "prediction": p, "ground_truth": l} for i, (prompt, p, l) in enumerate(zip(all_prompts, all_pred, all_label))]


        output_qa_file_path = Path(self.args.output_dir) / f"{metric_key_prefix}_predictions.json"
        with open(output_qa_file_path, "w", encoding="utf-8") as f:
            json.dump(qa_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved full evaluation inputs and predictions to {output_qa_file_path}")

        # 4. Compute and Aggregate Custom Metrics
        print("\n--- Custom Evaluation Metrics ---")
        
        # Initialize metrics dict with base metrics from the trainer (e.g., eval_loss)
        metrics = eval_output.metrics.copy()

        # A. Traffic Light Metrics
        tl_accuracy = eval_tl(all_pred, all_label)
        if tl_accuracy is not None:
            metrics[f"{metric_key_prefix}_tl_accuracy"] = tl_accuracy
            print(f"Traffic Light Accuracy: {tl_accuracy:.4f}")
        
        tl_dist_errors = get_eval_distance_errors(all_pred, all_label, r"It is (\d+(?:\.\d+)?)m ahead")
        if tl_dist_errors:
            tl_dist_mae = float(np.mean(tl_dist_errors))
            metrics[f"{metric_key_prefix}_tl_distance_mae"] = tl_dist_mae
            print(f"Traffic Light Distance MAE: {tl_dist_mae:.4f} (on {len(tl_dist_errors)} samples)")

        # B. Perception Metrics (Vehicle and Pedestrian Count)
        car_errors = get_eval_distance_errors(all_pred, all_label, r"observing (\d+) cars")
        if car_errors:
            car_count_mae = float(np.mean(car_errors))
            metrics[f"{metric_key_prefix}_vehicle_count_mae"] = car_count_mae
            print(f"Vehicle Count MAE: {car_count_mae:.4f} (on {len(car_errors)} samples)")

        ped_errors = get_eval_distance_errors(all_pred, all_label, r"and (\d+) pedestrians")
        if ped_errors:
            ped_count_mae = float(np.mean(ped_errors))
            metrics[f"{metric_key_prefix}_pedestrian_count_mae"] = ped_count_mae
            print(f"Pedestrian Count MAE: {ped_count_mae:.4f} (on {len(ped_errors)} samples)")

        # C. Action Metrics (Control Errors)
        lon_error, lat_error = eval_action(all_pred, all_label)
        if lon_error is not None and lat_error is not None:
            metrics[f"{metric_key_prefix}_lon_control_error"] = lon_error
            metrics[f"{metric_key_prefix}_lat_control_error"] = lat_error
            print(f"Longitudinal Control Error: {lon_error:.4f}")
            print(f"Lateral Control Error: {lat_error:.4f}")
        
        print("---------------------------------\n")

        self.log(metrics)
        eval_output.metrics.update(metrics)
        
        return eval_output

    

    
    
    
    
    
    #     return (weighted_loss, outputs) if return_outputs else weighted_loss


def eval_distance(all_pred, all_label, label_name, pattern):
    distance_errors = get_eval_distance_errors(all_pred, all_label, pattern)
    if len(distance_errors) > 0:
        mean_error = np.mean(distance_errors)
        print(
            f"{label_name}: Mean Absolute Error (MAE): {mean_error}, Total num: {len(distance_errors)}"
        )


def train(
    # model/data params
    base_model: str = "deepseek-ai/deepseek-coder-1.3b-base",  # the only required argument
    data_path: str = "/kaggle/input/training-dataset/vqa_train_10k.pkl",
    # training hyperparams
    batch_size: int = 16,
    micro_batch_size: int = 4,
    num_epochs: int = 4,
    learning_rate: float = 3e-4,
    val_set_size: int = 1e6,
    eval_steps: int = 10,
    # lora hyperparams
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    lora_target_modules: Tuple = ("q_proj", "v_proj"),
    group_by_length: bool = False,
    resume_from_checkpoint: Optional[str] = None,  # Changed to None as default
    augment_times: int = 0,
    output_dir: Optional[str] = "/kaggle/working/models/weights/stage1_full_model/",
    vqa: bool = False,
    eval_items: List[str] = DEFAULT_EVAL_ITEMS,
    mode: str = "train",
    load_pre_prompt_dataset: bool = False,
    val_data_path: str = "/kaggle/input/validation-dataset/vqa_valid_500.pkl",
):
    if output_dir is None:
        os.makedirs(output_dir, exist_ok=True)
    if mode == "eval":
        transformers.set_seed(42)

    # set DDP flags
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print("Training Alpaca-LoRA model with params:")
        for k in [
            "base_model",
            "data_path",
            "output_dir",
            "batch_size",
            "micro_batch_size",
            "num_epochs",
            "learning_rate",
            "val_set_size",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "group_by_length",
            "resume_from_checkpoint",
            "mode",
            "eval_items",
        ]:
            print(f"    {k}={eval(k)}")

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    model = load_model(
        base_model=base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_data, val_data = get_train_val_data(
        data_path,
        tokenizer,
        val_data_path=val_data_path,
        val_set_size=8,
        augment_times=augment_times,
        # load_pre_prompt_dataset=load_pre_prompt_dataset,
        vqa=vqa,
        eval_only=mode == "eval",
        eval_items=eval_items,
    )

    training_args = transformers.Seq2SeqTrainingArguments( # Assign to a variable first
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_strategy="steps",
        logging_steps=10,
       
        do_eval=True if val_data is not None else False,
        eval_strategy="steps", # CHANGED: Evaluate by steps
        eval_steps=625  if val_data is not None else None, # Frequency for both eval and save if strategies match

        save_strategy="steps",          # Valid for Seq2SeqTrainingArguments
        save_steps=1250,                 # Save a checkpoint every 500 steps (adjust)
        # save_total_limit=3,
        load_best_model_at_end=True if val_data is not None else False,
        metric_for_best_model="loss", # Or another metric like "accuracy", "rouge" etc.
        greater_is_better=False,      # Set to True if metric_for_best_model is like accuracy/ROUGE
                        
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=True,
        label_names=[
            "route_descriptors", "vehicle_descriptors", "pedestrian_descriptors",
            "ego_vehicle_descriptor", "user_input_ids", "user_attention_mask"
        ],
        prediction_loss_only=False,
        predict_with_generate=True,
        generation_max_length=512,
        generation_config=model.generation_config,
    )

    data_collator_instance = transformers.DataCollatorForSeq2Seq( # Assign to a variable
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    trainer = TrainerWithGeneration( # Now call the constructor
        model=model,
        args=training_args, # Pass the TrainingArguments object
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator_instance, # Pass the DataCollator object
        tokenizer=tokenizer, # Pass tokenizer if TrainerWithGeneration needs it directly (it does)
        # vqa=vqa,
        callbacks=[]
    )

    logging.set_verbosity_info()
    if mode == "train":
        # Check for existing checkpoints in the output directory
        checkpoint_dir = output_dir # This is self.args.output_dir for the Trainer
        determined_resume_path = None # Initialize


        trainer.train(resume_from_checkpoint=determined_resume_path) # Pass the variable

        if local_rank == 0:
            adapter_dir = os.path.join(output_dir, "lora_adapter")
            os.makedirs(adapter_dir, exist_ok=True)
        
            trainer.model.save_pretrained(adapter_dir)
            print(f"LoRA adapter saved to  {adapter_dir}")

        
            # ---- 2.  (optional) also save your two custom modules ---
            torch.save(model.vector_encoder.state_dict(),
                    os.path.join(adapter_dir, "vector_encoder.pth"))
            torch.save(model.llm_proj.state_dict(),
                    os.path.join(adapter_dir, "llm_proj.pth"))
            print("vector_encoder & llm_proj weights saved.")
        # --- END OF ADDED BLOCK ---
    elif mode == "eval":
        outputs = trainer.evaluate()
        print(outputs)

if __name__ == "__main__":
    import time
    import fire
    st = time.time()
    fire.Fire(lambda: train(base_model="deepseek-ai/deepseek-coder-1.3b-base"))
    print("Total time:", time.time() - st)
