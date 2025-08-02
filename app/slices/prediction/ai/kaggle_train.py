

# pylint: skip-file
import datetime
import os
import tempfile
import json
from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt

import fire
import numpy as np
import transformers
from peft import get_peft_model_state_dict  # noqa: E402
from transformers import logging  # noqa: F402
import glob # For finding latest checkpoint
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
# import wandb

KAGGLE_OUTPUT_DIR = "/kaggle/working/models/"


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


KAGGLE_OUTPUT_DIR = "/kaggle/working/result"

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

            if state.global_step > 0 and (state.global_step % args.save_steps == 0 or args.save_steps == 1): # Check if it's a regular save step
                print(f"First checkpoint saved at step {state.global_step} during this training run. Stopping training.")
                self.first_checkpoint_saved_this_run = True
                control.should_training_stop = True
class TrainerWithGeneration(transformers.Seq2SeqTrainer):
    """
    Custom Trainer class for sequence-to-sequence model with additional functionalities.
    Inherits from transformers.Seq2SeqTrainer.
    """

    def __init__(self, *args, **kwargs):
        self.vqa = kwargs.pop("vqa", False)
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs["data_collator"].tokenizer

    
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """
        Overrided method to perform evaluation loop with custom eval and logging.
        """

        # ensure prediction loss is set to False
        prediction_loss_only = False

        # call parent class method to get the evaluation outputs
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        input_lengths = []
        for batch in dataloader:
            if 'user_input_ids' in batch:
                # The length is the number of non-padding tokens
                input_lengths.extend(torch.sum(batch['user_attention_mask'], dim=1).tolist())
            else:
                input_lengths.extend(torch.sum(batch['attention_mask'], dim=1).tolist())
        

        # Perform additional operations based on evaluation output
        all_pred_tokens = (
            eval_output.predictions if self.vqa else eval_output.predictions[:, 77:]
        )  # remove the prompt for easier comparison
        all_pred = []
        for i, tokens in enumerate(all_pred_tokens):
            # Get the length of the original prompt for this example
            prompt_len = input_lengths[i]
            
            # Slice the token sequence to get ONLY the newly generated tokens
            generated_only_tokens = tokens[prompt_len:]
            
        pad_token_id = self.tokenizer.pad_token_id
        
        # Create a list of valid tokens to decode
        tokens_to_decode = [
            tok for tok in generated_only_tokens if tok not in [-100, pad_token_id]
        ]
        
        # Decode the cleaned list of tokens
        decoded_text = self.tokenizer.decode(tokens_to_decode, skip_special_tokens=True)
        all_pred.append(decoded_text.strip()) # .strip() is good practice
        
        # all_pred = decode_generation_seqeunces(self.tokenizer, all_pred_tokens)
        all_label = decode_generation_seqeunces(self.tokenizer, eval_output.label_ids)
        print("all_pred", all_pred)
        print("all_label", all_label)

        # Log the predictions

        
        if self.args.process_index == 0:  # Only save from the main process
            # Save Q&A predictions
            qa_results = []
            
            # Get original evaluation data
            if hasattr(self.eval_dataset, "_data"):  # For Hugging Face Dataset objects
                original_eval_data = self.eval_dataset._data.to_pylist()
            elif isinstance(self.eval_dataset, list):
                original_eval_data = self.eval_dataset
            else:  # Fallback
                print("Warning: Could not directly access original eval data for questions. QA output might be incomplete.")
                original_eval_data = [{"instruction": "UNKNOWN_QUESTION", "output": "UNKNOWN_GT_ANSWER"} for _ in range(len(all_pred))]

            for i in range(len(all_pred)):
                original_item = original_eval_data[i] if i < len(original_eval_data) else {}
                frame_num = original_item.get("frame_num", f"scenario_{i}")
                question = original_item.get("instruction", "Question not available")
                ground_truth_answer_from_dataset = original_item.get("output", "GT Answer not available")

                qa_results.append({
                    "scenario_id": frame_num,
                    "question": question,
                    "model_answer": all_pred[i],
                    "ground_truth_answer_decoded": all_label[i],
                    "ground_truth_answer_original": ground_truth_answer_from_dataset
                })

            output_qa_file_path = Path(self.args.output_dir) / f"{metric_key_prefix}_qa_predictions.json"
            with open(output_qa_file_path, "w", encoding="utf-8") as f:
                json.dump(qa_results, f, indent=4, ensure_ascii=False)
            print(f"Saved Q&A predictions to: {output_qa_file_path}")

            eval_distance(
                all_pred, all_label, "tl_distance", r"It is (\d+(?:\.\d+)?)m ahead"
            )
    
            # Evaluate perceptions
            eval_distance(
                all_pred, all_label, "car_error", r"observing (\d+(?:\.\d+)?) cars"
            )
            eval_distance(
                all_pred, all_label, "ped_error", r"and (\d+(?:\.\d+)?) pedestrians"
            )
            # Evaluate traffic light
            tl_accuracy = eval_tl(all_pred, all_label)
            if tl_accuracy is not None:
                print(f"TL accuracy: {tl_accuracy}")
            else:
                print("No traffic light states found in predictions.")
            # wandb.log({"tl_accuracy": tl_accuracy})


        

        # Evaluate actions
        average_error_lon, average_error_lat = eval_action(all_pred, all_label)
        if average_error_lon is not None and average_error_lat is not None:
            print(f"Average control error: {average_error_lon}, {average_error_lat}")
        return eval_output


def eval_distance(all_pred, all_label, label_name, pattern):
    distance_errors = get_eval_distance_errors(all_pred, all_label, pattern)
    if len(distance_errors) > 0:
        mean_error = np.mean(distance_errors)
        print(
            f"{label_name}: Mean Absolute Error (MAE): {mean_error}, Total num: {len(distance_errors)}"
        )
        # wandb.log({label_name: mean_error})
def plot_loss_history(history, output_dir):
    """
    Plots the training and evaluation loss from the Trainer's history.
    Saves the plot to a file.
    """
    if not history:
        print("No history to plot.")
        return

    # Extracting data
    train_logs = [log for log in history if 'loss' in log]
    eval_logs = [log for log in history if 'eval_loss' in log]

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot training loss
    if train_logs:
        train_steps = [log['step'] for log in train_logs]
        train_losses = [log['loss'] for log in train_logs]
        plt.plot(train_steps, train_losses, label='Training Loss', color='blue', marker='o', linestyle='-')

    # Plot evaluation loss
    if eval_logs:
        eval_steps = [log['step'] for log in eval_logs]
        eval_losses = [log['eval_loss'] for log in eval_logs]
        plt.plot(eval_steps, eval_losses, label='Evaluation Loss', color='red', marker='x', linestyle='--')

    # Formatting the plot
    plt.title('Training and Evaluation Loss Over Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to: {plot_path}")
    plt.close() # Close the figure to free up memory

def train(
    # model/data params
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # the only required argument
    data_path: str = "/kaggle/input/training-dataset/vqa_train_10k.pkl",
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 8,
    num_epochs: int = 5,
    learning_rate: float = 3e-4,
    val_set_size: int = 1e6,
    eval_steps: int = 10,
    # lora hyperparams
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    lora_target_modules: Tuple = ("q_proj", "v_proj"),
    group_by_length: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    augment_times: int = 0,
    output_dir: Optional[str] = "/kaggle/working/result",
    stage1_weights_path: Optional[str] = "/kaggle/input/stage1-weights/checkpoint-800",

    vqa: bool = True,
    eval_items: List[str] = DEFAULT_EVAL_ITEMS,
    mode: str = "train",
    load_pre_prompt_dataset: bool = False,
    val_data_path: str = "/kaggle/input/validation-dataset/vqa_valid_500.pkl",
):


    weights_to_load = stage1_weights_path if stage1_weights_path else resume_from_checkpoint

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
    tokenizer = load_llama_tokenizer(base_model)

    train_data, val_data = get_train_val_data(
        data_path,
        tokenizer,
        val_data_path=val_data_path,
        val_set_size=val_set_size,
        augment_times=augment_times,
        load_pre_prompt_dataset=load_pre_prompt_dataset,
        vqa=vqa,
        eval_only=mode == "eval",
        eval_items=["vqa", "caption", "action"] if vqa else ["caption"]
,
    )
    if val_data is not None:
        num_eval_samples = 100 
        val_data = val_data.shuffle(seed=42).select(range(num_eval_samples))


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
        eval_steps=2000 if val_data is not None else None, # Frequency for both eval and save if strategies match

        save_strategy="steps",          # Valid for Seq2SeqTrainingArguments
        save_steps=2000,                 # Save a checkpoint every 500 steps (adjust)
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
        generation_max_length=1024,
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
        vqa=vqa,
    )

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    logging.set_verbosity_info()
    if mode == "train":
        # Check for existing checkpoints in the output directory
        checkpoint_dir = output_dir # This is self.args.output_dir for the Trainer
        determined_resume_path = None # Initialize

        if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir): # Ensure it's a directory
            # Find the latest checkpoint
            checkpoints = [
                d for d in os.listdir(checkpoint_dir)
                if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
            ]
            if checkpoints:
                # Correctly extract step number for sorting
                try:
                    latest_checkpoint_dir_name = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                    determined_resume_path = os.path.join(checkpoint_dir, latest_checkpoint_dir_name)
                    print(f"Resuming from determined checkpoint: {determined_resume_path}")
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse step number from checkpoint names in {checkpoint_dir}")
                    determined_resume_path = True # Let Trainer try to find latest
            else:
                print(f"No 'checkpoint-*' subdirectories found in {checkpoint_dir}. Will train from scratch or initial model weights.")
                # determined_resume_path remains None
        else:
            print(f"Output directory {checkpoint_dir} does not exist or is not a directory. Will train from scratch.")
            # determined_resume_path remains None

        trainer.train(resume_from_checkpoint=determined_resume_path) # Pass the variable

        if local_rank == 0:
            print("🤗🤗🤗🤗🤗Model saved to:", output_dir, "🤗🤗🤗🤗🤗")
            model.save_pretrained(output_dir)
            print("LoRA adapters saved.")

            vector_encoder_weights = model.vector_encoder.state_dict()
            llm_proj_weights = model.llm_proj.state_dict()
            
            torch.save(vector_encoder_weights, os.path.join(output_dir, "vector_encoder.pth"))
            torch.save(llm_proj_weights, os.path.join(output_dir, "llm_proj.pth"))
            
            print("Custom modules (vector_encoder, llm_proj) saved successfully.")
               # 2. Save the training metrics
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            
            # 3. Save the trainer state (includes history)
            trainer.save_state()
            print(f"Trainer state and metrics saved to: {output_dir}")

            # 4. Plot the loss history
            plot_loss_history(trainer.state.log_history, output_dir)
            
        # --- END OF ADDED BLOCK ---
    elif mode == "eval":
        outputs = trainer.evaluate()
        print(outputs)
        state_path = os.path.join(output_dir, "trainer_state.json")
        if os.path.exists(state_path):
             with open(state_path, "r") as f:
                 state_data = json.load(f)
             plot_loss_history(state_data["log_history"], output_dir)
        

if __name__ == "__main__":
    import time
    import fire
    st = time.time()
    fire.Fire(train)
    print("Total time:", time.time() - st)
