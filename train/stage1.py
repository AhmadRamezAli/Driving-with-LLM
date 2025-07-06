import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from model import DrivingWithLLM
from utils.optimizer_utils import configure_optimiser
from utils.training_utils import get_train_val_data


def train_stage1(
    model_name="deepseek-ai/deepseek-coder-1.3b-base",
    train_data_path="data/train.pkl",
    output_dir="checkpoints/stage1",
    epochs=3,
    batch_size=4,
    lr=2e-5,
    weight_decay=0.01,
    val_set_size=100,
    device="cuda"
):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load training data
    train_data, val_data = get_train_val_data(
        data_path=train_data_path,
        tokenizer=tokenizer,
        val_set_size=val_set_size,
        add_input_prompt=True,
        eval_only=False,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Build model
    model = DrivingWithLLM(base_model=model_name).to(device)
    model.llm.requires_grad_(False)  # Freeze LLM for stage 1
    model.vector_encoder.requires_grad_(True)
    model.vector_former.requires_grad_(True)
    model.proj.requires_grad_(True)

    # Optimizer
    optimizer = configure_optimiser(model, lr=lr, weight_decay=weight_decay)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                route_descriptors=batch["route_descriptors"].to(device),
                vehicle_descriptors=batch["vehicle_descriptors"].to(device),
                pedestrian_descriptors=batch["pedestrian_descriptors"].to(device),
                ego_vehicle_descriptor=batch["ego_vehicle_descriptor"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

        # Save checkpoint
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch{epoch+1}.pt"))


if __name__ == "__main__":
    train_stage1()
