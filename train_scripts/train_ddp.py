#!/usr/bin/env python3
# type: ignore
"""OpenSloth - Simple Multi-GPU training with torchrun."""

import os
from typing import Any, Tuple
from unsloth import FastLanguageModel

from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer  # type: ignore

from opensloth.patching.ddp_patch import ddp_patch, patch_trainer_get_batch_samples
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
MODEL_NAME = "unsloth/Qwen3-0.6B"
MAX_SEQ_LENGTH = 16000
LORA_RANK = 8
NUM_EPOCHS = 10
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 32//WORLD_SIZE  # Adjust based on GPU memory
GRAD_ACCUM = 8

def init_model(local_rank: int) -> Tuple[FastLanguageModel, Any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{local_rank}",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    return model, tokenizer


def prepare_datasets(tokenizer: Any) -> Tuple[Dataset, Dataset]:
    """Load and format the Capybara dataset with the tokenizer chat template."""
    print("📊 Loading Capybara dataset...")
    train_dataset = load_dataset("trl-lib/Capybara", split="train[:2000]")
    val_dataset = load_dataset("trl-lib/Capybara", split="train[2000:2200]")

    def format_messages(examples):
        formatted = [
            tokenizer.apply_chat_template(messages, tokenize=False)
            for messages in examples["messages"]
        ]
        return {"text": formatted}

    train_columns = train_dataset.column_names
    val_columns = val_dataset.column_names

    train_dataset = train_dataset.map(format_messages, batched=True, remove_columns=train_columns)
    val_dataset = val_dataset.map(format_messages, batched=True, remove_columns=val_columns)

    print(f"✅ Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    return train_dataset, val_dataset


def print_batch_configuration(per_device_batch_size: int, grad_accum: int, world_size: int) -> None:
    effective_batch = per_device_batch_size * grad_accum * world_size
    print("📊 Batch configuration:")
    print(f"   - Per device batch size: {per_device_batch_size}")
    print(f"   - Gradient accumulation: {grad_accum}")
    print(f"   - Effective batch size: {effective_batch}")


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    val_dataset: Dataset,
    world_size: int,
) -> SFTTrainer:
    training_args = SFTConfig(
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        save_strategy="no",
        output_dir=f"outputs/debug_worldsize{world_size}",
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
        eval_strategy="epoch",
        dataset_num_proc=4,
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )


def main() -> None:
    ddp_patch()
    patch_trainer_get_batch_samples()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"🚀 Initializing training on GPU {local_rank}")

    model, tokenizer = init_model(local_rank)
    train_dataset, val_dataset = prepare_datasets(tokenizer)

    
    print(f"🌍 World size: {WORLD_SIZE} GPU(s)")

    print_batch_configuration(PER_DEVICE_BATCH_SIZE, GRAD_ACCUM, WORLD_SIZE)

    trainer = build_trainer(model, tokenizer, train_dataset, val_dataset, WORLD_SIZE)

    trainer.train()


if __name__ == "__main__":
    main()
