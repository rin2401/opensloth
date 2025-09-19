#!/usr/bin/env python3
# type: ignore
"""Minimal OpenSloth training script (no argparse, fixed hyperparameters)."""

import os
from typing import Any, Tuple, Optional
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset, load_from_disk
from trl import SFTConfig, SFTTrainer  # type: ignore
from opensloth.patching.ddp_patch import ddp_patch, patch_optimize_sft_trainer_batch_samples


# ------------------------------
# Hyperparameters
# ------------------------------
MODEL_NAME = "unsloth/Qwen3-0.6B-bnb-4bit"
MAX_SEQ_LENGTH = 4096
NUM_EPOCHS = 5
TRAIN_SAMPLES = 1000
VAL_SAMPLES = 10
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 8
GRAD_ACCUM = 4
EXPERIMENT_NAME = "opensloth_sanity"
USE_PATCHES = True
PREPARED_DATASET_PATH = None  # set path if using pre-processed dataset


def init_model(local_rank: int) -> Tuple[FastLanguageModel, Any]:
    device_map = f"cuda:{local_rank}" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "auto"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        device_map=device_map,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def prepare_datasets(tokenizer: Any) -> Tuple[Dataset, Optional[Dataset]]:
    print(f"📊 Loading Capybara dataset ({TRAIN_SAMPLES} train, {VAL_SAMPLES} val)...")
    train_dataset = load_dataset("trl-lib/Capybara", split=f"train[:{TRAIN_SAMPLES}]")
    val_dataset = load_dataset("trl-lib/Capybara", split=f"train[{TRAIN_SAMPLES}:{TRAIN_SAMPLES+VAL_SAMPLES}]") if VAL_SAMPLES > 0 else None

    def format_messages(examples):
        return {"text": [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in examples["messages"]]}

    train_dataset = train_dataset.map(format_messages, batched=True, remove_columns=train_dataset.column_names)
    if val_dataset is not None:
        val_dataset = val_dataset.map(format_messages, batched=True, remove_columns=val_dataset.column_names)
    return train_dataset, val_dataset


def build_trainer(model: Any, tokenizer: Any, train_dataset: Dataset, val_dataset: Optional[Dataset], world_size: int) -> SFTTrainer:
    run_name = f"{EXPERIMENT_NAME}_ws{world_size}"
    training_config = {
        "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "output_dir": f"outputs/{run_name}",
        "report_to": "tensorboard",
        "run_name": run_name,
        "eval_strategy": "epoch",
        "dataset_num_proc": 2,
        "max_seq_length": MAX_SEQ_LENGTH,
    }
    if world_size > 1:
        training_config["ddp_find_unused_parameters"] = False
    training_args = SFTConfig(**training_config)

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )


if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print("🔧 Applying DDP patches...")
    ddp_patch()
    patch_optimize_sft_trainer_batch_samples()

    print(f"🚀 Initializing training on GPU {local_rank}")

    model, tokenizer = init_model(local_rank)
    train_dataset, val_dataset = prepare_datasets(tokenizer)

    trainer = build_trainer(model, tokenizer, train_dataset, val_dataset, world_size)

    print("🎯 Starting training...")
    trainer.train()
    print("✅ Training completed!")
