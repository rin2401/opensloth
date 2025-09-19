#!/usr/bin/env python3
# type: ignore
"""OpenSloth - Configurable training script with wandb logging."""

import os
import argparse
from typing import Any, Tuple
from unsloth import FastLanguageModel

from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer  # type: ignore

from opensloth.patching.ddp_patch import ddp_patch, patch_optimize_sft_trainer_batch_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Configurable OpenSloth training")
    parser.add_argument("--model_name", default="unsloth/Qwen3-0.6B-bnb-4bit", help="Model name")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--train_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--test_samples", type=int, default=10, help="Number of test samples")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--per_device_batch_size", type=int, default=8, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--use_patches", action="store_true", default=True, help="Use DDP patches")
    parser.add_argument("--no_patches", action="store_true", help="Disable DDP patches")
    parser.add_argument("--experiment_name", default="opensloth_sanity", help="Experiment name for wandb")
    return parser.parse_args()


def init_model(args, local_rank: int) -> Tuple[FastLanguageModel, Any]:
    device_map = f"cuda:{local_rank}" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "auto"
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        device_map=device_map,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # LORA_RANK
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,  # LORA_RANK * 2
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    return model, tokenizer


def prepare_datasets(args, tokenizer: Any) -> Tuple[Dataset, Dataset]:
    """Load and format the Capybara dataset with the tokenizer chat template."""
    print(f"📊 Loading Capybara dataset ({args.train_samples} train, {args.test_samples} test)...")
    
    train_dataset = load_dataset("trl-lib/Capybara", split=f"train[:{args.train_samples}]")
    val_dataset = load_dataset("trl-lib/Capybara", split=f"train[{args.train_samples}:{args.train_samples + args.test_samples}]")

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


def print_configuration(args, world_size: int) -> None:
    effective_batch = args.per_device_batch_size * args.grad_accum * world_size
    print("📊 Training configuration:")
    print(f"   - Model: {args.model_name}")
    print(f"   - Max sequence length: {args.max_seq_length}")
    print(f"   - Epochs: {args.num_epochs}")
    print(f"   - World size: {world_size} GPU(s)")
    print(f"   - Per device batch size: {args.per_device_batch_size}")
    print(f"   - Gradient accumulation: {args.grad_accum}")
    print(f"   - Effective batch size: {effective_batch}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Training samples: {args.train_samples}")
    print(f"   - Test samples: {args.test_samples}")
    print(f"   - Use patches: {args.use_patches and not args.no_patches}")


def build_trainer(
    args,
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    val_dataset: Dataset,
    world_size: int,
) -> SFTTrainer:
    # Create experiment name with world size
    run_name = f"{args.experiment_name}_ws{world_size}"
    if args.no_patches:
        run_name += "_nopatches"
    
    # Configure training args based on world size
    training_config = {
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "output_dir": f"outputs/{run_name}",
        "report_to": "tensorboard",
        "run_name": run_name,
        "eval_strategy": "epoch",
        "dataset_num_proc": 2,
        "max_seq_length": args.max_seq_length,
    }
    
    # Only set DDP parameters for multi-GPU
    if world_size > 1:
        training_config["ddp_find_unused_parameters"] = False
    
    training_args = SFTConfig(**training_config)

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    # Apply patches unless explicitly disabled, and only for multi-GPU
    use_patches = args.use_patches and not args.no_patches and world_size > 1
    if use_patches:
        print("🔧 Applying DDP patches...")
        ddp_patch()
        patch_optimize_sft_trainer_batch_samples()
    elif world_size > 1:
        print("⚠️  Running multi-GPU without patches (may have issues)")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"🚀 Initializing training on GPU {local_rank}")

    print_configuration(args, world_size)

    model, tokenizer = init_model(args, local_rank)
    train_dataset, val_dataset = prepare_datasets(args, tokenizer)

    trainer = build_trainer(args, model, tokenizer, train_dataset, val_dataset, world_size)

    print("🎯 Starting training...")
    trainer.train()
    print("✅ Training completed!")


if __name__ == "__main__":
    main()