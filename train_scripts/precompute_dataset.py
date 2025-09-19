#!/usr/bin/env python3
# type: ignore
"""OpenSloth - Dataset Precomputation Tool."""

import argparse
import os
from pathlib import Path
from typing import Any, Optional, Tuple
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset, load_dataset
from trl import SFTTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Precompute and save tokenized datasets")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-0.6B-bnb-4bit",
                       help="Model name to use for tokenization")
    parser.add_argument("--max_seq_length", type=int, default=16000,
                       help="Maximum sequence length")
    # parser.add_argument("--load_in_4bit", action="store_true", default=True,
    #                    help="Load model in 4-bit precision")
    
    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, default="trl-lib/Capybara",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--train_split", type=str, default="train[:2000]",
                       help="Training split specification")
    parser.add_argument("--val_split", type=str, default="train[2000:2200]",
                       help="Validation split specification")
    
    # Tokenization options
    parser.add_argument("--target_token_only", action="store_true",
                       help="Only compute loss on target tokens (assistant responses)")
    parser.add_argument("--instruction_part", type=str, default="<|im_start|>user\n",
                       help="Instruction part template for target_token_only mode")
    parser.add_argument("--response_part", type=str, default="<|im_start|>assistant\n",
                       help="Response part template for target_token_only mode")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="data/precomputed",
                       help="Output directory for processed datasets")
    parser.add_argument("--dataset_name_suffix", type=str, default="",
                       help="Suffix to add to output dataset name")
    
    # Processing options
    parser.add_argument("--num_proc", type=int, default=4,
                       help="Number of processes for dataset mapping")
    parser.add_argument("--batch_size", type=int, default=1000,
                       help="Batch size for dataset processing")
    
    return parser.parse_args()


def init_model_and_tokenizer(args) -> Tuple[Any, Any]:
    """Initialize model and tokenizer for dataset preprocessing."""
    print(f"🤖 Loading model: {args.model_name}")
    # just init a fake model small for fast
    model, _ = FastLanguageModel.from_pretrained(
        model_name='unsloth/Qwen3-0.6B-bnb-4bit',
        max_seq_length=args.max_seq_length,
        device_map="auto",
    )# use small model to load fast (does not matter)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name) # load correct tokenizer for dataset
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # LORA_RANK
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,  # LORA_RANK * 2
        use_gradient_checkpointing="unsloth",
        random_state=42#,
    )# this to make trainer happy with trainable params
    print("✅ Model and tokenizer loaded successfully")
    return model, tokenizer


def load_raw_datasets(args) -> Tuple[Dataset, Optional[Dataset]]:
    """Load raw datasets from HuggingFace."""
    print(f"📊 Loading dataset: {args.dataset_name}")
    
    train_dataset = load_dataset(args.dataset_name, split=args.train_split)
    print(f"✅ Loaded {len(train_dataset)} training examples")
    
    val_dataset = None
    if args.val_split:
        try:
            val_dataset = load_dataset(args.dataset_name, split=args.val_split)
            print(f"✅ Loaded {len(val_dataset)} validation examples")
        except Exception as e:
            print(f"⚠️ Could not load validation split: {e}")
            val_dataset = None
    
    return train_dataset, val_dataset


def format_datasets(train_dataset: Dataset, val_dataset: Optional[Dataset], tokenizer: Any) -> Tuple[Dataset, Optional[Dataset]]:
    """Format datasets using tokenizer chat template."""
    print("🔄 Formatting datasets with chat template...")
    
    def format_messages(examples):
        formatted = []
        for messages in examples["messages"]:
            try:
                formatted_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=False
                )
                formatted.append(formatted_text)
            except Exception as e:
                print(f"⚠️ Error formatting message: {e}")
                formatted.append("")  # Fallback to empty string
        return {"text": formatted}
    
    # Format training dataset
    train_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        format_messages, 
        batched=True, 
        remove_columns=train_columns,
        desc="Formatting training dataset"
    )
    
    # Format validation dataset if it exists
    if val_dataset is not None:
        val_columns = val_dataset.column_names
        val_dataset = val_dataset.map(
            format_messages, 
            batched=True, 
            remove_columns=val_columns,
            desc="Formatting validation dataset"
        )
    
    print("✅ Dataset formatting completed")
    return train_dataset, val_dataset


def create_trainer_for_preprocessing(model: Any, tokenizer: Any, train_dataset: Dataset, 
                                   val_dataset: Optional[Dataset], args) -> SFTTrainer:
    """Create SFTTrainer for dataset preprocessing without training."""
    print("🔧 Creating trainer for dataset preprocessing...")
    
    # Minimal training args just for preprocessing
    from trl import SFTConfig
    training_args = SFTConfig(
        output_dir="temp_output",  # Temporary, won't be used
        per_device_train_batch_size=1,
        logging_steps=100,
        save_strategy="no",
        report_to=None,
        dataset_num_proc=args.num_proc,
        max_seq_length=args.max_seq_length,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        dataset_text_field="text",
    )
    
    # Apply target_token_only training using Unsloth's approach
    if args.target_token_only:
        print(f"🎯 Using target_token_only mode with instruction part: {args.instruction_part}")
        print(f"🎯 Using target_token_only mode with response part: {args.response_part}")
        trainer = train_on_responses_only(
            trainer,
            instruction_part=args.instruction_part,
            response_part=args.response_part,
        )
    
    print("✅ Trainer created for preprocessing")
    return trainer


def save_processed_datasets(trainer: SFTTrainer, args) -> None:
    """Save preprocessed datasets to disk."""
    print("💾 Saving processed datasets...")
    
    # Create output directory
    output_path = Path(args.output_dir)
    model_name_clean = args.model_name.replace("/", "_").replace("-", "_")
    
    suffix = f"_{args.dataset_name_suffix}" if args.dataset_name_suffix else ""
    target_suffix = "_target_only" if args.target_token_only else ""
    
    dataset_dir = output_path / f"{model_name_clean}_{args.dataset_name.replace('/', '_')}{target_suffix}{suffix}_processed"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {dataset_dir}")
    
    # Save training dataset
    if hasattr(trainer, 'train_dataset') and trainer.train_dataset is not None:
        train_path = dataset_dir / "train"
        trainer.train_dataset.save_to_disk(str(train_path))
        print(f"✅ Training dataset saved to: {train_path}")
        print(f"   - Number of examples: {len(trainer.train_dataset)}")
    else:
        print("⚠️ No training dataset found in trainer")
    
    # Save validation dataset if it exists
    if hasattr(trainer, 'eval_dataset') and trainer.eval_dataset is not None:
        val_path = dataset_dir / "val"
        trainer.eval_dataset.save_to_disk(str(val_path))
        print(f"✅ Validation dataset saved to: {val_path}")
        print(f"   - Number of examples: {len(trainer.eval_dataset)}")
    else:
        print("ℹ️ No validation dataset to save")
    
    # Save configuration for future reference
    config = {
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "dataset_name": args.dataset_name,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "target_token_only": args.target_token_only,
        "instruction_part": args.instruction_part if args.target_token_only else None,
        "response_part": args.response_part if args.target_token_only else None,
        "processed_at": str(Path.cwd()),
    }
    
    import json
    config_path = dataset_dir / "dataset_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to: {config_path}")
    print(f"\n🎉 Dataset preprocessing completed!")
    print(f"📊 Processed dataset location: {dataset_dir}")


def main() -> None:
    """Main function for dataset precomputation."""
    args = parse_args()
    
    print("🚀 Starting dataset precomputation...")
    print(f"   Model: {args.model_name}")
    print(f"   Dataset: {args.dataset_name}")
    print(f"   Target token only: {args.target_token_only}")
    print(f"   Output: {args.output_dir}")
    
    # Initialize model and tokenizer
    model, tokenizer = init_model_and_tokenizer(args)
    
    # Load raw datasets
    train_dataset, val_dataset = load_raw_datasets(args)
    
    # Format datasets with chat template
    train_dataset, val_dataset = format_datasets(train_dataset, val_dataset, tokenizer)
    
    # Create trainer for preprocessing
    trainer = create_trainer_for_preprocessing(model, tokenizer, train_dataset, val_dataset, args)
    
    # Save processed datasets
    save_processed_datasets(trainer, args)


if __name__ == "__main__":
    main()