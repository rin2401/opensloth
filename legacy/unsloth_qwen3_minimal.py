# -*- coding: utf-8 -*-
"""Minimal Qwen3 training script using Unsloth."""

import os

from opensloth.patching.patch_sampler import patch_sampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OPENSLOTH_LOCAL_RANK"] = "0"

CACHE_DATA_DIR = os.environ.get("OPENSLOTH_CACHE_DATA_DIR", "/tmp/opensloth_cache_data")


def train_qwen3_model():
    """Train Qwen3 model with minimal setup."""
    import torch
    from datasets import load_from_disk
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    # Load cached dataset
    cache_path = os.environ.get(
        "OPENSLOTH_CACHE_DATASET_PATH", "/tmp/opensloth_cache_data/qwen3_cache"
    )
    train_dataset = load_from_disk(cache_path)

    output_dir = "outputs/exps/qwen3-0.6b-FineTome-unsloth-no-packing"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-0.6B-bnb-4bit",
        max_seq_length=4096,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        learning_rate=1e-5,
        num_train_epochs=1,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        # max_steps=100,
        report_to="tensorboard",  # or "wandb"
        skip_prepare_dataset=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # type: ignore
        train_dataset=train_dataset,
        eval_dataset=None,
        args=args,
    )
    from unsloth_zoo.dataset_utils import train_on_responses_only

    trainer = train_on_responses_only(
        trainer,
        tokenizer=tokenizer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Train the model

    # from ._patch_sampler import patch_sampler

    trainer = patch_sampler(trainer)
    trainer_stats = trainer.train()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train_qwen3_model()
    print("Training completed successfully!")
