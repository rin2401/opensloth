#!/usr/bin/env python3
# type: ignore
"""Minimal Unsloth DPO training script with multi-GPU (DDP) support.

This converts the Unsloth Zephyr DPO notebook into a runnable script
following the OpenSloth example style (see qwen_sft_cache_dataset.py).

Key requirements satisfied:
- Import Unsloth FIRST and patch DPO trainer before anything else.
- Apply OpenSloth DDP patches early in main to set device and Trainer tweaks.
"""

import os
from typing import Any, Dict, List, Literal, Optional, Tuple

# --- Unsloth imports MUST come first ---
from unsloth import FastLanguageModel  # noqa: E402
from unsloth import PatchDPOTrainer  # noqa: E402

# Patch TRL's DPOTrainer before importing/creating it
PatchDPOTrainer()

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk  # noqa: E402
from datasets.builder import DatasetGenerationError  # noqa: E402
from trl import DPOTrainer, DPOConfig  # noqa: E402

# OpenSloth patches (safe to import AFTER Unsloth)
from opensloth.patching.ddp_patch import ddp_patch  # noqa: E402


# ------------------------------
# Hyperparameters
# ------------------------------
MODEL_NAME = "unsloth/zephyr-sft-bnb-4bit"
MAX_SEQ_LENGTH = 4096
NUM_EPOCHS = 3
LEARNING_RATE = 5e-6
PER_DEVICE_BATCH_SIZE = 2
GRAD_ACCUM = 4
BETA = 0.1
LOGGING_STEPS = 1
EXPERIMENT_NAME = "opensloth_zephyr_dpo"
REPORT_TO = "tensorboard"  # set to "none" to disable

# Dataset config: sample fraction on train/test splits
# Example mirrors the notebook using 0.5% (0.005) of UltraFeedback
DATASET_MIXER: Dict[str, float] = {
    "HuggingFaceH4/ultrafeedback_binarized": 0.005,
}
SPLITS: List[str] = ["train_prefs", "test_prefs"]


def init_model(local_rank: int) -> Tuple[Any, Any]:
    device_map = f"cuda:{local_rank}" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "auto"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        device_map=device_map,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


# --- Alignment Handbook style helpers (adapted) ---
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n"
    "{% endfor %}"
)


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"] = "sft",
    assistant_prefix: str = "<|assistant|>\n",
):
    import re

    def _strip_prefix(s, pattern):
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task in ["sft", "generation"]:
        messages = example["messages"]
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            if chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            else:
                prompt_messages.insert(0, example["chosen"][0])

            chosen_messages = example["chosen"][1:]
            rejected_messages = example["rejected"][1:]
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_chosen"] = _strip_prefix(
                example["text_chosen"], assistant_prefix
            )
            example["text_rejected"] = _strip_prefix(
                example["text_rejected"], assistant_prefix
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, must be one of ['sft', 'generation', 'rm', 'dpo']"
        )
    return example


def mix_datasets(
    dataset_mixer: Dict[str, float],
    splits: Optional[List[str]] = None,
    shuffle: bool = True,
) -> DatasetDict:
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs: List[float] = []
    for ds, frac in dataset_mixer.items():
        fracs.append(frac)
        for split in splits or []:
            try:
                dataset = load_dataset(ds, split=split)
            except DatasetGenerationError:
                dataset = load_from_disk(os.path.join(ds, split))

            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as train or test.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            n = int(frac * len(dataset))
            train_subset = dataset.select(range(max(n, 1)))
            train_subsets.append(train_subset)
        raw_datasets["train"] = (
            concatenate_datasets(train_subsets).shuffle(seed=42) if shuffle else concatenate_datasets(train_subsets)
        )

    if len(raw_val_datasets) > 0:
        raw_datasets["test"] = (
            concatenate_datasets(raw_val_datasets).shuffle(seed=42) if shuffle else concatenate_datasets(raw_val_datasets)
        )

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check formatting."
        )

    return raw_datasets


def get_datasets(data_config: Dict[str, float], splits: List[str]) -> DatasetDict:
    if not isinstance(data_config, dict):
        raise ValueError(f"Data config {data_config} not recognized.")
    return mix_datasets(data_config, splits=splits, shuffle=True)


def prepare_datasets(tokenizer: Any) -> Tuple[Any, Optional[Any]]:
    raw = get_datasets(DATASET_MIXER, splits=SPLITS)
    # Remember original columns, then map and rename for TRL DPO
    column_names = list(raw["train"].features)

    raw = raw.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
        num_proc=min(12, os.cpu_count() or 2),
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    for split in ["train", "test"]:
        if split in raw:
            raw[split] = raw[split].rename_columns(
                {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
            )

    train_ds = raw["train"]
    val_ds = raw.get("test")
    return train_ds, val_ds


def build_trainer(model: Any, tokenizer: Any, train_dataset: Any, val_dataset: Optional[Any], world_size: int) -> DPOTrainer:
    run_name = f"{EXPERIMENT_NAME}_ws{world_size}"

    training_config: Dict[str, Any] = {
        "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "logging_steps": LOGGING_STEPS,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "seed": 42,
        "output_dir": f"outputs/{run_name}",
        "report_to": REPORT_TO,
        "run_name": run_name,
    }
    if world_size > 1:
        training_config["ddp_find_unused_parameters"] = False

    args = DPOConfig(**training_config)

    return DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        beta=BETA,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_length=1024,
        max_prompt_length=512,
    )


if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print("🔧 Applying DDP patches…")
    ddp_patch()

    print(f"🚀 Initializing model on GPU {local_rank}")
    model, tokenizer = init_model(local_rank)

    print("📊 Preparing datasets (UltraFeedback → DPO format)…")
    train_dataset, val_dataset = prepare_datasets(tokenizer)

    print("🧰 Building DPO trainer…")
    trainer = build_trainer(model, tokenizer, train_dataset, val_dataset, world_size)

    print("🎯 Starting training…")
    trainer.train()
    print("✅ Training completed!")
