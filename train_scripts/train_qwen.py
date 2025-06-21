from opensloth.opensloth_config import (
    FastModelArgs,
    LoraArgs,
    OpenSlothConfig,
    TrainingArguments,
)
from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs

# 2 GPUs with packing configuration
GLOBAL_BZ = 32

DEVICES = [0, 1]

BZ = 1  # if sequence packing, then should be 1, larger does not contribute to speed
opensloth_config = OpenSlothConfig(
    data_cache_path="data/cache_qwen3_dataset/",
    devices=DEVICES,
    fast_model_args=FastModelArgs(
        model_name="unsloth/Qwen3-0.6B-Base-bnb-4bit",
        max_seq_length=16000,
        load_in_4bit=True,
    ),
    lora_args=LoraArgs(
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
        use_rslora=False,
    ),
    sequence_packing=True,
)

training_config = TrainingArguments(
    output_dir="outputs/exps/qwen3-0.6b-FineTome-2gpu-packing",
    per_device_train_batch_size=BZ,
    gradient_accumulation_steps=GLOBAL_BZ // (len(DEVICES) * BZ),
    learning_rate=1e-5,
    logging_steps=1,
    num_train_epochs=1,
    lr_scheduler_type="linear",
    warmup_steps=5,
    save_total_limit=1,
    weight_decay=0.01,
    optim="adamw_8bit",
    seed=3407,
    report_to="none",  # or wandb/tensorboard
)


if __name__ == "__main__":
    import os

    # Setup wandb
    os.environ["WANDB_PROJECT"] = "opensloth"
    os.environ["WANDB_NAME"] = (
        f"qwen3-0.6b_2gpu_packing_globalbz{GLOBAL_BZ}_samples10000"
    )

    print(
        f"Global batch size: {len(DEVICES) * BZ * training_config.gradient_accumulation_steps}"
    )
    print(f"Gradient accumulation steps: {training_config.gradient_accumulation_steps}")

    setup_envs(opensloth_config, training_config)
    run_mp_training(opensloth_config.devices, opensloth_config, training_config)
