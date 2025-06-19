"""
tested with:
pip install unsloth==2025.5.7 unsloth-zoo==2025.5.8


"""

from opensloth.opensloth_config import (
    FastModelArgs,
    HFDatasetConfig,
    LoraArgs,
    OpenSlothConfig,
    TrainingArguments,
)
from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs

# 2 GPUs with packing configuration
GLOBAL_BZ = 32

DEVICES = [0, 2]

BZ = 2  # if sequence packing, then should be 1, larger does not contribute to speed
opensloth_config = OpenSlothConfig(
    # Use Hugging Face dataset configuration
    data=HFDatasetConfig(
        tokenizer_name="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        chat_template="gemma3",  # known template: ['unsloth', 'zephyr', 'chatml', 'mistral', 'llama', 'vicuna', 'vicuna_old', 'vicuna old', 'alpaca', 'gemma', 'gemma_chatml', 'gemma2', 'gemma2_chatml', 'llama-3', 'llama3', 'phi-3', 'phi-35', 'phi-3.5', 'llama-3.1', 'llama-31', 'llama-3.2', 'llama-3.3', 'llama-32', 'llama-33', 'qwen-2.5', 'qwen-25', 'qwen25', 'qwen2.5', 'phi-4', 'gemma-3', 'gemma3', 'qwen-3', 'qwen3']
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
        num_samples=100,
        nproc=1,
        max_seq_length=16000,
        source_type="hf",
        dataset_name="mlabonne/FineTome-100k",
        split="train",
        cache=True,
    ),
    # Use PathDatasetConfig if you have a local dataset (only support sharegpt format for now)
    # data=PathDatasetConfig(
    #     path="sharegpt-format-dataset.json",  # we've just saved
    #     chat_template="qwen3",
    #     instruction_part="<|im_start|>user\n",
    #     response_part="<|im_start|>assistant\n",
    #     num_samples=10000,
    #     nproc=52,
    #     max_seq_length=16000,
    # ),
    devices=DEVICES,
    fast_model_args=FastModelArgs(
        model_name="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
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
    print(
        "→ Global batch size:",
        len(DEVICES) * BZ * training_config.gradient_accumulation_steps,
    )
    print("→ Grad accumulation:", training_config.gradient_accumulation_steps)
    setup_envs(opensloth_config, training_config)
    run_mp_training(opensloth_config.devices, opensloth_config, training_config)
    print("Training completed.")
