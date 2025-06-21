"""
COPY FROM: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb#scrollTo=5kyTw2n1edte
"""

from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def dump_data() -> Any:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-0.6B",  # Just for storing dataset, use smallest of its varient
        max_seq_length=2048,  # Context length - can be longer, but uses more memory
        load_in_4bit=True,  # 4bit uses much less memory
        load_in_8bit=False,  # A bit more accurate, uses 2x memory
        full_finetuning=False,  # We have full finetuning now!
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,  # Best to choose alpha = rank or rank*2
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")

    def to_conv(x):
        return [
            {"role": "user", "content": x["problem"]},
            {"role": "assistant", "content": x["generated_solution"]},
        ]

    df = pd.DataFrame(dataset).sample(1000, random_state=3407)
    df["messages"] = df.apply(to_conv, axis=1)
    df["text"] = df["messages"].apply(
        lambda m: tokenizer.apply_chat_template(m, tokenize=False)
    )
    ds = Dataset.from_pandas(df[["text"]])
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Use GA to mimic batch size!
            warmup_steps=5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=30,
            learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",  # Use this for WandB etc
        ),
    )
    trainer.train_dataset.save_to_disk("data/cache_qwen3_dataset")


if __name__ == "__main__":
    dump_data()
