# train_scripts/train_ddp.py
# type: ignore
import os
import random

import pandas as pd
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import random
from opensloth.patching.ddp_patch import ddp_patch
ddp_patch()

# -----------------------------
# Model config
# -----------------------------
max_seq_length = 512
lora_rank = 8
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# IMPORTANT: for 4-bit models under DDP you MUST load on the right GPU per rank
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B",  # any small model for testing
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    device_map={"": local_rank}, 
)

# Apply LoRA. Do NOT call .to(device) afterwards for 4-bit models.
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# -----------------------------
# Tiny fake dataset
# -----------------------------


def get_random_realistic_fake_data(n: int = 100, seed: int = 3407):
    """
    Generate a random but logical fake dataset with 'problem' and 'generated_solution'.
    
    Args:
        n (int): number of examples to generate
        seed (int): random seed for reproducibility
    
    Returns:
        dict: {"problem": [...], "generated_solution": [...]}
    """
    random.seed(seed)

    # Question templates
    math_templates = [
        ("What is {a} + {b}?", lambda a, b: f"The answer is {a+b}."),
        ("What is {a} - {b}?", lambda a, b: f"The answer is {a-b}."),
        ("Multiply {a} and {b}.", lambda a, b: f"The result is {a*b}."),
    ]

    trivia_templates = [
        ("What is the capital of {country}?", lambda country, capital: f"The capital of {country} is {capital}."),
        ("Who wrote {work}?", lambda work, author: f"{work} was written by {author}."),
        ("In which year did {event} happen?", lambda event, year: f"{event} happened in {year}."),
    ]

    countries = {"France": "Paris", "Japan": "Tokyo", "Germany": "Berlin", "Italy": "Rome"}
    works = {"Hamlet": "William Shakespeare", "1984": "George Orwell", "The Odyssey": "Homer"}
    events = {"WW2 end": 1945, "Moon landing": 1969, "Fall of Berlin Wall": 1989}

    problems, solutions = [], []
    for _ in range(n):
        if random.random() < 0.5:  # math
            tmpl, answer_fn = random.choice(math_templates)
            a, b = random.randint(1, 20), random.randint(1, 20)
            problems.append(tmpl.format(a=a, b=b))
            solutions.append(answer_fn(a, b))
        else:  # trivia
            tmpl, answer_fn = random.choice(trivia_templates)
            if "capital" in tmpl:
                country, capital = random.choice(list(countries.items()))
                problems.append(tmpl.format(country=country))
                solutions.append(answer_fn(country, capital))
            elif "wrote" in tmpl:
                work, author = random.choice(list(works.items()))
                problems.append(tmpl.format(work=work))
                solutions.append(answer_fn(work, author))
            else:  # event
                event, year = random.choice(list(events.items()))
                problems.append(tmpl.format(event=event))
                solutions.append(answer_fn(event, year))

    fake_data =  {"problem": problems, "generated_solution": solutions}
    df = pd.DataFrame(fake_data)
    df["Messages"] = df.apply(
        lambda x: [
            {"role": "user", "content": x["problem"]},
            {"role": "assistant", "content": x["generated_solution"]},
        ],
        axis=1,
    )
    df["text"] = tokenizer.apply_chat_template(df["Messages"].tolist(), tokenize=False)
    dataset = Dataset.from_pandas(df)
    return dataset



train_dataset = get_random_realistic_fake_data(n=100, seed=42)
val_dataset = get_random_realistic_fake_data(n=20, seed=2024)

# -----------------------------
# Trainer
# -----------------------------
# Set grad_accum to 1 for multi-GPU, 2 for single-GPU, to match effective batch if you want
world_size = int(os.environ.get("WORLD_SIZE", "1"))
grad_accum = 1 if world_size > 1 else 2

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=10,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        output_dir=f"outputs/debug_worldsize{world_size}",
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
        eval_strategy="epoch"
    ),  
)

trainer.train()
