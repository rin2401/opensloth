#   For code inside cache_unsloth_dataset: find them in https://docs.unsloth.ai/get-started/unsloth-notebooks
# Prepares and caches a text dataset for Unsloth fine-tuning.
# Loads, formats, and saves the dataset to 'unsloth_text_dataset'.


def dump_data():
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3-1b-it",  # Choose the smallest just to by pass
        max_seq_length=2048,  # Choose any for long context!
        load_in_4bit=True,  # 4 bit quantization to reduce memory
        load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning=False,  # [NEW!] We have full finetuning now!
        # token = "hf_...", # use one if using gated models
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Turn off for just text!
        finetune_language_layers=True,  # Should leave on!
        finetune_attention_modules=True,  # Attention good for GRPO
        finetune_mlp_modules=True,  # SHould leave on always!
        r=8,  # Larger = higher accuracy, but might overfit
        lora_alpha=8,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    from datasets import load_dataset
    from unsloth.chat_templates import get_chat_template, standardize_data_formats

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )
    dataset = load_dataset("mlabonne/FineTome-100k", split="train")
    dataset = standardize_data_formats(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {
            "text": texts,
        }

    dataset = dataset.map(formatting_prompts_func, batched=True)

    from trl import SFTConfig, SFTTrainer

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            dataset_num_proc=2,
        ),
    )

    from unsloth.chat_templates import train_on_responses_only

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    # now we dont train we save dataset and out
    trainer.train_dataset.save_to_disk("data/cache_gemma_responses_only")


if __name__ == "__main__":
    dump_data()
