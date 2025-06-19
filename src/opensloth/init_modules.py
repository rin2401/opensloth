"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os

from opensloth.dataset_utils import get_tokenized_dataset
from opensloth.patching.gemma import patch_gemma3_unsloth_for_sequence_packing

from .logging_config import get_opensloth_logger
from .opensloth_config import OpenSlothConfig, TrainingArguments


def init_model_and_tokenizer(opensloth_config: OpenSlothConfig):
    """Initialize and optionally set up LoRA for the model."""

    from unsloth import FastModel

    logger = get_opensloth_logger()

    logger.start_timing("model_loading")

    if opensloth_config.pretrained_lora:
        logger.info(
            f"Loading model from {opensloth_config.pretrained_lora} with LoRA weights"
        )
        opensloth_config.fast_model_args.model_name = opensloth_config.pretrained_lora
    from opensloth.nccl_grad_sync import setup_nccl_for_opensloth

    model, tokenizer = FastModel.from_pretrained(
        **opensloth_config.fast_model_args.model_dump()
    )
    if (
        "gemma-3" in opensloth_config.fast_model_args.model_name
        and opensloth_config.sequence_packing
    ):
        logger.info(
            "Detected Gemma3 model, applying Unsloth patch for sequence packing."
        )
        patch_gemma3_unsloth_for_sequence_packing()

    if not hasattr(tokenizer, "pad") and opensloth_config.sequence_packing:
        logger.info(
            "Tokenizer missing 'pad' method; attempting to patch using "
            "transformers.AutoTokenizer. This may indicate an Unsloth issue. "
            "See: https://github.com/unslothai/unsloth/issues/2056#event-17007147800"
        )
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(
            opensloth_config.fast_model_args.model_name,
        )
        tokenizer.pad = hf_tokenizer.pad

    logger.finish_timing("model_loading")

    logger.start_timing("nccl_setup")
    setup_nccl_for_opensloth(
        rank=int(os.environ["OPENSLOTH_LOCAL_RANK"]),
        gpus=opensloth_config.devices,
    )
    logger.finish_timing("nccl_setup")

    model_device = model.device
    logger.info(
        f"Model loaded on device {model_device}, tokenizer: {tokenizer.__class__.__name__}"
    )

    if (
        not opensloth_config.fast_model_args.full_finetuning
        and not opensloth_config.pretrained_lora
    ):
        logger.start_timing("lora_setup")
        model = FastModel.get_peft_model(
            model,
            **opensloth_config.lora_args.model_dump(),  # type: ignore
        )
        logger.finish_timing("lora_setup")

    # Allow custom chat templates
    if (
        hasattr(opensloth_config.data, "chat_template")
        and opensloth_config.data.chat_template is not None
    ):
        from unsloth.chat_templates import get_chat_template

        tokenizer = get_chat_template(
            tokenizer, chat_template=opensloth_config.data.chat_template
        )
        logger.info(f"Applied chat template: {opensloth_config.data.chat_template}")

    return model, tokenizer


def create_trainer(
    model,
    tokenizer,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """Load or prepare the dataset and create the SFTTrainer."""

    # Get enhanced logger for timing

    logger = get_opensloth_logger()

    logger.start_timing("trainer_setup")

    trainer = _get_trainer(
        model,
        tokenizer,
        opensloth_config,
        hf_train_args,
    )

    logger.finish_timing("trainer_setup")

    logger.start_timing("training_loop_patch")
    from opensloth.patching.inner_training_loop import patch_inner_training_loop
    from opensloth.patching.patch_log import patch_log
    from opensloth.patching.patch_sampler import patch_sampler

    patch_log(type(trainer))
    patch_inner_training_loop(opensloth_config)

    from .patching.get_batch_samples import patch_get_batch_samples

    patch_get_batch_samples(opensloth_config)

    # ====
    trainer = patch_sampler(trainer)  # type: ignore
    logger.finish_timing("training_loop_patch")

    # ===
    from .patching.patch_sampler import ShuffleData

    logger.info(f"Add callback ShuffleData to Trainer {trainer.__class__.__name__}")
    trainer.add_callback(ShuffleData())

    return trainer


def _get_trainer(
    model,
    tokenizer,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """
    Returns an SFTTrainer instance with a tokenized dataset.
    If cache=False, use untokenized dataset and custom SFTTrainer logic.
    """
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    from .logging_config import get_opensloth_logger

    logger = get_opensloth_logger()

    # If cache is False, use untokenized dataset and custom trainer logic
    if hasattr(opensloth_config.data, "cache") and opensloth_config.data.cache is False:
        from unsloth.chat_templates import train_on_responses_only

        from opensloth.dataset_utils import get_text_dataset

        logger.info(
            "cache=False: Using untokenized dataset and custom SFTTrainer setup."
        )
        text_dataset = get_text_dataset(opensloth_config.data)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,  # type: ignore
            train_dataset=text_dataset,
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
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
        logger.info("Custom SFTTrainer with train_on_responses_only applied.")
        return trainer

    # Default: use tokenized dataset
    tokenized_train_dataset = get_tokenized_dataset(
        config=opensloth_config.data,
    )

    logger.info("Creating final SFTTrainer with prepared dataset...")
    logger.start_timing("final_trainer_creation")
    hf_train_args.skip_prepare_dataset = True
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        args=hf_train_args,  # type: ignore
        tokenizer=tokenizer,  # type: ignore
    )
    logger.finish_timing("final_trainer_creation")

    if (
        hasattr(trainer, "data_collator")
        and not isinstance(trainer.data_collator, DataCollatorForSeq2Seq)
        and opensloth_config.sequence_packing
    ):
        logger.info(
            f"Replacing {type(trainer.data_collator).__name__} with "
            f"DataCollatorForSeq2Seq for better sequence handling"
        )
        trainer.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    else:
        logger.info(f"Data collator: {type(trainer.data_collator).__name__}")

    logger.info("Trainer setup completed successfully")
    return trainer


def configure_batch_size(hf_train_args, gpu_ith, num_gpus):
    if num_gpus != 1:
        hf_train_args.per_device_train_batch_size *= num_gpus  # This is the total batch size loaded by dataloader, the trainer later will chose the correct batch size for each GPU

    if not gpu_ith == 0:
        hf_train_args.report_to = "none"


__all__ = [
    "configure_batch_size",
    "init_model_and_tokenizer",
    "create_trainer",
]
