# type: ignore
# src/opensloth/patching/ddp_patch.py
# Torchrun/DDP monkey patch for TRL's SFTTrainer:
#   - deterministic SequentialSampler
#   - safe loss clone (fixes inplace *= bug)
#   - proper local_rank / device setup
# NOTE: Do NOT import unsloth here; user must import it first in their script.

_ddp_already_patched = False


def ddp_patch():
    """Setup torchrun environment and monkey-patch SFTTrainer."""
    import os
    import torch

    global _ddp_already_patched
    if _ddp_already_patched:
        # Already patched: return device anyway
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")

    _ddp_already_patched = True

    # --- Torchrun env setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"[opensloth.ddp_patch] rank={os.environ.get('RANK')} "
          f"local_rank={local_rank} -> {torch.cuda.get_device_name(local_rank)}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        print("[opensloth.ddp_patch] Multi-GPU detected -> prefer gradient_accumulation_steps=1")
    else:
        print("[opensloth.ddp_patch] Single GPU -> you may use gradient_accumulation_steps=2")



def patch_trainer_loss_scaling():
    """
    Replace 'loss *= self.accelerator.num_processes' with
    'loss = loss.clone() * self.accelerator.num_processes'
    inside HuggingFace Trainer.compute_loss.
    """
    import inspect
    import types
    import transformers.trainer as hf_trainer

    src = inspect.getsource(hf_trainer.Trainer.compute_loss)

    if "loss *= self.accelerator.num_processes" in src:
        new_src = src.replace(
            "loss *= self.accelerator.num_processes",
            "loss = loss.clone() * self.accelerator.num_processes",
        )
        code_obj = compile(new_src, filename="<patched_compute_loss>", mode="exec")
        ns = {}
        exec(code_obj, hf_trainer.__dict__, ns)

        hf_trainer.Trainer.compute_loss = types.MethodType(
            ns["compute_loss"], None, hf_trainer.Trainer
        )
        print("[opensloth.ddp_patch] Patched Trainer.compute_loss inplace loss bug")
    else:
        print("[opensloth.ddp_patch] No inplace loss scaling found, nothing patched")


def patch_trainer_deterministic_sampler():
    """Globally replace TRL’s SFTTrainer with a deterministic subclass for debugging"""
    from torch.utils.data import SequentialSampler
    from trl import SFTTrainer  # type: ignore
    from typing import Optional
    from datasets import Dataset  # type: ignore

    class DeterministicSafeSFTTrainer(SFTTrainer):
        def _get_train_sampler(self, train_dataset: Optional[Dataset] = None):
            if train_dataset is None:
                train_dataset = self.train_dataset
            return SequentialSampler(train_dataset)

        def _get_eval_sampler(self, eval_dataset: Optional[Dataset] = None):
            if eval_dataset is None:
                eval_dataset = self.eval_dataset
            return SequentialSampler(eval_dataset)

    import trl
    trl.SFTTrainer = DeterministicSafeSFTTrainer
    print("[opensloth.ddp_patch] Replaced trl.SFTTrainer with DeterministicSafeSFTTrainer")


__all__ = ["ddp_patch", "patch_trainer_loss_scaling", "patch_trainer_deterministic_sampler"]
