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
    patch_trainer_loss_scaling()
    return device


def patch_trainer_get_batch_samples() -> None:
    """Optimal dynamic programming repacking of variable-length samples to improve padding efficiency."""
    from typing import Dict, Iterable, List, Sequence, Tuple

    import torch
    from trl import SFTTrainer  # type: ignore

    if hasattr(SFTTrainer, "_opensloth_original_get_batch_samples"):
        return

    original_get_batch_samples = SFTTrainer.get_batch_samples

    class PackedBatch:
        """Holds variable-length samples before padding into tensors."""
        def __init__(self):
            self.input_ids: List[torch.Tensor] = []
            self.attention_mask: List[torch.Tensor] = []
            self.labels: List[torch.Tensor] = []

        def max_length(self) -> int:
            if not self.input_ids:
                return 0
            return max(t.size(0) for t in self.input_ids)

        def to_dict(self) -> Dict[str, torch.Tensor]:
            """Pad samples into a fixed-size batch dictionary."""
            if not self.input_ids:
                return {
                    "input_ids": torch.empty(0, dtype=torch.long),
                    "attention_mask": torch.empty(0, dtype=torch.long),
                    "labels": torch.empty(0, dtype=torch.long),
                }

            batch_size = len(self.input_ids)
            max_len = self.max_length()

            input_ids_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
            attention_mask_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
            labels_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long)

            for i, (ids, mask, lbls) in enumerate(zip(self.input_ids, self.attention_mask, self.labels)):
                L = ids.size(0)
                input_ids_tensor[i, :L] = ids
                attention_mask_tensor[i, :L] = mask
                labels_tensor[i, :L] = lbls

            return {
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor,
                "labels": labels_tensor,
            }

    def summarize_batches(batches: Sequence[Dict[str, torch.Tensor]], label: str) -> Tuple[int, int]:
        total_tokens = sum(batch["input_ids"].numel() for batch in batches)
        useful_tokens = sum((batch["labels"] != -100).sum().item() for batch in batches)
        efficiency = (useful_tokens / total_tokens) if total_tokens else 0.0
        if label:  # Only print if label is provided
            print(
                f"[opensloth.ddp_patch] {label}: total_tokens={total_tokens} "
                f"useful_tokens={useful_tokens} efficiency={efficiency:.2%}"
            )
        return total_tokens, useful_tokens

    def extract_samples(batches: Iterable[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        Flatten a list of padded batches into per-sample dicts with true (unpadded) length.
        Sorts samples by length descending (longest-first).
        """
        samples: List[Dict[str, torch.Tensor]] = []
        for batch in batches:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            lengths = attention_mask.sum(dim=1).tolist()

            for row, length in enumerate(lengths):
                length = int(length)
                samples.append(
                    {
                        "input_ids": input_ids[row, :length],
                        "attention_mask": attention_mask[row, :length],
                        "labels": labels[row, :length],
                        "length": length,
                    }
                )

        # Sort by sequence length descending (longest first)
        samples.sort(key=lambda item: item["length"], reverse=True)
        return samples

    def _optimal_partition_by_length(lengths: List[int], k: int) -> List[Tuple[int, int]]:
        """
        Partition a descending-sorted length array into k contiguous, non-empty groups
        minimizing sum over groups of (max_length_in_group * group_size).
        Returns list of (start_idx, end_idx) inclusive, covering 0..n-1.
        DP complexity: O(n * k^2) worst-case (fine for typical batch sizes).
        """
        n = len(lengths)
        if k <= 0:
            raise ValueError("k must be >= 1")
        if n < k:
            # It's impossible to have k non-empty groups if we have fewer samples than groups.
            # Caller ensures batches come from real dataloaders, so assert loudly here.
            raise ValueError(f"Not enough samples ({n}) to fill {k} non-empty batches")

        INF = 10**18
        # dp[t][j]: min cost to partition 0..j into t groups (1-index t)
        dp = [[INF] * n for _ in range(k)]
        prev = [[-1] * n for _ in range(k)]  # prev split index s for dp[t][j]: last group is s+1..j

        # Base: t=1 -> one group: cost is lengths[0] * (j+1)
        for j in range(n):
            dp[0][j] = lengths[0] * (j + 1)
            prev[0][j] = -1  # start at 0

        # Fill DP
        for t in range(1, k):  # groups 2..k
            # We need at least t items to form t groups (each non-empty), so j >= t
            for j in range(t, n):
                best_cost = INF
                best_s = -1
                # s is end index of previous partition; last group is s+1..j, so s ∈ [t-2 .. j-1]
                s_min = t - 2
                if s_min < -1:
                    s_min = -1
                for s in range(max(s_min, -1), j):
                    # cost of previous t groups on 0..s, plus cost of group (s+1..j)
                    # since lengths sorted desc, max of group (s+1..j) is lengths[s+1]
                    group_size = j - (s + 1) + 1  # = j - s
                    cost = (dp[t - 1][s] if s >= 0 else INF) + lengths[s + 1] * group_size
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s
                dp[t][j] = best_cost
                prev[t][j] = best_s

        # Reconstruct boundaries
        bounds: List[Tuple[int, int]] = []
        t = k - 1
        j = n - 1
        while t >= 0:
            s = prev[t][j]
            start = s + 1 if t > 0 else 0
            bounds.append((start, j))
            j = s
            t -= 1
        bounds.reverse()
        return bounds

    def repack_batches(batches: Sequence[Dict[str, torch.Tensor]], verbose: bool = False) -> List[Dict[str, torch.Tensor]]:
        """
        Optimal repacking (given fixed number of output batches and non-empty constraint).
        1) Extract and sort samples by length (desc).
        2) Use DP to partition into K contiguous groups minimizing sum(max_len * group_size).
        3) Assign groups to K new PackedBatch containers.
        """
        import time
        
        start_time = time.time()
        
        K = len(batches)
        samples = extract_samples(batches)
        lengths = [s["length"] for s in samples]
        bounds = _optimal_partition_by_length(lengths, K)

        if verbose:
            print("[opensloth.ddp_patch] DP group boundaries (start,end,len,max):")
        packed_batches = [PackedBatch() for _ in range(K)]

        for bi, (lo, hi) in enumerate(bounds):
            # max len inside the group is at 'lo' (descending order)
            if verbose:
                group_max = samples[lo]["length"]
                print(f"  Batch[{bi}] = [{lo}:{hi}] "
                      f"(size={hi-lo+1}, max={group_max})")
            target = packed_batches[bi]
            for idx in range(lo, hi + 1):
                s = samples[idx]
                target.input_ids.append(s["input_ids"])
                target.attention_mask.append(s["attention_mask"])
                target.labels.append(s["labels"])

        # Convert to padded tensors
        result = [pb.to_dict() for pb in packed_batches]
        
        if verbose:
            elapsed_time = time.time() - start_time
            print(f"[opensloth.ddp_patch] Repacking completed in {elapsed_time:.3f}s")
        
        return result

    def patched(self, *args, **kwargs):
        batches, count = original_get_batch_samples(self, *args, **kwargs)
        original_total, original_useful = summarize_batches(batches, "")
        repacked = repack_batches(batches, verbose=True)
        repacked_total, repacked_useful = summarize_batches(repacked, "")
        
        original_eff = (original_useful / original_total) if original_total else 0.0
        repacked_eff = (repacked_useful / repacked_total) if repacked_total else 0.0
        print(f"[opensloth.ddp_patch] Repacking improved efficiency from {original_eff:.1%} to {repacked_eff:.1%}")
        
        return repacked, count

    SFTTrainer._opensloth_original_get_batch_samples = original_get_batch_samples
    SFTTrainer.get_batch_samples = patched  # type: ignore[assignment]
    print("[opensloth.ddp_patch] Patched SFTTrainer.get_batch_samples with optimal DP repacking")



def patch_trainer_loss_scaling():
    """
    Replace 'loss *= self.accelerator.num_processes' with
    'loss = loss.clone() * self.accelerator.num_processes'
    inside HuggingFace Trainer.compute_loss.
    """
    import inspect
    import textwrap
    import transformers.trainer as hf_trainer

    src = inspect.getsource(hf_trainer.Trainer.compute_loss)

    if "loss *= self.accelerator.num_processes" in src:
        new_src = src.replace(
            "loss *= self.accelerator.num_processes",
            "loss = loss.clone() * self.accelerator.num_processes",
        )
        # Remove indentation to make it a module-level function
        new_src = textwrap.dedent(new_src)
        
        code_obj = compile(new_src, filename="<patched_compute_loss>", mode="exec")
        ns = {}
        exec(code_obj, hf_trainer.__dict__, ns)

        # Directly assign the function to the class
        hf_trainer.Trainer.compute_loss = ns["compute_loss"]
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


__all__ = [
    "ddp_patch",
    "patch_trainer_get_batch_samples",
    "patch_trainer_loss_scaling",
    "patch_trainer_deterministic_sampler",
]
