---
mode: agent
---

**Task:**
Convert the provided code snippets (originally from a Jupyter Notebook and adapted for OpenSloth) into a **standalone OpenSloth training script** that supports **multi-GPU training with Distributed Data Parallel (DDP)**.

**Requirements:**

1. Follow the structure and conventions demonstrated in **`#examples/qwen_sft_cache_dataset.py`**.
2. Ensure **correct import order**:

   - Import **Unsloth** first.
   - Apply **patchers** immediately after Unsloth imports.

3. Correctly set up **DDP initialization and wrapping** for multi-GPU training.
4. Ensure the final script is executable without requiring notebook-style constructs.

**Deliverable:**

- A **clean, runnable Python script** ready for multi-GPU training using OpenSloth with proper DDP setup.

---

This rewritten version:

- Clarifies **input context** (Jupyter → script).
- Defines **priority rules** (import order, patchers, DDP).
- Points to a **reference file** explicitly.
- Specifies the **output format** (a runnable script, not just fragments).
