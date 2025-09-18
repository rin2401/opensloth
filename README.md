<p align="center">
    <img src="images/opensloth.png" alt="opensloth Logo" width="200" />
</p>

# OpenSloth 🦥⚡

Scale [Unsloth](https://github.com/unslothai/unsloth) to multiple GPUs with just `torchrun`. No configuration files, no custom frameworks - pure PyTorch DDP.

- 🚀 **2-4x faster** than single GPU
- 🎯 **Zero configuration** - works out of the box
- 💾 **Same VRAM per GPU** as single GPU Unsloth
- 🔧 **Any Unsloth model** - Qwen, Llama, Gemma, etc.

## Installation

```bash
# Install dependencies
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv add unsloth datasets transformers trl
uv add git+https://github.com/anhvth/opensloth.git
```

## Quick Start

Replace `python` with `torchrun`:

```bash
# Single GPU
python train_scripts/train_ddp.py

# Multi-GPU 
torchrun --nproc_per_node=2 train_scripts/train_ddp.py  # 2 GPUs
torchrun --nproc_per_node=4 train_scripts/train_ddp.py  # 4 GPUs
```

OpenSloth automatically handles GPU distribution, gradient sync, and batch sizing.

## Performance

| Setup | Time | Speedup |
|-------|------|---------|
| 1 GPU | 19m 34s | 1.0x |
| 2 GPUs | 8m 28s | **2.3x** |

Expected scaling: 2 GPUs = ~2.3x, 4 GPUs = ~4.5x, 8 GPUs = ~9x

## Usage

```python
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from opensloth.patching.ddp_patch import ddp_patch

ddp_patch()  # Enable DDP compatibility

# Standard Unsloth setup
local_rank = int(os.environ.get("LOCAL_RANK", 0))
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-1.5B",
    device_map={"": local_rank},
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(model, r=16)
trainer = SFTTrainer(model=model, tokenizer=tokenizer, ...)
trainer.train()
```

Run: `torchrun --nproc_per_node=4 your_script.py`

## Migration from Old Approach

**Current (Recommended):** Simple `torchrun` + DDP patch
```python
from opensloth.patching.ddp_patch import ddp_patch
ddp_patch()
# ... standard Unsloth code
```

**Old Approach (v0.1.8):** For complex configuration files, use:
```bash
git checkout https://github.com/anhvth/opensloth/releases/tag/v0.1.8
```

## Links

- [Unsloth](https://github.com/unslothai/unsloth) - 2x faster training library
- [TRL](https://github.com/huggingface/trl) - Transformer Reinforcement Learning
- [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) - Distributed training

---

```bash
git clone https://github.com/anhvth/opensloth.git
cd opensloth  
torchrun --nproc_per_node=4 train_scripts/train_ddp.py
```

*Happy training! 🦥⚡*
