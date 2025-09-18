<p align="center">
    <img src="images/opensloth.png" alt="opensloth Logo" width="200" />
</p>

# OpenSloth 🦥⚡

**The Simplest Way to Scale Unsloth to Multiple GPUs**

OpenSloth makes multi-GPU training with [Unsloth](https://github.com/unslothai/unsloth) as simple as a single command. No complex configuration files, no custom multiprocessing - just pure PyTorch DDP with `torchrun`.

**Why OpenSloth?**
- 🚀 **2-4x faster** than single GPU Unsloth training
- 🎯 **Zero configuration** - works out of the box
- ⚡ **Standard PyTorch DDP** - no custom frameworks
- 💾 **Memory efficient** - same VRAM usage per GPU
- 🔧 **Works with any Unsloth model** - Qwen, Llama, Gemma, etc.

## 💾 Installation

### 1. Install uv (if not already installed)
First, install the `uv` package manager. See the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) for detailed instructions.

```bash
# Quick install (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

### 2. Create project and install dependencies
```bash
# Create Python environment and install dependencies in one command
uv init opensloth-project --python 3.11
cd opensloth-project

# Install PyTorch with CUDA support
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
uv add unsloth datasets transformers trl

# Install OpenSloth
uv add git+https://github.com/anhvth/opensloth.git
# or for development: git clone https://github.com/anhvth/opensloth.git && cd opensloth && uv add -e .

# Activate the environment
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
```

## ⚡ Quick Start

That's it! Just use `torchrun` instead of `python`:

```bash
# Single GPU (traditional)
python train_scripts/train_ddp.py

# Multi-GPU (the magic happens here! 🪄)
torchrun --nproc_per_node=2 train_scripts/train_ddp.py
torchrun --nproc_per_node=4 train_scripts/train_ddp.py
torchrun --nproc_per_node=8 train_scripts/train_ddp.py
```

That's literally it! OpenSloth automatically:
- ✅ Detects number of GPUs
- ✅ Distributes model across GPUs  
- ✅ Synchronizes gradients
- ✅ Adjusts batch sizes
- ✅ Handles all DDP setup

## 📊 Performance

**Real benchmark results** - Qwen3-8B on 2x RTX 4090:

| Setup | Time | Speedup |
|-------|------|---------|
| Unsloth (1 GPU) | 19m 34s | 1.0x |
| **OpenSloth (2 GPUs)** | **8m 28s** | **� 2.3x** |

*Note: >2x speedup thanks to efficient gradient synchronization and sequence packing*

**Expected scaling:**
- 2 GPUs: ~2.3x faster
- 4 GPUs: ~4.5x faster  
- 8 GPUs: ~9x faster

## 🎯 Usage Examples

### Basic Fine-tuning
```python
# train_my_model.py
import os
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from opensloth.patching.ddp_patch import ddp_patch

ddp_patch()  # Enable DDP compatibility

# Standard Unsloth setup
local_rank = int(os.environ.get("LOCAL_RANK", 0))
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-1.5B",
    device_map={"": local_rank},  # Key: assign each process to its GPU
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(model, r=16, ...)

# Standard TRL trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=your_dataset,
    args=SFTConfig(...),
)

trainer.train()
```

Run with: `torchrun --nproc_per_node=4 train_my_model.py`

### Different Model Sizes

```bash
# Small models (0.5B-1.5B) - Great for testing
torchrun --nproc_per_node=2 train_scripts/train_ddp.py

# Medium models (3B-8B) - Production training  
torchrun --nproc_per_node=4 train_scripts/train_ddp.py

# Large models (8B+) - Full utilization
torchrun --nproc_per_node=8 train_scripts/train_ddp.py
```

## 🔧 Key Features

### Automatic Scaling
OpenSloth automatically adjusts training parameters based on GPU count:

```python
world_size = int(os.environ.get("WORLD_SIZE", "1"))  # Auto-detected
grad_accum = 1 if world_size > 1 else 2  # Smart batching
effective_batch = batch_size * grad_accum * world_size
```

### Memory Efficiency
- Uses same VRAM per GPU as single-GPU Unsloth
- 4-bit quantization + LoRA adapters
- Gradient checkpointing for large models

### Monitoring
```bash
# Monitor training progress
tensorboard --logdir outputs/

# Check GPU utilization
nvidia-smi
```

## 🚀 Migration from Old OpenSloth

Migrating from the old complex approach? It's incredibly simple:

**Before (old approach):**
```python
# Complex configuration files
opensloth_config = OpenSlothConfig(...)
training_config = TrainingArguments(...)
run_mp_training(gpus, opensloth_config, training_config)
```

**After (new approach):**
```python
# Just add ddp_patch() and use torchrun!
from opensloth.patching.ddp_patch import ddp_patch
ddp_patch()

# Rest is standard Unsloth code
model, tokenizer = FastLanguageModel.from_pretrained(...)
trainer = SFTTrainer(...)
trainer.train()
```

Then run: `torchrun --nproc_per_node=N your_script.py`

## 🔧 Troubleshooting

**GPU not being utilized?**
```bash
# Check if all GPUs are visible
nvidia-smi

# Verify CUDA setup
python -c "import torch; print(torch.cuda.device_count())"
```

**Out of memory errors?**
```python
# Reduce batch size or sequence length
args=SFTConfig(
    per_device_train_batch_size=1,  # Start with 1
    max_length=512,                 # Reduce if needed
)
```

**Training hanging?**
```bash
# Check if torchrun is working
torchrun --nproc_per_node=1 -c "import torch; print('Working!')"
```

## � Advanced Usage

### Custom Datasets
```python
from datasets import Dataset

# Your data preparation
dataset = Dataset.from_dict({
    "text": [tokenizer.apply_chat_template(conv) for conv in conversations]
})

trainer = SFTTrainer(train_dataset=dataset, ...)
```

### Different Models
```python
# Works with any Unsloth model!
models = [
    "unsloth/Qwen3-0.5B",
    "unsloth/Llama-3.2-1B", 
    "unsloth/gemma-2-2b",
    "unsloth/Phi-3.5-mini",
]
```

## 🤝 Contributing

OpenSloth is open source! We welcome contributions:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📖 What's in the Box?

- `train_scripts/train_ddp.py` - **Main training script** (start here!)
- `src/opensloth/patching/ddp_patch.py` - DDP compatibility layer
- `legacy/` - Old complex approach (reference only)
- `examples/` - More usage examples

## 🔗 Links

- **[Unsloth](https://github.com/unslothai/unsloth)** - The amazing 2x faster training library
- **[Unsloth Docs](https://docs.unsloth.ai/)** - Official documentation  
- **[TRL](https://github.com/huggingface/trl)** - Transformer Reinforcement Learning
- **[PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)** - Distributed training docs

---

**Ready to scale up your training?** 

```bash
git clone https://github.com/anhvth/opensloth.git
cd opensloth  
torchrun --nproc_per_node=4 train_scripts/train_ddp.py
```

*Happy training! 🦥⚡*
