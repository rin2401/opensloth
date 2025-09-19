# OpenSloth Sanity Check Training

This setup provides a minimal, configurable training script that can run different GPU configurations sequentially.

## Quick Start

Run all configurations sequentially:

```bash
./scripts/sanity_check_training.sh
```

## Manual Usage

You can also run individual configurations:

### Single GPU

```bash
python train_scripts/train_configurable.py --experiment_name my_test
```

### 2 GPUs with patches

```bash
torchrun --nproc_per_node=2 train_scripts/train_configurable.py --experiment_name my_test --per_device_batch_size 4
```

### 4 GPUs with patches

```bash
torchrun --nproc_per_node=4 train_scripts/train_configurable.py --experiment_name my_test --per_device_batch_size 2
```

### 4 GPUs without patches

```bash
torchrun --nproc_per_node=4 train_scripts/train_configurable.py --experiment_name my_test --per_device_batch_size 2 --no_patches
```

## Configuration Options

- `--max_seq_length`: Maximum sequence length (default: 4096)
- `--num_epochs`: Number of training epochs (default: 20)
- `--train_samples`: Number of training samples (default: 500)
- `--test_samples`: Number of test samples (default: 50)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--per_device_batch_size`: Batch size per device (default: 8)
- `--grad_accum`: Gradient accumulation steps (default: 4)
- `--no_patches`: Disable DDP patches
- `--experiment_name`: Wandb experiment name

## Features

- ✅ Configurable parameters via command line
- ✅ Wandb logging (replaces tensorboard)
- ✅ Sequential execution of different GPU configurations
- ✅ Automatic batch size adjustment for different GPU counts
- ✅ Optional DDP patches
- ✅ Timestamped experiment names
