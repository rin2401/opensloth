#!/bin/bash
# Sanity check training script - runs all configurations sequentially

set -e  # Exit on any error

echo "🚀 Starting OpenSloth sanity check training..."
echo "This will run 4 different configurations sequentially:"
echo "1. Single GPU (Python)"
echo "2. 2 GPUs (torchrun with patches)"
echo "3. 4 GPUs (torchrun with patches)" 
echo "4. 4 GPUs (torchrun without patches)"
echo ""

# Base configuration
EXPERIMENT_NAME="opensloth_sanity_$(date +%Y%m%d_%H%M%S)"
TRAIN_SCRIPT="train_scripts/train_configurable.py"

# Common args
COMMON_ARGS="--max_seq_length 4096 --num_epochs 10 --train_samples 80 --test_samples 10 --experiment_name $EXPERIMENT_NAME"

echo "📊 Experiment name: $EXPERIMENT_NAME"
echo ""

# # 1. Single GPU training
# echo "=== 1/4: Single GPU Training ==="
# CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 LOCAL_RANK=0 RANK=0 MASTER_ADDR=localhost MASTER_PORT=29501 python $TRAIN_SCRIPT $COMMON_ARGS --per_device_batch_size 8 --no_patches
# echo "✅ Single GPU training completed!"
# echo ""

# 2. 2 GPU training with patches
# echo "=== 2/4: 2 GPU Training (with patches) ==="
# torchrun --nproc_per_node=2 --master_port=29500 $TRAIN_SCRIPT $COMMON_ARGS --per_device_batch_size 4
# echo "✅ 2 GPU training completed!"
# echo ""

# 3. 4 GPU training with patches
echo "=== 3/4: 4 GPU Training (with patches) ==="
torchrun --nproc_per_node=4 --master_port=29500 $TRAIN_SCRIPT $COMMON_ARGS --per_device_batch_size 2
echo "✅ 4 GPU training completed!"
echo ""

# # 4. 4 GPU training without patches
# echo "=== 4/4: 4 GPU Training (without patches) ==="
# torchrun --nproc_per_node=4 --master_port=29500 $TRAIN_SCRIPT $COMMON_ARGS --per_device_batch_size 2 --no_patches
# echo "✅ 4 GPU training (no patches) completed!"
# echo ""

echo "🎉 All sanity check training completed!"
echo "Check wandb for results under experiment: $EXPERIMENT_NAME"