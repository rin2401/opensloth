#!/usr/bin/env python3
"""
Dataset caching utilities for distributed training.

This module provides decorators and utilities for caching processed datasets
to avoid redundant processing in multi-GPU distributed training scenarios.
"""

import os
import time
import functools
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

try:
    from datasets import Dataset, load_from_disk
    from trl.trainer.sft_trainer import SFTTrainer
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    Dataset = Any
    SFTTrainer = Any


class CacheConfig:
    """Configuration for dataset caching behavior."""
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        timeout_seconds: int = 3600,
        ready_flag_name: str = ".ready",
        train_subdir: str = "train_dataset",
        val_subdir: str = "val_dataset",
        metadata_file: str = "cache_metadata.json",
        enable_validation: bool = True,
    ):
        """
        Initialize cache configuration.
        
        Args:
            cache_dir: Base directory for caching datasets
            timeout_seconds: Maximum time to wait for cache creation (default: 1 hour)
            ready_flag_name: Name of the file that signals cache is ready
            train_subdir: Subdirectory name for training dataset
            val_subdir: Subdirectory name for validation dataset
            metadata_file: File to store cache metadata
            enable_validation: Whether to validate cache integrity
        """
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.ready_flag_name = ready_flag_name
        self.train_subdir = train_subdir
        self.val_subdir = val_subdir
        self.metadata_file = metadata_file
        self.enable_validation = enable_validation


def _check_distributed_environment() -> Tuple[int, int, int]:
    """
    Check and validate distributed training environment.
    
    Returns:
        Tuple of (local_rank, world_size, global_rank)
        
    Raises:
        RuntimeError: If distributed environment is not properly configured
    """
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        global_rank = int(os.environ.get("RANK", 0))
    except ValueError as e:
        raise RuntimeError(
            "Invalid distributed environment variables. "
            "LOCAL_RANK, WORLD_SIZE, and RANK must be integers."
        ) from e
    
    # Validate environment consistency
    if world_size > 1:
        if local_rank >= world_size:
            raise RuntimeError(
                f"LOCAL_RANK ({local_rank}) must be less than WORLD_SIZE ({world_size})"
            )
        if global_rank >= world_size:
            raise RuntimeError(
                f"RANK ({global_rank}) must be less than WORLD_SIZE ({world_size})"
            )
    
    return local_rank, world_size, global_rank


def _generate_cache_key(**kwargs) -> str:
    """
    Generate a deterministic cache key based on parameters.
    
    Args:
        **kwargs: Key-value pairs to include in cache key
        
    Returns:
        String cache key
    """
    # Sort by key to ensure deterministic ordering
    sorted_items = sorted(kwargs.items())
    key_parts = []
    
    for key, value in sorted_items:
        if value is not None:
            # Clean and format the value
            if isinstance(value, str):
                clean_value = value.replace("/", "_").replace("\\", "_")
            else:
                clean_value = str(value)
            key_parts.append(f"{key}={clean_value}")
    
    return "_".join(key_parts)


def _wait_for_cache_ready(ready_flag_path: str, timeout_seconds: int, local_rank: int) -> None:
    """
    Wait for cache ready flag with timeout and progress logging.
    
    Args:
        ready_flag_path: Path to the ready flag file
        timeout_seconds: Maximum time to wait
        local_rank: Current process rank
        
    Raises:
        TimeoutError: If timeout is exceeded
    """
    start_time = time.time()
    last_log_time = start_time
    log_interval = 30  # Log every 30 seconds
    
    while not os.path.exists(ready_flag_path):
        elapsed = time.time() - start_time
        
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Rank {local_rank} timed out waiting for cache at {ready_flag_path} "
                f"after {timeout_seconds} seconds"
            )
        
        # Log progress periodically
        if elapsed - (last_log_time - start_time) >= log_interval:
            print(f"⏳ [Rank {local_rank}] Still waiting for cache... ({elapsed:.0f}s elapsed)")
            last_log_time = time.time()
        
        time.sleep(2)


def _save_cache_metadata(cache_dir: str, metadata: Dict[str, Any]) -> None:
    """Save cache metadata for validation and debugging."""
    import json
    
    metadata_path = os.path.join(cache_dir, "cache_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def _load_cache_metadata(cache_dir: str) -> Optional[Dict[str, Any]]:
    """Load cache metadata if it exists."""
    import json
    
    metadata_path = os.path.join(cache_dir, "cache_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            warnings.warn(f"Failed to load cache metadata: {e}")
    return None


def cache_dataset_wrapper(
    cache_config: Optional[CacheConfig] = None,
    cache_key_params: Optional[Dict[str, Any]] = None,
    validate_cache: bool = True,
) -> Callable:
    """
    Decorator for caching processed datasets in distributed training.
    
    This decorator caches the datasets created by an SFTTrainer factory function
    to avoid redundant processing across multiple GPU ranks in distributed training.
    
    **Preconditions:**
    - Must be used in a distributed training environment with proper environment variables
    - The decorated function must return an SFTTrainer instance
    - The SFTTrainer must have train_dataset and optionally eval_dataset attributes
    - The datasets library must be installed
    
    **Workflow:**
    - Rank 0: Checks cache, creates datasets if not cached, saves to cache
    - Other ranks: Wait for rank 0 to create cache, then load from cache
    
    Args:
        cache_config: Configuration for caching behavior
        cache_key_params: Additional parameters to include in cache key generation
        validate_cache: Whether to validate cache integrity
        
    Returns:
        Decorated function
        
    Raises:
        RuntimeError: If preconditions are not met
        ImportError: If required dependencies are missing
        TimeoutError: If cache creation times out
        
    Example:
        ```python
        @cache_dataset_wrapper(
            cache_config=CacheConfig(cache_dir="./my_cache"),
            cache_key_params={"model_name": "qwen", "seq_len": 4096}
        )
        def build_trainer(model, tokenizer, train_dataset, val_dataset, world_size):
            return SFTTrainer(...)
        ```
    """
    # Check dependencies
    if not HAS_DATASETS:
        raise ImportError(
            "The 'datasets' and 'trl' libraries are required for dataset caching. "
            "Install with: pip install datasets trl"
        )
    
    # Set defaults
    if cache_config is None:
        cache_config = CacheConfig()
    if cache_key_params is None:
        cache_key_params = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(model, tokenizer, train_dataset, val_dataset, world_size, *args, **kwargs):
            # Validate distributed environment
            local_rank, actual_world_size, global_rank = _check_distributed_environment()
            
            # Consistency check
            if world_size != actual_world_size:
                warnings.warn(
                    f"Function world_size ({world_size}) differs from environment "
                    f"WORLD_SIZE ({actual_world_size}). Using environment value."
                )
                world_size = actual_world_size
            
            # Generate cache key
            cache_key = _generate_cache_key(
                world_size=world_size,
                train_samples=len(train_dataset) if train_dataset else 0,
                val_samples=len(val_dataset) if val_dataset else 0,
                **cache_key_params
            )
            
            # Setup cache paths
            cache_dir = os.path.join(cache_config.cache_dir, f"{cache_key}_processed")
            train_cache_path = os.path.join(cache_dir, cache_config.train_subdir)
            val_cache_path = os.path.join(cache_dir, cache_config.val_subdir)
            ready_flag_path = os.path.join(cache_dir, cache_config.ready_flag_name)
            
            if local_rank == 0:
                # Rank 0: Check cache and save if needed
                if os.path.exists(ready_flag_path):
                    print(f"📦 [Rank {local_rank}] Loading cached datasets from {cache_dir}")
                    
                    # Load cached datasets
                    cached_train_dataset = load_from_disk(train_cache_path)
                    cached_val_dataset = None
                    if val_dataset is not None and os.path.exists(val_cache_path):
                        cached_val_dataset = load_from_disk(val_cache_path)
                    
                    # Validate cache if enabled
                    if validate_cache:
                        if len(cached_train_dataset) != len(train_dataset):
                            warnings.warn(
                                f"Cached train dataset size ({len(cached_train_dataset)}) "
                                f"differs from expected size ({len(train_dataset)})"
                            )
                    
                    # Build trainer with cached datasets
                    trainer = func(model, tokenizer, cached_train_dataset, cached_val_dataset, world_size, *args, **kwargs)
                else:
                    print(f"🔄 [Rank {local_rank}] Cache not found, processing datasets and creating cache...")
                    
                    # Build trainer normally
                    trainer = func(model, tokenizer, train_dataset, val_dataset, world_size, *args, **kwargs)
                    
                    # Validate trainer
                    if not hasattr(trainer, 'train_dataset'):
                        raise RuntimeError(
                            "The decorated function must return an object with 'train_dataset' attribute"
                        )
                    
                    # Save datasets to cache
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    # Save metadata
                    metadata = {
                        "cache_key": cache_key,
                        "world_size": world_size,
                        "train_samples": len(trainer.train_dataset),
                        "val_samples": len(trainer.eval_dataset) if trainer.eval_dataset else 0,
                        "created_at": time.time(),
                        "cache_key_params": cache_key_params,
                    }
                    _save_cache_metadata(cache_dir, metadata)
                    
                    # Save datasets
                    trainer.train_dataset.save_to_disk(train_cache_path)
                    if trainer.eval_dataset is not None:
                        trainer.eval_dataset.save_to_disk(val_cache_path)
                    
                    # Create ready flag to signal other ranks
                    with open(ready_flag_path, 'w') as f:
                        f.write(f"ready\ncreated_by_rank_0_at_{time.time()}\n")
                    
                    print(f"✅ [Rank {local_rank}] Datasets cached to {cache_dir}")
            else:
                # Non-zero ranks: Wait for cache and load
                print(f"⏳ [Rank {local_rank}] Waiting for rank 0 to create cache at {cache_dir}")
                
                # Wait for ready flag with timeout
                _wait_for_cache_ready(ready_flag_path, cache_config.timeout_seconds, local_rank)
                
                print(f"📦 [Rank {local_rank}] Loading cached datasets from {cache_dir}")
                
                # Load cached datasets
                cached_train_dataset = load_from_disk(train_cache_path)
                cached_val_dataset = None
                if val_dataset is not None and os.path.exists(val_cache_path):
                    cached_val_dataset = load_from_disk(val_cache_path)
                
                # Build trainer with cached datasets
                trainer = func(model, tokenizer, cached_train_dataset, cached_val_dataset, world_size, *args, **kwargs)
            
            return trainer
        
        return wrapper
    
    return decorator


# Convenience function for common use cases
def simple_cache_dataset_wrapper(
    cache_dir: str = "data/cache",
    timeout_minutes: int = 60,
    **cache_key_params
) -> Callable:
    """
    Simplified version of cache_dataset_wrapper with common defaults.
    
    Args:
        cache_dir: Directory for caching datasets
        timeout_minutes: Timeout in minutes (converted to seconds)
        **cache_key_params: Additional parameters for cache key generation
        
    Returns:
        Decorator function
    """
    config = CacheConfig(
        cache_dir=cache_dir,
        timeout_seconds=timeout_minutes * 60
    )
    
    return cache_dataset_wrapper(
        cache_config=config,
        cache_key_params=cache_key_params
    )