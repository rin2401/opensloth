import pathlib


OPENSLOTH_DATA_DIR = pathlib.Path("~/.cache/opensloth").expanduser().resolve()

# Import utilities for convenient access
from .utils import cache_dataset_wrapper, simple_cache_dataset_wrapper, CacheConfig

__all__ = [
    "OPENSLOTH_DATA_DIR",
    "cache_dataset_wrapper",
    "simple_cache_dataset_wrapper", 
    "CacheConfig",
]
