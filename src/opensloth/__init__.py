import pathlib


OPENSLOTH_DATA_DIR = pathlib.Path("~/.cache/opensloth").expanduser().resolve()

# Import utilities for convenient access
__all__ = [ "OPENSLOTH_DATA_DIR"]
