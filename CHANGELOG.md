# Changelog

## vNEXT (2025-06-21)

### Added
- **Dataset caching scripts**
  - `cache_unsloth_dataset/README.md`, `cache_unsloth_dataset/gemma.py`, `cache_unsloth_dataset/qwen3.py`: Scripts and docs for preparing and caching datasets for Unsloth fine-tuning.

### Changed
- **Training scripts**
  - `scripts/train_gemma.py`, `scripts/train_qwen.py`: Improved dataset caching, sequence packing, and multi-GPU support. Updated configs for Gemma-3 and Qwen3 models.
- **Configuration and patching**
  - `src/opensloth/dataset_utils.py`, `src/opensloth/init_modules.py`, `src/opensloth/opensloth_config.py`, `src/opensloth/patching/inner_training_loop.py`: Internal improvements for dataset handling, model initialization, and training loop patching.
- **Documentation**
  - `README.md`: Updated installation, usage, and dataset preparation instructions.

### Other
- `.gitignore`: Updated to reflect new cache and notebook files.

---

See the README for usage and upgrade notes.

## vNEXT (2025-06-19)

### Added
- **Gemma-3 support**  
  - `scripts/train_gemma.py`: Now supports Gemma-3 models and templates (`gemma-3`, `gemma3`).
- **Tokenized dataset caching**  
  - (See option in: `scripts/train_gemma.py`, possibly `src/opensloth/dataset_utils.py`)
- **Unsloth Gemma-3 patch for sequence packing**  
  - `src/opensloth/patching/gemma.py`: Patch for Gemma-3 sequence packing
  - `src/opensloth/init_modules.py`: Detects Gemma-3 and applies patch when sequence packing is enabled

### Changed
- Internal logic to detect Gemma-3 and apply the patch automatically when sequence packing is enabled  
  - `src/opensloth/init_modules.py`

---

See the README for usage and upgrade notes.
