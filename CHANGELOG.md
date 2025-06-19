# Changelog

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
