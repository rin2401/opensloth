from datasets import load_from_disk

from opensloth.init_modules import _ensure_data_correct

if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    train_dataset = load_from_disk(path)
    _ensure_data_correct(train_dataset)
