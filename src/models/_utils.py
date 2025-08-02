# models/_utils.py
from pathlib import Path
import pandas as pd

def find_logo_indices(logo_root: Path) -> list[int]:
    # supports both pipeline_results/splits_logo/train_split_0.tsv
    # and pipeline_results/splits_logo/splits/train_split_0.tsv
    idx = set()
    for pat in ["train_split_*.tsv", "splits/train_split_*.tsv"]:
        for p in (logo_root).glob(pat):
            try:
                idx.add(int(p.stem.split("_")[-1]))
            except:
                pass
    return sorted(idx)

def load_logo_split(base_dir: Path, i: int):
    # try with /splits/, then fallback
    train_fp = base_dir / "splits_logo" / "splits" / f"train_split_{i}.tsv"
    test_fp  = base_dir / "splits_logo" / "splits" / f"test_split_{i}.tsv"
    if not train_fp.exists():
        train_fp = base_dir / "splits_logo" / f"train_split_{i}.tsv"
        test_fp  = base_dir / "splits_logo" / f"test_split_{i}.tsv"
    train = pd.read_csv(train_fp, sep="\t")
    test  = pd.read_csv(test_fp,  sep="\t")
    return train, test

def load_stratified_split(base_dir: Path):
    # supports both train_data.tsv/test_data.tsv and train.tsv/test.tsv
    for train_name, test_name in [("train.tsv","test.tsv"), ("train_data.tsv","test_data.tsv")]:
        train_fp = base_dir / "splits_stratified" / train_name
        test_fp  = base_dir / "splits_stratified" / test_name
        if train_fp.exists() and test_fp.exists():
            return (pd.read_csv(train_fp, sep="\t"), pd.read_csv(test_fp, sep="\t"))
    raise FileNotFoundError("Could not find stratified split files in splits_stratified/")