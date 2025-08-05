# models/XGBoost/train.py
import argparse, pickle
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from .._utils import find_logo_indices, load_logo_split, load_stratified_split
from .._prep import prepare_xy, align_columns


def train_and_save(Xtr, ytr, Xte, yte, out_dir: Path, idx: int):
    le = LabelEncoder(); le.fit(["early","middle","late"])
    ytr = le.transform(ytr); yte = le.transform(yte)

    xgb = XGBClassifier(
        objective="multi:softprob", num_class=3, random_state=42,
        n_estimators=400, learning_rate=0.1, max_depth=6,
        subsample=0.9, colsample_bytree=0.7,
        tree_method="hist", n_jobs=-1
    )
    xgb.fit(Xtr, ytr)
    ypred = xgb.predict(Xte)
    acc = accuracy_score(yte, ypred)
    rep = classification_report(yte, ypred, target_names=le.classes_, zero_division=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"model_split_{idx}.joblib", "wb") as f:
        pickle.dump(xgb, f)
    with open(out_dir / f"split_{idx}_data.pkl", "wb") as f:
        pickle.dump((Xtr.values, ytr, Xte.values, yte, le, list(Xtr.columns)), f)
    print(f"[XGB] split {idx}: acc={acc:.4f}")
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="pipeline_results")
    ap.add_argument("--out-dir",  default="pipeline_results/models/xgboost")
    ap.add_argument("--splits", choices=["logo","stratified","both"], default="both")
    args = ap.parse_args()

    base = Path(args.base_dir); out_dir = Path(args.out_dir)

    if args.splits in ("logo","both"):
        for i in find_logo_indices(base / "splits_logo"):
            tr, te = load_logo_split(base, i)
            Xtr, ytr, _ = prepare_xy(tr); Xte, yte, _ = prepare_xy(te)
            Xtr, Xte = align_columns(Xtr, Xte)
            train_and_save(Xtr, ytr, Xte, yte, out_dir, i)

    if args.splits in ("stratified","both"):
        tr, te = load_stratified_split(base)
        Xtr, ytr, _ = prepare_xy(tr); Xte, yte, _ = prepare_xy(te)
        Xtr, Xte = align_columns(Xtr, Xte)
        train_and_save(Xtr, ytr, Xte, yte, out_dir, 6)

if __name__ == "__main__":
    main()
