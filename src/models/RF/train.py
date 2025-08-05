# models/RF/train.py
import argparse, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from .._utils import find_logo_indices, load_logo_split, load_stratified_split
from .._prep import prepare_xy, align_columns


def train_and_save(Xtr, ytr, Xte, yte, out_dir: Path, idx: int, top_k: int | None):
    le = LabelEncoder(); le.fit(["early","middle","late"])
    ytr = le.transform(ytr); yte = le.transform(yte)

    rf = RandomForestClassifier(
        random_state=42,
        n_estimators=389, max_depth=13,
        min_samples_split=5, min_samples_leaf=1,
        max_features="sqrt", bootstrap=True,
        criterion="gini", class_weight="balanced_subsample",
        n_jobs=-1,
    )

    feat_cols = list(Xtr.columns)
    if top_k and top_k > 0:
        rf.fit(Xtr, ytr)
        order = np.argsort(rf.feature_importances_)[::-1][:top_k]
        feat_cols = [feat_cols[i] for i in order]

    # final fit
    rf.fit(Xtr[feat_cols], ytr)
    ypred = rf.predict(Xte[feat_cols])
    acc = accuracy_score(yte, ypred); rep = classification_report(yte, ypred, target_names=le.classes_, zero_division=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"model_split_{idx}.joblib", "wb") as f:
        pickle.dump(rf, f)  # joblib is fine too; pickle keeps it simple here
    with open(out_dir / f"split_{idx}_data.pkl", "wb") as f:
        pickle.dump((Xtr[feat_cols].values, ytr, Xte[feat_cols].values, yte, le, feat_cols), f)
    print(f"[RF] split {idx}: acc={acc:.4f}")
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="pipeline_results")
    ap.add_argument("--out-dir",  default="pipeline_results/models/RandomForest")
    ap.add_argument("--splits", choices=["logo","stratified","both"], default="both")
    ap.add_argument("--top-k", type=int, default=36)
    args = ap.parse_args()

    base = Path(args.base_dir); out_dir = Path(args.out_dir)

    if args.splits in ("logo","both"):
        for i in find_logo_indices(base / "splits_logo"):
            tr, te = load_logo_split(base, i)
            Xtr, ytr, _ = prepare_xy(tr); Xte, yte, _ = prepare_xy(te)
            Xtr, Xte = align_columns(Xtr, Xte)
            train_and_save(Xtr, ytr, Xte, yte, out_dir, i, args.top_k)

    if args.splits in ("stratified","both"):
        tr, te = load_stratified_split(base)
        Xtr, ytr, _ = prepare_xy(tr); Xte, yte, _ = prepare_xy(te)
        Xtr, Xte = align_columns(Xtr, Xte)
        train_and_save(Xtr, ytr, Xte, yte, out_dir, 6, args.top_k)

if __name__ == "__main__":
    main()

