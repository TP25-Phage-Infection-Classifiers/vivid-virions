# models/_prep.py
import pandas as pd

def prepare_xy(df: pd.DataFrame):
    # choose label column robustly
    label_col = "classification" if "classification" in df.columns else "classification_x"
    drop_cols = {"Unnamed: 0","Geneid","DNASequence","classification_y","group", label_col}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols].copy(), df[label_col].copy(), feature_cols

def align_columns(Xtr: pd.DataFrame, Xte: pd.DataFrame):
    # add missing test columns (0.0), drop extras, reorder to train
    for c in Xtr.columns:
        if c not in Xte.columns:
            Xte[c] = 0.0
    extra = [c for c in Xte.columns if c not in Xtr.columns]
    if extra:
        Xte = Xte.drop(columns=extra)
    return Xtr, Xte[Xtr.columns]
