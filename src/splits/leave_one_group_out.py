# src/splits/leave_one_group_out.py
import os, glob
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut

INPUT_DIR = "pipeline_results/feature_tables"
OUTPUT_DIR = "pipeline_results/splits_logo"
SPLIT_DIR = os.path.join(OUTPUT_DIR, "splits")

def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # 1) combine + group 
    files = sorted(glob.glob(f"{INPUT_DIR}/*.tsv"))
    if not files:
        raise FileNotFoundError(f"No *.tsv found in {INPUT_DIR}")

    df = pd.concat(
        [pd.read_csv(f, sep="\t").assign(group=os.path.basename(f)) for f in files],
        ignore_index=True,
    )
    df.to_csv(os.path.join(OUTPUT_DIR, "combined.tsv"), sep="\t", index=False)

    # Label column
    if "classification" in df.columns:
        label_col = "classification"
    elif "classification_x" in df.columns:
        label_col = "classification_x"
    else:
        label_col = None 

    # 2) LOGO-Splits 
    groups = df["group"].unique()
    if len(groups) < 2:
        raise ValueError(
            f"LeaveOneGroupOut needs >=2 groups, but found {len(groups)} "
            f"({list(groups)}). Add at least one more feature table."
        )

    logo = LeaveOneGroupOut()
    results = []

    for i, (train_idx, test_idx) in enumerate(logo.split(df, groups=df["group"])):
        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]

        # Save split data
        train_df.to_csv(f"{SPLIT_DIR}/train_split_{i}.tsv", sep="\t", index=False)
        test_df.to_csv(f"{SPLIT_DIR}/test_split_{i}.tsv",  sep="\t", index=False)

        # Sanity check
        left_out = df.iloc[test_idx]["group"].iloc[0]
        overlap = len(set(train_df.get("Geneid", [])).intersection(set(test_df.get("Geneid", []))))

        if label_col is not None:
            train_counts = train_df[label_col].value_counts().to_dict()
            test_counts  = test_df[label_col].value_counts().to_dict()
        else:
            train_counts = {}
            test_counts  = {}

        results.append({
            "split": i,
            "left_out_group": left_out,
            "train_total": int(len(train_df)),
            "test_total": int(len(test_df)),
            "overlap_geneids": int(overlap),
            "train_class_counts": train_counts,
            "test_class_counts":  test_counts,
        })

        print(f"Split {i}: left-out = {left_out} | train={len(train_df)} test={len(test_df)} | overlap={overlap}")

    # Save
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "logo_class_distributions.tsv"),
                                 sep="\t", index=False)

if __name__ == "__main__":
    main()
