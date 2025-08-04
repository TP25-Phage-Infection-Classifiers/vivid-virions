# src/pipeline_steps/data_quality_assessment.py
import pandas as pd
import numpy as np
from pathlib import Path


def _detect_empty_cells(df: pd.DataFrame):
    """
    Identical to the notebook:
      - Count total missing values.
      - Build the special row that starts with the literal string
        'Replicates with missing value' followed by 1/0 flags
        for EACH column from index 1 onward indicating if that column
        contains at least one NaN (not restricted to count columns).
      - Fill NaNs in the data with -1.
    """
    missing_values_total = int(df.isna().sum().sum())
    
    flags = df.iloc[:, 1:].isna().any(axis=0).astype(int).tolist()
    marker_row = ["Replicates with missing value"] + flags
    
    df_filled = df.fillna(-1)
    return df_filled, missing_values_total, marker_row

def _too_high_zero_value(df: pd.DataFrame):
    """
    Notebook behavior:
      - Look for columns that start with '0_R'
      - Mark a row as 1 if ANY of those columns has a value > 500
      - Otherwise 0
    """
    zero_cols = [c for c in df.columns if str(c).startswith("0_R")]
    if not zero_cols:
        return [0] * len(df)

    marks = []
    for _, row in df.iterrows():
        flag = 0
        for c in zero_cols:
            try:
                if float(row[c]) > 500:
                    flag = 1
                    break
            except Exception:
                
                pass
        marks.append(flag)
    return marks

def _detect_relative_changes(df: pd.DataFrame, threshold: float):
    """
    Notebook behavior:
      - 'count_cols' = df.columns[1:-2]  (first column is Geneid,
        last two are text/meta columns in the raw tables)
      - For each row, compute coefficient of variation (std/mean)
        across those columns; if mean == 0 or no numbers -> 0
      - negligible_changes = 1 if CV < threshold else 0
      - Return both arrays.
    """
    
    count_cols = df.columns[1:-2]
    rel_changes = []
    negligible = []

    for _, row in df.iterrows():
        nums = []
        for val in row[count_cols]:
            try:
                nums.append(float(val))
            except Exception:
                # ignore non-numeric
                continue
        if nums and np.mean(nums) != 0:
            mean = float(np.mean(nums))
            std = float(np.std(nums))
            cv = std / mean
            rel_changes.append(cv)
            negligible.append(1 if cv < threshold else 0)
        else:
            rel_changes.append(0.0)
            negligible.append(0)

    return rel_changes, negligible

def _collect_outlier_genes(df: pd.DataFrame):
    """
    Notebook behavior:
      Outlier = 1 if:
        - negligible changes == 1
        - OR 'Genes with too high Zero count' is truthy
        - OR Entity == 'host'
      (Applied AFTER the marker row is appended in the notebook, so
       the marker row’s Outlier becomes NaN in the saved file.)
    """
    
    if "Entity" in df.columns:
        outlier = (
            (df["negligible changes"] == 1) |
            (df["Genes with too high Zero count"]) |
            (df["Entity"] == "host")
        ).astype(int)
    else:
        outlier = (
            (df["negligible changes"] == 1) |
            (df["Genes with too high Zero count"])
        ).astype(int)

    df["Outlier"] = outlier
    return df

def run_quality_assessment(normalized_dir, marked_dir, threshold: float = 0.1):
    
    # Make marked tables exactly like the notebook:
      
    normalized_dir = Path(normalized_dir)
    marked_dir = Path(marked_dir)
    marked_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(normalized_dir.glob("*.tsv"))
    if not files:
        print(f"No normalized .tsv files found in {normalized_dir}")
        return marked_dir

    overview_lines = []

    for in_path in files:
        try:
            df = pd.read_csv(in_path, sep="\t")
        except Exception as e:
            print(f"Failed to read {in_path.name}: {e}")
            continue

        # 1) Missing cells marker and fill with -1 
        df, missing_values, marker_row = _detect_empty_cells(df)

        # 2) Relative changes & negligible changes
        rel, neg = _detect_relative_changes(df, threshold=threshold)

        # 3) '0_R*' > 500 flags
        high0 = _too_high_zero_value(df)

        # Add computed columns
        df["relative changes"] = rel
        df["negligible changes"] = neg
        df["Genes with too high Zero count"] = high0

        # 4) Append the marker row BEFORE adding 'Outlier'
        # Pad marker_row to match current number of columns
        while len(marker_row) < len(df.columns):
            marker_row.append(0)
        df.loc[len(df)] = marker_row

        # 5) Now add Outlier column 
        df = _collect_outlier_genes(df)

        # Overview stats (same counts as notebook)
        negligible_genes = int(sum(neg))
        zero_count_genes = int(sum(high0))
        overview_lines.append(
            f"{in_path.name}: missing values: {missing_values}, "
            f"negligible genes: {negligible_genes}, "
            f"genes with a too high count at time 0: {zero_count_genes}"
        )

        # 6) Save marked file
        out_path = marked_dir / (in_path.stem + "_marked.tsv")
        df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
        print(f"Saved: {out_path.name}")

    # 7) overview file
    with open(marked_dir / "overview_marks.txt", "w", encoding="utf-8") as fh:
        for line in overview_lines:
            fh.write(line + "\n")

    return marked_dir

