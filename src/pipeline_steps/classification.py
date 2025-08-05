#src/pipeline_steps/classification.py
import pandas as pd
from pathlib import Path
import os
import re

def average_runs(df):
    timepoint_groups = {}
    averaged_data = pd.DataFrame()

    for col in df.columns:
        match = re.match(r"(.+)_R\d+", col)
        if match:
            timepoint = match.group(1)
            timepoint_groups.setdefault(timepoint, []).append(col)
        else:
            averaged_data[col] = df[col]

    for timepoint, cols in timepoint_groups.items():
        averaged_data[timepoint + "_T"] = df[cols].mean(axis=1)

    return averaged_data


def classify_timepoint(col, filename):
    if "Lood" in filename:
        if col.startswith("5_"): return "early"
        elif col.startswith("15_"): return "middle"
        elif col.startswith("25_"): return "late"
    elif "Yang" in filename:
        if col.startswith("5_"): return "early"
        elif col.startswith("10_"): return "middle"
        elif col.startswith("20_"): return "late"
    elif "Finstrlova" in filename:
        if any(col.startswith(tp) for tp in ["2_", "5_", "10_"]): return "early"
        elif col.startswith("20_"): return "middle"
        elif col.startswith("30_"): return "late"
    elif "Brandao" in filename:
        if col.startswith("5_"): return "early"
        elif col.startswith("10_"): return "middle"
        elif col.startswith("15_"): return "late"
    elif "Guegler_T4" in filename:
        if col.startswith("2.5_"): return "early"
        elif col.startswith("5_"): return "middle"
        elif any(col.startswith(tp) for tp in ["10_", "20_", "30_"]): return "late"
    elif "Guegler_T7" in filename:
        if any(col.startswith(tp) for tp in ["2.5_", "5_", "10_"]): return "early"
        elif col.startswith("20_"): return "middle"
        elif col.startswith("30_"): return "late"
    elif "Sprenger" in filename:
        if col.startswith("0_"): return "early"
        elif col.startswith("30_"): return "middle"
        elif col.startswith("60_"): return "late"
    return None


def classify_datasets(input_path, schema, base_output_path):
    """
    Classify genes based on peak expression timepoint.

    - input_path: folder with *_filtered.tsv files
    - base_output_path: root folder (e.g., 'pipeline_results') where classified output goes
    """
    input_path = Path(input_path)
    base_output_path = Path(base_output_path)
    classified_path = base_output_path / "classified"
    classified_path.mkdir(parents=True, exist_ok=True)

    datasets = []
    for file in input_path.glob("*_filtered.tsv"):
        df = pd.read_csv(file, sep="\t")
        datasets.append((df, file.stem))

    if not datasets:
        raise FileNotFoundError(f"No *_filtered.tsv files found in {input_path}")

    for df, filename in datasets:
        averaged_df = average_runs(df)
        timepoint_cols = [col for col in averaged_df.columns if "_T" in col]
        if not timepoint_cols:
            raise ValueError(f"No timepoint columns found in {filename}")
        max_col = averaged_df[timepoint_cols].idxmax(axis=1)
        classification = max_col.apply(lambda z: classify_timepoint(z, filename))
        averaged_df["classification"] = classification
        averaged_df.to_csv(classified_path / f"{filename}_classified.tsv", sep="\t", index=False)

    return classified_path
