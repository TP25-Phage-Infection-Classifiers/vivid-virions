# src/models/comparison.py
import argparse
import warnings
from math import pi
from pathlib import Path
import pickle

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.exceptions import InconsistentVersionWarning


# Config defaults 
DEFAULT_BASE_MODEL_DIR = Path("pipeline_results/models")
DEFAULT_REPORT_DIR = Path("pipeline_results/report")
DEFAULT_SPLITS = list(range(7))  # 0..5 = LOGO, 6 = stratified

COLOR_PALETTE = {
    "xgboost": "#1f77b4",
    "mlp": "#ff7f0e",
    "RandomForest": "#2ca02c",
}

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MLP definition 
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.drop(F.relu(self.fc3(x)))
        return self.out(x)


# Helpers
def _ensure_axes(n_models: int):
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    return fig, axes

def _save_confusion_grid(report_dir: Path, split_number: int, entries: list):
    """
    entries: list of dicts with keys {name, y_true, y_pred, labels, ax}
    """
    for e in entries:
        ConfusionMatrixDisplay.from_predictions(
            e["y_true"], e["y_pred"], display_labels=e["labels"],
            normalize="true", ax=e["ax"]
        )
        e["ax"].set_title(f"{e['name']} (normalized)")

    plt.suptitle(f"Normalized Confusion Matrices – Split {split_number}")
    plt.tight_layout()
    out_fp = report_dir / f"confusion_matrices_split_{split_number}.png"
    plt.savefig(out_fp, dpi=200)
    plt.close()

def _load_artifacts(base_model_dir: Path, model_name: str, split: int):
    """
    Returns: (X_test, y_test, label_encoder, feature_cols, model_path)
    """
    sub = {"xgboost": "xgboost", "mlp": "mlp", "RandomForest": "RandomForest"}[model_name]
    model_dir = base_model_dir / sub
    model_file = f"model_split_{split}"
    data_file = f"split_{split}_data.pkl"

    data_path = model_dir / data_file
    model_path = model_dir / (model_file + (".pt" if model_name == "mlp" else ".joblib"))

    if not data_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing files for {model_name} split {split}: "
                                f"{model_path.name}, {data_path.name}")

    with open(data_path, "rb") as f:
        # Tuple format from trainers:
        # (X_train, y_train, X_test, y_test, le, feature_cols)
        _Xtr, _ytr, X_test, y_test, le, feature_cols = pickle.load(f)

    return X_test, y_test, le, feature_cols, model_path

def _predict(model_name: str, X_test, le, model_path):
    """
    Predict labels (encoded ints) for any model.
    - For sklearn models, ensure feature names/ordering match the model's training schema
      by using model.feature_names_in_ when available.
    """
    if model_name == "mlp":
        # Torch MLP 
        if isinstance(X_test, pd.DataFrame):
            X_used = X_test.values
        else:
            X_used = X_test
        model = MLP(input_dim=X_used.shape[1], num_classes=len(le.classes_)).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_used, dtype=torch.float32).to(DEVICE)
            y_pred = model(X_tensor).argmax(dim=1).cpu().numpy()
        return y_pred

    # sklearn models (RF / XGB)
    model = joblib.load(model_path)
    if hasattr(model, "feature_names_in_"):
        # Prefer passing a DataFrame with the exact training columns
        if isinstance(X_test, pd.DataFrame):
            X_used = X_test.reindex(columns=model.feature_names_in_, fill_value=0.0)
        else:
            # Turn array back into DF using training feature names, then order
            X_used = pd.DataFrame(X_test, columns=model.feature_names_in_)
    else:
        # Fallback: use numpy array (older sklearn versions or models without names)
        X_used = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    return model.predict(X_used)


# Main evaluation
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-dir", default=str(DEFAULT_BASE_MODEL_DIR),
                    help="Root folder with saved models (mlp/ RandomForest/ xgboost subfolders).")
    ap.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR),
                    help="Output folder for metrics CSV and plots.")
    ap.add_argument("--splits", default="0,1,2,3,4,5,6",
                    help="Comma-separated split indices to evaluate (default: 0..6).")
    args = ap.parse_args()

    base_model_dir = Path(args.base_model_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Parse splits
    try:
        splits = [int(s.strip()) for s in args.splits.split(",") if s.strip() != ""]
    except Exception:
        splits = DEFAULT_SPLITS

    all_results = []
    model_names = ["xgboost", "mlp", "RandomForest"]

    # Evaluate each split
    for split_number in splits:
        print(f"=== Split {split_number} ===")
        fig, axes = _ensure_axes(len(model_names))
        entries = []
        metrics_list = []

        for i, name in enumerate(model_names):
            try:
                X_test, y_test, le, feature_cols, model_path = _load_artifacts(
                    base_model_dir, name, split_number
                )
                y_pred = _predict(name, X_test, le, model_path)

                # Gather entries for confusion grid
                entries.append({
                    "name": name,
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "labels": le.classes_,
                    "ax": axes[i],
                })

                # Metrics 
                metrics_list.append({
                    "Split": split_number,
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                })

            except FileNotFoundError as e:
                print(f"  ⚠ {e}")
                axes[i].set_visible(False)
                continue

        if entries:
            _save_confusion_grid(report_dir, split_number, entries)
            all_results.extend(metrics_list)
        else:
            print("  No models evaluated for this split.")

    if not all_results:
        print("No results to summarize. Exiting.")
        return

    
    # Aggregate & save metrics table
    df_all = pd.DataFrame(all_results)
    csv_path = report_dir / "metrics_all_splits.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"Saved metrics table -> {csv_path}")


    # Summary plots
    sns.set_style("whitegrid")

    # 1) Lines across splits
    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        plt.figure(figsize=(8, 5))
        for model in df_all["Model"].unique():
            subset = df_all[df_all["Model"] == model].sort_values("Split")
            plt.plot(subset["Split"], subset[metric], label=model, marker="o", linewidth=2)
        plt.title(f"{metric} Across Splits")
        plt.xlabel("Split")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        out_fp = report_dir / f"metrics_lines_{metric.lower()}.png"
        plt.savefig(out_fp, dpi=200)
        plt.close()

    # 2) Boxplots 
    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            x="Model",
            y=metric,
            hue="Model",           
            data=df_all,
            palette=COLOR_PALETTE,
            dodge=False,            
            legend=False           
        )
        plt.title(f"{metric} Distribution Across Splits")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.ylim(0, 1)
        plt.tight_layout()
        out_fp = report_dir / f"metrics_boxplot_{metric.lower()}.png"
        plt.savefig(out_fp, dpi=200)
        plt.close()

    # 3) Radar plot (mean across splits)
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    metrics_mean = df_all.groupby("Model")[metrics].mean().reset_index()

    labels = metrics
    num_metrics = len(labels)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for _, row in metrics_mean.iterrows():
        model = row["Model"]
        values = row[metrics].tolist() + [row[metrics].tolist()[0]]
        color = COLOR_PALETTE.get(model, None)
        ax.plot(angles, values, label=model, linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

        # annotate values
        for i in range(num_metrics):
            angle_rad = angles[i]
            val = values[i]
            ha = "left" if 0 < angle_rad < np.pi else "right"
            ax.text(angle_rad, val, f"{val:.2f}", color="black", fontsize=8, ha=ha, va="center")

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["polar"].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=11)
    plt.title("Model Performance (Mean Across Splits)", fontsize=16, pad=20)
    plt.tight_layout()
    out_fp = report_dir / "metrics_radar.png"
    plt.savefig(out_fp, dpi=200)
    plt.close()

    print(f"Report assets saved in: {report_dir.resolve()}")

if __name__ == "__main__":
    main()
