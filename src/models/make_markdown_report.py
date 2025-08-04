from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from math import pi
import datetime as dt

# Optional: suppress sklearn version warnings
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# MLP class for loading saved models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def plot_radar(metrics_df, title, color_palette, metrics=("Accuracy", "Precision", "Recall", "F1")):
    labels = list(metrics)
    num_metrics = len(labels)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    label_offsets = {
        "RandomForest": -0.09,
        "mlp": 0.001,
        "xgboost": 0.01
    }

    for _, row in metrics_df.iterrows():
        model = row["Model"]
        vals = [row[m] for m in metrics]
        values = vals + [vals[0]]
        color = color_palette.get(model, None)

        ax.plot(angles, values, label=model, linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

        offset = label_offsets.get(model, 0.01)
        for i in range(num_metrics):
            angle_rad = angles[i]
            value = values[i]
            ha = 'left' if 0 < angle_rad < pi else 'right'
            ax.text(
                angle_rad,
                value + offset,
                f"{value:.2f}",
                color="black",
                fontsize=8,
                ha=ha,
                va='center'
            )

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=13)
    ax.yaxis.grid(True)
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=12)
    plt.title(title, fontsize=16, pad=25)
    plt.tight_layout()
    return fig

def fmt_mean_std(series: pd.Series) -> str:
    return f"{series.mean():.3f} ± {series.std():.3f}"

def build_tables(df: pd.DataFrame) -> tuple[str, str]:
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    rows = []
    for model, g in df.groupby("Model"):
        row = {"Model": model}
        for m in metrics:
            row[m] = fmt_mean_std(g[m])
        rows.append(row)
    summary_df = pd.DataFrame(rows, columns=["Model"] + metrics).sort_values("Model")
    summary_md = summary_df.to_markdown(index=False)

    parts = []
    for m in metrics:
        pivot = df.pivot(index="Split", columns="Model", values=m).sort_index()
        parts.append(f"\n**{m} per Split**\n\n{pivot.to_markdown(index=True)}\n")
    per_split_md = "\n".join(parts)
    return summary_md, per_split_md

def write_markdown(report_dir: Path, summary_md: str, per_split_md: str) -> None:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    md_path = report_dir / "report.md"
    lines: list[str] = []
    lines.append("# Model Evaluation Report\n")
    lines.append(f"_Generated: {ts}_\n")
    lines.append("\n---\n")
    lines.append("## Summary (Mean ± Std across Splits)\n")
    lines.append(summary_md)
    lines.append("\n")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        lines.append(f"![metrics_lines_{metric}](metrics_lines_{metric}.png)\n")
        lines.append(f"![metrics_boxplot_{metric}](metrics_boxplot_{metric}.png)\n")
    lines.append("\n## Radar Plots\n")
    lines.append("![Radar Mean](metrics_radar.png)\n")
    lines.append("![Radar Split 6](metrics_radar_split_6.png)\n")
    lines.append("\n## Per-Split Metrics\n")
    lines.append(per_split_md)
    lines.append("\n")
    lines.append("## Confusion Matrices per Split\n")
    for p in sorted(report_dir.glob("confusion_matrices_split_*.png")):
        lines.append(f"### {p.stem.replace('_', ' ').title()}\n")
        lines.append(f"![{p.name}]({p.name})\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")

def main(input_dir="data", output_dir="pipeline_results/report"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    model_names = ["xgboost", "mlp", "RandomForest"]
    class_labels = ["early", "middle", "late"]
    splits = range(7)
    all_results = []
    color_palette = {"xgboost": "#1f77b4", "mlp": "#ff7f0e", "RandomForest": "#2ca02c"}

    for split_number in splits:
        metrics_list = []
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, name in enumerate(model_names):
            model_file = f"model_split_{split_number}"
            data_file = f"split_{split_number}_data.pkl"
            model_path = input_dir / name / (model_file + (".pt" if name == "mlp" else ".joblib"))
            data_path = input_dir / name / data_file

            if not model_path.exists() or not data_path.exists():
                continue

            with open(data_path, "rb") as f:
                _, _, X_test, y_test, le, _ = pickle.load(f)

            if name == "mlp":
                model = MLP(input_dim=X_test.shape[1], num_classes=len(le.classes_)).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                    y_pred = model(X_tensor).argmax(dim=1).cpu().numpy()
            else:
                import joblib
                model = joblib.load(model_path)
                y_pred = model.predict(X_test)

            metrics = {
                "Split": split_number,
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
            metrics_list.append(metrics)

            cm = confusion_matrix(y_test, y_pred, normalize="true")
            ax = axes[idx]
            sns.heatmap(cm, annot=False, fmt=".2f", cmap="viridis",
                        xticklabels=class_labels, yticklabels=class_labels,
                        ax=ax, vmin=0, vmax=1)
            for y in range(cm.shape[0]):
                for x in range(cm.shape[1]):
                    val = cm[y, x]
                    color = "black" if val > 0.5 else "white"
                    ax.text(x + 0.5, y + 0.5, f"{val:.2f}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        fig.suptitle(f"Confusion Matrices (normalized) – Split {split_number}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(output_dir / f"confusion_matrices_split_{split_number}.png")
        plt.close(fig)

        all_results.extend(metrics_list)

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(output_dir / "metrics_all_splits.csv", index=False)

    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for model in df_all["Model"].unique():
            subset = df_all[df_all["Model"] == model]
            plt.plot(subset["Split"], subset[metric], label=model, marker='o')
        plt.title(f"{metric} Across Splits")
        plt.xlabel("Split")
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"metrics_lines_{metric.lower()}.png")
        plt.close()

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="Model", y=metric, hue="Model", data=df_all, palette=color_palette)
        plt.title(f"{metric} Distribution Across Splits")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / f"metrics_boxplot_{metric.lower()}.png")
        plt.close()

    radar_df = df_all.groupby("Model")[metrics].mean().reset_index()
    fig = plot_radar(radar_df, "Model Performance (Mean Across Splits)", color_palette, metrics)
    fig.savefig(output_dir / "metrics_radar.png")
    plt.close(fig)

    split_id = 6
    df_split = df_all[df_all["Split"] == split_id]
    if not df_split.empty:
        radar_split_df = df_split.groupby("Model")[metrics].mean().reset_index()
        fig = plot_radar(radar_split_df, f"Model Performance (80/20 Stratified Split)", color_palette, metrics)
        fig.savefig(output_dir / f"metrics_radar_split_{split_id}.png")
        plt.close(fig)

    summary_md, per_split_md = build_tables(df_all)
    write_markdown(output_dir, summary_md, per_split_md)

    print(f"✅ Report generated in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="pipeline_results/report")
    args = parser.parse_args()

    main(input_dir=args.input_dir, output_dir=args.output_dir)