# src/phage_classification_pipeline.py
import argparse
import os
import sys
import webbrowser
import subprocess
from pathlib import Path
from importlib import import_module

from pipeline_steps.data_normalize import normalize_datasets
from pipeline_steps.data_quality_assessment import run_quality_assessment
from pipeline_steps.data_cleaning import curate_dataset_folder
from pipeline_steps.classification import classify_datasets
from pipeline_steps.feature_extraction import extract_features, report_failed_extractions

PIPELINE_ROOT = Path("pipeline_results")

GENOME_MAPPING = {
    "Brandao_MCCM_full_raw_counts_tpm_filtered_classified.tsv": "NC_010326",
    "Finstrlova_Newman_full_raw_counts_tpm_filtered_classified.tsv": "NC_005880",
    "Guegler_T4_minusToxIN_full_raw_counts_tpm_filtered_classified.tsv": "NC_000866",
    "Guegler_T7_plusToxIN_full_raw_counts_tpm_filtered_classified.tsv":  "NC_001604",
    "Lood_full_raw_counts_tpm_filtered_classified.tsv": "MK797984.1",
    "Sprenger_VC_WT_VP882_delta_cpdS_full_raw_counts_tpm_filtered_classified.tsv": "NC_009016.1",
    "Yang_full_raw_counts_tpm_filtered_classified.tsv": "NC_021316",
}

# ---------- helpers ----------
def _open_path(p: Path):
    """Open a file or folder with the system default app."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(p)], check=False)
        else:
            subprocess.run(["xdg-open", str(p)], check=False)
    except Exception:
        try:
            webbrowser.open_new_tab(p.resolve().as_uri())
        except Exception:
            pass


def _run_logo_split():
    print("[6] Running LOGO split …")
    from splits.leave_one_group_out import main as run_logo
    run_logo()
    print(f"   ➜ {(PIPELINE_ROOT / 'splits_logo').resolve()}")


def _run_stratified_split():
    print("[6] Running stratified 80/20 split …")
    from splits.stratified_split import main as run_stratified
    run_stratified()
    print(f"   ➜ {(PIPELINE_ROOT / 'splits_stratified').resolve()}")


def _train_models(run_mlp: bool, run_rf: bool, run_xgb: bool):
    """
    Call the training modules you created. They read splits from pipeline_results
    and save artifacts under pipeline_results/models/<name>.
    """
    if run_mlp:
        print("[7] Training MLP (LOGO + stratified) …")
        mod = import_module("models.MLP.train")
        mod.main.__wrapped__ if hasattr(mod.main, "__wrapped__") else None  # noop for linters
        mod_args = argparse.Namespace(
            base_dir=str(PIPELINE_ROOT),
            out_dir=str(PIPELINE_ROOT / "models" / "mlp"),
            splits="both",
            epochs=100,
            batch_size=64,
            patience=10,
            seed=42,
        )
        mod.main.__globals__["argparse"].ArgumentParser = _frozen_parser(mod_args)
        mod.main()

    if run_rf:
        print("[7] Training RandomForest (LOGO + stratified) …")
        mod = import_module("models.RF.train")
        mod_args = argparse.Namespace(
            base_dir=str(PIPELINE_ROOT),
            out_dir=str(PIPELINE_ROOT / "models" / "RandomForest"),
            splits="both",
            top_k=36,
        )
        mod.main.__globals__["argparse"].ArgumentParser = _frozen_parser(mod_args)
        mod.main()

    if run_xgb:
        print("[7] Training XGBoost (LOGO + stratified) …")
        mod = import_module("models.XGBoost.train")
        mod_args = argparse.Namespace(
            base_dir=str(PIPELINE_ROOT),
            out_dir=str(PIPELINE_ROOT / "models" / "xgboost"),
            splits="both",
        )
        mod.main.__globals__["argparse"].ArgumentParser = _frozen_parser(mod_args)
        mod.main()


def _build_comparison_and_report(regenerate: bool):
    """
    1) Run comparison to produce metrics CSV and images.
    2) Build Markdown report that embeds those artifacts.
    3) Auto-open the report and the folder.
    """
    report_dir = PIPELINE_ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("[8] Generating metrics & plots (comparison) …")
    comp = import_module("models.comparison")
    comp_args = argparse.Namespace(
        base_model_dir=str(PIPELINE_ROOT / "models"),
        report_dir=str(report_dir),
        splits="0,1,2,3,4,5,6",
    )
    comp.main.__globals__["argparse"].ArgumentParser = _frozen_parser(comp_args)
    comp.main()

    print("[9] Writing Markdown report …")
    md = import_module("models.make_markdown_report")
    md_args = argparse.Namespace(
        report_dir=str(report_dir),
        title="Phage Classification – Model Comparison Report",
        regenerate=False,  # we just ran comparison above
    )
    md.main.__globals__["argparse"].ArgumentParser = _frozen_parser(md_args)
    md.main()

    # Auto-open the report and the folder
    report_md = report_dir / "report.md"
    if report_md.exists():
        _open_path(report_md)
    _open_path(report_dir)


def _frozen_parser(args_ns: argparse.Namespace):
    """
    Replace argparse.ArgumentParser in imported modules to feed fixed args.
    This avoids subprocess calls and keeps logs in one place.
    """
    class _Frozen:
        def __init__(self, *_, **__):
            pass
        def add_argument(self, *_, **__):
            return None
        def parse_args(self, *_, **__):
            return args_ns
    return _Frozen


def main():
    parser = argparse.ArgumentParser(description="End-to-end phage pipeline")
    parser.add_argument("--input", required=True,
                        help="Folder containing raw '*_full_raw_counts.tsv' tables (searched recursively)")

    # Optional: create splits
    parser.add_argument("--split-stratified", action="store_true",
                        help="Run stratified 80/20 split after feature extraction")
    parser.add_argument("--split-logo", action="store_true",
                        help="Run leave-one-group-out split after feature extraction")

    # Optional: training flags
    parser.add_argument("--train-mlp", action="store_true", help="Train MLP after splits")
    parser.add_argument("--train-rf", action="store_true", help="Train RandomForest after splits")
    parser.add_argument("--train-xgb", action="store_true", help="Train XGBoost after splits")
    parser.add_argument("--train-all", action="store_true", help="Train all (MLP, RF, XGBoost) after splits")

    # Optional: build comparison and markdown report
    parser.add_argument("--report", action="store_true",
                        help="Generate metrics/plots and build Markdown report")

    args = parser.parse_args()
    input_dir = Path(args.input)

    # Output folders
    normalized_dir = PIPELINE_ROOT / "datasets_normalized"
    marked_dir     = PIPELINE_ROOT / "marked"
    filtered_dir   = PIPELINE_ROOT / "filtered"
    features_dir   = PIPELINE_ROOT / "feature_extraction"

    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)

    print("[1/5] Normalizing raw count tables …")
    normalize_datasets(input_root=input_dir, output_dir=normalized_dir, keep_only_phage=False)

    print("[2/5] Running data quality assessment …")
    run_quality_assessment(normalized_dir=normalized_dir, marked_dir=marked_dir)

    print("[3/5] Cleaning datasets …")
    curate_dataset_folder(input_path=marked_dir, output_path=filtered_dir)

    print("[4/5] Classifying genes …")
    classified_dir = classify_datasets(input_path=filtered_dir, schema=None, base_output_path=PIPELINE_ROOT)

    print("[5/5] Extracting features …")
    extract_features(
        classified_dir=classified_dir,
        out_feature_dir=features_dir,
        genome_mapping=GENOME_MAPPING,
        base_output_path=PIPELINE_ROOT,
    )
    report_failed_extractions(features_dir)

    # Splits (read from pipeline_results/feature_tables)
    ran_any_split = False
    if args.split_stratified:
        _run_stratified_split()
        ran_any_split = True
    if args.split_logo:
        _run_logo_split()
        ran_any_split = True

    # If user wants to train but forgot to request splits, run both by default
    wants_training = args.train_all or args.train_mlp or args.train_rf or args.train_xgb or args.report
    if wants_training and not ran_any_split:
        print("ℹ No split flags provided; running both splits by default so models can train …")
        _run_stratified_split()
        _run_logo_split()

    # Train models
    if wants_training:
        _train_models(
            run_mlp = args.train_all or args.train_mlp,
            run_rf  = args.train_all or args.train_rf,
            run_xgb = args.train_all or args.train_xgb,
        )

    # Build comparison plots + Markdown report
    if args.report:
        _build_comparison_and_report(regenerate=False)

    print("\n✅ Pipeline finished.")
    print(f"- Normalized:   {normalized_dir.resolve()}")
    print(f"- Marked:       {marked_dir.resolve()}")
    print(f"- Filtered:     {filtered_dir.resolve()}")
    print(f"- Classified:   {(PIPELINE_ROOT / 'classified').resolve()}")
    print(f"- Features:     {features_dir.resolve()}")
    if args.split_stratified or wants_training:
        print(f"- Splits (stratified): {(PIPELINE_ROOT / 'splits_stratified').resolve()}")
    if args.split_logo or wants_training:
        print(f"- Splits (LOGO):       {(PIPELINE_ROOT / 'splits_logo').resolve()}")
    if wants_training:
        print(f"- Models:        {(PIPELINE_ROOT / 'models').resolve()}")
    if args.report:
        print(f"- Report:        {(PIPELINE_ROOT / 'report').resolve()}")


if __name__ == "__main__":
    main()




