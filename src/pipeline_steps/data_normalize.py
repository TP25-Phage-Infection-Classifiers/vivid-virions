# src/pipeline_steps/data_normalize.py
from pathlib import Path
import re
import pandas as pd

SAMPLE_COL_RE = re.compile(r"^\d+(\.\d+)?_R\d+$")

def _tpm_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Divide each sample column by its column sum / 1e6 (TPM-like scaling on counts matrix).
    Only columns matching ^<time>_R<rep> are normalized.
    """
    df = df.copy()
    sample_cols = [c for c in df.columns if SAMPLE_COL_RE.match(str(c))]
    if not sample_cols:
        return df 

    totals_million = df[sample_cols].sum().div(1_000_000)
    totals_million = totals_million.replace(0, 1)  
    df[sample_cols] = df[sample_cols].div(totals_million, axis=1)
    return df

def normalize_datasets(input_root, output_dir, keep_only_phage: bool = False) -> Path:
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_root.is_file():
        raw_files = [input_root] if input_root.name.endswith("_full_raw_counts.tsv") else []
    else:
        raw_files = list(input_root.rglob("*_full_raw_counts.tsv"))

    if not raw_files:
        raise FileNotFoundError(
            f"No '*_full_raw_counts.tsv' found under {input_root.resolve()}"
        )

    print(f"{len(raw_files)} raw file(s) found for normalization.")
    for fp in raw_files:
        try:
            df = pd.read_csv(fp, sep="\t")

            
            if keep_only_phage and "Entity" in df.columns:
                df = df[df["Entity"] == "phage"].copy()

            df = _tpm_normalize(df)

            out_name = fp.stem + "_tpm.tsv"
            out_path = output_dir / out_name
            df.to_csv(out_path, sep="\t", index=False)
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"Skipped {fp}: {e}")

    return output_dir

