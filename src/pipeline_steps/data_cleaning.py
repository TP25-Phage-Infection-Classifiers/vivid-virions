# src/pipeline_steps/data_cleaning.py
import pandas as pd
from pathlib import Path

def curate_dataset_folder(input_path, output_path):
    """
    Input: folder with *_full_raw_counts_tpm_marked.tsv
    - Removes rows with Outlier == 1
    - Removes replicate columns where the marker row has value 1
    Output: *_filtered.tsv in output_path
    """
    input_folder = Path(input_path)
    output_folder = Path(output_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    tsv_files = list(input_folder.glob("*_full_raw_counts_tpm_marked.tsv"))
    print(f"{len(tsv_files)} file(s) found.")

    log_lines = []
    for file_path in tsv_files:
        try:
            df = pd.read_csv(file_path, sep="\t")

            # 1) remove outliers
            original_rows = len(df)
            if "Outlier" in df.columns:
                df = df[df["Outlier"] != 1]
            removed_rows = original_rows - len(df)

            # 2) remove columns flagged by marker row
            marker_row = df[df["Geneid"] == "Replicates with missing value"]
            removed_columns = []
            if not marker_row.empty:
                marker = marker_row.iloc[0]
                removed_columns = [col for col in df.columns if (col in marker.index and marker[col] == 1)]
                df = df[df["Geneid"] != "Replicates with missing value"]
                if removed_columns:
                    df = df.drop(columns=removed_columns)
                marker_status = "Found column marker"
            else:
                marker_status = "No column marker found"

            out_file = output_folder / file_path.name.replace("_marked.tsv", "_filtered.tsv")
            df.to_csv(out_file, sep="\t", index=False)
            print(f"Saved: {out_file.name}")

            log_lines.append(f"File: {file_path.name}")
            log_lines.append(f"- removed outlier rows: {removed_rows}")
            log_lines.append(f"- {marker_status}")
            log_lines.append(f"- removed columns: {', '.join(removed_columns) if removed_columns else 'none'}")
            log_lines.append("")
        except Exception as e:
            print(f"Error at {file_path.name}: {e}")
            log_lines.append(f"File: {file_path.name} – ERROR: {e}")
            log_lines.append("")

    
    if tsv_files:
        log_path = output_folder / "filter_protocoll.txt"
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("\n".join(log_lines))

    return output_folder


