# src/splits/stratified_split.py
import os, glob
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIR = "pipeline_results/feature_tables"
OUTPUT_DIR = "pipeline_results/splits_stratified"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined_file = os.path.join(OUTPUT_DIR, "combined.tsv")

    # 1) merge
    tsv_files = glob.glob(os.path.join(INPUT_DIR, "*.tsv"))
    with open(combined_file, "w", encoding="utf-8") as out_file:
        for i, fname in enumerate(tsv_files):
            with open(fname, "r", encoding="utf-8") as in_file:
                lines = in_file.readlines()
                out_file.writelines(lines if i == 0 else lines[1:])
    print(f"✅ Kombiniert: {len(tsv_files)} Dateien → {combined_file}")

    # 2) split
    df = pd.read_csv(combined_file, sep="\t")
    label_col = "classification" if "classification" in df.columns else "classification_x"

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[label_col], random_state=42
    )

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.tsv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.tsv"), sep="\t", index=False)

    print("Class distribution (Train):")
    print(train_df[label_col].value_counts(normalize=True))
    print("\nClass distribution (Test):")
    print(test_df[label_col].value_counts(normalize=True))

    overlap = set(train_df.get("Geneid", [])).intersection(set(test_df.get("Geneid", [])))
    print("{} overlapping gene!".format(len(overlap)) if overlap else
          "No overlapping genes")

if __name__ == "__main__":
    main()


