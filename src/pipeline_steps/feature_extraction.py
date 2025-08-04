# src/pipeline_steps/feature_extraction.py
from pathlib import Path
import pandas as pd
from Bio import Entrez, SeqIO
from Bio.SeqUtils import gc_fraction
from Bio.Seq import Seq
from collections import Counter
from itertools import product
import tarfile


# ----- optional imports -----
try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    HAS_PROTPARAM = True
except Exception:
    HAS_PROTPARAM = False

try:
    # your local DSSP parser module (leave out if you don't have it)
    from DSSPparser import parseDSSP
    HAS_DSSP = True
except Exception:
    HAS_DSSP = False


Entrez.email = "david.martin@student.uni-tuebingen.de"


# helpers

def _get_sequences_from_geneid(genome_accession: str, geneid: str):
    """
    Fetch protein and DNA sequence from a GenBank record using multiple tags
    (gene, product, locus_tag, and the trailing part after the last underscore).
    Returns (protein_seq, dna_seq) or ('NOT_FOUND'/'ERROR_FETCH', ...).
    """
    try:
        handle = Entrez.efetch(db="nucleotide", id=genome_accession,
                               rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
    except Exception as e:
        print(f"Failed to fetch {genome_accession}: {e}")
        return ("ERROR_FETCH", "ERROR_FETCH")

    tag = str(geneid).replace("gene-", "").strip()
    short_tag = tag.split("_")[-1]

    for feature in record.features:
        if feature.type != "CDS":
            continue

        locus_tag = feature.qualifiers.get("locus_tag", [""])[0]
        gene = feature.qualifiers.get("gene", [""])[0]
        product = feature.qualifiers.get("product", [""])[0]

        if tag in (locus_tag, gene, product) or short_tag in (locus_tag, gene, product):
            protein = feature.qualifiers.get("translation", ["TRANSLATION_NOT_FOUND"])[0]
            dna_seq = feature.location.extract(record.seq)
            return (protein, str(dna_seq))

    return ("NOT_FOUND", "NOT_FOUND")


def _count_kmers(seq: str, k: int, alphabet=("A", "T", "G", "C")):
    seq = (seq or "").upper()
    valid = set(alphabet)
    counts = Counter(
        seq[i:i+k] for i in range(len(seq) - k + 1)
        if set(seq[i:i+k]).issubset(valid)
    )
    all_kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    return [counts.get(kmer, 0) for kmer in all_kmers], all_kmers

def _extract_pdb_archive(archive_path: Path, output_dir: Path):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

        for member in tar.getmembers():
            if member.name.endswith(".pdb"):
                member.name = Path(member.name).name  # flatten path
                tar.extract(member, path=output_dir)

# main API

def extract_features(
    classified_dir,
    out_feature_dir,
    genome_mapping,
    base_output_path: str | Path = "pipeline_results",
    dssp_dir: str | Path | None = None,
):
    """
    End-to-end feature extraction.

    1) Read *_classified.tsv from `classified_dir`
    2) Fetch ProteinSequence & DNASequence -> save to `out_feature_dir`
    3) Build DNA features -> `pipeline_results/dna_feature_table/`
    4) Build protein (k=3 + physchem) -> `pipeline_results/features/protein/protein_primary_table_k=3/`
    5) (Optional) Build secondary-structure (DSSP) features -> `pipeline_results/protein_structures/features/`
    6) Merge DNA + protein-primary + structure -> `pipeline_results/feature_tables/`
    7) Print a short failure report
    """
    classified_dir = Path(classified_dir)
    out_feature_dir = Path(out_feature_dir)
    out_feature_dir.mkdir(parents=True, exist_ok=True)
    base_output_path = Path(base_output_path)

    # 1) + 2) fetch sequences
    files = list(classified_dir.glob("*_classified.tsv"))
    if not files:
        print(f"No *_classified.tsv found in {classified_dir}")
        return

    for file_path in files:
        fname = file_path.name
        print(f"Fetching sequences: {fname}")
        df = pd.read_csv(file_path, sep="\t")

        genome_acc = genome_mapping.get(fname)
        if not genome_acc:
            print(f"No genome accession mapped for {fname} — skipped.")
            continue

        prot_list, dna_list = [], []
        for geneid in df["Geneid"]:
            p, d = _get_sequences_from_geneid(genome_acc, geneid)
            prot_list.append(p)
            dna_list.append(d)

        df["ProteinSequence"] = prot_list
        df["DNASequence"] = dna_list

        out_fp = out_feature_dir / fname
        df.to_csv(out_fp, sep="\t", index=False)
        print(f"✅ Saved sequences -> {out_fp}")

    # 3) DNA features
    dna_out = base_output_path / "dna_feature_table"
    _build_dna_feature_tables(out_feature_dir, dna_out, k=3)

    # 4) Protein k=3 + physchem
    protein_out = base_output_path / "features" / "protein" / "protein_primary_table_k=3"
    _build_protein_k3_features(out_feature_dir, protein_out)

    # 5) Secondary structure 
    struct_out = base_output_path / "protein_structures" / "features"
    if dssp_dir is None:
        dssp_dir = Path("data/pipeline_data_set/protein")

    input_dir = Path("data/pipeline_data_set")

    _build_structure_features(input_dir=input_dir, dssp_dir=dssp_dir, out_dir=struct_out)

    # 6) Merge ALL features
    merged_out = base_output_path / "feature_tables"
    _merge_all_features(
        dna_dir=dna_out,
        protein_dir=protein_out,
        struct_dir=struct_out,
        out_dir=merged_out,
    )

    # 7) Failure report
    _report_failed_extractions(out_feature_dir)

    print("✨ Feature extraction complete.")
    print(f"   DNA features:       {dna_out}")
    print(f"   Protein k=3:        {protein_out}")
    print(f"   Structure features: {struct_out}")
    print(f"   Merged tables:      {merged_out}")


# step builders

def _build_dna_feature_tables(seq_dir: Path, out_dir: Path, k=3):
    out_dir.mkdir(parents=True, exist_ok=True)

    for file in seq_dir.glob("*.tsv"):
        df = pd.read_csv(file, sep="\t")

        # Minimal columns required
        if not {"Geneid", "DNASequence", "classification"}.issubset(df.columns):
            print(f"⚠ Skipping (missing columns) {file.name}")
            continue

        feats = df[["Geneid", "DNASequence", "classification"]].copy()

        # GC content & length
        feats["GC_Content"] = feats["DNASequence"].apply(
            lambda s: gc_fraction(Seq(str(s))) if isinstance(s, str) else 0
        )
        feats["Seq_length"] = feats["DNASequence"].apply(
            lambda s: len(s) if isinstance(s, str) else 0
        )

        # Base fractions
        def base_counts(s: str):
            s = (s or "").upper()
            return {
                "A_Content": s.count("A"),
                "T_Content": s.count("T"),
                "G_Content": s.count("G"),
                "C_Content": s.count("C"),
            }

        base_df = feats["DNASequence"].apply(base_counts).apply(pd.Series)
        base_frac = base_df.div(feats["Seq_length"].replace(0, 1), axis=0)
        feats = pd.concat([feats, base_frac], axis=1)

        # Purine / Pyrimidine fraction
        feats["Purin_Content"] = feats["DNASequence"].apply(
            lambda s: (s.upper().count("A") + s.upper().count("G")) / len(s)
            if isinstance(s, str) and len(s) > 0 else 0
        )
        feats["Pyrimidin_Content"] = feats["DNASequence"].apply(
            lambda s: (s.upper().count("C") + s.upper().count("T")) / len(s)
            if isinstance(s, str) and len(s) > 0 else 0
        )

        # CpG bias
        def cpg_bias(s: str):
            s = (s or "").upper()
            L = len(s)
            if L == 0:
                return 0
            c = s.count("C")
            g = s.count("G")
            cpg = sum(1 for i in range(L - 1) if s[i:i+2] == "CG")
            expected = (c * g) / L if L > 0 else 0
            return (cpg / expected) if expected > 0 else 0

        feats["CpG_bias"] = feats["DNASequence"].apply(cpg_bias)

        # k=3 k-mers (normalized)
        km_counts, all_kmers = zip(*feats["DNASequence"].apply(lambda s: _count_kmers(s or "", k)))
        km_df = pd.DataFrame(list(km_counts), columns=[f"kmer_{kmer}" for kmer in all_kmers[0]])
        row_sums = km_df.sum(axis=1).replace(0, 1)
        km_df = km_df.div(row_sums, axis=0)

        df_out = pd.concat([feats, km_df], axis=1)
        out_file = out_dir / file.name
        df_out.to_csv(out_file, sep="\t", index=False)
        print(f"DNA features -> {out_file}")


def _build_protein_k3_features(seq_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not HAS_PROTPARAM:
        print("Skipping protein features (Bio.SeqUtils.ProtParam not available).")
        return

    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    TRIPEPTIDES = ["".join(p) for p in product(AA_LIST, repeat=3)]

    def extract_k3_features(seq: str):
        seq = (seq or "").upper().replace("*", "")
        # Tripeptide frequencies
        tri_counts = Counter(seq[i:i+3] for i in range(len(seq) - 2))
        total_tri = sum(tri_counts.values()) or 1
        tri = {f"kmer3_{tp}": tri_counts.get(tp, 0) / total_tri for tp in TRIPEPTIDES}
        # Physicochemical
        try:
            analysis = ProteinAnalysis(seq)
            pI = analysis.isoelectric_point()
            gravy = analysis.gravy()
            instability = analysis.instability_index()
        except Exception:
            pI = gravy = instability = None
        return tri | {
            "IsoelectricPoint": pI,
            "GRAVY": gravy,
            "InstabilityIndex": instability,
            "seq_length": len(seq),
        }

    for file in seq_dir.glob("*.tsv"):
        df = pd.read_csv(file, sep="\t")
        required = {"Geneid", "classification", "ProteinSequence"}
        if not required.issubset(df.columns):
            print(f"Skipping protein features for {file.name} (missing columns)")
            continue

        feats = []
        for _, row in df.iterrows():
            f = extract_k3_features(row["ProteinSequence"])
            feats.append({"Geneid": row["Geneid"], "classification": row["classification"], **f})

        out_df = pd.DataFrame(feats)
        out_path = out_dir / file.name.replace(".tsv", "_k3_physchem.tsv")
        out_df.to_csv(out_path, sep="\t", index=False)
        print(f"Protein k=3 features -> {out_path}")



def _build_structure_features(input_dir: Path, dssp_dir: Path, out_dir: Path):
    import subprocess
    from collections import Counter
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    dssp_dir = Path(dssp_dir)

    pdb_archive = input_dir / "protein" / "pdbs_teamprojekt.tar.gz"
    _extract_pdb_archive(pdb_archive, dssp_dir)

    if not HAS_DSSP:
        print("Skipping structure features (DSSP parser not available).")
        return

    def convert_to_dssp(input_file: Path, output_file: Path):
        try:
            subprocess.run(["mkdssp", "-i", str(input_file), "-o", str(output_file)], check=True)
            print(f" Converted {input_file.name} → {output_file.name}")
        except Exception as e:
            print(f"DSSP conversion failed for {input_file.name}: {e}")

    # Convert .pdb/.cif to .dssp if missing
    for file in dssp_dir.glob("*"):
        if file.suffix.lower() in [".pdb", ".cif"]:
            out_path = dssp_dir / (file.stem + ".dssp")
            if not out_path.exists():
                convert_to_dssp(file, out_path)

    dssp_files = list(dssp_dir.glob("*.dssp"))
    if not dssp_files:
        print(f"ℹ No .dssp files found in {dssp_dir}, skipping structure features.")
        return

    MAX_ASA = {
        "A": 121.0, "R": 265.0, "N": 187.0, "D": 187.0, "C": 148.0,
        "Q": 214.0, "E": 214.0, "G": 97.0,  "H": 216.0, "I": 195.0,
        "L": 191.0, "K": 230.0, "M": 203.0, "F": 228.0, "P": 154.0,
        "S": 143.0, "T": 163.0, "W": 264.0, "Y": 255.0, "V": 165.0,
    }

    # Load protein sequences from prior step
    protein_map = {}
    for file in Path("pipeline_results/feature_extraction").glob("*.tsv"):
        try:
            df = pd.read_csv(file, sep="\\t")
            for _, row in df.iterrows():
                protein_map[str(row["Geneid"])] = row.get("ProteinSequence", "")
        except Exception:
            pass

    def protein_analysis(seq: str):
        try:
            if isinstance(seq, str) and "X" not in seq and len(seq) > 0:
                analysis = ProteinAnalysis(seq)
                return {
                    "molecular_weight": analysis.molecular_weight(),
                    "aromaticity": analysis.aromaticity(),
                }
        except Exception:
            return {}
        return {}

    def dssp_features(dssp_file: Path):
        parser = parseDSSP(dssp_file)
        parser.parse()
        df = parser.dictTodataframe()

        # Clean and normalize secondary structure annotations
        df["struct"] = df["struct"].apply(lambda x: "C" if str(x).strip() == "" else str(x).strip())
        df["struct"] = df["struct"].apply(lambda x: "H" if x in ["G", "I"] else x)

        structure_counts = df["struct"].value_counts()
        total = len(df)
        struc_content = {k: structure_counts.get(k, 0) / total for k in ["H", "E", "C"]}
        for frac in ["H", "E", "C"]:
            struc_content.setdefault(frac, 0.0)

        df["acc"] = pd.to_numeric(df["acc"], errors="coerce")
        df["rsa"] = df.apply(lambda row: row["acc"] / MAX_ASA.get(row["aa"], np.nan), axis=1)
        rsa_mean = df["rsa"].mean(skipna=True)
        frac_exposed = (df["rsa"] > 0.2).mean()

        structs = df["struct"].tolist()
        transitions = sum(1 for i in range(1, len(structs)) if structs[i] != structs[i - 1])
        transitions_per_res = transitions / len(structs) if structs else 0.0

        return {
            "H_frac": struc_content["H"],
            "E_frac": struc_content["E"],
            "C_frac": struc_content["C"],
            "rsa_mean": rsa_mean,
            "frac_exposed": frac_exposed,
            "transitions_per_residue": transitions_per_res,
        }

    for fp in dssp_files:
        gid = fp.stem
        struct_feats = dssp_features(fp)
        prot_feats = protein_analysis(protein_map.get(gid, ""))
        feats = pd.DataFrame([{**struct_feats, **prot_feats, "Geneid": gid}])
        out_path = out_dir / f"{gid}_structure_features.tsv"
        feats.to_csv(out_path, sep="\\t", index=False)
        print(f"Structure features -> {out_path}")



# merging

def _merge_all_features(dna_dir: Path, protein_dir: Path, struct_dir: Path, out_dir: Path):
    """
    Merge DNA + protein primary (k3+physchem) + (optional) structure features into one table per dataset.
    Output naming stays identical: *_classified.tsv -> *_feature_table.tsv
    """
    from functools import reduce

    out_dir.mkdir(parents=True, exist_ok=True)

    dna_files = sorted(Path(dna_dir).glob("*.tsv"))
    if not dna_files:
        print(f"No DNA feature tables found in {dna_dir}; skipping merge.")
        return

    def norm_key(stem: str) -> str:
        # Align stems so all three sources map to the same dataset key
        return stem.replace("_k3_physchem", "").replace("_structure_features", "")

    prot_map = {}
    if Path(protein_dir).exists():
        for p in Path(protein_dir).glob("*_k3_physchem.tsv"):
            prot_map[norm_key(p.stem)] = p

    struct_map = {}
    if Path(struct_dir).exists():
        for s in Path(struct_dir).glob("*.tsv"):
            struct_map[norm_key(s.stem)] = s

    for dna_fp in dna_files:
        df_dna = pd.read_csv(dna_fp, sep="\t")
        key = norm_key(dna_fp.stem)

        dfs = [df_dna]
        sources = ["DNA"]

        if key in prot_map:
            try:
                df_p = pd.read_csv(prot_map[key], sep="\t")
                dfs.append(df_p)
                sources.append("Protein")
            except Exception as e:
                print(f"Could not read protein features for {dna_fp.name}: {e}")

        if key in struct_map:
            try:
                df_s = pd.read_csv(struct_map[key], sep="\t")
                dfs.append(df_s)
                sources.append("Structure")
            except Exception as e:
                print(f"Could not read structure features for {dna_fp.name}: {e}")

        def _m(left, right):
            return left.merge(right, on="Geneid", how="outer", suffixes=("", "_dup"))

        try:
            df_merged = reduce(_m, dfs)
        except Exception as e:
            print(f"Merge failed for {dna_fp.name}: {e}")
            continue

        # Coalesce classification columns if duplicates appear
        cls_cols = [c for c in df_merged.columns if c.startswith("classification")]
        if cls_cols:
            df_merged["classification"] = df_merged[cls_cols].bfill(axis=1).iloc[:, 0]
            df_merged = df_merged.drop(columns=[c for c in cls_cols if c != "classification"])

        # Drop exact duplicate columns from suffixing
        dup_cols = [c for c in df_merged.columns if c.endswith("_dup")]
        for c in dup_cols:
            base = c[:-4]
            if base in df_merged.columns and df_merged[base].equals(df_merged[c]):
                df_merged = df_merged.drop(columns=[c])

        out_path = Path(out_dir) / dna_fp.name.replace("_classified.tsv", "_feature_table.tsv")
        df_merged.to_csv(out_path, sep="\t", index=False)
        print(f"Merged features ({', '.join(sources)}) -> {out_path}")


# reporting 

def _report_failed_extractions(results_folder):
    results_folder = Path(results_folder)
    failure_keywords = {"NOT_FOUND", "ERROR_FETCH", "TRANSLATION_NOT_FOUND"}
    found_any = False

    for file_path in results_folder.glob("*.tsv"):
        df = pd.read_csv(file_path, sep="\t")
        if not {"ProteinSequence", "DNASequence"}.issubset(df.columns):
            continue

        failed = df[
            df["ProteinSequence"].isin(failure_keywords) |
            df["DNASequence"].isin(failure_keywords)
        ]
        if not failed.empty:
            found_any = True
            print(f"\nFailures in {file_path.name} ({len(failed)})")
            print(failed[["Geneid", "ProteinSequence", "DNASequence"]].to_string(index=False))

    if not found_any:
        print("All sequence extractions succeeded. No issues found.")


# keep backward-compat for the pipeline import
def report_failed_extractions(results_folder):
    return _report_failed_extractions(results_folder)
