"""
ZINC250k Downloader — Stable Dataset for SMILES Pre-training
==============================================================
Downloads ZINC250k from HuggingFace (zpn/zinc250k), validates with RDKit,
canonicalises, filters to drug-like range, and saves as zinc250k_clean.csv.

Usage:
    python data/zinc_downloader.py [--output data/zinc250k_clean.csv]

Output: CSV with columns [smiles, mw, logp], ~247k rows
"""

import os
import sys
import argparse
import time

import numpy as np
import pandas as pd

# Add parent dir for config imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_v4 import ZINC_DATA_PATH

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[ERROR] RDKit is required. Install: pip install rdkit-pypi")
    sys.exit(1)


def download_zinc250k() -> pd.DataFrame:
    """Download ZINC250k from HuggingFace with fallback."""
    print("[1/4] Downloading ZINC250k dataset...")

    # Primary: HuggingFace datasets
    try:
        from datasets import load_dataset

        ds = load_dataset("zpn/zinc250k", split="train")
        df = ds.to_pandas()
        # The HF dataset usually has a 'smiles' column
        if "smiles" not in df.columns:
            # Try common column name variants
            for col in df.columns:
                if "smile" in col.lower():
                    df = df.rename(columns={col: "smiles"})
                    break
        if "smiles" not in df.columns:
            # Use first column as SMILES
            df = df.rename(columns={df.columns[0]: "smiles"})
        print(f"  Downloaded {len(df)} molecules from HuggingFace")
        return df
    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        print("  Trying fallback source...")

    # Fallback: Direct CSV download
    try:
        import requests

        url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Save temporarily and read
        tmp_path = "/tmp/zinc250k_raw.csv"
        with open(tmp_path, "w") as f:
            f.write(response.text)
        df = pd.read_csv(tmp_path)
        if "smiles" not in df.columns:
            for col in df.columns:
                if "smile" in col.lower():
                    df = df.rename(columns={col: "smiles"})
                    break
        print(f"  Downloaded {len(df)} molecules from GitHub fallback")
        return df
    except Exception as e2:
        print(f"  Fallback download also failed: {e2}")
        raise RuntimeError(
            "Could not download ZINC250k from any source. "
            "Please manually download and place as zinc250k_clean.csv"
        )


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate SMILES with RDKit, canonicalise, and filter to drug-like range."""
    print(f"[2/4] Validating {len(df)} SMILES with RDKit...")

    valid_rows = []
    invalid_count = 0

    for idx, row in df.iterrows():
        smi = str(row["smiles"]).strip()
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            invalid_count += 1
            continue

        # Canonicalise
        canonical = Chem.MolToSmiles(mol, canonical=True)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()

        valid_rows.append(
            {
                "smiles": canonical,
                "mw": round(mw, 2),
                "logp": round(logp, 3),
                "heavy_atoms": heavy_atoms,
            }
        )

    print(f"  Valid: {len(valid_rows)}, Invalid: {invalid_count}")

    df_clean = pd.DataFrame(valid_rows)

    # Filter to drug-like range
    print("[3/4] Filtering to drug-like range (MW 150-600, ≥5 heavy atoms)...")
    before = len(df_clean)
    df_clean = df_clean[
        (df_clean["mw"] >= 150)
        & (df_clean["mw"] <= 600)
        & (df_clean["heavy_atoms"] >= 5)
    ].copy()

    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    # Drop heavy_atoms column (not needed for training)
    df_clean = df_clean.drop(columns=["heavy_atoms"])

    print(f"  Before filter: {before}, After: {len(df_clean)}")

    return df_clean


def main():
    parser = argparse.ArgumentParser(description="Download and clean ZINC250k")
    parser.add_argument(
        "--output",
        type=str,
        default=str(ZINC_DATA_PATH),
        help="Output path for cleaned CSV",
    )
    args = parser.parse_args()

    output_path = args.output

    # Check if already exists
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        print(f"[CACHED] {output_path} already exists with {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample:\n{df.head(3).to_string()}")
        return

    t0 = time.time()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Download
    df_raw = download_zinc250k()

    # Validate and clean
    df_clean = validate_and_clean(df_raw)

    # Save
    print(f"[4/4] Saving to {output_path}...")
    df_clean.to_csv(output_path, index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ZINC250k Download Complete")
    print(f"  Rows:    {len(df_clean):,}")
    print(f"  Columns: {list(df_clean.columns)}")
    print(f"  Path:    {output_path}")
    print(f"  Time:    {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n  Sample:")
    print(df_clean.head(5).to_string(index=False))

    # SMILES length stats (important for MAX_SMILES_LEN config)
    lengths = df_clean["smiles"].str.len()
    print(f"\n  SMILES length stats:")
    print(f"    Mean:   {lengths.mean():.1f}")
    print(f"    Median: {lengths.median():.1f}")
    print(f"    Max:    {lengths.max()}")
    print(f"    99.9th: {lengths.quantile(0.999):.0f}")


if __name__ == "__main__":
    main()
