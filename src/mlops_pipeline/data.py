"""Load raw IEEE-CIS CSVs, join transaction and identity tables, save as Parquet."""

import argparse
from pathlib import Path

import pandas as pd

from mlops_pipeline.config import load_config


def load_merged(raw_dir: Path) -> pd.DataFrame:
    """Left-join identity onto transactions and sort by time."""
    raw_dir = Path(raw_dir)
    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    idn = pd.read_csv(raw_dir / "train_identity.csv")
    df = tx.merge(idn, on="TransactionID", how="left", validate="one_to_one")
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df.sort_values("TransactionDT").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    df = load_merged(cfg.data.raw_dir)
    out = Path(cfg.data.merged_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} shape={df.shape} fraud_rate={df['isFraud'].mean():.4f}")


if __name__ == "__main__":
    main()
