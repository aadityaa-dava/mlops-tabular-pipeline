"""Load raw IEEE-CIS CSVs, join transaction and identity tables, save as Parquet."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/train_merged.parquet")


def load_merged(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Left-join identity onto transactions and sort by time."""
    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    idn = pd.read_csv(raw_dir / "train_identity.csv")
    df = tx.merge(idn, on="TransactionID", how="left", validate="one_to_one")
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df.sort_values("TransactionDT").reset_index(drop=True)


def main() -> None:
    df = load_merged()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} shape={df.shape} fraud_rate={df['isFraud'].mean():.4f}")


if __name__ == "__main__":
    main()
