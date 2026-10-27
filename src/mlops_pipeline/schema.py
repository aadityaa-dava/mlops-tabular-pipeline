"""Data contract for the merged IEEE-CIS transaction table, plus a validation CLI."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from mlops_pipeline.config import load_config

REPORT_PATH = Path("reports/data_validation.json")

transaction_schema = pa.DataFrameSchema(
    {
        "TransactionID": pa.Column(int, unique=True, nullable=False),
        "isFraud": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "TransactionDT": pa.Column(int, pa.Check.ge(0), nullable=False),
        "TransactionAmt": pa.Column(checks=pa.Check.gt(0), nullable=False),
        "ProductCD": pa.Column(checks=pa.Check.isin(["W", "C", "R", "H", "S"]), nullable=False),
    },
    checks=pa.Check(
        lambda df: df["TransactionDT"].is_monotonic_increasing,
        error="rows must be sorted by TransactionDT",
    ),
    strict=False,  # the other ~430 columns are allowed, and may contain nulls
)


def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Raise pandera.errors.SchemaErrors listing every violation, else return df."""
    return transaction_schema.validate(df, lazy=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    df = pd.read_parquet(cfg.data.merged_path)
    validate_transactions(df)  # any violation raises, so the pipeline stage fails
    report = {
        "passed": True,
        "rows": len(df),
        "columns": df.shape[1],
        "fraud_rate": round(float(df["isFraud"].mean()), 6),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"data validation passed: {report}")


if __name__ == "__main__":
    main()
