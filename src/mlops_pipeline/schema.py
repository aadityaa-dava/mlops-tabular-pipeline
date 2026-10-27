"""Data contract for the merged IEEE-CIS transaction table."""

import pandas as pd
import pandera.pandas as pa

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
