"""Row-wise (stateless) feature engineering.

Every feature here is computed from a single row's own values. Nothing is learned
from the data (no means, counts or label statistics), so these transforms cannot
leak information between train, validation and test, and serving can reuse them
unchanged. Stateful features (frequency encodings, aggregations) come later and
must be fit on the training window only.
"""

import re

import numpy as np
import pandas as pd

FAMILY_PATTERNS = {
    "D": r"^D\d+$",
    "M": r"^M\d+$",
    "V": r"^V\d+$",
    "id": r"^id_\d+$",
}
SECONDS_PER_HOUR = 3_600
SECONDS_PER_DAY = 86_400


def family_columns(columns) -> dict[str, list[str]]:
    return {
        name: [c for c in columns if re.match(pattern, c)]
        for name, pattern in FAMILY_PATTERNS.items()
    }


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with engineered columns added. Needs no label column."""
    out = df.copy()
    dt = df["TransactionDT"]
    out["hour"] = (dt // SECONDS_PER_HOUR) % 24
    out["dow"] = (dt // SECONDS_PER_DAY) % 7  # origin unknown: a 7-day cycle label

    amt = df["TransactionAmt"]
    out["log_amt"] = np.log1p(amt)
    out["amt_frac"] = amt - np.floor(amt)

    for name, cols in family_columns(df.columns).items():
        out[f"nan_{name}"] = df[cols].isna().sum(axis=1) if cols else 0
    out["has_identity"] = df["id_01"].notna().astype(int)

    p, r = df["P_emaildomain"], df["R_emaildomain"]
    out["email_match"] = np.where(p.isna() | r.isna(), np.nan, (p == r).astype("float32"))
    return out
