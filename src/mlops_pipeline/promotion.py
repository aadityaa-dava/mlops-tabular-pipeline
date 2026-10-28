"""Promotion rule: should a candidate model replace the current champion?

Pure functions only, so the rule is easy to test and the retraining job can run the
same code. Metric keys are the MLflow names without the "val_" prefix.
"""

from dataclasses import dataclass

PR_AUC_MARGIN = 0.01  # a bit above the 0.0076 seed-to-seed range measured on validation
RECALL_TOLERANCE = 0.01
ROUNDTRIP_TOLERANCE = 1e-6
_EPS = 1e-9  # absorbs floating point error at the boundaries


@dataclass(frozen=True)
class Decision:
    promote: bool
    reason: str


def should_promote(
    candidate: dict[str, float],
    champion: dict[str, float] | None = None,
    baseline: dict[str, float] | None = None,
    margin: float = PR_AUC_MARGIN,
    recall_tolerance: float = RECALL_TOLERANCE,
) -> Decision:
    """Compare the candidate with the champion, or with the baseline if none exists yet."""
    reference, label = (champion, "champion") if champion is not None else (baseline, "baseline")
    if reference is None:
        raise ValueError("need a champion or a baseline to compare against")
    if candidate["roundtrip_max_abs_diff"] > ROUNDTRIP_TOLERANCE:
        return Decision(False, "reloaded model does not reproduce its own scores")
    gain = candidate["pr_auc"] - reference["pr_auc"]
    if gain < margin - _EPS:
        return Decision(
            False, f"PR-AUC gain {gain:+.4f} is below the required {margin:.4f} over the {label}"
        )
    recall_drop = reference["recall_at_5pct"] - candidate["recall_at_5pct"]
    if recall_drop > recall_tolerance + _EPS:
        return Decision(
            False,
            f"recall@5% drops {recall_drop:.4f}, more than the allowed {recall_tolerance:.4f}",
        )
    return Decision(True, f"PR-AUC gain {gain:+.4f} over the {label} meets the {margin:.4f} margin")


WINDOW_PARAMS = ("train_end_day", "val_end_day", "val_rows")


def window_mismatch(
    candidate_params: dict[str, str], reference_params: dict[str, str]
) -> str | None:
    """Describe the first difference in the evaluation window, or None if they match."""
    for key in WINDOW_PARAMS:
        a, b = candidate_params.get(key), reference_params.get(key)
        if a is None or b is None or a != b:
            return f"{key} differs or is missing (candidate {a}, reference {b})"
    return None


PAIRED_MARGIN = 0.02  # about twice the no-change retrain wobble measured on days 135-150
PAIRED_CONFIDENCE = "95% paired bootstrap interval must exclude zero"


def should_promote_paired(
    diff: float,
    interval_low: float,
    recall_drop: float,
    roundtrip_diff: float,
    margin: float = PAIRED_MARGIN,
    recall_tolerance: float = RECALL_TOLERANCE,
) -> Decision:
    """Promote only on a clear, paired improvement on the same rows.

    diff is candidate minus champion PR-AUC; interval_low is the lower end of its paired
    bootstrap interval. The gain must reach the margin AND the interval must exclude zero.
    """
    if roundtrip_diff > ROUNDTRIP_TOLERANCE:
        return Decision(False, "candidate does not reproduce its own scores after reloading")
    if diff < margin - _EPS:
        return Decision(False, f"PR-AUC gain {diff:+.4f} is below the required {margin:.4f}")
    if interval_low <= 0:
        reason = f"gain {diff:+.4f} is not separable from noise (interval low {interval_low:+.4f})"
        return Decision(False, reason)
    if recall_drop > recall_tolerance + _EPS:
        return Decision(
            False,
            f"recall@5% drops {recall_drop:.4f}, more than the allowed {recall_tolerance:.4f}",
        )
    return Decision(True, f"gain {diff:+.4f} meets the margin and its interval excludes zero")
