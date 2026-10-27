# Model promotion policy

A registered model version becomes `champion` (the version the API serves) only if
all checks pass. The rule is implemented in `src/mlops_pipeline/promotion.py` and
is the same code the retraining job will run.

## Checks
1. **Reproducibility**: the model reloaded from the registry must reproduce its own
   scores (max absolute difference <= 1e-6).
2. **Primary metric**: PR-AUC must beat the reference by at least 0.01.
   Reference = the current champion, or the logistic-regression baseline if there is
   no champion yet. The margin is a judgment call set slightly above the seed-to-seed
   range measured on validation (0.6280-0.6356, i.e. 0.0076).
3. **Guardrail**: recall at a 5% alert budget may not fall more than 0.01 below the
   reference, so a PR-AUC gain cannot hide a worse operating point.

Candidate and reference must be scored on the same fixed window.

## Use of the data windows
- Validation is used for early stopping and model comparison, so its scores are
  slightly optimistic.
- The test window is scored once for the final model, for reporting only. It is never
  used to choose between candidates.

## Known limitations
- The margin and tolerance were chosen after seeing our own experiment history, so
  replaying that history through the rule is a sanity check, not independent proof.
- Validation scores are about 0.07 PR-AUC higher than with a 14-day gap before the
  window (docs/results.md), so absolute numbers overstate production performance.
- Open question for Phase 12: which window does the retraining job use to compare a new
  candidate with the champion? Not decided yet.
