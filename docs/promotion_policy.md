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

## Amendment (Step 46)
- A candidate is compared with the current champion (alias `champion`). The baseline is
  the reference only while no champion exists. A candidate that already is the champion
  is a no-op.
- Candidate and reference runs must share the evaluation window (train_end_day,
  val_end_day, val_rows); otherwise the comparison is refused.
- The test window is scored at most once per project, for the first champion. Once any
  run in the experiment carries the tag test_scored=yes, later candidates are decided on
  validation only and the test window is not touched. The guard is registry-wide, not
  per run: looking at a second candidate's test score would turn the test window into a
  selection set.
- A rejected candidate is a normal outcome (exit code 0, reason printed), not an error.
- Still open (Phase 12): a fresh holdout for comparing retrained models.

## Amendment (Step 80): paired comparison for retrained candidates
Measured on days 135-150 (39,533 rows, 1,299 frauds): a retrain with the same training days and
a different seed moved PR-AUC by +0.0036 (paired 95% interval -0.0027 to +0.0111), so the
original 0.01 margin sits at the noise level of a window this size. A candidate trained on 15
more recent days scored +0.0878 (interval +0.0776 to +0.0981) over the champion.
For retrained candidates the rule is therefore: PR-AUC gain of at least 0.02 over the champion
on the same rows, a paired bootstrap interval that excludes zero, recall@5% not more than 0.01
lower, and a passing reload round trip. The 0.02 was set after seeing these numbers, so they
cannot validate it. The B result is largely the recency effect (its training data ends the
day before the comparison window), so it shows what retraining on fresh data buys, not a
better model in a deeper sense. Early stopping used the comparison window for both models.

## Retraining run (Step 81): candidate trained on days 0-135
`make retrain` trains a candidate on days 0-135 and compares it with the champion on days
135-150 (39,533 rows, 1,299 frauds), same rows for both, using the paired rule above.
Result: champion PR-AUC 0.5257, candidate 0.6135, paired difference +0.0878 (interval +0.0776
to +0.0981), recall@5% 8.3 points higher, round trip exact. The rule accepts the candidate,
which was written to artifacts/candidate/ (git-ignored staging folder). The champion, the
registry and the deployed image were not touched.

Limits that stay attached to this result:
- One comparison window. Days 120-135 were in the candidate's training data, so there is no
  held-out evidence for them, and the test window (150-183) has not been used.
- The candidate's training data ends the day before the comparison window starts, the champion's
  ended 15 days earlier. Earlier experiments showed this recency alone moves PR-AUC by about
  0.07, so most of the gain is probably the recency effect.
- Early stopping used the comparison window for both models.
- Promotion is deliberately not automatic: replacing the champion needs a person to look at
  reports/retrain_report.json first. No promote command exists yet.
