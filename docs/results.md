# Results log

Splits: time-based (train days 0-120, validation 120-150, test 150-183).
Test set is not used until the final model is chosen.

## Baseline: logistic regression (validation window)
Features: log amount, hour of day (sin/cos), has_identity, log C1-C14, ProductCD.

| metric | value |
|---|---|
| random-guess PR-AUC (base rate) | 0.0347 |
| PR-AUC | 0.3832 |
| ROC-AUC | 0.8212 |
| recall @ top 1% | 0.2340 (max possible is about 0.29) |
| precision @ top 1% | 0.8124 |

## LightGBM on raw columns (validation window)
No feature engineering. 431 raw columns (TransactionID, TransactionDT and isFraud
dropped). Text columns become categoricals using only values seen >= 100 times in
train; rare or unseen values become NaN. Early stopping on validation picked 913
trees, so validation scores are slightly optimistic. The test window is still unused.

| model | PR-AUC | ROC-AUC | recall@1% | precision@1% | recall@5% | precision@5% |
|---|---|---|---|---|---|---|
| logistic regression | 0.3832 | 0.8212 | 0.2340 | 0.8124 | 0.4227 | 0.2936 |
| LightGBM (raw) | 0.6280 | 0.9304 | 0.2687 | 0.9332 | 0.6614 | 0.4593 |

Notes:
- Recall@1% is capped at about 0.29 (the 1% budget is 853 alerts and validation has
  about 2,960 frauds), so recall@5% is the better operating-point metric.
- Top feature by gain is V258 (9.3% of total gain), nearly double the next one.
  Candidate for drift monitoring in Phase 11.
- Open question: card1/card2/addr1 act like entity identifiers and there is no gap
  between train and validation, so scores may be inflated by shared cards. To test
  with a gap experiment.

## Feature engineering batch 1: stateless row-wise features (validation window)
Added ten columns via src/mlops_pipeline/features.py: hour, dow, log_amt, amt_frac,
nan_D / nan_M / nan_V / nan_id (missing counts per column family), has_identity,
email_match (three-valued). No state is learned, so the features cannot leak across
splits (tested: transforming rows alone equals transforming them inside the full frame).

| config | columns | trees | PR-AUC | ROC-AUC | recall@5% | precision@5% |
|---|---|---|---|---|---|---|
| LightGBM raw | 431 | 913 | 0.6280 | 0.9304 | 0.6614 | 0.4593 |
| LightGBM + batch 1 | 441 | 1712 | 0.6347 | 0.9273 | 0.6644 | 0.4614 |

Notes:
- PR-AUC +0.0067 but ROC-AUC -0.0031, and trees nearly doubled: not a clear win.
- None of the new columns reached the top-10 by gain. The ranking reshuffled among
  correlated V columns (V257 now first, V258 share 9.3% -> 3.9%); gain importance is
  unreliable across correlated features.
- All results are a single seed (42) on a single validation window, so no noise estimate
  yet. Next: repeat across seeds to measure noise before judging further feature batches.

## Seed noise: raw vs batch 1 (validation PR-AUC, same single validation window)

| seed | raw | + batch 1 | difference | recall@5% raw | recall@5% batch 1 |
|---|---|---|---|---|---|
| 42 | 0.6280 | 0.6347 | +0.0067 | 0.6614 | 0.6644 |
| 1 | 0.6322 | 0.6304 | -0.0018 | 0.6648 | 0.6577 |
| 2 | 0.6356 | 0.6312 | -0.0044 | 0.6759 | 0.6607 |
| 3 | 0.6298 | 0.6321 | +0.0023 | 0.6570 | 0.6583 |
| mean | 0.6314 | 0.6321 | +0.0007 | 0.6648 | 0.6603 |

- Raw PR-AUC varies 0.6280-0.6356 across seeds (sd about 0.003). Differences below about
  0.008 are not distinguishable from seed noise on this window.
- Batch 1 shows no reliable gain: the sign flips across seeds, and recall@5% is slightly
  lower on average. The seed-42 gain of +0.0067 reported earlier was the most favourable
  of four pairings.
- Decision: production default stays on raw columns. features.py is kept as tested
  experiment code. Seeds only measure model randomness; they do not measure sensitivity
  to the choice of validation window.

## Recency experiment: gap before validation (raw features, validation PR-AUC)
Question: do shared cards / recent data inflate validation scores? Run with
`--gap-days 14` (drop the last 14 training days, so a 14-day gap precedes validation) and
a control `--trim-early-days 14` (drop the first 14 training days: same amount of data
lost, nearest training day stays adjacent to validation). Seeds 1, 2, 3.

| seed | no gap | control | gap-14 |
|---|---|---|---|
| 1 | 0.6322 | 0.6263 | 0.5558 |
| 2 | 0.6356 | 0.6300 | 0.5589 |
| 3 | 0.6298 | 0.6352 | 0.5615 |
| mean | 0.6325 | 0.6305 | 0.5587 |
| rows | 410,601 | 358,988 | 368,632 |
| recall@5% (mean) | 0.6659 | 0.6621 | 0.6080 |

- The control stays within seed noise of no-gap, so lost data volume does not explain
  the drop. Gap-14 loses about 0.07 PR-AUC with more rows than the control.
- The mechanism is not identified. Shared cards across the cutoff and fast-changing
  fraud patterns would both produce this result.
- Implication: validation scores assume training data right up to the cutoff. With
  delayed labels in production, expect something closer to the gap-14 number. Retraining
  cadence matters (Phase 12).
- Decision: batch 2 entity features (frequency encodings, aggregations) are skipped.
  This is a time-budget call, not a measured result. Production default: raw columns.

## Promotion and final test-window score: fraud-lgbm v1
Rule: docs/promotion_policy.md, implemented in src/mlops_pipeline/promotion.py.
v1 vs the logistic-regression baseline on validation: PR-AUC gain +0.2449 (required
0.0100), recall@5% above the baseline, round trip diff 0.0, so v1 was promoted and
the registry alias `champion` points to it. The test window was scored once, after the
decision, and the result is logged on the same MLflow run.

| metric | validation | test (94,636 rows, fraud rate 0.0347) |
|---|---|---|
| PR-AUC | 0.6280 | 0.5490 |
| ROC-AUC | 0.9304 | 0.9037 |
| recall@1% | 0.2687 | 0.2587 |
| precision@1% | 0.9332 | 0.8975 |
| recall@5% | 0.6614 | 0.5893 |
| precision@5% | 0.4593 | 0.4088 |

Notes:
- Headline number to quote: test PR-AUC 0.549 (random guess: 0.0347). Validation scores
  are optimistic because early stopping used them.
- The validation-to-test drop (-0.079) is similar in size to the 14-day-gap experiment
  (-0.07). Consistent with a recency effect, not proof: different time periods also differ
  in fraud mix and volume.
- recall@1% is capped at about 0.29 on this window (946 alerts, about 3,280 frauds), so the
  1% budget barely separates models. Degradation shows at 5%: recall 0.661 -> 0.589.
- The test window is now spent for model selection. Retraining comparisons (Phase 12)
  need a fresh holdout.
