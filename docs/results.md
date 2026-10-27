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
