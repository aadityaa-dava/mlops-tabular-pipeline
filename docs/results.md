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
