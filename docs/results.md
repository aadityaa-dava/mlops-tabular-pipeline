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
