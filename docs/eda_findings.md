# EDA findings

Data: IEEE-CIS train_transaction joined with train_identity
(590,540 rows, 434 columns, 183 days, fraud rate 3.5%).

## What we learned
- Missingness is structural. C columns: 0% missing. D, M, V: 43-58% on
  average. id_*: ~85%. Only 24.4% of transactions have an identity record.
- has_identity is strong: fraud 7.85% with identity vs 2.09% without.
- ProductCD: C = 11.7% fraud (68k rows), W = 2.0% fraud (440k rows).
- TransactionAmt overlaps heavily between classes; spikes at standard prices.
- Weekly fraud rate swings between ~1.9% and ~5.1%, well beyond sampling
  noise (~0.25 pp). Expect validation and test windows to differ.

## Decisions for Phase 4 (modelling)
- Time-based split on TransactionDT; never shuffle.
- Do not use raw TransactionDT or TransactionID as features (they encode
  time). Derive hour-of-day and day-of-week instead.
- Add has_identity and per-family missing-count features.
- Keep NaNs; use tree models that handle them natively.
- Fit any encoders or statistics on the training window only.
- Serving must send missing identity fields as NaN (avoid train/serve skew).
- Report PR-AUC over a multi-week test window, and avoid over-reading small
  differences between models.
