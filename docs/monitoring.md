# Drift monitoring notes

## Environment
Evidently 0.7.23 crashes on import under pandas 3.1 (it needs a pandas internal that no
longer exists), although its metadata allows any pandas >= 1.3.5, so the resolver did not
warn. Monitoring therefore lives in its own project (monitoring/) on pandas 2.x. The main
pipeline stays on pandas 3. The two communicate through files, not imports.

## First probe on real data (10 most important model features, Evidently defaults)
Reference: 100,000 random training rows (days 0-120). Current: up to 50,000 rows per
window. Default method: normalised Wasserstein distance (numeric) and Jensen-Shannon
distance (categorical), drift threshold 0.1 per column.

| window | columns over 0.1 | largest score |
|---|---|---|
| null: two random halves of train | 0 of 10 | 0.020 (V258) |
| train days 0-30 | 2 of 10 (C1, C14) | 0.111 (C1) |
| train days 30-60 | 0 of 10 | 0.070 (V258) |
| train days 60-90 | 0 of 10 | 0.054 (V258) |
| train days 90-120 | 0 of 10 | 0.051 (V258) |
| validation days 120-150 | 0 of 10 | 0.080 (D2) |
| test days 150-183 (features only) | 0 of 10 | 0.079 (D2) |

## What this says
- The 0.1 default sits just above normal block-to-block variation (0.03-0.08), and it
  already fires on days 0-30, which are inside the reference window.
- No feature-drift alert would have fired for validation or test, yet PR-AUC fell by 0.079
  from validation to test. Feature drift on these 10 columns does not explain or predict the
  performance drop (consistent with, not proof of, entity or recency effects). Detecting
  decay needs labels, which arrive late in production.
- Only 10 columns were checked, with the default threshold and one sampling seed.

## Monitoring code (monitoring/): first run
Watchlist: the 10 features with the largest gain share (artifacts/champion/feature_importance.json).
Reference: 100,000 random rows from days 0-120. A current window alerts when any watched
column has an Evidently drift score of at least 0.2, or when its share of missing values
moves by at least 0.2 (absolute) from the reference.

First run, same windows as the probe: the drift scores reproduce the probe. The null row
differs slightly on V258 (0.016 vs 0.020), probably because reference and current are swapped
for that row (not checked). Normal change in missing share is up to about 0.09 per column,
except V258 on days 0-30 (0.161).

Threshold history:
- missing_rate_shift started at 0.15, a guess made before seeing numbers. It fired on days
  0-30, which are inside the reference (V258: 0.161), so it was raised to 0.2.
- Both thresholds are tuned on the data they are judged on. With them nothing alerts on the
  training blocks, validation or test. That shows the rules are not trigger-happy on this
  data, not that they would catch real production drift.

Open observations:
- card1 on test scores 0.037 against at most 0.012 in the training blocks, and D2 on
  validation and test scores about 0.08 against at most 0.048. These are 3x and 1.7x the
  largest in-reference scores but far below the global 0.2, so they do not alert. Per-column
  thresholds would flag them. Whether that is signal or noise is untested.
- Next: inject controlled shifts into a window to measure how small a shift the 0.2 rule
  detects.
