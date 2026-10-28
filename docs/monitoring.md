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
