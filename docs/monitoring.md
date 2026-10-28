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

## Sensitivity simulation (monitoring/simulate.py)
Base window: validation days 120-150 (50,000 rows), which fires nothing. The reference and
thresholds (0.2 and 0.2) are the monitor's. One column is altered at a time.

| alteration | target drift score | caught |
|---|---|---|
| TransactionAmt x1.05 / 1.1 / 1.2 / 1.5 / 2.0 | 0.031 / 0.058 / 0.116 / 0.291 / 0.583 | no / no / no / yes / yes |
| C13 + 0.1 / 0.25 / 0.5 / 1.0 std | 0.144 / 0.293 / 0.541 / 1.039 | no / yes / yes / yes |
| P_emaildomain: 5 / 10 / 20 / 40% of rows get an unseen value | 0.149 / 0.209 / 0.297 / 0.431 | no / yes / yes / yes |
| V258: 5 / 10 / 20 / 30 / 50% of present values go missing | score 0.072 / 0.075 / 0.062 / 0.066 / 0.053 | no for all |

For the V258 rows the missing-share shift was 0.100 / 0.108 / 0.123 / 0.138 / 0.169
(baseline 0.092), below the 0.2 rule in every case.

## What the simulation shows
- The drift score behaves like the shift in reference standard deviations plus the column's
  own baseline score (C13: 0.045 + 0.1 / 0.25 / 0.5 / 1.0 gives 0.145 / 0.295 / 0.545 /
  1.045, measured 0.144 / 0.293 / 0.541 / 1.039).
- A 20% inflation of transaction amounts (0.116) is missed. No single global threshold
  separates it from normal variation, because C1 reached 0.111 on days 0-30 inside the
  reference while TransactionAmt never exceeded 0.039 in any training block. Columns have
  different natural noise, so per-column thresholds look necessary.
- The missingness rule is nearly blind on sparse columns. Blanking half of V258's present
  values moves its missing share by about 0.077, which implies V258 is present in about 15%
  of rows (inferred, not counted). Even losing every value would move the share by only about
  0.15. The unit test for this rule uses a column that is 40% missing, which is much easier
  than the real data, so it overstates the rule's power.
- The drift score for V258 does not rise when its values go missing (0.072 down to 0.053), so
  the score appears to ignore missing values (inferred from these numbers, not checked in
  Evidently's code). A field that stops arriving is only visible to the missingness rule.
- The "worst other" column of the simulation output is D2's baseline (0.080) in every row: a
  shift in one column does not disturb the others, and the column carries nothing else.

## Next
- Calibrate per-column thresholds from the four training blocks only, and judge them on the
  validation and test windows. The window split then keeps calibration data separate from
  judging data.
- Replace the absolute missingness rule with one that scales with how often the column is
  present, then re-run this simulation.
