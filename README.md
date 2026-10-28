# mlops-tabular-pipeline

An end-to-end ML pipeline for card-transaction fraud detection (IEEE-CIS dataset):
versioned data, a config-driven reproducible pipeline, tracked experiments, a model
registry with a written promotion policy, and tests. Serving, CI/CD, deployment and
drift monitoring are in progress.

## Status

| phase | state |
|---|---|
| data, EDA, validation, DVC | done |
| baseline and LightGBM, MLflow tracking and registry | done |
| config-driven DVC pipeline | done |
| FastAPI serving, Docker image, CI/CD to GHCR | done |
| cloud deployment (Render free tier) | done |
| drift monitoring (own environment, per-column limits, sensitivity simulation) | done |
| retraining trigger | not started |

## Results (test window, scored once)

Model: LightGBM on raw columns. Random-guess PR-AUC is 0.035 (3.5% fraud).

| metric | validation | test |
|---|---|---|
| PR-AUC | 0.628 | **0.549** |
| ROC-AUC | 0.930 | 0.904 |
| recall @ top 5% | 0.661 | 0.589 |
| precision @ top 5% | 0.459 | 0.409 |

Validation scores are optimistic (early stopping used them). More detail, including
experiments that did not help, is in `docs/results.md`; the decision rules are in
`docs/promotion_policy.md`.

## Reproduce

Requires macOS or Linux, [uv](https://docs.astral.sh/uv/), and Git. Data comes from
Kaggle (the competition rules must be accepted with your own account).

    git clone https://github.com/aadityaa-dava/mlops-tabular-pipeline.git
    cd mlops-tabular-pipeline
    make setup
    # place train_transaction.csv and train_identity.csv from
    # https://www.kaggle.com/c/ieee-fraud-detection/data into data/raw/
    make test && make lint
    make repro

The DVC remote is currently a folder on the author's machine, so `dvc pull` only works
there. A shared remote comes with the cloud-deployment phase. DVC checks the raw files
against the hashes in `data/raw/*.csv.dvc`, so wrong or modified data is detected.

## Serve

Every push to `main` runs lint and tests, builds the image, starts it, checks its health,
scores a request through it, and publishes it to GitHub's container registry tagged with the
commit SHA. To run it locally:

    docker run -p 8000:8000 ghcr.io/aadityaa-dava/mlops-tabular-pipeline:latest
    curl -s localhost:8000/health

The service scores transactions (it returns probabilities; choosing an alert threshold is
left to the caller). Only `TransactionAmt` and `ProductCD` are required, other columns are
optional and a missing value is treated as missing, as in training. Columns the model does
not use (including `isFraud`) are rejected. There is no authentication or rate limiting yet.

The model ships inside the image (`artifacts/champion/`, exported from the registry with
`make export`), so serving needs neither MLflow nor the data. A test confirms the HTTP path
scores 2,000 real validation rows identically to batch scoring.

## Live demo

The service is deployed on Render's free tier from the image CI publishes:

    https://fraud-api-bvtt.onrender.com/health

Things to know before calling it:

- A free instance spins down after 15 minutes without traffic, so the first request after
  a quiet spell can take about a minute.
- It has no authentication or rate limiting: anyone with the URL can call it. It scores
  transactions and returns probabilities; no data is stored.
- The disk is ephemeral, so nothing written by the service survives a restart.
- Redeploys are manual: the service pins one image tag (a commit SHA), and Render does not
  redeploy prebuilt images automatically. See `docs/deployment.md`.

## Monitoring

Drift checks run in `monitoring/`, a separate environment with pandas 2, because Evidently
0.7 does not import under pandas 3 (the main pipeline stays on pandas 3). The two sides share
files, not code: the model export folder lists which features matter
(`feature_importance.json`) and which are categorical (`categories.json`).

    make monitor     compare time windows with a training-period reference
    make simulate    inject controlled shifts and see where the rules start to fire

- It watches the model's 10 highest-gain features. A window alerts when a column's Evidently
  drift score, or the change in its share of missing values, reaches that column's own limit.
  Limits are 1.25x the largest value seen in four training blocks, with floors.
- In the simulation the per-column limits caught a 10% inflation of transaction amounts, a
  shift of 0.1 standard deviations in `C13`, and 5% of rows carrying an unseen email domain.
  A single global threshold needed roughly 50%, 0.25 and 10% for the same three. They missed
  a 5% inflation of amounts.
- Known gaps: a sparse column that stops arriving (`V258`, present in about a quarter of rows)
  is not detected at all; the limits were calibrated on windows inside the reference sample,
  so they are optimistic; and the per-column rule already fires on the validation window
  (`D2`), which may be real drift or a limit that is too tight.
- Feature drift did not predict the performance drop: nothing alerted under the global rule
  when test PR-AUC fell by 0.08. Detecting decay needs labels, which arrive late in
  production.
- The deployed service is not wired to the monitor. Newer images log one JSON line per
  `/predict` call to stdout (aggregates only, no field values), but the live Render service
  runs an older image and its disk is ephemeral.

Details and numbers: `docs/monitoring.md`.

## Layout

    configs/config.yaml     all settings (split cutoffs, seed, hyperparameters, MLflow names)
    dvc.yaml                pipeline: merge -> validate -> train
    src/mlops_pipeline/     data, schema, split, features, models, tracking, registry
    tests/                  unit tests
    docs/                   problem framing, EDA findings, results log, promotion policy

## Honest notes

- Test PR-AUC is about 0.08 below validation, close to the 0.07 drop seen when a
  14-day gap separates training from validation. That suggests recent data matters a lot;
  it does not prove why.
- Engineered row-wise features did not help beyond seed noise and were not adopted.
