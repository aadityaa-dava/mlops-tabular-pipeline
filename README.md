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
| cloud deployment | not started |
| drift monitoring, retraining trigger | not started |

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
