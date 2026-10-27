# Problem framing

## Business problem
An online payments company wants to flag fraudulent card-not-present
transactions in near real time, so a fraud team can review or block them
before money is lost.

## Prediction task
Binary classification: given a transaction's features, predict
`isFraud` (1 = fraud). Output is a probability; a threshold turns it into an
alert.

## Why accuracy is the wrong metric
Fraud is rare (roughly 3-4% of rows; to be verified in EDA). A model that
predicts "not fraud" for everything scores ~96% accuracy and catches nothing.

## Costs (assumptions, to be tuned)
- Missed fraud (false negative): we lose the transaction amount.
- False alarm (false positive): a fixed review cost, assumed $5 per alert
  (analyst time plus customer friction).
- The review team can only handle a limited alert volume.

## Success metrics
- Primary (model selection): PR-AUC (average precision) on a time-based
  held-out test set. It is threshold-free and honest under class imbalance.
- Operational: recall at a fixed alert budget (e.g. when flagging the top 1%
  of transactions by score), plus expected cost at the chosen threshold.
- Guardrails: calibration of scores, and inference latency of the API.

## Evaluation protocol
- Split by time (`TransactionDT`): train on the earliest transactions,
  validate on the next window, test on the latest. No random shuffling,
  because fraud patterns shift and random splits leak the future.
- Test set is touched once, at the end of each modelling phase.

## Baseline to beat
A logistic regression on a handful of numeric features. Everything we build
must be compared against it.
