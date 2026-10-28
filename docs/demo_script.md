# Demo video script (about 4 minutes)

Before recording: wake the live service by opening its /health page a few minutes earlier (a
sleeping free instance can take about a minute to answer), start `make mlflow-ui`, and have the
GitHub repository, the Actions page and docs/architecture.md open in tabs. The live /docs page
was not verified when this script was written: check it before relying on it.

1. The problem (30 s). Card-fraud scoring, 3.5% of transactions are fraud, so accuracy is
   useless. The metric is PR-AUC (random guess 0.035), the split is by time, and the test
   window was scored once: 0.549. Say plainly that it is lower than validation (0.628) and why
   that matters.
2. The pipeline (60 s). Show the architecture diagram. Run `make repro` and point at the skipped
   stages: nothing changed, so nothing reruns. Open the MLflow UI and show a run with its git
   commit, data hashes and metrics.
3. The live service (45 s). curl /health, then /predict with two transactions, then one request
   with an `isFraud` field to show it is rejected. Mention the cold start and that there is no
   authentication.
4. CI/CD (30 s). Show a green Actions run: lint, tests, monitoring tests, image build, a health
   check and a scored request, then the push to the registry.
5. Monitoring and its limits (45 s). Run `make simulate` and show the table: what the per-column
   limits catch, and the V258 row that they miss. Say that feature drift did not predict the
   drop from validation to test.
6. Retraining (30 s). Show reports/retrain_report.json: +0.0878, and the sentence about
   recency. Say why promotion stays manual.
7. What broke (30 s). Pick two: the DVC pointer files that were ignored, and Evidently not
   importing under pandas 3.

Keep every claim to what the repository shows, and say which numbers are measured and which
are judgment calls.
