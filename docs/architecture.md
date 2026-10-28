# Architecture

What exists in this repository today and how the pieces connect. Solid arrows are code that runs; dotted arrows are file reads between separate environments.

```mermaid
flowchart LR
    subgraph data["Data (versioned)"]
        raw["Raw CSVs<br/>DVC pointers in git<br/>remote: local folder"] --> merge["merge<br/>(DVC stage)"]
        merge --> validate["validate<br/>pandera schema"]
        validate --> train["train<br/>LightGBM, config-driven"]
    end
    train --> mlflow["MLflow<br/>params, metrics, data hashes,<br/>registry (champion alias)"]
    mlflow --> export["export<br/>artifacts/champion/<br/>model + categories + importance"]
    export --> git["git: code + exported model"]
    git --> ci["GitHub Actions<br/>lint, tests, monitoring tests,<br/>build, health check"]
    ci --> ghcr["GHCR image<br/>tagged with commit SHA"]
    ghcr --> render["Render free tier<br/>FastAPI /health /predict"]
    export -. "watchlist" .-> monitor["monitoring/<br/>Evidently, pandas 2,<br/>per-column limits"]
    raw -. "windows" .-> monitor
    monitor --> report["drift reports"]
    export --> retrain["retrain<br/>paired comparison on<br/>days 135-150"]
    retrain --> candidate["artifacts/candidate<br/>staging, manual promotion"]
```

## What is not connected

- The deployed service does not feed the monitor, and its disk is ephemeral.
- Retraining runs locally and stops at a staged candidate; promoting it to champion, re-registering it in MLflow and redeploying are manual.
- The DVC remote is a folder on one machine, so cloud jobs cannot pull the data.
- Monitoring and retraining read data that only exists locally, so CI tests them with synthetic data, not the real windows.
