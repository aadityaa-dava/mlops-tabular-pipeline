import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text()
DOCKERIGNORE = (ROOT / ".dockerignore").read_text().split()
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
SERVE_REQS = (ROOT / "requirements-serve.txt").read_text()


def requirement_name(spec: str) -> str:
    return re.split(r"[<>=!~\[ ]", spec, maxsplit=1)[0].lower().replace("_", "-")


def test_serving_requirements_cover_the_runtime_dependencies():
    pinned = {m.group(1).lower() for m in re.finditer(r"^([A-Za-z0-9_.-]+)==", SERVE_REQS, re.M)}
    wanted = {requirement_name(d) for d in PYPROJECT["project"]["dependencies"]}
    assert wanted <= pinned, f"requirements-serve.txt is stale; missing {wanted - pinned}"


def test_serving_requirements_exclude_training_packages():
    pinned = {m.group(1).lower() for m in re.finditer(r"^([A-Za-z0-9_.-]+)==", SERVE_REQS, re.M)}
    heavy = {requirement_name(d) for d in PYPROJECT["dependency-groups"]["train"]}
    assert not heavy & pinned, f"training packages in the serving image: {heavy & pinned}"


def test_dockerfile_runs_as_non_root_with_a_healthcheck():
    assert "USER appuser" in DOCKERFILE
    assert "HEALTHCHECK" in DOCKERFILE
    assert "libgomp1" in DOCKERFILE  # LightGBM needs the OpenMP runtime


def test_dockerfile_copies_the_model_and_not_the_data():
    assert "COPY artifacts/champion" in DOCKERFILE
    assert "COPY data" not in DOCKERFILE


def test_dockerignore_keeps_data_and_secrets_out_but_not_the_model():
    for entry in ("data", ".git", "mlflow.db", ".env"):
        assert entry in DOCKERIGNORE, entry
    assert "artifacts" not in DOCKERIGNORE
