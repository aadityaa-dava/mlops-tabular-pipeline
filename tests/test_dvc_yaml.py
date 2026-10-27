from pathlib import Path

import yaml

from mlops_pipeline.config import load_config
from mlops_pipeline.schema import REPORT_PATH

ROOT = Path(__file__).resolve().parents[1]
STAGES = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]
RAW_CONFIG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
CFG = load_config(ROOT / "configs" / "config.yaml")


def entry_path(entry):
    """DVC entries are either a plain path or a one-key mapping {path: options}."""
    return next(iter(entry)) if isinstance(entry, dict) else entry


def test_merge_stage_matches_config_paths():
    merge = STAGES["merge"]
    raw = Path(CFG.data.raw_dir)
    assert [entry_path(o) for o in merge["outs"]] == [str(CFG.data.merged_path)]
    for name in ("train_transaction.csv", "train_identity.csv"):
        assert str(raw / name) in merge["deps"]


def test_validate_stage_matches_code_and_config():
    validate = STAGES["validate"]
    assert str(CFG.data.merged_path) in validate["deps"]
    assert [entry_path(m) for m in validate["metrics"]] == [str(REPORT_PATH)]


def test_tracked_params_exist_in_the_config():
    for stage_name, stage in STAGES.items():
        for entry in stage.get("params", []):
            for file, keys in entry.items():
                assert file == "configs/config.yaml", stage_name
                for key in keys:
                    node = RAW_CONFIG
                    for part in key.split("."):
                        assert part in node, f"{stage_name}: {key} not in config"
                        node = node[part]
