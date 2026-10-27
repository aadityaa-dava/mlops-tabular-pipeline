from mlops_pipeline.experiments import CONFIGS


def test_configs_are_well_formed():
    for name, cfg in CONFIGS.items():
        assert cfg["model"] in {"logreg", "lgbm"}, name
        assert cfg["gap_days"] >= 0 and cfg["trim_early_days"] >= 0, name


def test_gap_and_control_remove_the_same_number_of_days():
    assert CONFIGS["gap14"]["gap_days"] == CONFIGS["control14"]["trim_early_days"] == 14.0
    assert CONFIGS["gap14"]["trim_early_days"] == 0.0
    assert CONFIGS["control14"]["gap_days"] == 0.0
