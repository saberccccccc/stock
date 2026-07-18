import pytest

from experiments.factor_baselines import audit_factor_baselines, get_factor_baseline


def test_compact_price_volume_baseline_is_ready_from_its_own_features():
    spec = get_factor_baseline("alpha158_compact_price_volume_v1")
    report = audit_factor_baselines(spec["features"])
    assert report["alpha158_compact_price_volume_v1"]["ready"] is True
    assert report["alpha158_broad_price_volume_v1"]["ready"] is False


def test_baseline_lookup_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown factor baseline"):
        get_factor_baseline("not_a_baseline")
