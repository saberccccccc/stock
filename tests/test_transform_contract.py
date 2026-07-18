import json
import pickle

import pytest

from data.transform_contract import build_v14_transform_contract, write_transform_contract


def _meta():
    return {
        "all_dates": ["2024-01-02", "2024-01-03"],
        "all_codes": ["000001.SZ"],
        "x_dim": 3,
        "risk_full_dim": 2,
        "feature_cols": ["x1", "x2"],
        "label_families": {"oo": {"norm_path": "oo.dat", "date_shift": 0}, "oo_lag1": {"alias_of": "oo", "date_shift": 1}},
    }


def test_build_contract_records_daily_normalization_and_declared_gaps(tmp_path):
    meta_path = tmp_path / "meta.pkl"
    with meta_path.open("wb") as handle:
        pickle.dump(_meta(), handle)

    contract = build_v14_transform_contract(_meta(), meta_path=meta_path)

    assert contract["feature_transforms"]["fit_scope"] == "daily_cross_section"
    assert contract["feature_transforms"]["global_train_fitted_scaler"] is False
    assert contract["labels"]["families"]["oo_lag1"]["date_shift"] == 1
    assert contract["status"] == "audited_with_declared_gaps"


def test_contract_writer_is_immutable(tmp_path):
    meta_path = tmp_path / "meta.pkl"
    meta_path.write_bytes(b"meta")
    contract = build_v14_transform_contract(_meta(), meta_path=meta_path)
    output = tmp_path / "transform_contract.json"

    write_transform_contract(contract, output)

    assert json.loads(output.read_text(encoding="utf-8"))["kind"] == "v14_transform_contract"
    with pytest.raises(FileExistsError):
        write_transform_contract(contract, output)
