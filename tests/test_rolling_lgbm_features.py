from run.rolling_lgbm_alpha import build_experiment_contract, resolve_feature_indices


def test_factor_baseline_resolves_named_v14_feature_indices():
    meta = {"feature_cols": ["ret_5d", "ret_20d", "vol_10d", "vol_60d", "price_momentum", "log_volume", "volume_spike", "gap", "amplitude", "sma5_gap", "sma20_gap", "rsi_norm", "macd_pct", "atr_pct", "volume_ratio"]}
    indices, names, source = resolve_feature_indices(meta, {"data": {"factor_baseline": "alpha158_compact_price_volume_v1"}})
    assert source == "alpha158_compact_price_volume_v1"
    assert indices == list(range(15))
    assert names[0] == "ret_5d"


def test_rolling_contract_freezes_label_and_cache_boundaries():
    meta = {"meta_path": "cache/meta.pkl", "all_dates": ["2010-01-04", "2026-05-18"], "x_dim": 2}
    spec = {"data": {"research_end": "2026-05-18", "label_family": "oo_lag1", "horizon_index": 4}, "model": {}, "windows": []}
    contract = build_experiment_contract(spec, meta, "compact", ["ret_5d"], 7)
    assert contract["protocol"]["forward_is_observation_only"] is True
    assert contract["protocol"]["research_end"] == "2025-12-31"
    assert contract["protocol"]["cache_read_ceiling"] == "2026-05-18"
    assert contract["cache_contract"]["data_end"] == "2026-05-18"


def test_rolling_contract_clamps_scope_to_first_physical_trading_date():
    meta = {
        "meta_path": "cache/meta.pkl",
        "all_dates": ["2010-01-04", "2024-12-31"],
        "x_dim": 2,
    }
    spec = {
        "data": {
            "research_end": "2025-12-31",
            "label_family": "oo_lag1",
            "horizon_index": 4,
        },
        "model": {},
        "windows": [],
    }
    window = {
        "name": "predict_2024",
        "train_start": "2010-01-01",
        "train_end": "2022-12-31",
        "valid_start": "2023-01-01",
        "valid_end": "2023-12-31",
        "predict_start": "2024-01-01",
        "predict_end": "2024-12-31",
    }
    from experiments.rolling import RollingWindow

    contract = build_experiment_contract(
        spec,
        meta,
        "compact",
        ["ret_5d"],
        7,
        windows=[RollingWindow.from_mapping(window)],
    )

    scope = contract["experiment_scope"]["ranges"]
    assert scope["feature_warmup"]["start"] == "2010-01-04"
    assert scope["feature_warmup"]["end"] == "2010-01-04"
    assert scope["train"]["start"] == "2010-01-04"
