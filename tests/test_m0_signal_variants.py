import pytest
import numpy as np

from run.generate_m0_signal_variants import (
    main,
    parse_args,
    parse_variant_specs,
    pending_variant_specs,
)
from backtest.predictors import PersistentPredictor
from run.v9_long_only_optimization import V9RankPredictor


class CountingPredictor:
    def __init__(self):
        self.calls = 0
        self.base = type("Base", (), {"score_source": "alpha"})()

    def predict_alpha(self, sample, valid, regime):
        self.calls += 1
        return np.asarray(sample["scores"], dtype=np.float32)


def test_parse_m0_signal_variants():
    assert parse_variant_specs("raw,avg2,avg3,avg5") == [
        ("raw", "none", 1),
        ("avg2", "average", 2),
        ("avg3", "average", 3),
        ("avg5", "average", 5),
    ]


def test_parse_m0_signal_variants_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown signal variant"):
        parse_variant_specs("ema3")


def test_pending_variants_skip_existing_filtered_output(tmp_path):
    specs = parse_variant_specs("raw,avg2")
    completed = tmp_path / "m0" / "raw" / "alpha_maxret095.jsonl"
    completed.parent.mkdir(parents=True)
    completed.write_text("complete\n", encoding="utf-8")

    assert pending_variant_specs(tmp_path, "m0", specs) == [("avg2", "average", 2)]


def test_m0_variants_score_each_date_once(monkeypatch, tmp_path):
    predictor = CountingPredictor()
    samples = [
        {
            "date": "2024-01-02",
            "codes": [f"A{i}" for i in range(10)],
            "scores": list(range(10)),
            "X": np.zeros((10, 1)),
        },
        {
            "date": "2024-01-03",
            "codes": [f"A{i}" for i in range(10)],
            "scores": list(range(1, 11)),
            "X": np.zeros((10, 1)),
        },
    ]
    written = []
    monkeypatch.setattr(
        "run.generate_m0_signal_variants.load_v9_samples_and_predictor",
        lambda args: (None, samples, predictor),
    )
    monkeypatch.setattr(
        "run.backtest_v9_retention.detect_regime",
        lambda sample: None,
    )
    monkeypatch.setattr(
        "run.generate_m0_signal_variants.assert_alpha_rows_within_research",
        lambda rows, context: None,
    )
    monkeypatch.setattr(
        "run.generate_m0_signal_variants.write_alpha",
        lambda path, rows: written.append((path, rows)),
    )
    monkeypatch.setattr(
        "run.generate_m0_signal_variants.apply_chase_filter",
        lambda raw, filtered, data_dir: None,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_m0_signal_variants.py",
            "--checkpoint",
            "model.pt",
            "--output-dir",
            str(tmp_path),
            "--name",
            "m0",
            "--variants",
            "raw,avg2,avg3",
        ],
    )

    main()

    assert predictor.calls == len(samples)
    assert len(written) == 3


def test_legacy_variant_loop_already_reused_v9_score_cache():
    base = CountingPredictor()
    cached = V9RankPredictor(base, cache={})
    samples = [
        {"date": "2024-01-02", "codes": ["A", "B"], "scores": [1.0, 2.0]},
        {"date": "2024-01-03", "codes": ["A", "B"], "scores": [2.0, 3.0]},
    ]
    variants = [
        cached,
        PersistentPredictor(cached, window=2, mode="average"),
        PersistentPredictor(cached, window=3, mode="average"),
    ]

    for predictor in variants:
        for sample in samples:
            predictor.predict_alpha(
                sample,
                np.ones(len(sample["codes"]), dtype=bool),
                regime=None,
            )

    assert base.calls == len(samples)


def test_parse_args_supports_frozen_test_dates(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_m0_signal_variants.py",
            "--checkpoint",
            "model.pt",
            "--output-dir",
            "out",
            "--name",
            "m0",
            "--split",
            "test",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2026-05-18",
        ],
    )
    args = parse_args()
    assert args.split == "test"
    assert args.start_date == "2025-01-01"
    assert args.end_date == "2026-05-18"
