import numpy as np
import torch

from backtest.engine import DLPredictor


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inference_modes = []
        self.input_shapes = []

    def forward(self, x, risk, mask, industry_ids):
        self.inference_modes.append(torch.is_inference_mode_enabled())
        self.input_shapes.append((tuple(x.shape), tuple(risk.shape), tuple(mask.shape)))
        alpha = x[..., 0]
        horizons = torch.stack((alpha + 1.0, alpha + 2.0), dim=-1)
        return alpha, horizons, None


def sample():
    return {
        "X": np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        "risk": np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        "industry_ids": np.asarray([1, 2, 3], dtype=np.int64),
    }


def test_dl_predictor_reuses_one_input_path_for_all_heads():
    model = RecordingModel()
    predictor = DLPredictor(model, torch.device("cpu"), regime_dim=1)
    valid = np.asarray([True, False, True])

    alpha = predictor.predict_alpha(sample(), valid, regime=None)
    raw = predictor.predict_raw_alpha(sample(), valid, regime=None)
    horizon = predictor.predict_horizon_alpha(sample(), valid, regime=None, horizon_idx=1)

    np.testing.assert_allclose(raw, [1.0, 3.0])
    np.testing.assert_allclose(horizon, [3.0, 5.0])
    assert alpha.shape == (2,)
    assert model.inference_modes == [True, True, True]
    assert model.input_shapes == [((1, 2, 2), (1, 2, 1), (1, 2))] * 3


def test_dl_predictor_returns_empty_scores_without_a_model_call():
    model = RecordingModel()
    predictor = DLPredictor(model, torch.device("cpu"), regime_dim=1)

    result = predictor.predict_alpha(sample(), np.asarray([False, False, False]), regime=None)

    assert result.dtype == np.float32
    assert result.size == 0
    assert model.inference_modes == []
