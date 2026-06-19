from types import SimpleNamespace

import torch

from core.train_utils import downside_top_loss, total_loss_v7


def test_downside_loss_prefers_low_downside_stocks():
    n_stocks = 40
    mask = torch.ones(1, n_stocks, dtype=torch.bool)
    paths = torch.full((1, n_stocks, 4), 0.02)
    paths[:, :5, 1] = -0.20

    high_risk_alpha = torch.zeros(1, n_stocks)
    high_risk_alpha[:, :5] = 5.0
    low_risk_alpha = -high_risk_alpha

    high_risk_loss = downside_top_loss(high_risk_alpha, paths, mask)
    low_risk_loss = downside_top_loss(low_risk_alpha, paths, mask)

    assert high_risk_loss > low_risk_loss


def test_downside_loss_is_zero_without_enough_stocks():
    alpha = torch.zeros(1, 20)
    paths = torch.zeros(1, 20, 4)
    mask = torch.ones(1, 20, dtype=torch.bool)

    assert downside_top_loss(alpha, paths, mask).item() == 0.0


def test_lag1_losses_have_independent_activation_flags():
    n_stocks = 40
    alpha = torch.linspace(-1.0, 1.0, n_stocks).reshape(1, n_stocks)
    alphas = alpha.unsqueeze(-1).repeat(1, 1, 2)
    horizon_preds = alpha.unsqueeze(-1)
    y = alpha.clone()
    y_seq = alpha.unsqueeze(-1)
    lag1_y_seq = alpha.unsqueeze(-1)
    mask = torch.ones(1, n_stocks, dtype=torch.bool)
    cfg = SimpleNamespace(
        horizon_indices=(0,),
        horizon_weights=(1.0,),
        industry_loss_weight=0.0,
        multi_loss_weight=0.0,
        diversity_loss_weight=0.0,
        spread_loss_weight=0.0,
        top_focus_loss_weight=0.0,
        downside_loss_weight=0.0,
        lag1_loss_weight=1.0,
        lag1_top_focus_loss_weight=1.0,
        lag1_top_focus_temperature=0.75,
        pairwise_top_loss_weight=0.0,
    )

    _, components = total_loss_v7(
        alpha,
        alphas,
        horizon_preds,
        y,
        y_seq,
        mask,
        cfg,
        lag1_y_seq=lag1_y_seq,
        lag1_enabled=False,
        lag1_top_focus_enabled=True,
    )

    assert components["lag1"].item() == 0.0
    assert components["lag1_top_focus"].abs().item() > 0.0
