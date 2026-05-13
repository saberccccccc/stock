# backtest.py - 多周期市场中性组合回测与优化
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import lightgbm as lgb
from scipy.stats import spearmanr
import warnings
import pickle
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

from core.config import DataConfig
from core.model import UltimateV7Model
from core.train_utils import get_regime_dim
from data.pipeline import build_cross_section_dataset, N_AGGS, INDUSTRY_REL_FEATURES

# ==================== 多周期模型训练与加载 ====================
def _stack_horizon_samples(samples, h):
    X_list, y_list = [], []
    for s in samples:
        if s['y_seq'].shape[1] < h:
            continue
        y_vals = s['y_seq'][:, h - 1]
        if np.all(np.isnan(y_vals)):
            continue
        X_list.append(s['X'])
        y_list.append(y_vals)
    if not X_list:
        return None, None
    return np.vstack(X_list), np.hstack(y_list)


def train_multi_horizon_models(train_samples, val_samples, horizon_list=(1, 3, 5, 10),
                               model_dir="models_multi_v9_tech_macro"):
    os.makedirs(model_dir, exist_ok=True)
    models = {}
    ic_by_horizon = []

    for h in horizon_list:
        print(f"\n训练 Horizon={h} 模型...")
        X_train, y_train = _stack_horizon_samples(train_samples, h)
        if X_train is None:
            print(f"Horizon {h}: 无可用训练样本")
            ic_by_horizon.append(0.0)
            continue

        print(f"训练数据 shape: {X_train.shape}")

        dtrain = lgb.Dataset(X_train, label=y_train)
        X_val, y_val = _stack_horizon_samples(val_samples, h)
        params = {
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
            'num_leaves': 63, 'learning_rate': 0.03, 'feature_fraction': 0.7,
            'bagging_fraction': 0.7, 'bagging_freq': 5, 'lambda_l1': 0.2,
            'lambda_l2': 0.2, 'min_data_in_leaf': 30, 'max_depth': -1,
            'verbose': -1, 'num_threads': 8,
        }
        if X_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            model = lgb.train(
                params, dtrain, num_boost_round=300,
                valid_sets=[dtrain, dval], valid_names=['train', 'valid'],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
            )
        else:
            model = lgb.train(params, dtrain, num_boost_round=300)
        model.save_model(f"{model_dir}/lgb_h{h}.txt")
        models[h] = model

        # 验证 IC
        ic_list = []
        for s in val_samples:
            if s['y_seq'].shape[1] < h:
                continue
            y_true = s['y_seq'][:, h - 1]
            if np.all(np.isnan(y_true)):
                continue
            pred = model.predict(s['X'])
            ic, _ = spearmanr(pred, y_true)
            if np.isfinite(ic):
                ic_list.append(ic)
        mean_ic = np.mean(ic_list) if ic_list else 0.0
        ic_by_horizon.append(mean_ic)
        print(f"Horizon {h} 验证 IC = {mean_ic:.4f}")

    np.save(f"{model_dir}/ic_decay.npy", ic_by_horizon)
    return models, ic_by_horizon


def load_multi_horizon_models(horizon_list=(1, 3, 5, 10), model_dir="models_multi_v9_tech_macro"):
    models = {}
    for h in horizon_list:
        path = f"{model_dir}/lgb_h{h}.txt"
        if os.path.exists(path):
            models[h] = lgb.Booster(model_file=path)
        else:
            raise FileNotFoundError(f"model {path} not found")
    ic_decay = (np.load(f"{model_dir}/ic_decay.npy")
                if os.path.exists(f"{model_dir}/ic_decay.npy") else None)
    return models, ic_decay


def models_match_feature_dim(models, expected_dim):
    return all(model.num_feature() == expected_dim for model in models.values())


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用CUDA，但当前PyTorch不可用CUDA")
    return torch.device(device_arg)


def load_v9_checkpoint(checkpoint_path, train_samples, cfg, device_arg="auto"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"V9 checkpoint不存在 {checkpoint_path}")

    device = resolve_device(device_arg)
    input_dim = train_samples[0]["X"].shape[1]
    industry_rel_dim = len(INDUSTRY_REL_FEATURES)
    total_agg = (input_dim - industry_rel_dim) // 2
    base_feat_dim = total_agg // N_AGGS
    regime_dim = get_regime_dim(cfg)
    num_industries = train_samples[0]["risk"].shape[1] - regime_dim
    if num_industries <= 0:
        raise ValueError(
            f"行业维度异常: risk_dim={train_samples[0]['risk'].shape[1]}, regime_dim={regime_dim}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # 从state_dict自动检测是否有GAT层，不依赖cfg.use_gat
    has_gat = any(
        k.startswith("gat_conv") or k.startswith("fusion_gate")
        for k in state_dict.keys()
    )

    model = UltimateV7Model(
        input_dim, base_feat_dim, n_aggs=N_AGGS,
        hidden_dim=256, n_heads=8, n_layers=4,
        n_horizons=len(cfg.horizon_indices), n_alpha=4,
        use_gat=has_gat,
        regime_dim=regime_dim, num_industries=num_industries,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    print(f"加载V9深度模型: {checkpoint_path} {'[GAT]' if has_gat else '[Transformer]'}")
    print(
        f"DL维度: input_dim={input_dim}, base_feat_dim={base_feat_dim}, "
        f"regime_dim={regime_dim}, num_industries={num_industries}, device={device}"
    )
    if isinstance(checkpoint, dict):
        print(f"checkpoint epoch={checkpoint.get('epoch')}, val_ics={checkpoint.get('val_ics')}")
    return model, device, regime_dim


class DLPredictor:
    name = "dl"

    def __init__(self, model, device, regime_dim):
        self.model = model
        self.device = device
        self.regime_dim = regime_dim

    def predict_alpha(self, sample, valid, regime):
        X_np = sample["X"][valid].astype(np.float32)
        risk_np = sample["risk"][valid].astype(np.float32)
        industry_np = sample["industry_ids"][valid].astype(np.int64)
        if X_np.shape[0] == 0:
            return np.array([], dtype=np.float32)

        X = torch.from_numpy(X_np).unsqueeze(0).to(self.device)
        risk = torch.from_numpy(risk_np).unsqueeze(0).to(self.device)
        industry_ids = torch.from_numpy(industry_np).unsqueeze(0).to(self.device)
        mask = torch.ones(1, X_np.shape[0], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            alpha_raw, _, _ = self.model(
                X, risk[..., :self.regime_dim], mask, industry_ids
            )
        pred = alpha_raw[0].detach().cpu().numpy()
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        if pred.size > 1:
            pred = (pred - np.mean(pred)) / (np.std(pred) + 1e-8)
        return np.tanh(pred)


class LGBPredictor:
    name = "lgb"

    def __init__(self, models, ic_decay=None):
        self.models = models
        self.ic_decay = ic_decay

    def predict_alpha(self, sample, valid, regime):
        return fused_alpha(self.models, sample["X"][valid], regime, self.ic_decay)

# ==================== 市场状态识别====================
def detect_regime(sample):
    risk_mean = np.mean(sample['risk'], axis=0)
    vol_z = risk_mean[1]
    mom_z = risk_mean[2]
    if mom_z > 0.2 and vol_z < 0.5:
        return "trend_up"
    elif mom_z < -0.2 and vol_z > 0.5:
        return "panic"
    return "sideways"

# ==================== 多Alpha融合 ====================
def fused_alpha(models, X, regime, ic_decay=None):
    alpha_dict = {}
    for h, model in models.items():
        pred = model.predict(X)
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        pred = (pred - np.mean(pred)) / (np.std(pred) + 1e-8)
        alpha_dict[h] = np.tanh(pred)

    regime_weights = {
        "trend_up": {10: 0.5, 5: 0.3, 3: 0.2},
        "panic": {1: 0.6, 3: 0.3, 5: 0.1},
        "sideways": {5: 0.4, 3: 0.3, 1: 0.3},
    }
    weights = dict(regime_weights.get(regime, regime_weights["sideways"]))

    if ic_decay is not None:
        horizons = list(models.keys())
        adjusted = {}
        for h, weight in weights.items():
            if h not in horizons:
                continue
            idx = horizons.index(h)
            ic_weight = max(float(ic_decay[idx]), 0.0) if idx < len(ic_decay) else 0.0
            adjusted[h] = weight * ic_weight
        if sum(adjusted.values()) > 1e-12:
            weights = adjusted

    weight_sum = sum(weights.values()) + 1e-8
    alpha = np.zeros(X.shape[0], dtype=np.float32)
    for h, weight in weights.items():
        if h in alpha_dict:
            alpha += (weight / weight_sum) * alpha_dict[h]
    alpha = (alpha - np.mean(alpha)) / (np.std(alpha) + 1e-8)
    return np.tanh(alpha)

# ==================== 风险模型 ====================
def compute_risk_model(B, R, lam=1e-3, decay=0.94):
    T = R.shape[0]
    weights = np.array([decay ** (T - 1 - i) for i in range(T)])
    weights /= weights.sum()
    BtB = B.T @ B + lam * np.eye(B.shape[1])
    factor_returns = np.linalg.solve(BtB, B.T @ R.T)
    F_cov = np.zeros((B.shape[1], B.shape[1]))
    for t in range(T):
        f = factor_returns[:, t].reshape(-1, 1)
        F_cov += weights[t] * (f @ f.T)
    diag = np.diag(np.diag(F_cov))
    F_cov = 0.2 * diag + 0.8 * F_cov
    f_scale = np.mean(np.diag(F_cov))
    if not np.isfinite(f_scale) or f_scale < 1e-8:
        f_scale = 1.0
    F_cov = np.nan_to_num(F_cov / f_scale, nan=0.0, posinf=0.0, neginf=0.0)
    fitted = (B @ factor_returns).T
    residuals = R - fitted
    D_diag = np.var(residuals, axis=0) + 1e-8
    d_scale = np.mean(D_diag)
    if not np.isfinite(d_scale) or d_scale < 1e-8:
        d_scale = 1.0
    D_diag = np.nan_to_num(D_diag / d_scale, nan=1.0, posinf=1.0, neginf=1.0)
    return F_cov, D_diag


def ewma_beta(rets, idx_ret, half_life=20):
    N, T = rets.shape
    decay = np.exp(-np.log(2) / half_life)
    weights = np.array([decay ** (T - 1 - i) for i in range(T)])
    weights /= weights.sum()
    wmean_rets = np.sum(rets * weights, axis=1, keepdims=True)
    wmean_idx = np.sum(idx_ret * weights)
    rets_dm = rets - wmean_rets
    idx_dm = idx_ret - wmean_idx
    cov = np.sum((rets_dm * idx_dm) * weights, axis=1)
    var_idx = np.sum(idx_dm ** 2 * weights)
    beta = cov / (var_idx + 1e-8)
    return np.clip(beta, -3, 3)

def build_simple_weights(alpha, mode="simple_ls", top_frac=0.10, max_leverage=1.0, regime="sideways", market_mult=1.0):
    N = len(alpha)
    w = np.zeros(N, dtype=np.float64)
    if N == 0:
        return w
    order = np.argsort(alpha)
    k = max(1, int(N * top_frac))
    top_idx = order[-k:]

    if mode == "simple_long":
        regime_mult = {"panic": 0.5, "sideways": 1.0, "trend_up": 1.0}
        lev = max_leverage * regime_mult.get(regime, 1.0) * market_mult
        alpha_top = np.maximum(alpha[top_idx], 0)
        alpha_sum = np.sum(alpha_top) + 1e-12
        w[top_idx] = alpha_top / alpha_sum * lev
    elif mode == "simple_ls":
        bottom_idx = order[:k]
        alpha_top = np.maximum(alpha[top_idx], 0)
        w[top_idx] = alpha_top / (np.sum(alpha_top) + 1e-12) * (max_leverage / 2)
        alpha_bot = np.maximum(-alpha[bottom_idx], 0)
        w[bottom_idx] = -alpha_bot / (np.sum(alpha_bot) + 1e-12) * (max_leverage / 2)
    else:
        raise ValueError(f"未知简单组合模式 {mode}")

    return w

# ==================== 风险预算与优化====================
def risk_budget_allocation(alpha, D_diag, target_vol=0.15):
    """基于alpha强度和个股残差风险分配风险预算

    Args:
        alpha: alpha信号强度
        D_diag: 个股残差风险（来自风险模型）
        target_vol: 目标波动率

    Returns:
        风险预算分配
    """
    # 按alpha强度分配
    strength = np.abs(alpha)
    strength /= (np.sum(strength) + 1e-8)

    # 残差风险加权：残差方差大的股票分配更少风险预算
    inv_risk = 1.0 / (D_diag + 1e-8)
    inv_risk /= (np.sum(inv_risk) + 1e-8)

    # 混合：50%按alpha强度，50%按风险倒数
    blended = 0.5 * strength + 0.5 * inv_risk
    blended /= (np.sum(blended) + 1e-8)

    target_var = (target_vol / np.sqrt(252)) ** 2
    return blended * target_var


def optimize_with_risk_budget(alpha, B, beta, prev_w, F_cov, D_diag, risk_budget,
                              lambda_t, lambda_b, max_weight, max_leverage,
                              lr0=0.02, n_iter=200, return_diagnostics=False):
    N = len(alpha)
    w = prev_w.copy() if prev_w is not None else np.zeros(N)
    max_w_arr = (np.full(N, max_weight) if np.isscalar(max_weight)
                 else max_weight)

    # 检测是否为初始建仓
    is_initial = prev_w is None or np.sum(np.abs(prev_w)) < 1e-8
    lambda_init = 0.01 if is_initial else 0.0  # 初始建仓时添加轻微的仓位规模惩罚

    converged = False
    final_iter = n_iter
    for i in range(n_iter):
        sigma_w = B @ (F_cov @ (B.T @ w)) + D_diag * w
        risk_contrib = w * sigma_w
        grad_risk = (risk_contrib - risk_budget) / (np.sum(risk_budget) + 1e-8)
        grad = grad_risk - alpha

        if lambda_t > 0 and prev_w is not None:
            grad += 2 * lambda_t * (w - prev_w)
        if lambda_b > 0:
            beta_exp = np.dot(w, beta)
            grad += 2 * lambda_b * beta_exp * beta
        if lambda_init > 0:
            # 初始建仓惩罚：惩罚总仓位规模，鼓励平滑建仓
            grad += lambda_init * np.sign(w)

        lr = lr0 / np.sqrt(i + 1)
        w_new = w - lr * grad
        w_new = np.clip(w_new, -max_w_arr, max_w_arr)

        lev = np.sum(np.abs(w_new))
        if lev > max_leverage:
            w_new = w_new / lev * max_leverage
        w_new = w_new - np.mean(w_new)

        # 使用相对收敛判据，对大规模组合更稳健
        rel_change = np.linalg.norm(w_new - w) / (np.linalg.norm(w) + 1e-8)
        if rel_change < 1e-4:
            w = w_new
            converged = True
            final_iter = i + 1
            break
        w = w_new

    w[np.abs(w) < 1e-6] = 0.0
    if not np.all(np.isfinite(w)):
        if return_diagnostics:
            return np.zeros(N), {"converged": False, "iterations": 0, "risk_budget_match": 0.0}
        return np.zeros(N)

    if return_diagnostics:
        sigma_w = B @ (F_cov @ (B.T @ w)) + D_diag * w
        risk_contrib = w * sigma_w
        total_risk_budget = np.sum(risk_budget)
        total_risk_contrib = np.sum(risk_contrib)
        risk_budget_match = 1.0 - np.abs(total_risk_contrib - total_risk_budget) / (total_risk_budget + 1e-8)

        diagnostics = {
            "converged": converged,
            "iterations": final_iter,
            "risk_budget_match": float(risk_budget_match),
            "beta_exposure": float(np.dot(w, beta)) if beta is not None else 0.0,
        }
        return w, diagnostics

    return w

# ==================== 撮合引擎 ====================
def execute_order_with_impact(w_target, prev_w, price, volume,
                              adv_ratio=0.02, impact_coeff=0.1,
                              portfolio_value=1e8, regime="sideways"):
    """执行订单并计算冲击成本

    Args:
        regime: 市场状态("panic", "sideways", "trend_up")，用于调整冲击成本
    """
    price = np.asarray(price, dtype=float)
    volume = np.asarray(volume, dtype=float)
    w_target = np.nan_to_num(w_target, nan=0.0, posinf=0.0, neginf=0.0)
    prev_w = np.nan_to_num(prev_w, nan=0.0, posinf=0.0, neginf=0.0)
    trade = w_target - prev_w

    valid_liquidity = np.isfinite(price) & np.isfinite(volume) & (price > 0) & (volume > 0)
    # volume单位是手，乘以100股
    dollar_vol = np.where(valid_liquidity, price * volume * 100, 0.0)
    max_trade_weight = np.where(
        dollar_vol > 0,
        dollar_vol * adv_ratio / portfolio_value,
        0.0,
    )
    trade_weight_abs = np.abs(trade)
    fill_ratio = np.minimum(1.0, max_trade_weight / (trade_weight_abs + 1e-8))
    fill_ratio = np.where(valid_liquidity, fill_ratio, 0.0)
    trade_exec = trade * fill_ratio
    turnover_ratio = np.where(
        dollar_vol > 0,
        np.abs(trade_exec) * portfolio_value / (dollar_vol + 1e-8),
        0.0,
    )
    # Cap turnover_ratio to prevent extreme impact cost from illiquid days
    turnover_ratio = np.clip(turnover_ratio, 0.0, 1.0)

    # 市场状态调节系数：恐慌时流动性枯竭，冲击成本更高
    regime_multiplier = {
        "panic": 1.5,      # 恐慌市场：流动性枯竭
        "sideways": 1.0,   # 正常市场：基准
        "trend_up": 1.0    # 趋势市场：基准
    }
    impact_coeff_adj = impact_coeff * regime_multiplier.get(regime, 1.0)

    impact_cost = impact_coeff_adj * (turnover_ratio ** 2) * np.abs(trade_exec)
    impact_cost = np.nan_to_num(impact_cost, nan=0.0, posinf=0.0, neginf=0.0)
    w_filled = np.nan_to_num(prev_w + trade_exec, nan=0.0, posinf=0.0, neginf=0.0)
    return w_filled, float(np.sum(impact_cost)), fill_ratio, trade_exec

# ==================== 价格与成交量数据加载 ====================
def load_price_volume(config):
    data_dir = config.data_dir
    excluded = {'all_data_jq.csv', 'stable_stocks.csv', 'stable_stocks_industry.csv'}
    files = [f for f in os.listdir(data_dir)
             if f.endswith('.csv') and f not in excluded and f[0].isdigit()]
    if config.max_stocks:
        files = files[:config.max_stocks]

    price_dict, vol_dict = {}, {}
    for fname in tqdm(files, desc="load_price_vol"):
        code = fname.replace('.csv', '')
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, usecols=['trade_date', 'close', 'volume'])
            df.columns = df.columns.str.strip().str.lower()
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            price_dict[code] = df['close']
            vol_dict[code] = df['volume']
        except Exception:
            continue
    return price_dict, vol_dict


def build_universe_matrix(price_dict, vol_dict, all_codes):
    all_dates = sorted(set().union(*[price_dict[c].index
                                      for c in all_codes if c in price_dict]))
    all_dates = pd.DatetimeIndex(all_dates)
    N, T = len(all_codes), len(all_dates)
    price_mat = np.full((N, T), np.nan)
    vol_mat = np.full((N, T), np.nan) if vol_dict else None
    code2idx = {c: i for i, c in enumerate(all_codes)}
    for code, ser in price_dict.items():
        i = code2idx.get(code)
        if i is None:
            continue
        price_mat[i] = ser.reindex(all_dates).values
        if vol_dict and code in vol_dict:
            vol_mat[i] = vol_dict[code].reindex(all_dates).values
    return price_mat, vol_mat, all_dates, code2idx

# ==================== 回测主函数====================
def run_backtest_production(predictor, val_samples, price_dict, vol_dict,
                            future_len=2, rebalance_freq=None, hist_window=60,
                            ewma_hl=20, adv_limit_ratio=0.02, adv_mode="execution",
                            portfolio_mode="optimizer", top_frac=0.10,
                            max_weight=0.05, lambda_t=0.05, lambda_b=0.2,
                            target_vol=0.15, impact_coeff=0.1, config=None,
                            portfolio_value=1e8):
    # 自动确定调仓频率
    ic_decay = getattr(predictor, "ic_decay", None)
    if ic_decay is not None:
        ic_decay = np.asarray(ic_decay, dtype=float)
    if rebalance_freq is None and ic_decay is not None and len(ic_decay) > 0 and abs(ic_decay[0]) > 1e-8:
        ic_norm = ic_decay / (ic_decay[0] + 1e-8)
        for i in range(len(ic_norm)):
            # 方法1：绝对阈值 - IC衰减到50%以下
            if ic_norm[i] < 0.5:
                rebalance_freq = i + 1
                break
            # 方法2：趋势检测 - IC显著下降（下降超过30%）
            if i > 0 and ic_norm[i] < ic_norm[i-1] * 0.7:
                rebalance_freq = i + 1
                break
        else:
            # 都未触发：使用默认中等周期
            rebalance_freq = max(3, min(5, len(ic_decay)))
    if rebalance_freq is None:
        rebalance_freq = 5
    print(f"预测器 {predictor.name}")
    print(f"{rebalance_freq}")
    print(f"组合模式: {portfolio_mode} | top_frac={top_frac:.2%}")
    print(f"ADVģʽ: {adv_mode}")
    print("return metric: next_close_to_next_close")

    all_codes = sorted(set(c for s in val_samples for c in s['codes']))
    price_mat, vol_mat, all_dates, code2idx = build_universe_matrix(
        price_dict, vol_dict, all_codes)
    T_total = len(all_dates)

    # 加载指数数据
    config = config or DataConfig()
    idx_path = os.path.join(config.data_dir, "hs300_index.csv")
    if os.path.exists(idx_path):
        idx_df = pd.read_csv(idx_path)
        date_col = 'trade_date' if 'trade_date' in idx_df.columns else 'date'
        idx_df[date_col] = pd.to_datetime(idx_df[date_col])
        idx_df.set_index(date_col, inplace=True)
        idx_close = idx_df['close'].reindex(all_dates)
        idx_daily = idx_close.pct_change().fillna(0)
    else:
        idx_close = pd.Series(np.nan, index=all_dates)
        idx_daily = pd.Series(0.0, index=all_dates)

    # 日期映射
    date2idx = {}
    for s in val_samples:
        dt = s['date']
        pos = all_dates.searchsorted(dt, side='right') - 1
        date2idx[dt] = max(pos, 0)

    # 日收益率矩阵
    ret_daily = np.full((len(all_codes), T_total - 1), np.nan)
    for i in range(len(all_codes)):
        p = price_mat[i]
        ret_daily[i] = p[1:] / p[:-1] - 1
    ret_daily[~np.isfinite(ret_daily)] = 0.0

    daily_total_weights = [np.zeros(len(all_codes)) for _ in range(T_total)]
    daily_costs = np.zeros(T_total)
    last_signal_idx = -1
    total_cost = 0.0
    diag = defaultdict(list)
    diag_counts = defaultdict(int)

    for t_idx, sample in enumerate(tqdm(val_samples, desc=f"回测-{predictor.name}")):
        dt = sample['date']
        col_cur = date2idx[dt]
        if col_cur < hist_window + future_len:
            continue
        if t_idx - last_signal_idx >= rebalance_freq:
            diag_counts['rebalance_attempts'] += 1
            codes = sample['codes']
            idx_full = [code2idx[c] for c in codes]
            N = len(codes)

            hist_start = col_cur - hist_window
            hist_end = col_cur - 1
            if hist_start < 0:
                continue

            price_hist = price_mat[idx_full, hist_start:hist_end + 1]
            valid = np.sum(~np.isnan(price_hist), axis=1) >= 0.7 * hist_window
            if not np.any(valid):
                continue

            price_hist = price_hist[valid]
            codes = [codes[i] for i in range(N) if valid[i]]
            idx_full = [idx_full[i] for i in range(N) if valid[i]]
            N = len(codes)

            with np.errstate(divide='ignore', invalid='ignore'):
                rets = np.log(price_hist[:, 1:] / price_hist[:, :-1])
                rets[~np.isfinite(rets)] = 0
            R_mat = rets.T

            regime = detect_regime(sample)
            alpha = predictor.predict_alpha(sample, valid, regime)
            alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            if alpha.shape[0] != N:
                raise ValueError(f"预测长度不匹配 alpha={alpha.shape[0]}, N={N}")
            diag['valid_names'].append(N)
            diag['alpha_std'].append(float(np.std(alpha)))

            # 市场择时：沪深300指数低于60日均线时降低整体敞口
            market_mult = 1.0
            if portfolio_mode == "simple_long":
                if col_cur >= 60 and np.isfinite(idx_close.iloc[col_cur]):
                    idx_ma60 = idx_close.iloc[col_cur - 60:col_cur].mean()
                    idx_cur = idx_close.iloc[col_cur]
                    if idx_cur < idx_ma60:
                        market_mult = 0.7
                    # 极度弱势：连续6个月下跌，降至0.3
                    if col_cur >= 120:
                        idx_ret_6m = idx_close.iloc[col_cur] / idx_close.iloc[col_cur - 120] - 1
                        if idx_ret_6m < -0.10:
                            market_mult = min(market_mult, 0.3)

            prev_w = daily_total_weights[col_cur][idx_full]

            if portfolio_mode == "optimizer":
                idx_ret_hist = idx_daily.iloc[hist_start + 1:hist_end + 1].values
                beta_vec = ewma_beta(rets, idx_ret_hist, ewma_hl)

                regime_dim = get_regime_dim(config)
                B_style = np.hstack([
                    sample['risk'][valid, :3],
                    sample['risk'][valid, regime_dim:]
                ])
                F_cov, D_diag = compute_risk_model(B_style, R_mat)

                max_w_arr = max_weight
                # ⚠️ DEPRECATED: weight_cap模式已弃用，存在设计缺陷
                # 问题：优化器用历史ADV约束权重，但执行时100%成交(exec_adv_ratio=1e9)
                # 当日流动性差时冲击成本爆炸，导致回测表现极差
                # 推荐使用execution模式（默认）
                if adv_mode in ("weight_cap", "both") and vol_mat is not None and adv_limit_ratio > 0:
                    vol_hist = vol_mat[idx_full, hist_start:hist_end + 1]
                    lookback = min(20, vol_hist.shape[1])
                    adv = np.nanmean(vol_hist[:, -lookback:], axis=1)
                    adv = np.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
                    dollar_vol = np.where(
                        (adv > 0) & np.isfinite(price_hist[:, -1]) & (price_hist[:, -1] > 0),
                        adv * price_hist[:, -1] * 100,  # volume是手数，需要乘以100
                        0.0,
                    )
                    max_w_adv = dollar_vol * adv_limit_ratio / portfolio_value
                    max_w_arr = np.minimum(max_weight, max_w_adv)
                    max_w_arr = np.where(dollar_vol > 0, max_w_arr, max_weight)
                    diag['avg_adv_weight_cap'].append(float(np.mean(max_w_arr)))

                risk_budget = risk_budget_allocation(alpha, D_diag, target_vol)
                w_target, opt_diag = optimize_with_risk_budget(
                    alpha, B_style, beta_vec, prev_w,
                    F_cov, D_diag, risk_budget,
                    lambda_t, lambda_b,
                    max_weight=max_w_arr, max_leverage=1.0,
                    return_diagnostics=True,
                )
                diag['opt_converged'].append(opt_diag['converged'])
                diag['opt_iterations'].append(opt_diag['iterations'])
                diag['risk_budget_match'].append(opt_diag['risk_budget_match'])
                diag['beta_exposure'].append(opt_diag['beta_exposure'])
            else:
                w_target = build_simple_weights(alpha, portfolio_mode, top_frac,
                                                 max_leverage=1.0, regime=regime,
                                                 market_mult=market_mult)

            if not np.all(np.isfinite(w_target)):
                continue
            diag['target_leverage'].append(float(np.sum(np.abs(w_target))))

            entry_day = col_cur + 1
            exit_day = entry_day + future_len
            # 边界检查：确保exit_day不超出范围
            if exit_day >= T_total:
                exit_day = T_total - 1
            if entry_day >= T_total:
                continue

            price_next = price_mat[idx_full, entry_day]
            vol_next = (vol_mat[idx_full, entry_day]
                        if vol_mat is not None else np.ones(N) * 1e9)
            tradable = np.isfinite(price_next) & (vol_next > 0)
            w_target_tradable = np.where(tradable, w_target, prev_w)

            exec_adv_ratio = adv_limit_ratio if adv_mode in ("execution", "both") else 1e9
            w_filled, impact_cost, fill_ratio, trade_exec = execute_order_with_impact(
                w_target_tradable, prev_w, price_next, vol_next,
                adv_ratio=exec_adv_ratio, impact_coeff=impact_coeff,
                portfolio_value=portfolio_value,
                regime=regime,  # 传入市场状态用于调整冲击成本
            )
            total_cost += impact_cost
            daily_costs[entry_day] += impact_cost
            diag_counts['successful_rebalances'] += 1
            diag['filled_leverage'].append(float(np.sum(np.abs(w_filled))))
            diag['turnover'].append(float(np.sum(np.abs(trade_exec))))
            active_trade = np.abs(w_target_tradable - prev_w) > 1e-8
            if np.any(active_trade):
                diag['fill_ratio'].append(float(np.mean(fill_ratio[active_trade])))
            diag['untradable_ratio'].append(float(1.0 - np.mean(tradable)))

            hold_start = entry_day + 1
            hold_end = min(exit_day, T_total - 1)
            diag['holding_days_written'].append(max(0, hold_end - hold_start + 1))
            for d in range(hold_start, hold_end + 1):
                day_weights = daily_total_weights[d].copy()
                day_weights[idx_full] = w_filled
                daily_total_weights[d] = day_weights

            last_signal_idx = t_idx

    # 计算每日收益
    daily_port_ret = []
    for day in range(1, T_total):
        w_day = daily_total_weights[day].copy()
        stock_ret = ret_daily[:, day - 1]
        valid = np.isfinite(stock_ret)
        w_day[~valid] = 0.0
        stock_ret = np.nan_to_num(stock_ret, nan=0.0)
        lev = np.sum(np.abs(w_day))
        if lev > 1.0:
            w_day = w_day / lev
        port_ret_day = np.dot(w_day, stock_ret) - daily_costs[day]
        daily_port_ret.append(port_ret_day)

    port_ret = np.nan_to_num(np.array(daily_port_ret), nan=0.0)
    idx_ret_arr = np.nan_to_num(idx_daily.iloc[1:len(daily_port_ret) + 1].values, nan=0.0)

    # 滚动Beta中性化
    neu_ret = []
    beta_estimates = []
    for i in range(len(port_ret)):
        start = max(0, i - 60)
        if i - start < 20:
            beta = 0.0
        else:
            cov_ = np.cov(port_ret[start:i], idx_ret_arr[start:i])[0, 1]
            var_ = np.var(idx_ret_arr[start:i])
            beta = cov_ / (var_ + 1e-8) if var_ > 1e-8 else 0.0
            beta_estimates.append(beta)
        neu_ret.append(port_ret[i] - beta * idx_ret_arr[i])

    leverage_values = [np.sum(np.abs(w)) for w in daily_total_weights if np.sum(np.abs(w)) > 0]
    avg_leverage = np.mean(leverage_values) if leverage_values else 0.0
    nonzero_days = len(leverage_values)

    def diag_mean(key):
        return float(np.mean(diag[key])) if diag[key] else 0.0

    print("\n========== 回测诊断 ==========")
    print(f"预测器 {predictor.name} | 组合模式: {portfolio_mode}")
    print(f"rebalance: attempts {diag_counts['rebalance_attempts']} | success {diag_counts['successful_rebalances']}")
    print(f"平均有效股票数 {diag_mean('valid_names'):.1f}")
    print(f"Alpha标准差 {diag_mean('alpha_std'):.4f}")
    print(f"目标杠杆: {diag_mean('target_leverage'):.3f} | 成交后杠杆 {diag_mean('filled_leverage'):.3f}")
    print(f"平均换手/调仓: {diag_mean('turnover'):.3f} | 平均填充率 {diag_mean('fill_ratio'):.3f}")
    print(f"不可交易比例: {diag_mean('untradable_ratio'):.3%}")
    if diag['avg_adv_weight_cap']:
        print(f"平均ADV权重上限: {diag_mean('avg_adv_weight_cap'):.5f}")
    if portfolio_mode == "optimizer" and diag.get('opt_converged'):
        converged_rate = np.mean(diag['opt_converged']) if diag['opt_converged'] else 0.0
        print(f"\n优化器诊断")
        print(f"  收敛率 {converged_rate:.1%}")
        print(f"  平均迭代次数: {diag_mean('opt_iterations'):.1f}")
        print(f"  风险预算匹配度 {diag_mean('risk_budget_match'):.3f}")
        print(f"  平均Beta暴露: {diag_mean('beta_exposure'):.4f}")
    print(f"非零持仓天数: {nonzero_days} / {T_total}")
    print(f"平均杠杆: {avg_leverage:.3f}")
    print(f"平均估计Beta: {(np.mean(beta_estimates) if beta_estimates else 0.0):.3f}")
    print(f"总冲击成本 {total_cost:.4f}")
    active = np.array([np.sum(np.abs(w)) > 0 for w in daily_total_weights[1:]], dtype=bool)
    if active.any():
        first_active = int(np.argmax(active))
        last_active = len(active) - int(np.argmax(active[::-1]))
        port_ret = port_ret[first_active:last_active]
        neu_ret = np.array(neu_ret)[first_active:last_active]
    else:
        neu_ret = np.array(neu_ret)

    # 返回详细数据用于保存
    backtest_data = {
        'daily_returns': port_ret,
        'neutral_returns': neu_ret,
        'daily_weights': daily_total_weights,
        'daily_costs': daily_costs,
        'dates': all_dates,
        'codes': all_codes,
        'diagnostics': diag,
        'diagnostic_counts': dict(diag_counts),
        'beta_estimates': beta_estimates,
        'leverage_values': leverage_values,
    }
    return port_ret, neu_ret, backtest_data


def calc_metrics(returns):
    if len(returns) == 0:
        return 0, 0, 0
    returns = np.nan_to_num(returns, nan=0.0)
    cum = np.cumprod(1 + returns)
    days = len(returns)
    years = days / 252
    ann = (cum[-1] ** (1 / years) - 1) * 100 if years > 0 and cum[-1] > 0 else np.nan
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
    peak = np.maximum.accumulate(cum)
    mdd = ((peak - cum) / (peak + 1e-12)).max()
    return ann, sharpe, mdd


def calc_extended_metrics(returns):
    """计算扩展的风险指标"""
    if len(returns) == 0:
        return {}

    returns = np.nan_to_num(returns, nan=0.0)
    cum = np.cumprod(1 + returns)
    days = len(returns)
    years = days / 252

    # 基础指标
    ann_ret = (cum[-1] ** (1 / years) - 1) * 100 if years > 0 and cum[-1] > 0 else 0.0
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
    peak = np.maximum.accumulate(cum)
    mdd = ((peak - cum) / (peak + 1e-12)).max()

    # Calmar比率
    calmar = ann_ret / (mdd * 100 + 1e-8) if mdd > 0 else 0.0

    # Sortino比率（下行波动率）
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
    sortino = returns.mean() / (downside_std + 1e-8) * np.sqrt(252)

    # 胜率
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0.0

    # 盈亏比
    avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.0
    avg_loss = np.mean(np.abs(returns[returns < 0])) if np.any(returns < 0) else 1e-8
    profit_loss_ratio = avg_win / (avg_loss + 1e-8)

    return {
        'ann_return': ann_ret,
        'sharpe': sharpe,
        'max_drawdown': mdd * 100,
        'calmar': calmar,
        'sortino': sortino,
        'win_rate': win_rate * 100,
        'profit_loss_ratio': profit_loss_ratio,
        'total_days': days,
    }


def analyze_by_period(returns, dates, period='year'):
    """鎸夋椂闂存?分析回测表现"""
    if len(returns) == 0 or len(dates) == 0:
        return {}

    returns = np.array(returns)
    dates = pd.DatetimeIndex(dates[:len(returns)])

    results = {}
    if period == 'year':
        for year in dates.year.unique():
            mask = dates.year == year
            year_returns = returns[mask]
            if len(year_returns) > 0:
                results[str(year)] = calc_extended_metrics(year_returns)

    return results


def compare_predictors(all_results, output_dir="backtest_results"):
    """对比多个预测器的表现"""
    if len(all_results) < 2:
        return

    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*60)
    print("Predictor comparison")
    print("="*60)

    # Comparison table
    comparison = []
    for name, (data, metrics) in all_results.items():
        ext_raw = calc_extended_metrics(data['daily_returns'])
        ext_neu = calc_extended_metrics(data['neutral_returns'])
        comparison.append({
            'predictor': name,
            'ann_return_raw': f"{ext_raw.get('ann_return', 0):.2f}%",
            'sharpe_raw': f"{ext_raw.get('sharpe', 0):.2f}",
            'mdd_raw': f"{ext_raw.get('max_drawdown', 0):.2f}%",
            'calmar_raw': f"{ext_raw.get('calmar', 0):.2f}",
            'ann_return_neu': f"{ext_neu.get('ann_return', 0):.2f}%",
            'sharpe_neu': f"{ext_neu.get('sharpe', 0):.2f}",
            'mdd_neu': f"{ext_neu.get('max_drawdown', 0):.2f}%",
        })

    df_comp = pd.DataFrame(comparison)
    print("\n" + df_comp.to_string(index=False))

    # 保存对比结果
    df_comp.to_csv(f"{output_dir}/comparison_{timestamp}.csv", index=False)
    print(f"\n对比结果已保存到: {output_dir}/comparison_{timestamp}.csv")


def save_backtest_results(backtest_data, metrics, predictor_name, portfolio_mode, output_dir="backtest_results"):
    """保存回测结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{output_dir}/{predictor_name}_{portfolio_mode}_{timestamp}"

    # 保存日收益率
    returns_df = pd.DataFrame({
        'date': backtest_data['dates'][1:len(backtest_data['daily_returns'])+1],
        'raw_return': backtest_data['daily_returns'],
        'neutral_return': backtest_data['neutral_returns'],
        'cost': backtest_data['daily_costs'][1:len(backtest_data['daily_returns'])+1],
    })
    returns_df.to_csv(f"{prefix}_returns.csv", index=False)

    # 保存诊断信息（处理长度不一致的列表）
    diag_dict = backtest_data['diagnostics']
    if diag_dict:
        max_len = max(len(v) for v in diag_dict.values() if isinstance(v, list))
        diag_dict_padded = {}
        for k, v in diag_dict.items():
            if isinstance(v, list):
                diag_dict_padded[k] = v + [np.nan] * (max_len - len(v))
            else:
                diag_dict_padded[k] = v
        diag_df = pd.DataFrame(diag_dict_padded)
        diag_df.to_csv(f"{prefix}_diagnostics.csv", index=False)

    # 保存扩展指标
    ext_metrics_raw = calc_extended_metrics(backtest_data['daily_returns'])
    ext_metrics_neu = calc_extended_metrics(backtest_data['neutral_returns'])

    # 保存分年度分析
    yearly_raw = analyze_by_period(
        backtest_data['daily_returns'],
        backtest_data['dates'][1:len(backtest_data['daily_returns'])+1],
        period='year'
    )
    yearly_neu = analyze_by_period(
        backtest_data['neutral_returns'],
        backtest_data['dates'][1:len(backtest_data['daily_returns'])+1],
        period='year'
    )

    # 保存汇总指标
    with open(f"{prefix}_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"预测器 {predictor_name}\n")
        f.write(f"组合模式: {portfolio_mode}\n")
        f.write(f"\n========== 整体表现 ==========\n")
        f.write(f"\n原始多空:\n")
        for k, v in ext_metrics_raw.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\n修正中性\n")
        for k, v in ext_metrics_neu.items():
            f.write(f"  {k}: {v:.4f}\n")

        f.write(f"\n========== 分年度表现（原始）==========\n")
        for year, m in yearly_raw.items():
            f.write(f"\n{year}:\n")
            for k, v in m.items():
                f.write(f"  {k}: {v:.4f}\n")

        f.write(f"\n========== 诊断统计 ==========\n")
        for k, v in backtest_data['diagnostic_counts'].items():
            f.write(f"{k}: {v}\n")

    # 保存完整数据（pickle）
    with open(f"{prefix}_full_data.pkl", 'wb') as f:
        pickle.dump(backtest_data, f)

    print(f"\n结果已保存到: {prefix}_*")
    return prefix


def parse_args():
    parser = argparse.ArgumentParser(description="V9 DL model / LightGBM production backtest")
    parser.add_argument("--model-type", choices=["dl", "lgb", "both"], default="dl")
    parser.add_argument("--checkpoint", default="ultimate_v7_best.pt")
    parser.add_argument("--lgb-dir", default="models_multi_v9_tech_macro")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--adv-mode", choices=["execution", "weight_cap", "both"], default="execution",
                        help="ADV约束模式 (默认: execution, 推荐). weight_cap已弃用，存在设计缺陷导致冲击成本过高")
    parser.add_argument("--portfolio-mode", choices=["optimizer", "simple_ls", "simple_long"], default="optimizer")
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--rebalance-freq", type=int, default=None)
    return parser.parse_args()


def load_or_train_lgb(train, val, model_dir, horizon_list):
    expected_dim = train[0]['X'].shape[1]
    if not os.path.exists(f"{model_dir}/lgb_h1.txt"):
        print("训练多周期LightGBM模型...")
        return train_multi_horizon_models(train, val, horizon_list, model_dir)
    print("加载已有LightGBM模型...")
    models, ic_decay = load_multi_horizon_models(horizon_list, model_dir)
    if not models_match_feature_dim(models, expected_dim):
        print("已有LightGBM模型特征维度不匹配，重新训练...")
        models, ic_decay = train_multi_horizon_models(train, val, horizon_list, model_dir)
    return models, ic_decay


def run_and_print_metrics(predictor, val, price_dict, vol_dict, cfg, args):
    print(f"\n开始生产级回测: {predictor.name}")
    raw_ret, neu_ret, backtest_data = run_backtest_production(
        predictor, val, price_dict, vol_dict,
        future_len=cfg.target_horizon, rebalance_freq=args.rebalance_freq, hist_window=60,
        ewma_hl=20, adv_limit_ratio=0.02, adv_mode=args.adv_mode,
        portfolio_mode=args.portfolio_mode, top_frac=args.top_frac,
        max_weight=0.05, lambda_t=0.05, lambda_b=0.2,
        target_vol=0.15, impact_coeff=0.1,
        config=cfg,
    )

    ann_raw, sharpe_raw, mdd_raw = calc_metrics(raw_ret)
    ann_neu, sharpe_neu, mdd_neu = calc_metrics(neu_ret)

    print(f"\n========== 生产级回测绩效({predictor.name}, {args.portfolio_mode}) ==========")
    print(f"原始多空: 年化 {ann_raw:.2f}% | 夏普 {sharpe_raw:.2f} | 回撤 {mdd_raw * 100:.2f}%")
    print(f"修正中性 年化 {ann_neu:.2f}% | 夏普 {sharpe_neu:.2f} | 回撤 {mdd_neu * 100:.2f}%")

    # 计算并显示扩展指标
    ext_metrics_raw = calc_extended_metrics(raw_ret)
    ext_metrics_neu = calc_extended_metrics(neu_ret)

    print(f"\n========== 扩展指标 ==========")
    print(f"原始多空:")
    print(f"  Calmar比率: {ext_metrics_raw.get('calmar', 0):.3f}")
    print(f"  Sortino比率: {ext_metrics_raw.get('sortino', 0):.3f}")
    print(f"  胜率: {ext_metrics_raw.get('win_rate', 0):.2f}%")
    print(f"  盈亏比 {ext_metrics_raw.get('profit_loss_ratio', 0):.3f}")
    print(f"修正中性")
    print(f"  Calmar比率: {ext_metrics_neu.get('calmar', 0):.3f}")
    print(f"  Sortino比率: {ext_metrics_neu.get('sortino', 0):.3f}")
    print(f"  胜率: {ext_metrics_neu.get('win_rate', 0):.2f}%")
    print(f"  盈亏比 {ext_metrics_neu.get('profit_loss_ratio', 0):.3f}")

    # 保存结果
    metrics = {
        'ann_raw': ann_raw, 'sharpe_raw': sharpe_raw, 'mdd_raw': mdd_raw,
        'ann_neu': ann_neu, 'sharpe_neu': sharpe_neu, 'mdd_neu': mdd_neu,
    }
    save_backtest_results(backtest_data, metrics, predictor.name, args.portfolio_mode)

    return backtest_data, metrics


# ==================== 主程序====================
def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    cfg = DataConfig()
    cfg.use_technical_features = True
    cfg.use_market_features = True
    cfg.use_macro_features = True
    cfg.min_stocks_per_time = 30
    cfg.target_horizon = 5
    cfg.seq_len = 40
    cfg.max_horizon = 10

    print("构建截面数据集...")
    train, val = build_cross_section_dataset(cfg, use_cache=True)

    print("加载价格与成交量数据...")
    price_dict, vol_dict = load_price_volume(cfg)
    print(f"股票数 {len(price_dict)}")

    predictors = []
    if args.model_type in ("dl", "both"):
        model, device, regime_dim = load_v9_checkpoint(args.checkpoint, train, cfg, args.device)
        predictors.append(DLPredictor(model, device, regime_dim))

    if args.model_type in ("lgb", "both"):
        horizon_list = [1, 3, 5, 10]
        models, ic_decay = load_or_train_lgb(train, val, args.lgb_dir, horizon_list)
        print("IC Decay:", ic_decay)
        predictors.append(LGBPredictor(models, ic_decay))

    all_results = {}
    for predictor in predictors:
        data, metrics = run_and_print_metrics(predictor, val, price_dict, vol_dict, cfg, args)
        all_results[predictor.name] = (data, metrics)

    # 对比分析
    if len(all_results) > 1:
        compare_predictors(all_results)


if __name__ == "__main__":
    main()
