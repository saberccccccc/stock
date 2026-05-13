# risk_controller.py - 动态风险预算与风控
import numpy as np


class DynamicRiskBudget:
    """
    动态风险预算控制器
    - 波动率自适应目标波动率
    - 回撤控制自动降杠杆
    - IC衰减动态调仓频率
    """

    def __init__(self, base_target_vol=0.15, max_dd_limit=0.15, vol_lookback=60,
                 dd_lookback=252, min_target_vol=0.05, max_target_vol=0.25):
        self.base_target_vol = base_target_vol
        self.max_dd_limit = max_dd_limit
        self.vol_lookback = vol_lookback
        self.dd_lookback = dd_lookback
        self.min_target_vol = min_target_vol
        self.max_target_vol = max_target_vol

        self.mu_vol = base_target_vol
        self.current_dd = 0.0
        self.vol_ema = base_target_vol
        self.ema_decay = 0.94

    def update(self, daily_returns):
        """
        根据每日收益序列更新状态。
        daily_returns: np.array of recent returns (建议最近60个交易日)
        """
        if len(daily_returns) < 5:
            return

        realized_vol = np.std(daily_returns) * np.sqrt(252)
        if np.isfinite(realized_vol):
            self.vol_ema = (self.ema_decay * self.vol_ema +
                            (1 - self.ema_decay) * realized_vol)

        cum_ret = np.cumprod(1 + np.clip(daily_returns, -0.5, 0.5))
        peak = np.maximum.accumulate(cum_ret)
        dd = float((peak[-1] - cum_ret[-1]) / (peak[-1] + 1e-8))
        if np.isfinite(dd):
            self.current_dd = max(0.0, dd)

    def get_target_vol(self):
        """
        波动率自适应目标波动率：
        实际波动率 < 基准 -> 适度提升目标（但不超过上限）
        实际波动率 > 基准 -> 降低目标波动率
        """
        if self.vol_ema > 0:
            ratio = self.base_target_vol / self.vol_ema
            adjusted = self.base_target_vol * np.sqrt(np.clip(ratio, 0.3, 2.0))
        else:
            adjusted = self.base_target_vol

        if self.current_dd > self.max_dd_limit * 0.5:
            dd_penalty = max(0.3, 1 - self.current_dd / self.max_dd_limit)
            adjusted *= dd_penalty

        return np.clip(adjusted, self.min_target_vol, self.max_target_vol)

    def get_leverage_multiplier(self):
        """
        回撤控制杠杆倍数：
        dd < 5%  -> 满杠杆(1.0)
        dd 5-10% -> 0.8x
        dd 10-15% -> 0.5x
        dd > 15%  -> 0.3x（强制降仓）
        """
        if self.current_dd < 0.05:
            return 1.0
        elif self.current_dd < 0.10:
            return 0.8
        elif self.current_dd < self.max_dd_limit:
            return 0.5
        else:
            return 0.3

    def get_rebalance_freq(self, ic_decay_curve, default_freq=5):
        """
        根据IC衰减速度动态调仓：
        快速衰减 -> 缩短调仓周期
        缓慢衰减 -> 延长调仓周期
        """
        if ic_decay_curve is None or len(ic_decay_curve) < 2:
            return default_freq

        ic_norm = ic_decay_curve / (ic_decay_curve[0] + 1e-8)
        for i in range(len(ic_norm)):
            if ic_norm[i] < 0.5:
                half_life = i + 1
                break
            if i > 0 and ic_norm[i] < ic_norm[i-1] * 0.7:
                half_life = i + 1
                break
        else:
            half_life = max(3, min(5, len(ic_decay_curve)))

        half_life = max(half_life, 1)
        return int(np.clip(half_life, 2, 20))


if __name__ == "__main__":
    print("=== 动态风险预算测试 ===\n")

    rbc = DynamicRiskBudget(base_target_vol=0.15)

    calm_rets = np.random.normal(0.001, 0.01, 60)
    rbc.update(calm_rets)
    print(f"平静市场: target_vol={rbc.get_target_vol():.3f}, "
          f"leverage={rbc.get_leverage_multiplier():.2f}, dd={rbc.current_dd:.3f}")

    high_vol_rets = np.random.normal(0.000, 0.03, 60)
    rbc.update(high_vol_rets)
    print(f"高波动:   target_vol={rbc.get_target_vol():.3f}, "
          f"leverage={rbc.get_leverage_multiplier():.2f}, dd={rbc.current_dd:.3f}")

    dd_rets = np.array([-0.02] * 5 + [0.005] * 10 + [-0.03] * 5)
    rbc.update(dd_rets)
    print(f"回撤中:   target_vol={rbc.get_target_vol():.3f}, "
          f"leverage={rbc.get_leverage_multiplier():.2f}, dd={rbc.current_dd:.3f}")

    ic_decay = np.array([0.15, 0.10, 0.07, 0.04, 0.02, 0.01, 0.005, 0.003])
    freq = rbc.get_rebalance_freq(ic_decay)
    print(f"\nIC衰减: {ic_decay}")
    print(f"建议调仓周期: {freq} 天")
