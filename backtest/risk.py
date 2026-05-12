# risk_controller.py 鈥?鍔ㄦ€侀?闄╅?绠椾笌椋庢帶
import numpy as np


class DynamicRiskBudget:
    """
    鍔ㄦ€侀?闄╅?绠楁帶鍒跺櫒
    - 娉㈠姩鐜囪嚜閫傚簲鐩?爣娉㈠姩鐜?    - 鍥炴挙鎺у埗鑷?姩闄嶆潬鏉?    - IC琛板噺鍔ㄦ€佽皟浠撻?鐜?    """

    def __init__(self, base_target_vol=0.15, max_dd_limit=0.15, vol_lookback=60,
                 dd_lookback=252, min_target_vol=0.05, max_target_vol=0.25):
        self.base_target_vol = base_target_vol
        self.max_dd_limit = max_dd_limit
        self.vol_lookback = vol_lookback
        self.dd_lookback = dd_lookback
        self.min_target_vol = min_target_vol
        self.max_target_vol = max_target_vol

        # 鐘舵€佽拷韪?        self.mu_vol = base_target_vol         # 闀挎湡娉㈠姩鐜囧潎鍊硷紙EMA鏇存柊锛?        self.current_dd = 0.0
        self.vol_ema = base_target_vol
        self.ema_decay = 0.94

    def update(self, daily_returns):
        """
        鏍规嵁姣忔棩鏀剁泭搴忓垪鏇存柊鐘舵€?        daily_returns: np.array of recent returns (寤鸿?鏈€杩?0涓?氦鏄撴棩)
        """
        if len(daily_returns) < 5:
            return

        # 鏇存柊娉㈠姩鐜嘐MA
        realized_vol = np.std(daily_returns) * np.sqrt(252)
        if np.isfinite(realized_vol):
            self.vol_ema = (self.ema_decay * self.vol_ema +
                            (1 - self.ema_decay) * realized_vol)

        # 鏇存柊鏈€澶у洖鎾?        cum_ret = np.cumprod(1 + np.clip(daily_returns, -0.5, 0.5))
        peak = np.maximum.accumulate(cum_ret)
        dd = float((peak[-1] - cum_ret[-1]) / (peak[-1] + 1e-8))
        if np.isfinite(dd):
            self.current_dd = max(0.0, dd)

    def get_target_vol(self):
        """
        娉㈠姩鐜囪嚜閫傚簲鐩?爣娉㈠姩鐜囷細
        瀹為檯娉㈠姩鐜?< 鍩哄噯 鈫?閫傚害鎻愬崌鐩?爣锛堜絾涓嶈秴杩囦笂闄愶級
        瀹為檯娉㈠姩鐜?> 鍩哄噯 鈫?闄嶄綆鐩?爣娉㈠姩鐜?        """
        if self.vol_ema > 0:
            ratio = self.base_target_vol / self.vol_ema
            # ratio > 1 鈫?甯傚満姣旈?鏈熷钩闈欙紝鍙?€傚綋鏀惧ぇ
            # ratio < 1 鈫?甯傚満姣旈?鏈熸尝鍔ㄥぇ锛岄渶瑕佹敹绱?            adjusted = self.base_target_vol * np.sqrt(np.clip(ratio, 0.3, 2.0))
        else:
            adjusted = self.base_target_vol

        # 鍥炴挙鎯╃綒
        if self.current_dd > self.max_dd_limit * 0.5:
            dd_penalty = max(0.3, 1 - self.current_dd / self.max_dd_limit)
            adjusted *= dd_penalty

        return np.clip(adjusted, self.min_target_vol, self.max_target_vol)

    def get_leverage_multiplier(self):
        """
        鍥炴挙鎺у埗鏉犳潌鍊嶆暟锛?        dd < 5%  鈫?婊℃潬鏉?(1.0)
        dd 5-10% 鈫?0.8x
        dd 10-15% 鈫?0.5x
        dd > 15%  鈫?0.3x锛堝己鍒堕檷浠擄級
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
        鏍规嵁IC琛板噺閫熷害鍔ㄦ€佽皟浠擄細
        蹇?€熻“鍑?鈫?缂╃煭璋冧粨鍛ㄦ湡
        缂撴參琛板噺 鈫?寤堕暱璋冧粨鍛ㄦ湡
        """
        if ic_decay_curve is None or len(ic_decay_curve) < 2:
            return default_freq

        ic_norm = ic_decay_curve / (ic_decay_curve[0] + 1e-8)
        # 缁撳悎缁濆?闃堝€煎拰瓒嬪娍妫€娴嬶紙涓巈ngine.py淇濇寔涓€鑷达級
        for i in range(len(ic_norm)):
            # 鏂规硶1锛氱粷瀵归槇鍊?- IC琛板噺鍒?0%浠ヤ笅
            if ic_norm[i] < 0.5:
                half_life = i + 1
                break
            # 鏂规硶2锛氳秼鍔挎?娴?- IC鏄捐憲涓嬮檷锛堜笅闄嶈秴杩?0%锛?            if i > 0 and ic_norm[i] < ic_norm[i-1] * 0.7:
                half_life = i + 1
                break
        else:
            # all ic >= 0.5: use default mid-range cycle
            half_life = max(3, min(5, len(ic_decay_curve)))

        half_life = max(half_life, 1)
        if half_life == 0:
            half_life = default_freq

        # 闄愬埗鍦?[2, 20] 澶?        return int(np.clip(half_life, 2, 20))


# ==================== 鐙?珛杩愯?锛堟煡鐪嬬姸鎬侊級 ====================
if __name__ == "__main__":
    print("=== 鍔ㄦ€侀?闄╅?绠楁祴璇?===\n")

    rbc = DynamicRiskBudget(base_target_vol=0.15)

    # 妯℃嫙骞抽潤甯傚満
    calm_rets = np.random.normal(0.001, 0.01, 60)
    rbc.update(calm_rets)
    print(f"骞抽潤甯傚満: target_vol={rbc.get_target_vol():.3f}, "
          f"leverage={rbc.get_leverage_multiplier():.2f}, dd={rbc.current_dd:.3f}")

    # 妯℃嫙楂樻尝鍔?    high_vol_rets = np.random.normal(0.000, 0.03, 60)
    rbc.update(high_vol_rets)
    print(f"楂樻尝鍔?   target_vol={rbc.get_target_vol():.3f}, "
          f"leverage={rbc.get_leverage_multiplier():.2f}, dd={rbc.current_dd:.3f}")

    # 妯℃嫙鍥炴挙
    dd_rets = np.array([-0.02] * 5 + [0.005] * 10 + [-0.03] * 5)
    rbc.update(dd_rets)
    print(f"鍥炴挙涓?   target_vol={rbc.get_target_vol():.3f}, "
          f"leverage={rbc.get_leverage_multiplier():.2f}, dd={rbc.current_dd:.3f}")

    # 娴嬭瘯鍔ㄦ€佽皟浠撻?鐜?    ic_decay = np.array([0.15, 0.10, 0.07, 0.04, 0.02, 0.01, 0.005, 0.003])
    freq = rbc.get_rebalance_freq(ic_decay)
    print(f"\nIC琛板噺: {ic_decay}")
    print(f"[FIXED]")
