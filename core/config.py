# config.py 鈥?鎴?潰鏁版嵁闆嗛厤缃?from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class DataConfig:
    """Cross-section multi-factor data pipeline config"""
    # 鏁版嵁璺?緞
    data_dir: str = "data/raw"
    max_stocks: Optional[int] = None
    force_rebuild: bool = False

    # 窗口参数
    seq_len: int = 40
    target_horizon: int = 5       # 涓绘爣绛撅細棰勬祴鏈?潵N鏃ユ敹鐩婏紙榛樿?5日）
    max_horizon: int = 10

    # 鍏煎?鏃т唬鐮?    @property
    def future_len(self):
        return self.target_horizon

    # 鎶€鏈?寚鏍?    use_technical_features: bool = False
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # 鎴?潰鏋勫缓
    min_stocks_per_time: int = 30
    normalize_features: bool = True

    # DataLoader
    batch_size: int = 512
    num_workers: int = 8

    # ==================== 鏂板?锛氭墿灞曞洜瀛?====================
    use_macro_features: bool = False            # 瀹忚?/璧勯噾娴佸洜瀛愶紙鍖楀悜銆佽瀺璧勮瀺鍒搞€丳MI锛?    use_fundamental_features: bool = False      # 季报基本面因子（ROE銆佽惀鏀跺?閫熴€丳E分位数）
    tushare_token: Optional[str] = None         # 基本面数据Token锛涗负绌烘椂灏濊瘯璇诲彇鐜??变量TUSHARE_TOKEN

    # ==================== 鏂板?锛氭ā鍨嬪?寮?====================
    use_gat: bool = False                       # GAT浜т笟閾惧浘缃戠粶
    use_stacking: bool = False                  # Stacking集成（LGB→Transformer锛?
    # ==================== 鏂板?锛氬姩鎬侀?鎺?====================
    dynamic_risk_budget: bool = False           # 鍔ㄦ€侀?闄╅?绠?    max_drawdown_limit: float = 0.15            # 鏈€澶у洖鎾ゅ?忍度
    base_target_vol: float = 0.15               # 鍩虹?鐩?爣娉㈠姩鐜?
    # ==================== V8鏂板?锛氬競鍦烘暣浣撳睘鎬?====================
    use_market_features: bool = True      # 甯傚満鏁翠綋灞炴€э紙鎸囨暟鏀剁泭銆佸?搴︺€佺?鏁ｅ害锛?
    # ==================== V7鏂板?锛氬?鍛ㄦ湡鑱斿悎璁?粌 ====================
    test_mode: bool = False                     # 娴嬭瘯妯″紡锛堜粎鍔犺浇鍓峃鍙?偂绁?級
    test_stocks: int = 1000                     # 娴嬭瘯妯″紡鑲＄エ鏁?    use_multi_horizon: bool = True              # 澶氬懆鏈熻仈鍚堣?缁?    horizon_indices: Tuple[int, ...] = (0, 2, 4, 6)   # y_seq涓?殑鍒楃储寮? h1,h3,h5,h7
    horizon_weights: Tuple[float, ...] = (0.15, 0.25, 0.35, 0.25)  # 各周期loss权重
