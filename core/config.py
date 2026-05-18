# config.py - 截面数据集配置
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

# ── 全局常量 ──────────────────────────────────────────────
TRADING_DAYS = 252          # 年化交易日数
ADV_LIMIT_RATIO = 0.02      # 单日成交量占比上限
MAX_WEIGHT = 0.05           # 单只持仓权重上限
TARGET_VOL = 0.15           # 目标年化波动率
IMPACT_COEFF = 0.1          # 交易冲击成本系数
EPS = 1e-8                  # 数值稳定小量

# 模型 checkpoint 路径（实验分支用 exp 子目录，与主项目隔离）
V9_CKPT = "checkpoints_exp/ultimate_v7_best.pt"
GAT_CKPT = "checkpoints_exp/ultimate_v7_gat_best.pt"
LEGACY_CKPT = "checkpoints_exp/ultimate_v7_legacy_best.pt"


@dataclass
class DataConfig:
    """Cross-section multi-factor data pipeline config"""
    # 数据路径
    data_dir: str = "data/raw"
    max_stocks: Optional[int] = None
    force_rebuild: bool = False

    # 窗口参数
    seq_len: int = 40
    target_horizon: int = 5       # 主标签：预测未来N日收益（默认5日）
    max_horizon: int = 10

    # 兼容旧代码
    @property
    def future_len(self):
        return self.target_horizon

    # 技术指标
    use_technical_features: bool = False
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # 截面构建
    min_stocks_per_time: int = 30
    normalize_features: bool = True

    # DataLoader
    batch_size: int = 512
    num_workers: int = 8

    # ==================== 新增：扩展因子 ====================
    use_macro_features: bool = False            # 宏观/资金流因子（北向、融资融券、PMI）
    use_fundamental_features: bool = False      # 季报基本面因子（ROE、营收增速、PE分位数）
    tushare_token: Optional[str] = None         # 基本面数据Token，为空时尝试读取环境变量TUSHARE_TOKEN

    # ==================== 新增：模型开关 ====================
    use_gat: bool = False                       # GAT产业链图网络
    max_drawdown_limit: float = 0.15            # 最大回撤容忍度
    base_target_vol: float = 0.15               # 基础目标波动率
    # ==================== V8新增：市场整体属性 ====================
    use_market_features: bool = True      # 市场整体属性（指数收益、宽度、离散度）
    # ==================== V7新增：多周期联合训练 ====================
    test_mode: bool = False                     # 测试模式（仅加载前N只股票）
    test_stocks: int = 1000                     # 测试模式股票数
    use_multi_horizon: bool = True              # 多周期联合训练
    horizon_indices: Tuple[int, ...] = (0, 2, 4, 6)   # y_seq中的列索引 h1,h3,h5,h7
    horizon_weights: Tuple[float, ...] = (0.225, 0.225, 0.3, 0.25)  # 各周期loss权重


# ── 项目环境初始化 ──────────────────────────────────────
def setup_project_environment():
    """设置 sys.path 和 cwd 到项目根目录，返回 PROJECT_ROOT Path。"""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    warnings.filterwarnings("ignore")
    return root
