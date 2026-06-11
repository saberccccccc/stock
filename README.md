# deepseek_optimized - 股票多因子Alpha预测系统

## 项目结构

```
├── core/                          # 核心模型与配置
│   ├── config.py                  #   DataConfig + 全局常量 + setup_project_environment()
│   ├── model.py                   #   UltimateV7Model (Transformer + 可选GAT)
│   └── train_utils.py             #   CrossSectionDataset + collate + train_model()
├── data/                          # 数据层
│   ├── pipeline.py                #   训练/推理截面构建
│   ├── market_features.py         #   市场宽度/离散度
│   ├── fundamental_factors.py     #   基本面因子 (PIT)
│   ├── macro_factors.py           #   宏观因子 (北向/两融/PMI)
│   ├── api_utils.py               #   SafeAPICaller + resolve_tushare_token()
│   ├── validate.py                #   每日数据校验
│   ├── update.py                  #   全量数据更新
│   └── update_daily.py            #   日频增量更新
├── backtest/                      # 回测引擎
│   ├── engine.py                  #   DLPredictor/LGBPredictor + 优化器
│   ├── layered_engine.py          #   分层持仓回测
│   ├── runtime.py                 #   回测运行时加载
│   ├── runners.py                 #   ProductionBacktestParams + run函数
│   ├── predictors.py              #   V9GATEnsemble/Intersection Predictor
│   ├── reports.py                 #   指标计算 + CSV保存 + 可视化
│   └── tracking.py                #   PnL跟踪 + 持仓表 + 批量对比
├── run/                           # 入口脚本
│   ├── train.py                   #   统一训练入口 --model {v9,gat,gat_v2,legacy}
│   ├── backtest.py                #   统一回测入口 --experiment {ensemble,intersection,concentrated,persistent}
│   ├── backtest_layered_holdings.py # 分层持仓回测
│   ├── track_backtest_holdings.py #   回测持仓盯市跟踪
│   ├── recommend_daily.py         #   单日推荐/单股查询
│   ├── recommend_persistent.py    #   多日持久/动量/一致性推荐
│   ├── rank_watchlist.py          #   自选股多日排名
│   ├── daily_top10.py             #   每日自动 Top10 推荐 (cron)
│   └── recommend_utils.py         #   推荐公共工具
├── tests/                         # 单元测试
│   ├── test_config.py             #   DataConfig + build_v9_config
│   ├── test_metrics.py            #   calc_metrics / calc_extended_metrics
│   └── test_pipeline.py           #   特征计算 + 归一化 + 行业加载
├── checkpoints/                   # 模型权重 (.pt)
├── recommendations/               # 推荐输出 (CSV + 日志)
├── cache/                         # 运行时缓存
├── data/raw/                      # 原始日线CSV (gitignored)
├── data/forward_raw/              # 2026-05-19起的前向回测数据 (gitignored)
├── data/tracking_raw/             # 盯市跟踪数据 (gitignored)
└── backtest_results*/             # 回测输出 (gitignored)
```

## 快速开始

```bash
# 1. 初始化数据
export TUSHARE_TOKEN="<your-token>"
python data/update.py --init
python data/update.py

# 2. 训练
python run/train.py --model v9

# 3. 回测
python run/backtest.py --experiment ensemble
```

## 每日推荐

```bash
# 自动更新数据 + 推荐
python run/daily_top10.py

# 手动推荐
python run/recommend_daily.py --as-of latest --predictor top_union_bottom_intersection --top-n 10
python run/recommend_daily.py --as-of latest --predictor avg_score --code 000001.SZ

# 持久推荐（多日平均）
python run/recommend_persistent.py --ndates 5 --top-n 30

# 自选股排名
python run/rank_watchlist.py --from-date 2026-05-09 --to-date 2026-05-15
```

## 回测

```bash
python run/backtest.py --experiment ensemble          # 3策略 × 3模式
python run/backtest.py --experiment intersection      # 1预测器 × 6模式
python run/backtest.py --experiment concentrated      # 集中持仓 1%/2%/3%/5%
python run/backtest.py --experiment persistent        # 多日持续性信号
python run/backtest_layered_holdings.py               # 分层持仓
```

## 环境

- PyTorch 2.7.1+cu118, RTX 2060 6GB
- Python: `C:/Users/x/miniconda3/envs/torch/python`
- CUDA 训练: `batch_size=2, accum_steps=8` (V9), `batch_size=4, accum_steps=4` (GAT)
- AMP 必须禁用（否则 Loss NaN）

## 测试

```bash
C:/Users/x/miniconda3/envs/torch/python tests/test_config.py
C:/Users/x/miniconda3/envs/torch/python tests/test_metrics.py
C:/Users/x/miniconda3/envs/torch/python tests/test_pipeline.py
```

## 关键不变量

- 回测收益指标: `next_close_to_next_close`
- 研究截止日: `2026-05-18`，训练/验证/测试不得超过该日期
- 数据边界: `data/raw` (冻结研究集) vs `data/forward_raw` (真实前向回测)，不可混淆
- 资金规模: 容量评估默认使用 50万元和100万元
- 特征维度: X=236, risk=170 (get_regime_dim 动态)

完整规则见 `RESEARCH_PROTOCOL.md`。
