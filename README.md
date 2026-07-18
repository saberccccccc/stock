# deepseek_model_exp - 股票多因子 Alpha 预测与执行研究

## 项目结构

```
├── core/                          # 核心模型与配置
│   ├── config.py                  #   DataConfig + 全局常量 + setup_project_environment()
│   ├── model.py                   #   UltimateV7Model (Transformer + 可选GAT)
│   └── train_utils.py             #   CrossSectionDataset + collate + train_model()
├── alpha/                         # Alpha I/O、变换、市场覆盖与诊断
├── experiments/                   # 实验注册、排行榜与 checkpoint 选择
├── data/                          # 数据层
│   ├── pipeline.py                #   训练/推理截面构建
│   ├── market_features.py         #   市场宽度/离散度
│   ├── fundamental_factors.py     #   基本面因子 (PIT)
│   ├── macro_factors.py           #   宏观因子 (北向/两融/PMI)
│   ├── api_utils.py               #   SafeAPICaller + resolve_tushare_token()
│   ├── st_status.py               #   PIT 历史 ST 状态事件契约与校验
│   ├── dataset_runtime.py         #   流式 Dataset/DataHandler/Processor 运行时
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
│   ├── tracking.py                #   PnL跟踪 + 持仓表 + 批量对比
│   ├── open_ledger.py             #   Open-price share-ledger 引擎
│   ├── execution.py               #   现金、整手、费用、ADV 与涨跌停约束
│   └── presets.py                 #   正式执行基线与研究预设
├── run/                           # 入口脚本
│   ├── train.py                   #   统一训练入口 --model {v9,gat,gat_v2,legacy}
│   ├── backtest.py                #   统一回测入口 --experiment {ensemble,intersection,concentrated,persistent}
│   ├── audit_execution_coverage.py # 执行输入覆盖审计
│   ├── download_historical_st_events.py # 下载并冻结历史 ST 事件
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
├── data/forward_raw/              # 兼容前向缓存；2026全年仅观察 (gitignored)
├── data/tracking_raw/             # 盯市跟踪数据 (gitignored)
└── backtest_results*/             # 回测输出 (gitignored)
```

## 快速开始

```powershell
# Required for all Torch/CUDA commands
$env:PYTHON = "$env:USERPROFILE\miniconda3\envs\torch\python.exe"

# 1. 初始化数据
export TUSHARE_TOKEN="<your-token>"
& $env:PYTHON data/update.py --init
& $env:PYTHON data/update.py

# 2. 训练
& $env:PYTHON run/train.py --model v9

# 3. 回测
& $env:PYTHON run/backtest.py --experiment ensemble
```

当前兼容阶段，`data/update.py` 写入 `data/raw` 时把选择数据限制到
`2025-12-31`；2026 数据属于 Forward 观察。物理文件可以覆盖更长历史，
但实验必须用逻辑 split 和实际日期字段约束读取，不能由目录名推断用途。
可用下面的命令检查实际边界：

```powershell
python run/audit_data_boundary.py --dataset-role research --effective-end-date 2025-12-31
```

`data/update_daily.py` 仍禁止选择数据写过 `2025-12-31`；当前 Forward 更新
继续使用兼容目录 `data/forward_raw`，后续由统一 Provider 消除双目录依赖。

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

- 当前已验证: PyTorch 2.11.0+cu128，RTX 5070 Laptop 8GB
- PowerShell 脚本优先使用 `$env:PYTHON`，其次寻找当前用户的
  `miniconda3/envs/torch/python.exe`，最后回退到 `python`
- CUDA 训练: `batch_size=2, accum_steps=8` (V9), `batch_size=4, accum_steps=4` (GAT)
- AMP 必须禁用（否则 Loss NaN）

## 测试

```bash
python -m pytest -q
```

若默认 `python` 不是训练环境，可在 PowerShell 中先设置：

```powershell
$env:PYTHON = "$env:USERPROFILE\miniconda3\envs\torch\python.exe"
& $env:PYTHON -m pytest -q
```

## Qlib 对齐

当前优先建设项目原生的 Qlib 式研究框架，而不是引入 Qlib 默认数据或
回测器。Q0 对齐基线、组件矩阵和 Workflow v2 草案见：

- `reports/qlib_alignment_20260717/QLIB_ALIGNMENT_REPORT_ZH.md`
- `reports/qlib_alignment_20260717/QLIB_TERMINOLOGY_ZH.md`
- `schemas/workflow_v2.schema.json`
- `configs/workflow_v2_golden.json`

`experiments/workflow.py` 现在兼容 Workflow schema v1，并可校验、归一化
和编译 v2 的 `rolling_lgbm_alpha` 与 `frozen_artifact`。尚未绑定具体训练器
的模型会明确拒绝，不会隐式回退。Workflow schema 版本与
`experiments/recording.py` 的实验 manifest schema v2 是两个独立契约。
正式执行仍只使用 realistic `open_ledger`，不使用 Qlib Executor。

Q2 已提供 `ProjectDataset.prepare(segment, col_set, data_key)`，支持
raw/infer/learn 数据视图、shared/infer/learn Processor、Train-only fit 和
冻结状态重放，并直接包装现有 v14 memmap。正式 LightGBM rolling 已可通过
`data.dataset_runtime=project_dataset` 使用该路径；2024 Compact 单窗口的
模型与 alpha 已和 legacy 达到字节一致。由于当前仍有约 11.6% 性能开销，
默认继续使用 `legacy_iter`，待 2025 第二窗口 parity 后再决定切换。Q3 的
`experiments/model_adapters.py` 已统一 LightGBM、PyTorch strong alpha、冻结
信号和 legacy 只读信号的生命周期与 `PredictionFrame`；Q5 再绑定真实 e19
训练器并进行月度 Rolling，当前训练入口保持兼容。

Q4 的 `experiments/record_templates.py` 与
`run/materialize_standard_records.py` 已提供 Signal -> SignalAnalysis ->
Portfolio -> RiskAttribution/Stress -> Decision 的不可变证据链。组合记录
只接受 official open-price ledger 的真实产物；旧报告缺少订单等明细时会
明确不完整，不会自动补造。

## 关键不变量

- 当前正式组合层基线: `ledger_path_v3_t0001_nolookahead`；
  T 日收盘后生成信号，T+1 开盘按现金、股数、整手、费用、ADV 与涨跌停约束成交
- 历史执行基线: V9 `avgw3` + `maxret095` + open-price share-ledger，仅作为 legacy 参照
- `next_close_to_next_close` 只属于旧版 close-based 连续性回测，不作为当前正式执行口径
- 评价划分: 2024 Val、2025 Test、2026 全年 Forward；Forward 当前到
  `2026-06-30`，只观察，不参与选择
- 完整 2026 Forward 的父模型、变换和策略必须在 `2025-12-31` 前冻结；
  `2026-05-18` 只是旧缓存/报告日期，不是 Forward 起点
- 数据边界: 物理数据覆盖与逻辑 split 分开记录；当前双目录仅为兼容实现
- 历史 ST: 正式历史执行必须有 `data/raw/st_status_events.csv` 及匹配
  manifest；`data/stock_industry.csv` 只是当前快照，不能证明 2024/2025
  的历史 ST 状态。下载器默认使用 Tushare `st`；只有显式使用
  `--endpoint namechange` 并通过独立覆盖审计时，才允许使用名称区间重建；
  其 manifest 必须标注 `source_label=由历史股票名称区间重建`，不能误读为
  直接 ST 事件源。
- 历史 ST 下载器对权限/积分错误立即失败，不会重复重试；网络和临时频率
  错误仍按退避策略重试。官方接口未在页面中承诺通用 `offset/limit` 分页，
  因此正式 CLI 使用 `namechange_date_range` 或 `st_by_ts_code` 两种明确
  模式；未获得权限前不把任何下载结果当作完整研究证据。
  研究来源按选择边界 `2025-12-31` 截断，Forward 来源只作观察。
- 资金规模: 容量评估默认使用 50万元和100万元
- 特征维度: X=236, risk=170 (get_regime_dim 动态)

完整规则见 `RESEARCH_PROTOCOL.md`。当前候选、baseline 与报告路径登记在
`registry/`，项目当前索引见 `PROJECT_CURRENT_INDEX_20260710.md`，重构蓝图见
`PROJECT_REFACTOR_BLUEPRINT_20260710.md`。

项目唯一有效的总执行顺序见
`MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md`。旧 Qlib 计划和长期路线图
只保留技术设计与历史证据，不再分别决定“下一步”。当前总计划位置为 P2：
正式训练主线收敛。

工程开发前请先阅读 `PROJECT_RULES.md`、`ARCHITECTURE.md`、
`DEVELOPMENT_LOG.md` 与 `ADR/`；它们定义当前工程治理和决策记录规则。
