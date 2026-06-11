# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## PowerShell 编码注意事项

**所有 PowerShell 中读写 `.py` / `.md` 文件的操作必须加 `-Encoding UTF8`。**

```powershell
Get-Content -LiteralPath "file.py" -Raw -Encoding UTF8 | Set-Content -LiteralPath "file.py" -Encoding UTF8
```

Python 3 自身按 UTF-8 读写即可。不要用 PowerShell 默认编码批量处理源码。

## Environment and commands

Use the existing Miniconda environment on this machine:

```bash
C:/Users/x/miniconda3/envs/torch/python -m py_compile core/config.py core/model.py core/train_utils.py data/pipeline.py data/fundamental_factors.py data/macro_factors.py data/update.py data/update_daily.py data/api_utils.py backtest/engine.py backtest/layered_engine.py backtest/runtime.py backtest/runners.py backtest/predictors.py backtest/reports.py backtest/tracking.py run/train.py run/backtest.py run/recommend_daily.py run/recommend_persistent.py run/rank_watchlist.py run/daily_top10.py run/recommend_utils.py run/backtest_layered_holdings.py run/track_backtest_holdings.py
C:/Users/x/miniconda3/envs/torch/python run/train.py --model v9
C:/Users/x/miniconda3/envs/torch/python run/train.py --model gat
C:/Users/x/miniconda3/envs/torch/python run/backtest.py --experiment ensemble
C:/Users/x/miniconda3/envs/torch/python run/recommend_daily.py --as-of latest --predictor top_union_bottom_intersection --top-n 10
C:/Users/x/miniconda3/envs/torch/python run/daily_top10.py
```

The environment has PyTorch `2.7.1+cu118` with CUDA available on an RTX 2060-class 6GB GPU.

Data update commands require a Tushare token supplied outside source code:

```bash
export TUSHARE_TOKEN="65209d394f51051f94f8a9eeeb3396048121ecf94080eda0e33d06e5"
C:/Users/x/miniconda3/envs/torch/python data/update.py
C:/Users/x/miniconda3/envs/torch/python data/update_daily.py
C:/Users/x/miniconda3/envs/torch/python data/update.py --init
```

## Architecture overview

- `run/train.py --model {v9,gat,gat_v2,legacy}` — unified training entry.
- `run/backtest.py --experiment {ensemble,intersection,concentrated,persistent}` — unified backtest entry.
- `core/model.py` — `UltimateV7Model`: FeatureGrouper + cross-stock Transformer + optional GAT branch + multi-horizon heads.
- `core/train_utils.py` — `CrossSectionDataset`, collate functions, loss/evaluation, `get_regime_dim()`, `train_model()`.
- `core/config.py` — `DataConfig` dataclass + global constants (`TRADING_DAYS`, etc.) + `setup_project_environment()`.
- `data/pipeline.py` — builds cross-sectional samples from `data/raw/`, constructs features/labels, exposes `build_inference_sample()` for label-free inference.
- `data/fundamental_factors.py`, `data/macro_factors.py` — PIT fundamental/macro factors (do not change PIT logic casually).
- `data/api_utils.py` — shared `SafeAPICaller` + `resolve_tushare_token()` for all API calls.
- `backtest/engine.py` — DL/LGB predictors, portfolio optimizers, `run_backtest_production()`.
- `backtest/layered_engine.py` — layered holding backtest engine.
- `backtest/runtime.py` — `build_v9_backtest_config()`, `load_backtest_runtime()`, `load_v9_gat_predictors()`.
- `backtest/runners.py` — `ProductionBacktestParams`, `LayeredBacktestParams`, `run_production_backtest_once()`, `run_layered_backtest_once()`.
- `backtest/predictors.py` — `V9GATEnsemblePredictor`, `V9GATIntersectionPredictor`.
- `backtest/reports.py` — `save_summary_csv()`, `calc_metrics()`, `calc_extended_metrics()`, `save_backtest_results()`.
- `backtest/tracking.py` — PnL calculation, holdings tables, batch comparison helpers.
- `run/recommend_utils.py` — shared board classification, predictor construction, scoring/ranking, date helpers.
- `run/daily_top10.py` — cron-friendly daily wrapper: update data → clear cache → recommend Top-10 → append to log.

## Training

```bash
C:/Users/x/miniconda3/envs/torch/python run/train.py --model v9
C:/Users/x/miniconda3/envs/torch/python run/train.py --model gat
C:/Users/x/miniconda3/envs/torch/python run/train.py --model gat_v2
C:/Users/x/miniconda3/envs/torch/python run/train.py --model legacy
```

Optional flags: `--test-stocks N`, `--epochs N`, `--lr X`, `--device {auto,cpu,cuda}`, `--output-dir DIR`.

- **AMP must stay disabled.** Mixed precision causes `Train Loss: nan`.
- RTX 2060 / 6GB: V9 uses `batch_size=2, accum_steps=8`; GAT uses `batch_size=4, accum_steps=4, keep_ratio=0.7`.
- **No BatchNorm in current model stack** (Transformer/heads use LayerNorm). Gradient accumulation changes optimizer step frequency but does **not** introduce BatchNorm running-stat mismatch.
- Checkpoints: `checkpoints/ultimate_v7_{best,gat_best,legacy_best}.pt`.
- If feature schema, label logic, or factor PIT logic changes, old checkpoints should be invalidated.

## Backtest

```bash
C:/Users/x/miniconda3/envs/torch/python run/backtest.py --experiment ensemble
C:/Users/x/miniconda3/envs/torch/python run/backtest.py --experiment intersection
C:/Users/x/miniconda3/envs/torch/python run/backtest.py --experiment concentrated
C:/Users/x/miniconda3/envs/torch/python run/backtest.py --experiment persistent
C:/Users/x/miniconda3/envs/torch/python run/backtest_layered_holdings.py  # separate engine
```

Return metric is `next_close_to_next_close`. ADV mode `execution` is default. Do not change `entry_day`/hold-window semantics.

### Key benchmarks (2026-05-18, ~4940 stocks)

| Experiment | Strategy | Mode | Ann | Sharpe | MDD |
|-----------|----------|------|-----|--------|-----|
| ensemble | top_union_bottom_intersection | simple_ls | 41.09% | 3.76 | 8.39% |
| ensemble | top_union_bottom_intersection | optimizer_projected | 38.79% | 3.72 | 8.13% |
| ensemble | avg_score | simple_ls | 38.18% | 3.50 | 8.96% |
| intersection | — | optimizer | 42.91% | 4.20 | 6.78% |
| concentrated | avg_score 1% | simple_ls | 108.62% | 5.44 | 8.17% |
| persistent | — | baseline | 38.33% | 3.51 | 8.96% |

Key insight: alpha strength increases as selection narrows; 2-3% offers best practical balance. Multi-date alpha averaging destroys signal. `simple_long` consistently underperforms `simple_ls` by 10-20%.

## Recommendation system

```bash
# Daily Top-10 (auto-updates data first)
C:/Users/x/miniconda3/envs/torch/python run/daily_top10.py

# Single-day: top-N, single-stock query, batch mode
C:/Users/x/miniconda3/envs/torch/python run/recommend_daily.py --as-of latest --predictor top_union_bottom_intersection --top-n 10
C:/Users/x/miniconda3/envs/torch/python run/recommend_daily.py --as-of latest --predictor avg_score --code 000001.SZ
C:/Users/x/miniconda3/envs/torch/python run/recommend_daily.py --from-date 2026-05-11 --to-date 2026-05-15 --top-n 20

# Multi-date persistence/momentum/consistency
C:/Users/x/miniconda3/envs/torch/python run/recommend_persistent.py --ndates 5 --top-n 30
C:/Users/x/miniconda3/envs/torch/python run/recommend_persistent.py --all-boards

# Watchlist ranking
C:/Users/x/miniconda3/envs/torch/python run/rank_watchlist.py --from-date 2026-05-09 --to-date 2026-05-15
```

Predictor choices: `v9`, `gat`, `avg_score`, `union`, `intersection`, `top_union_bottom_intersection`. Output includes board classification (主板/创业板/科创板/北交所) and Chinese stock names. Alpha is a cross-sectional ranking score, not expected return. Scripts do NOT auto-update `data/raw` (except `daily_top10.py`).

## Data and cache rules

- `data/raw/` — training/backtest dataset. `data/tracking_raw/` — daily post-close tracking data. Never mix.
- Rebuild caches after changing data semantics: `rm -f cache/cross_section_v13_*_meta.pkl cache/cross_section_v13_*.dat cache/fundamental_features*.parquet`
- Cache files per build:
  - `*_meta.pkl` — metadata (dimensions, indices, file paths)
  - `*_feat.dat` — raw aggregated features float32 (129 dims)
  - `*_risk.dat` — raw risk/market/macro float32 (59 dims)
  - `*_ret.dat` — forward return sequences float32 (10 horizons)
  - `*_X_norm.dat` — precomputed normalized X int16 scale=1000 (250 dims)
  - `*_risk_full.dat` — precomputed normalized risk int16 (59 dims)
  - `*_y_norm.dat` — precomputed labels int16
  - `*_y_seq_norm.dat` — precomputed multi-horizon labels int16
- Cache key does NOT encode `test_stocks` or `max_stocks`：same key for test and full runs → delete cache when switching.
- `test_mode=True` truncates loaded CSVs to `test_stocks`.

## Current feature structure

### X features (264 dims, exp-006 计划)
- 高频量价 base(12)+tech(11)=23 → 5-way agg (last/sma5/sma20/vol5/vol20) = 115
- 低频基本面 extra(7) → last+qoq only = 14（sma/vol 对季频数据无意义）
- agg total: 115 + 14 = 129
- rank: 对全部 129 个 agg 特征做截面排序 = **129**（之前仅高频 115）
- ind_rel = 6
- Total `X`: 129 + 129 + 6 = **264**
- Extra: fund(roe,revenue_yoy) + shareholder(sh_conc_ratio,sh_per_capita_ratio) + restricted(next_inv_days,next_ratio,mv_ratio_90d) = 7

**同时砍掉 FiLM MLP**：低频 raw+rank 直接作为 extra 通道送入 FeatureGrouper → Transformer 自己学。FiLM 强制低频只能"调控"高频解读方式，但 ROE、营收增速等本身携带独立截面信号，应在股票间比较排序。

### Risk features (59 dims)
- stock(6): log_volume, vol_60d, ret_20d, ret_5d, volume_ratio, amplitude
- market(50): 宽基指数(16) + 宽度(3) + 申万行业指数收益(31)
- macro(3): north_net_zscore, margin_balance_change, pmi_zscore
- ~~industry onehot(83)~~ — 已移除。行业信息由 `industry_ids` 单独传递（embedding + GAT 边）

### Normalization (2026-05-26 改动)
- **X features**: winsorize(1%/99%) + z-score + clip(±4)。替代旧 MAD 归一化。
  - agg 特征（129 维）：winsorize + z-score（解决低频长尾压缩问题）
  - rank 特征（129 维）：z-score only（已均匀 [0,1]，不 winsorize 避免抹掉极端排序）
  - ind_rel 特征（6 维）：z-score only（已均值中心化）
- **Risk**: 前 6 列 z-score + clip(±4)，其余不变
- **存储**: int16 × 1000（均匀 0.001 精度，max error=0.001, 0% 超限）

### Dataset 构建参数
- `valid_times`: `range(max(seq_len, 80), num_dates - max_horizon)` — vol_60d 需要 60 天 + sma20 聚合 20 天 = 最少 80 天历史
- `min_stocks_per_time`: 从 DataConfig 读取，默认 30
- `PrecomputedMemmapDataset` 初始化时预过滤有效股票 < min_stocks 的截面

## Label semantics

- `sample['raw_y']` — unstandardized forward return for main target horizon
- `sample['y']` — cross-section standardized training label
- `sample['y_seq']` — multi-horizon forward return sequence
- Backtests should use `raw_y` or recomputed returns, not standardized `y`

## Point-in-time data constraints

- Fundamental factors keyed by announcement date, not report `end_date` alone.
- PMI available from following month; northbound z-score uses shifted rolling mean/std.
- Macro features stay in `risk/regime` channel, not per-stock `X`.

## Industry embedding and GAT

- `num_industries` inferred dynamically (not hard-coded). Unknown industries = `-1`.
- GAT branch builds same-industry edges on CPU (`max_edges_per_stock=5`).
- `run/train.py --model gat` uses `keep_ratio=0.7`; `--model gat_v2` uses `keep_ratio=0.5` and `resume=False`.

## Refactoring summary

| Phase | Date | Detail |
|-------|------|--------|
| Backtest refactor | 05-15 | Added `predictors.py`, `runtime.py`, `runners.py`, `reports.py`; slimmed experiment scripts |
| Dead code cleanup | 05-18 | Deleted `model_gat.py`, `risk.py`, 50 `*_feat.parquet`, 4.3GB old backtest results |
| API rate-limit unification | 05-18 | `data/api_utils.py` — single `SafeAPICaller` for 5 call sites |
| Recommendation dedup | 05-18 | `run/recommend_utils.py` — shared board/predictor/scoring/date helpers |
| Tracking split | 05-18 | `backtest/tracking.py` — extracted from 669-line `track_backtest_holdings.py` |
| Constants + path boilerplate | 05-18 | `TRADING_DAYS`, etc. + `setup_project_environment()` in `core/config.py` |
| Directory cleanup | 05-18 | `checkpoints/` + `logs/`, 20 path refs updated, dead config options removed |
| Train unification | 05-19 | `run/train.py --model` replaces 4 scripts (validated, old scripts deleted) |
| Backtest unification | 05-19 | `run/backtest.py --experiment` replaces 4 scripts (CSV parity verified) |
| engine.py extract | 05-19 | `calc_metrics` → `calc_extended_metrics` dedup; 5 report functions → `reports.py`; `_sanitize()` helper added |
| pipeline.py dedup | 05-19 | `_compute_base_features()` + `_load_industry_map()` extracted; train/inference dim parity verified (X=236, risk=170) |

**New files this refactor:** `data/api_utils.py`, `run/recommend_utils.py`, `backtest/tracking.py`, `run/daily_top10.py`, `run/train.py`, `run/backtest.py`

**Deferred (high risk):** Pipeline inference deep dedup, engine optimizer dispatch dedup, engine main loop split.

**Invariant:** Labels, return metric, `entry_day`/hold-window semantics, ADV constraints, portfolio weight construction, and `data/raw` vs `data/tracking_raw` boundary must not change.

## Model experiment branch

Model experiments are isolated in a separate git worktree to avoid breaking the stable main pipeline:

| | Main | Experiments |
|---|------|-------------|
| Path | `C:/Users/x/code/stock_prediction/deepseek_optimized` | `C:/Users/x/code/stock_prediction/deepseek_model_exp` |
| Branch | `master` | `model-experiments` |
| Checkpoints | `checkpoints/` | `checkpoints_exp/` |
| Backtest | `backtest_results_*/` | `backtest_results_exp_*/` |
| Log | — | `experiments.log` |

```bash
cd C:/Users/x/code/stock_prediction/deepseek_model_exp
python run/train.py --model v9 --epochs 25
python run/backtest.py --experiment ensemble
```

Results go to `checkpoints_exp/` and `backtest_results_exp_*/`, never touching the main checkpoint or benchmark data.

## Model experiment log

Current model: `UltimateV7Model` — Multi-group FeatureGrouper (高频5-agg + 低频2-agg, per-group残差) + 4-layer Transformer (dropout=0.35) + 双重X残差 + optional GAT branch (2 blocks, 2-layer GATConv+residual) + CrossIndustryAttention + multi-alpha/horizon heads.

### exp-001: Head LayerNorm + Dropout (DONE, minimal gain)

在 `alpha_heads` 和 `horizon_heads` 的 GELU 前加了 `nn.LayerNorm(hidden_dim)` + `nn.Dropout(0.1)`。

**结论**：提升不大。两个 head 都是微型 MLP（hidden→hidden→1），本身不易过拟合。Transformer 的 dropout=0.35 已是主力正则化。已验证过拟合曲线确认无效。

### exp-002: 行业感知损失 + Spread Loss (DONE, best alpha=0.0940)

复合 loss 改造 `core/train_utils.py` 的 `total_loss_v7`：

**行业感知 IC** (`DataConfig.industry_loss_weight`, 默认 0.1)：
```
main = (1-w) × global_PearsonIC + w × mean(within_industry_PearsonIC)
```

**Top-Bottom spread** (`DataConfig.spread_loss_weight`, 默认 0.001)：
- `top_bottom_spread_loss()` — softmax 加权多头-空头收益差
- temperature 1.0，前 10 epoch 关闭（`spread_delay_epochs=10`），避免早期主导梯度
- 最终权重 0.001，贡献可忽略——spread loss 太容易优化，总是压倒 IC

**权重调优历程**：
- 初版 ind=0.7 spread=0.05 → spread 主导，IC 上不去
- 调整为 ind=0.1 spread=0.001 → epoch 12 alpha_IC=0.0940（GAT 最优）
- 关键教训：**spread 必须极低权重或关闭**，否则模型投机取巧

**配套改动**：
- `CosineAnnealingLR` (eta_min=1e-5) + `lr_warmup_epochs=10`（前 10 epoch 固定 lr）
- `residualize_labels=True` — 标签剥离行业均值+规模效应（`_residualize_labels()` in pipeline.py）
- 训练日志打印 loss 分量（global_ic, within_ic, spread）
- 验证新增 per-horizon top-bottom spread 指标（`topbot_h1/h3/h5/h7`）
- 输出分离：Val IC 和 Val TB 分行打印

**已修复 bug**：
- SW_INDUSTRIES 英文名→中文名，31 个行业指数文件从读不到→正确加载，62 个市场特征从 0→真实数据

### exp-003: 标签残差化 + LR Warmup (DONE, IC slower → reverted)

**Label residualization** (`_residualize_labels()` in `data/pipeline.py`)：
- 标签剥离行业均值+规模效应后，IC 增速明显变慢（0.070→0.076 vs 之前的 0.074→0.094）
- 行业 beta 是真实可交易信号，残差化等于扔掉一块可预测收益
- 专业量化不做标签残差化——行业中性化放在组合构建阶段
- **结论**：已关闭（`residualize_labels=False`），保留代码备查

**LR warmup** (`lr_warmup_epochs=10`)：
- 前 10 epoch 固定 lr=3e-4，之后余弦衰减到 1e-5
- **保留**，让模型早期充分探索

### exp-004: 架构大修 (DONE, 训练中)

多项改动涉及 `core/model.py` + `core/train_utils.py` + `core/config.py`：

**模型架构：**

| # | 改动 | 说明 |
|---|------|------|
| 1 | **Multi-group FeatureGrouper** | 高频23×5 + 低频7×2 两组，各自 per-agg投影+cross-agg attention。per_slot_dim=32，高频160维+低频64维+extra32维=256维 |
| 2 | **FeatureGrouper 组内残差** | 每组 `Linear(group_dim, output_dim)` 投影后加到 attention 输出 |
| 3 | **双重 X 残差** | `input_proj`(X→256) 加在 FeatureGrouper 输出；`trans_input_proj`(X→256) 加在 Transformer 入口 |
| 4 | **Industry embedding 残差** | `h = h + industry_proj(cat(h, ind_emb))`，不再全量替换 |
| 5 | **移除 FiLM** | 低频特征进 FeatureGrouper 低频组，Transformer 自己学跨特征关系 |
| 6 | **Head Dropout 移除** | alpha_heads 和 horizon_heads 只保留 LayerNorm，Dropout(0.1) 删除 |

**GAT 架构（V9 训练时 `use_gat=False`，以下仅 GAT 模式生效）：**

| # | 改动 | 说明 |
|---|------|------|
| 7 | GAT 3层残差 | ①GATConv×2+residual ②CrossIndustryAttention+residual ③整体GAT residual=`fallback+x`，保留Transformer信号 |
| 8 | 训练时边缓存 | `_edge_cache` LRU 512，GPU tensor hash |
| 9 | GAT 输入读 Transformer 输出 | `trans1→GAT1→trans2→GAT2` 串联 |
| 10 | 2个 GAT Block | suffix='1'/'2' 独立参数 |
| 11 | Cross-Industry Attention | 按行业pool→跨行业MHA→逐股票gate，残差注入 |

**正则化参数：**

| 参数 | 值 |
|------|:---:|
| Transformer dropout | 0.35 |
| Transformer 层数 | 4 |
| FeatureGrouper cross-agg dropout | 0.1 (高频) / 0 (低频) |
| GATConv dropout | 0.2 |
| Head dropout | 无 |
| weight_decay | 2e-3 |
| grad_clip | 0.2 |

**训练性能：**

| 参数 | 值 |
|------|:---:|
| AdamW fused | True (CUDA kernel 融合, ~5-10% 提速) |
| GPU cache cleanup | 关闭 (cleanup_cache_interval=0, 模型小无需) |
| LR | 1e-4, warmup=5, CosineAnnealing to 1e-5 |

**数据流（V9 模式）：**
```
X(250) ──input_proj──→ (256) ──────────────────────┐
X(250) ──FeatureGrouper──→ (256) ──┐               │
           组内: + group_residual    │               │
           ├─高频组 23×5 → 160      │               │
           ├─低频组 7×2  →  64      │               │
           └─extra   121  →  32      │               │
           └── (+) ──────────────────┘               │
                    └── (+) ──→ ind_emb (残差)       │
                    └── (+) ──→ rank_emb             │
                    └── (+) ──→ trans_input_proj ────┘
                                    │
                              4层 Transformer (dropout=0.35)
                                    │
                              Alpha/Horizon heads
```

### 财报数据下载

独立脚本 `scripts/download_fundamentals_akshare.py`：
- 用 akshare 免费接口（`stock_profit_sheet_by_report_em` + `stock_balance_sheet_by_report_em`）
- 多线程并行（`--workers 12`），关闭 tqdm 避免冲突
- 断点续传（每 500 只自动保存 parquet）
- `--update` 模式：仅更新 90 天内未刷新的股票
- 输出 `cache/fundamental_features_akshare.parquet`

用法：
```bash
C:/Users/x/miniconda3/envs/torch/python scripts/download_fundamentals_akshare.py              # 全量
C:/Users/x/miniconda3/envs/torch/python scripts/download_fundamentals_akshare.py --update      # 增量
C:/Users/x/miniconda3/envs/torch/python scripts/download_fundamentals_akshare.py --test 10     # 测试
```

**当前实验配置**（`C:/Users/x/code/stock_prediction/deepseek_model_exp`）：
```
industry_loss_weight=0.1
spread_loss_weight=0.001, spread_temperature=1.0, spread_delay_epochs=5
residualize_labels=False
lr_warmup_epochs=5
transformer_dropout=0.35
n_transformer_layers=4
early_stop_patience=5
use_fundamental_features=True
use_shareholder_features=True
use_restricted_features=True
use_fused_adam=True
cleanup_cache_interval=0
```

### 特征维度优化 (2026-05-25)

四项改动：

**1. 删除 `sw_*_available`（31个死特征）**
- 同一截面所有股票值完全相同，对排序零贡献
- N_MARKET 81→50，risk 170→**139**

**2. 按频率拆分聚合方式**
- 高频量价 (23) → 5-way agg (last/sma5/sma20/vol5/vol20) = 115
- 低频基本面 (7) → last + qoq only = 14（sma/vol 对季频数据无意义）
- qoq = 当前窗口最新值 - 10天前值，捕获变化方向
- agg total: 115+14=**129**

**3. rank 仅对高频**
- 低频 rank 无意义（FiLM 用 raw 值），rank 150→115
- X = 129(agg) + 115(rank) + 6(ind_rel) = **250**

**4. FiLM 调制（pre + post transformer）**
- 低频 14维 → MLP(14→64→128→1024) → γ₁,β₁,γ₂,β₂
- Pre-transformer: `h = h × γ₁ + β₁`（低频调控量价信号解读）
- Post-transformer: `h_out = h_out × γ₂ + β₂`（低频调控市场共识解读）
- 改动：`core/model.py` 的 `__init__` + `forward`（+15行）

维度汇总：
| | 改前 | 改后 |
|---|------|------|
| agg | 150 | 129 |
| rank | 150 | 115 |
| X | 306 | 250 |
| risk | 170 | 139 |

GAT 兼容：MemmapDataset 支持 `risk_trim` 参数，GAT 训练时 risk 自动截到 `regime_dim=59`。

### 股东户数特征 (2026-05-24)

新增 `data/shareholder_features.py` — PIT 安全的筹码集中度因子：
- 数据源：akshare `stock_zh_a_gdhs`，53 个季度（2013Q1~2026Q1），202k 行，5441 只股票
- PIT 对齐：按公告日期过滤，公告日前数据不可见
- 两个特征：`sh_conc_ratio`（筹码集中度）、`sh_per_capita_ratio`（户均持股比例）
- 接入方式：`DataConfig.use_shareholder_features=True`
- 缓存文件：`cache/shareholder_features.parquet`

### 数据管道 + 防过拟合修复 (2026-05-26)

经过对数据构建和训练代码的全面审查，完成了以下修复和改进：

**数据质量修复：**

1. **MAD → Win+Z 归一化**：`_normalize_and_assemble` 改用 winsorize(1%/99%) + z-score + clip(±4)。旧 MAD 归一化对低频特征有长尾压缩问题（`mv_ratio_90d` 97% 值挤在 0，`sh_per_capita_ratio` 85% 为 0）。Win+Z 分特征组处理：agg 用 winsorize+z-score，rank/ind_rel 用 z-score only（rank 已均匀分布，winsorize 会抹掉极端排序信号）。

2. **valid_times 修正**：从 `range(seq_len, ...)` 改为 `range(max(seq_len, 80), ...)`。vol_60d 需要 60 天 + sma20 聚合 20 天 = 最少 80 天历史。旧代码 t=40→79 全是空截面（之前静默，现在预过滤后报 0 过滤）。

3. **砍掉 risk 中 83 维 industry one-hot**：行业信息已由 `industry_ids` 单独传递（embedding + GAT 边），one-hot 从来没被模型用到（被 `risk[..., :regime_dim]` 裁掉）。risk 从 139→59 维，省 ~5.6GB 磁盘。

4. **risk 因子从 3→6 维**：新增 `ret_5d`（反转）、`volume_ratio`（换手率代理）、`amplitude`（振幅）。risk 结构：stock(6)+market(50)+macro(3)=59。

**数据存储优化：**

5. **float16 → int16 × 1000 存储**：均匀 0.001 精度，max error=0.001，0% 超限，信噪比 3171x。比 float16 更可控（float16 精度不均匀，边缘差）。

6. **预计算截面并行化**：`_precompute_all` 用 ThreadPoolExecutor（8-12 workers）并行处理日期。NumPy 操作释放 GIL，实测加速 ~6x。

7. **矩阵填充并行化**：`_fill_one_stock` 和收盘价矩阵也用 ThreadPoolExecutor，~3x 加速。

**训练防过拟合（exp-005）：**

8. **dropout 0.35→0.5**（`DataConfig.transformer_dropout`）
9. **Transformer 层 4→2**（`DataConfig.n_transformer_layers`），参数 4.0M→2.5M
10. **LR warmup 10→5**（`DataConfig.lr_warmup_epochs`），lr 保持 3e-4
11. **Early stopping**（`DataConfig.early_stop_patience=5`）

**代码质量修复：**

12. `_precompute_all` 的 `min_stocks` 从硬编码 30 改为从 config 读取
13. `current_arch` 补全 `low_feat_dim`，防止配置变更时 checkpoint 恢复静默失败
14. 缓存恢复验证所有 `.dat` 文件存在
15. Scheduler 恢复不兼容时 catch 异常（`CosineAnnealingLR` vs `LambdaLR`）

**当前实验配置**（`C:/Users/x/code/stock_prediction/deepseek_model_exp`）：
```
industry_loss_weight=0.1
spread_loss_weight=0.001, spread_temperature=1.0, spread_delay_epochs=5
residualize_labels=False
lr_warmup_epochs=5
transformer_dropout=0.5
n_transformer_layers=2
early_stop_patience=5
use_fundamental_features=True
use_shareholder_features=True
use_restricted_features=True
```

### Prioritized next ideas

**V9 优化（exp-006 计划中）：**

0. **低频 rank + 砍 FiLM → Transformer 自己学** — rank 扩展到全部 129 agg 特征（含低频 14），低频 raw+rank 直接进 FeatureGrouper。FiLM 强制低频只能调控高频，限制了低频特征的独立截面信号。X: 250→264 维

**GAT 继续优化：**

1. **Industry Super-Node** — 每行业一个虚拟节点，行业内股票双向连 Super-Node，Super-Node 间全连接（83×83）。信息跨行业需经过 Super-Node 中转
2. **Industry Embedding 边** — 用 learnable embedding 相似度定义跨行业连接。但需谨慎：相似度是模型自己学的，可能循环论证

**其他方向：**

3. **移除 rank embedding** — `_build_rank_embed` 对截面排序特征做嵌入，可能引入前视偏差
4. ~~**FiLM 门控融合**~~ — exp-004 已实现
5. ~~**Cross-Industry Attention**~~ — exp-004 已实现
6. ~~**限售解禁特征**~~ — 已实现（`data/restricted_features.py`）

### 限售解禁特征 (2026-05-24)

新增 `data/restricted_features.py` — 未来解禁压力因子：
- 数据源：akshare `stock_restricted_release_detail_em`，2010~2026 全覆盖
- 三个特征：`restricted_next_inv_days`（1/(1+距下次解禁天数)）、`restricted_next_ratio`（下次解禁稀释比例）、`restricted_mv_ratio_90d`（未来90天累计稀释）
- 接入方式：`DataConfig.use_restricted_features=True`
- 缓存文件：`cache/restricted_features.parquet`
- 增量更新：`python data/restricted_features.py --update`

### 增量更新接口

所有数据源支持增量更新（`--update`），避免每次全量重拉：

| 数据 | 更新命令 |
|------|---------|
| 个股日线 | `python data/update_daily.py` |
| 全量基础数据 | `python data/update.py` |
| 财报基本面 | `python scripts/download_fundamentals_akshare.py --update` |
| 股东户数 | `python data/shareholder_features.py --update` |
| 限售解禁 | `python data/restricted_features.py --update` |

## Recent fixes (experiment branch)

- **`data/market_features.py:101`**: 新高比例窗口从 `t-19:t+1`（含当日）改为 `t-20:t`（不含当日），消除前视偏差。
- **`data/update_daily.py:89`**: `safe_to_csv` 条件从 `len(existing) > max(len(df), min_rows)` 改为 `len(existing) >= len(df)`，修复覆盖丢行。
- 主线已 resolved 的其他 bug（括号/device/前缀/KeyError/tracking_raw/max_workers/列表类型）本分支同样已修。

### Codex review fixes (2026-05-26)

**训练/回测链路：**
- `build_cross_section_dataset()` 现在返回 memmap metadata 时，`backtest/runtime.py` 和 `backtest/engine.py` 会通过 `samples_from_precomputed_metadata()` 转回旧 sample list，修复回测/推荐入口不兼容。
- checkpoint 加载优先读取 `arch_config`（`low_feat_dim`、`n_layers`、`dropout`、`num_industries` 等），不再硬编码旧 V9 架构。
- 实验分支默认 checkpoint 路径统一到 `checkpoints_exp/`；smoke test 可用 `--output-dir checkpoints_exp_smoke`。
- `run/train.py --device cpu/cuda` 现在会传入 `train_model()`，不再被训练函数内部自动 CUDA 覆盖。
- `cfg.low_feat_dim` 对所有模型都会设置，避免低频特征开关变化时模型切片错位。

**缓存规则：**
- cache key 现在包含 `test_mode/test_stocks/max_stocks`，修复小样本 smoke test 误用全量缓存的问题。
- 因限售解禁 PIT 生效日修正、risk/feature schema、预计算语义变更，训练前必须清理 `cache/cross_section_*` 和 `cache/inference_matrices_cache.pkl` 让缓存重建。

**PIT / 数据安全：**
- AkShare 财报缺失公告日时不再用报告期末日生效，改用保守披露延迟：Q1/Q3 +45d，半年报 +60d，年报 +120d。
- 限售解禁缓存新增/自动补 `effective_date = release_date - 30d`，历史截面只使用已披露窗口内的未来解禁事件。
- `data/update_daily.py` 断点续传修复：失败股票不再写入 updated progress，skip 统计不再重复累加。

**新增诊断/基线：**
- `scripts/data_quality_report.py --universe`：输出 raw/PIT 覆盖率、ST、短上市、零成交、涨跌停频率等诊断。
- `run/baseline_eval.py --include-lgb`：Ridge/简单因子/LightGBM baseline，同一数据集输出 IC、Rank IC、Top-Bottom spread。
- 小样本 baseline（30只）参考：LightGBM `mean_ic=0.07266, rank_ic=0.08197, topbot=0.18272`；Ridge `mean_ic=0.05975`。

**回测约束：**
- `run/backtest.py` 新增 `--exclude-st --min-listing-days N --block-limit-trades --capacity-values 1e7,5e7,1e8,5e8`。
- 底层回测输出 Universe 过滤比例、涨跌停买卖阻断比例、各容量档 fill/turnover/impact。默认不传参数时保持旧语义。

## Known issues

1. **正式实验模型尚未重训** — 目前只有 `checkpoints_exp_smoke/` 的 1 epoch smoke checkpoint；正式结果需要清理 cross-section 缓存后训练 `checkpoints_exp/`。

### Design notes

2. **`data/pipeline.py` 推理/训练 CSV 长度阈值不同** — 训练 `seq_len + max_horizon + 50`，推理 `seq_len + 50`。故意为之。
