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
- Rebuild caches after changing data semantics: `rm -f cache/cross_section_*.pkl cache/fundamental_features*.parquet`
- Cache names encode feature groups, stock universe, `seq_len`, `target_horizon`, `max_horizon`, min stocks, normalization mode.
- `test_mode=True` truncates loaded CSVs to `test_stocks`.

## Current feature structure

- Base feature dim: 23 → aggregated (5 ways): 115 → rank features: 115 → industry-relative: 6 → Total `X`: 236
- `risk`: stock-level (3) + market (81) + macro (3) + industry one-hot → `get_regime_dim(cfg)` for slicing

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

Current model: `UltimateV7Model` — FeatureGrouper + 4-layer Transformer (dropout=0.35) + optional GAT branch + multi-alpha/horizon heads.

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

### exp-004: GAT 架构优化 (IN PROGRESS)

三项改动（`core/model.py`）：

1. **GAT 2 层 + 残差**：`h → GATConv(4 heads) → GELU → Dropout → GATConv(4 heads) + residual → GELU → LayerNorm`
   - heads 2→4，丰富邻居聚合
   - 2 层 → 行业消息 2 跳传递（供应商→客户→竞争对手）

2. **训练时边缓存**：`build_industry_edges` 统一的 `_edge_cache`（max 512 entries）
   - 以前只 eval 缓存，训练每 epoch 重复构建
   - 同截面不同 epoch 遇到同一批行业+股票组合 → 命中缓存

3. **GAT 输入改为 Transformer 输出**：`gat_out = self._gat_forward(trans_out, ...)`
   - 以前 GAT 读 pre-transformer `h`，与 Transformer 并行
   - 现在 GAT 读 `trans_out`（已含跨股票注意力上下文）

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

当前实验配置（`C:/Users/x/code/stock_prediction/deepseek_model_exp`）：
```
industry_loss_weight=0.1
spread_loss_weight=0.001, spread_temperature=1.0, spread_delay_epochs=10
residualize_labels=False
lr_warmup_epochs=10
use_fundamental_features=False  # 等 akshare 下载完再开
```

### Prioritized next ideas

**高优先级：**

1. **移除 rank embedding** — `_build_rank_embed` 对截面排序特征做嵌入，可能引入前视偏差
2. **FiLM 门控融合** — 市场状态向量门控调控 Transformer 每层输出

**中优先级：**

3. **增加 GAT 层数 + 残差连接** — 当前 GAT 仅 1 层
4. **Transformer 深度实验**（2/4/6层）

## Known issues

### Resolved (2026-05-19)

1. **`core/train_utils.py:350` 括号不匹配** → **FIXED.**
2. **`run/train.py` `--device cuda` 无 GPU 时崩溃** → **FIXED.** Added `torch.cuda.is_available()` check.
3. **`run/recommend_utils.py:49` 打印前缀信息格式** → **FIXED.** Joins prefix strings with `、`.
4. **`backtest/layered_engine.py` KeyError 风险** → **FIXED.** Changed to `diag.get(...)`.
5. **`run/daily_top10.py` 向 `data/raw` 写入日更数据** → **FIXED.** Changed to `data/tracking_raw`.
6. **`data/update.py` `TushareProLite` 死参数 `max_workers`** → **FIXED.** Removed.

### Active

7. **`run/train.py` v9/GAT 的 `num_industries` 计算方式不同** — v9 从 `risk_dim - regime_dim` 推算（one-hot 列数），GAT 从 `industry_ids.max()+1` 推算（离散 ID 个数）。若某些行业有 one-hot 列但训练数据中没有对应股票（被过滤掉），两者会不一致，导致 `nn.Embedding` 错位。
8. **诊断列表填充类型风险** → **FIXED (2026-05-19).** `reports.py` 中 `isinstance(v, (list, np.ndarray))` 统一接受两种类型，并用 `list(v)` 保证 list concatenation。
9. **`data/update_daily.py:89` `safe_to_csv` 条件缺陷可能丢失数据** — `if len(existing) > max(len(df), min_rows)` 条件有缺口：当 `len(existing)` 介于 `len(df)` 和 `min_rows` 之间时，合并被跳过，文件被覆盖为更少的行。例如 `existing=150` 行、`df=100` 行、`min_rows=200` → 合并跳过 → 150 行被覆盖为 100 行 → 丢失 50 行。应改为 `if len(existing) > len(df)` 无条件合并保护。
10. **`data/market_features.py:101` 新高比例窗口含当日** — `close_matrix[valid_c, t-19:t+1]` 包含 t 日自身，使当前收盘价总能匹配 20 日最高价（含当日）。应改为 `t-20:t` 排除当日，避免微小前视偏差。

### Design notes (intentional)

11. **`data/pipeline.py` 推理和训练 CSV 长度阈值不一致** — 训练需要 `seq_len + max_horizon + 50`（需要未来标签），推理只需 `seq_len + 50`。故意为之。
