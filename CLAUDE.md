# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## PowerShell 编码注意事项

**所有 PowerShell 中读写 `.py` / `.md` 文件的操作必须加 `-Encoding UTF8`。**

```powershell
# 正确
Get-Content -LiteralPath "file.py" -Raw -Encoding UTF8 | Set-Content -LiteralPath "file.py" -Encoding UTF8

# 错误：中文 Windows 默认 GBK，会把 UTF-8 文件读坏
Get-Content -LiteralPath "file.py" -Raw | Set-Content -LiteralPath "file.py" -Encoding UTF8
```

Python 3 自身按 UTF-8 读写即可。不要用 PowerShell 默认编码批量处理源码。

## Environment and commands

Use the existing Miniconda environment on this machine:

```bash
F:/miniconda3/envs/pytorch/python -m py_compile core/config.py core/model.py core/train_utils.py data/pipeline.py data/fundamental_factors.py data/macro_factors.py data/update.py data/update_daily.py backtest/engine.py backtest/ensemble.py backtest/risk.py backtest/layered_engine.py backtest/runtime.py backtest/runners.py backtest/predictors.py backtest/reports.py run/train_v9.py run/train_gat.py run/train_gat_v2.py run/train_legacy.py run/hyper_search.py run/recommend_daily.py run/backtest_v9_gat_ensemble_modes.py run/backtest_v9_gat_intersection_modes.py run/backtest_v9_gat_concentrated_modes.py run/backtest_layered_holdings.py
F:/miniconda3/envs/pytorch/python run/train_v9.py
F:/miniconda3/envs/pytorch/python run/train_gat.py
F:/miniconda3/envs/pytorch/python backtest/engine.py --model-type both
F:/miniconda3/envs/pytorch/python run/hyper_search.py
F:/miniconda3/envs/pytorch/python run/backtest_v9_gat_ensemble_modes.py
F:/miniconda3/envs/pytorch/python run/backtest_v9_gat_concentrated_modes.py
F:/miniconda3/envs/pytorch/python run/recommend_daily.py --as-of latest --predictor avg_score --top-n 20
F:/miniconda3/envs/pytorch/python run/recommend_daily.py --as-of latest --predictor avg_score --code 000001.SZ
```

The environment currently has PyTorch `2.7.1+cu118` with CUDA available on an RTX 2060-class 6GB GPU.

If dependencies need to be installed:

```bash
F:/miniconda3/envs/pytorch/python -m pip install -r requirements.txt
```

Data update commands require a Tushare token supplied outside source code:

```bash
export TUSHARE_TOKEN="65209d394f51051f94f8a9eeeb3396048121ecf94080eda0e33d06e5"
F:/miniconda3/envs/pytorch/python data/update.py
F:/miniconda3/envs/pytorch/python data/update_daily.py
F:/miniconda3/envs/pytorch/python data/update.py --init
```

On Windows CMD use `set TUSHARE_TOKEN=65209d394f51051f94f8a9eeeb3396048121ecf94080eda0e33d06e5` instead of `export`.

## Architecture overview

- `run/` contains entry scripts. `run/train_v9.py` is the main Transformer training entry; `run/train_gat.py` and `run/train_gat_v2.py` train the GAT variant; `run/train_legacy.py` is a legacy backup entry; `run/hyper_search.py` runs LightGBM/backtest parameter search.
- `core/model.py` defines `UltimateV7Model`: `FeatureGrouper` + cross-stock Transformer + optional industry GAT branch + multi-horizon heads.
- `core/train_utils.py` defines `CrossSectionDataset`, collate functions, loss/evaluation, `get_regime_dim()`, and `train_model()`.
- `run/recommend_daily.py` loads the latest available cross-section, scores it with V9/GAT ensemble predictors, and prints a ranked recommendation table with optional single-stock query.
- `run/backtest_v9_gat_concentrated_modes.py` runs 5%/3% concentrated ensemble backtests (narrower stock selection than the default 10%).
- `data/pipeline.py` builds cross-sectional samples from `data/raw/`, joins market/fundamental/macro factors, creates labels, and returns `(train_samples, val_samples)`. Also exposes `build_inference_sample(config, stock_universe=None, as_of_date=None)` for label-free latest/as-of cross-section inference.
- `data/fundamental_factors.py` and `data/macro_factors.py` must remain point-in-time (PIT).
- `backtest/engine.py` trains/loads multi-horizon LightGBM models, loads DL checkpoints, and runs production backtests through `DLPredictor` / `LGBPredictor`.
- `backtest/ensemble.py` handles stacking/blending experiments.
- `backtest/runners.py` provides `ProductionBacktestParams`, `LayeredBacktestParams`, `run_production_backtest_once()`, and `run_layered_backtest_once()`.
- `backtest/runtime.py` provides `build_v9_backtest_config()`, `load_backtest_runtime()`, and `load_dl_predictor()` / `load_v9_gat_predictors()`.
- `backtest/predictors.py` provides `V9GATIntersectionPredictor` and `V9GATEnsemblePredictor`.
- `backtest/reports.py` provides `save_summary_csv()` for tabular summary output.

## Architecture refactor notes (2026-05-15)

First-stage low-risk refactor has been completed to reduce duplicated V9/GAT backtest experiment code without changing strategy semantics or historical comparability.

Completed first-stage changes:

- Added `backtest/predictors.py` for reusable V9/GAT combined predictors:
  - `V9GATIntersectionPredictor`
  - `V9GATEnsemblePredictor`
- Added `backtest/runtime.py` for shared V9/GAT backtest setup:
  - default V9 backtest `DataConfig`
  - cross-section dataset loading
  - price/volume loading
  - V9 and GAT checkpoint loading
- Added `backtest/runners.py` for a single production backtest wrapper:
  - `ProductionBacktestParams`
  - `metrics_from_returns()`
  - `run_production_backtest_once()`
- Added `backtest/reports.py` for summary CSV saving and printing.
- Slimmed these entry scripts so they mostly define experiment configs and call shared modules:
  - `run/backtest_v9_gat_intersection_modes.py`
  - `run/backtest_v9_gat_ensemble_modes.py`

Important invariant: this refactor must not change labels, return metric, `entry_day` / hold-window semantics, ADV execution constraints, portfolio weight construction, or the `data/raw` vs `data/tracking_raw` boundary.

Validation performed after the first-stage refactor:

```bash
F:/miniconda3/envs/pytorch/python -m py_compile backtest/predictors.py backtest/runtime.py backtest/runners.py backtest/reports.py run/backtest_v9_gat_intersection_modes.py run/backtest_v9_gat_ensemble_modes.py
F:/miniconda3/envs/pytorch/python -c "from backtest.runtime import build_v9_backtest_config; c=build_v9_backtest_config(); print(c.use_technical_features, c.use_market_features, c.use_macro_features, c.min_stocks_per_time, c.target_horizon, c.seq_len, c.max_horizon)"
F:/miniconda3/envs/pytorch/python -c "from backtest.predictors import V9GATIntersectionPredictor, V9GATEnsemblePredictor; from backtest.runtime import build_v9_backtest_config; from backtest.runners import ProductionBacktestParams; print('ok')"
```

Expected config check output:

```text
True True True 30 5 40 10
```

Completed second-stage refactor (2026-05-15):

- Added `LayeredBacktestParams` and `run_layered_backtest_once()` to `backtest/runners.py` without changing the existing overwrite runner.
- `run/backtest_layered_holdings.py` now reuses `backtest/runtime.py` for config/data/model loading and `backtest/runners.py` for the layered runner wrapper. CLI defaults and output labels are preserved.
- Added `build_inference_sample()` to `data/pipeline.py` for label-free latest/as-of cross-section inference (no future returns required, same feature construction as training).
- Added `run/recommend_daily.py`: daily stock recommendation CLI + single-stock alpha/rank/percentile query.
- Added `run/backtest_v9_gat_concentrated_modes.py`: 5%/3% concentrated V9/GAT ensemble backtests, output to `backtest_results_concentrated/`.

Validation:

```bash
F:/miniconda3/envs/pytorch/python -m py_compile backtest/runtime.py backtest/runners.py data/pipeline.py run/backtest_layered_holdings.py run/recommend_daily.py run/backtest_v9_gat_concentrated_modes.py
F:/miniconda3/envs/pytorch/python -c "from backtest.runtime import build_v9_backtest_config; c=build_v9_backtest_config(target_horizon=5); print(c.use_technical_features, c.use_market_features, c.use_macro_features, c.min_stocks_per_time, c.target_horizon, c.seq_len, c.max_horizon)"
F:/miniconda3/envs/pytorch/python -c "from data.pipeline import build_inference_sample; from backtest.runners import LayeredBacktestParams, run_layered_backtest_once; print('ok')"
```

Expected config output: `True True True 30 5 40 10`

Recommended next architecture stages:

1. Split shared portfolio/date/index-return helpers out of `backtest/engine.py` and `backtest/layered_engine.py` only after result equivalence is verified.
2. Split `run/track_backtest_holdings.py` into smaller units for tracking-data update, holdings table generation, tracking PnL calculation, and report output. Keep the safety guard that refuses writing tracking updates into `data/raw`.
3. Consider a single lightweight experiment CLI only after the duplicated scripts are stable; do not merge all experiments into one complex command before result parity is established.

## Training workflow (critical)

Always check for existing training processes before starting a new training run. Multiple concurrent training processes will compete for GPU memory.

```bash
ps aux | grep -E "python.*(train|run)" | grep -v grep
```

If starting a fresh run, stop any old training process first and delete old logs if needed:

```bash
rm -f *.log
```

Do not start a new full training run while another training process is still running.

## Training configuration

- **AMP must stay disabled.** Mixed precision causes `Train Loss: nan` with the correlation-based loss. Use `use_amp=False`; `train_model()` defaults to AMP off.
- On RTX 2060 / 6GB, V9 Transformer training should use `batch_size=2, accum_steps=8` (effective batch 16).
- On RTX 2060 / 6GB, GAT training should use `batch_size=4, accum_steps=4, keep_ratio=0.7, val_batch_size=1` by default. This keeps enough industry graph structure while avoiding shared-GPU-memory pressure.
- `run/train_v9.py` defaults to `test_mode=False` for full training. For smoke tests, set `cfg.test_mode=True` and reduce `cfg.test_stocks`.
- GAT training uses a separate checkpoint `ultimate_v7_gat_best.pt`; V9 Transformer uses `ultimate_v7_best.pt`.
- If label logic, feature schema, PIT factor logic, market/macro/fundamental factor logic, or industry IDs change, old checkpoints should be considered invalid and models should be retrained.

## Data and cache rules

Runtime artifacts are gitignored: `data/raw/`, `data/tracking_raw/`, `cache/`, logs, model weights, LightGBM model folders, and backtest outputs.

Daily post-close tracking data should be written to `data/tracking_raw/`, not `data/raw/`. `data/raw/` is the training/backtest dataset; mixing daily live updates into it changes the train/validation universe and can invalidate cached datasets. `data/tracking_raw/` only needs data from the trading day after the train/validation dataset ends; use it for holding/PnL tracking, not model training.

Rebuild caches after changing data semantics:

```bash
rm -f cache/cross_section_*.pkl
rm -f cache/fundamental_features*.parquet
rm -f cache/north_flow.csv cache/margin_balance.csv cache/pmi_pit_v2.csv
```

Current cache behavior:

- `data/pipeline.py` uses short descriptive cross-section cache names, e.g. `cross_section_v13_config_key_tech_market_macro_all_s40_t5_h10_min30_mad.pkl`.
- Cache names encode enabled feature groups, stock universe, `seq_len`, `target_horizon`, `max_horizon`, minimum stocks per cross-section, and normalization mode.
- `test_mode=True` actually truncates loaded CSV files to `test_stocks`.
- `data/fundamental_factors.py` fundamental cache keys include `CACHE_VERSION`, `FACTOR_SCHEMA_VERSION`, and a digest of `FACTOR_SCHEMA`.
- Macro factor files are cached separately. If PIT/z-score logic changes, delete the corresponding macro CSV cache so it is recomputed.

## Point-in-time data constraints

- Fundamental factors must be keyed by announcement/effective date, not report `end_date` alone.
- `merge_to_daily()` should only forward-fill data that was already published by the current daily date.
- PMI is treated as available from the following month and uses historical expanding statistics shifted by one period.
- Northbound flow z-score uses rolling mean/std shifted by one day; do not include the current day's value in its own normalization baseline.
- Macro features are market-wide values and should stay in the `risk/regime` channel, not per-stock `X`, because cross-sectional standardization/ranking would erase or distort identical per-stock values.

## Current feature structure

Base stock features currently include momentum/volatility, volume, microstructure, and normalized technical indicators. With technical features enabled the usual dimensions are:

- Base feature dimension: 23
- Aggregations: `N_AGGS = 5` (`last`, `sma5`, `sma20`, `vol5`, `vol20`)
- Aggregated features: `23 × 5 = 115`
- Rank features: `115`
- Industry-relative features: `len(INDUSTRY_REL_FEATURES) = 6`
- Total `X` dimension: `236`

`risk` contains:

- First 3 stock-level risk/style fields (`size`, `vol`, `mom`; `size` is currently volume-derived, not market-cap-derived)
- Market-wide regime features (`N_MARKET`, currently 81 when market features include broad indices, breadth, Shenwan industry returns, and availability masks)
- Optional macro features: 3 (`north`, `margin`, `PMI`)
- Industry one-hot features after `get_regime_dim(cfg)`

Use `get_regime_dim(cfg)` instead of hard-coding `87` when slicing `risk`.

## Industry embedding and GAT

- Samples include `industry_ids`; unknown industries are `-1`.
- `UltimateV7Model` uses `nn.Embedding(num_industries + 1, 16)`. The final index is the unknown-industry bucket.
- `num_industries` should be inferred dynamically from the risk industry one-hot dimension or observed non-negative `industry_ids`; do not hard-code `82` or `83`.
- `core/model.py` GAT branch builds same-industry edges on CPU with `max_edges_per_stock=5` by default and uses a bounded eval edge cache.
- `run/train_gat.py` currently uses `keep_ratio=0.7`, `batch_size=4`, `accum_steps=4`, and `val_batch_size=1` on 6GB GPUs. `keep_ratio=0.7` is the current compromise between preserving industry subgraphs and avoiding 5.7GB+ VRAM peaks.
- Full-universe GAT training loads the cross-section cache into RAM; ~17GB RSS can be a stable plateau. Treat it as a leak only if RSS keeps increasing across epochs instead of staying near that level.
- `run/train_gat_v2.py` is a no-stdout variant using `keep_ratio=0.5` and `resume=False`.

## Backtest system

`backtest/engine.py` supports:

```bash
F:/miniconda3/envs/pytorch/python backtest/engine.py --model-type dl --checkpoint ultimate_v7_best.pt
F:/miniconda3/envs/pytorch/python backtest/engine.py --model-type lgb --lgb-dir models_multi_v9_tech_macro
F:/miniconda3/envs/pytorch/python backtest/engine.py --model-type both
```

Portfolio modes:

```bash
F:/miniconda3/envs/pytorch/python backtest/engine.py --portfolio-mode optimizer
F:/miniconda3/envs/pytorch/python backtest/engine.py --portfolio-mode optimizer_projected
F:/miniconda3/envs/pytorch/python backtest/engine.py --portfolio-mode optimizer_mvo
F:/miniconda3/envs/pytorch/python backtest/engine.py --portfolio-mode simple_ls
F:/miniconda3/envs/pytorch/python backtest/engine.py --portfolio-mode simple_long
```

ADV modes:

- `execution` is the default and recommended mode. It constrains actual fills at execution.
- `both` is a conservative variant.
- `weight_cap` is retained for comparison only and is deprecated because it can understate execution liquidity risk in the optimizer while forcing full execution later.

Backtest return metric is `next_close_to_next_close`. Do not change the `entry_day` / hold window semantics casually; changing it alters historical performance comparability.

### Backtest benchmarks (2026-05-16, current data/code)

Validation sample from `cache/cross_section_v13_config_key_tech_market_macro_all_s40_t5_h10_min30_mad.pkl` (built 2026-05-16), 776 val samples, ~4940 stocks per cross-section, ~2000 tradeable after liquidity filter, 155 rebalances, ~775 active return days.

**Model similarity (776 trading days, unchanged from 05-15):**

| Metric | Mean | Std | Min | 25% | 50% | 75% | Max |
|--------|------|-----|-----|-----|-----|-----|-----|
| Top 10% Jaccard | 0.339 | 0.121 | 0.014 | 0.260 | 0.340 | 0.434 | 0.601 |
| Bottom 10% Jaccard | 0.520 | 0.102 | 0.089 | 0.465 | 0.531 | 0.593 | 0.719 |
| Spearman rank corr | 0.734 | — | — | — | — | — | — |

**Single model comparison (engine default params, top_frac=0.10):**

| Model | Checkpoint | Mode | Raw ann | Raw Sharpe | Raw MDD |
|------|-----------|------|---------|------------|---------|
| V9 | `ultimate_v7_best.pt` | simple_ls | 33.69% | 3.19 | 9.25% |
| V9 | `ultimate_v7_best.pt` | optimizer | 35.36% | 3.44 | 7.24% |
| V9 | `ultimate_v7_best.pt` | simple_long | 24.47% | 1.00 | 21.23% |
| GAT | `ultimate_v7_gat_best.pt` | simple_ls | 37.19% | 3.23 | 9.29% |
| GAT | `ultimate_v7_gat_best.pt` | optimizer | 29.29% | 2.90 | 6.56% |
| GAT | `ultimate_v7_gat_best.pt` | simple_long | 30.14% | 1.23 | 22.23% |

**Intersection predictor (V9GATIntersectionPredictor, top_frac=0.10, 交集选股):**

| Mode | Raw ann | Raw Sharpe | Raw MDD |
|------|---------|------------|---------|
| **optimizer** | **42.91%** | **4.20** | 6.78% |
| simple_ls | 42.49% | 3.56 | 8.71% |
| optimizer_projected | 39.84% | 3.51 | 8.52% |
| simple_long | 29.83% | 1.20 | 22.41% |
| optimizer_mvo_ra0p1 | 28.78% | 3.36 | 6.87% |
| optimizer_mvo_ra1 | 27.07% | 3.80 | 4.82% |

**Ensemble predictor (V9GATEnsemblePredictor, 3 strategies × 3 modes + simple_long):**

| Strategy | Mode | Raw ann | Raw Sharpe | Raw MDD |
|----------|------|---------|------------|---------|
| top_union_bottom_intersection | simple_ls | **41.08%** | **3.76** | 8.39% |
| top_union_bottom_intersection | optimizer_projected | 38.78% | 3.72 | 8.13% |
| top_union_bottom_intersection | optimizer_mvo_ra0p1 | 30.12% | 3.62 | 6.50% |
| top_union_bottom_intersection | simple_long | 26.61% | 1.11 | 21.31% |
| avg_score | simple_ls | 38.33% | 3.51 | 8.96% |
| avg_score | optimizer_projected | 36.03% | 3.47 | 8.72% |
| avg_score | optimizer_mvo_ra0p1 | 22.68% | 3.06 | 8.47% |
| avg_score | simple_long | 28.53% | 1.16 | 21.48% |
| union | simple_ls | 30.18% | 3.11 | 9.23% |
| union | optimizer_projected | 28.43% | 3.07 | 8.97% |
| union | optimizer_mvo_ra0p1 | 24.21% | 3.06 | 7.46% |
| union | simple_long | 26.64% | 1.11 | 21.19% |

**Concentrated ensemble (all top_frac tiers, simple_ls and optimizer_projected):**

| Strategy | top_frac | Mode | Raw ann | Raw Sharpe | Raw MDD |
|----------|----------|------|---------|------------|---------|
| top_union_bottom | 1% | simple_ls | **112.55%** | **5.49** | 9.06% |
| avg_score | 1% | simple_ls | 108.62% | 5.44 | 8.17% |
| top_union_bottom | 1% | optimizer_projected | 104.90% | 5.40 | 8.71% |
| avg_score | 1% | optimizer_projected | 101.17% | 5.34 | 7.92% |
| top_union_bottom | 2% | simple_ls | 79.36% | 4.80 | 7.37% |
| avg_score | 2% | simple_ls | 76.92% | 4.74 | 7.48% |
| top_union_bottom | 2% | optimizer_projected | 74.33% | 4.73 | 7.18% |
| avg_score | 2% | optimizer_projected | 71.78% | 4.66 | 7.33% |
| avg_score | 3% | simple_ls | 65.26% | 4.47 | 7.77% |
| top_union_bottom | 3% | simple_ls | 64.91% | 4.39 | 8.37% |
| avg_score | 3% | optimizer_projected | 60.94% | 4.39 | 7.55% |
| top_union_bottom | 3% | optimizer_projected | 60.64% | 4.32 | 8.08% |
| avg_score | 5% | simple_ls | 52.32% | 4.14 | 7.32% |
| top_union_bottom | 5% | simple_ls | 52.87% | 4.08 | 7.96% |
| avg_score | 5% | optimizer_projected | 48.97% | 4.08 | 7.15% |
| top_union_bottom | 5% | optimizer_projected | 49.48% | 4.01 | 7.74% |
| top_union_bottom_intersection | 10% | simple_ls | 41.68% | 3.67 | 8.63% |
| top_union_bottom_intersection | 10% | optimizer_projected | 39.31% | 3.63 | 8.37% |
| avg_score | 10% | simple_ls | 38.33% | 3.51 | 8.96% |
| avg_score | 10% | optimizer_projected | 36.03% | 3.47 | 8.72% |

Key insight: alpha signal strength increases monotonically as selection narrows. 1% top_frac achieves 112%+ ann with Sharpe 5.49, but avg_long drops to ~29 stocks — execution liquidity becomes the binding constraint. 2-3% offers the best practical balance.

**Persistent multi-date signal (alpha averaging over 5-day window):**

| Strategy | Mode | Raw ann | Raw Sharpe | Raw MDD |
|----------|------|---------|------------|---------|
| avg_score | baseline | 38.33% | 3.51 | 8.96% |
| avg_score | average w5 | 19.20% | 2.23 | 7.71% |
| avg_score | composite w5 | 22.80% | 2.75 | 7.09% |
| top_union_bottom | baseline | 41.08% | 3.76 | 8.39% |
| top_union_bottom | average w5 | 14.97% | 2.68 | 4.28% |
| top_union_bottom | composite w5 | 8.13% | 1.81 | 4.01% |

**Interpretation:**
- All metrics significantly higher than the 05-15 benchmarks due to data update (new daily raw data).
- Best production: `top_union_bottom_intersection` + `simple_ls` (41.08% ann, 3.76 Sharpe).
- Best aggressive: `avg_score` + 1% concentrated + `simple_ls` (108.62% ann, 5.44 Sharpe). Practical sweet spot at 2-3%.
- Lowest drawdown: `intersection` + `optimizer_mvo_ra1` (27.07% ann, 4.82% MDD).
- simple_long consistently underperforms simple_ls by 10-20% ann — the short leg contributes significant alpha.
- Multi-date alpha averaging destroys signal — single-day alpha remains the best input.
- `optimizer_mvo` with higher risk_aversion reduces drawdown but at significant return cost.

### Layered vs Overwrite Rebalance Comparison (2026-05-16)

Tested layered portfolio rebalance strategy against traditional overwrite baseline using GAT model (`ultimate_v7_gat_best.pt`) with `optimizer_projected` mode.

**Experiment design:**
- **Overwrite**: Every r days, completely re-optimize portfolio and replace old holdings
- **Layered**: Every r days, create a new "layer" with fresh optimization; old layers decay over h=5 days and are aggregated

**Results (GAT, optimizer_projected, h=5):**

| Frequency | Strategy | Ann Return | Sharpe | Max DD |
|-----------|----------|------------|--------|--------|
| r=1 | Overwrite | 49.24% | 4.06 | 10.83% |
| r=1 | Layered | 34.82% | 3.57 | 5.63% |
| r=2 | Overwrite | 44.67% | 3.74 | 10.43% |
| r=2 | Layered | 32.30% | 3.64 | 4.88% |
| r=5 | Overwrite | 34.93% | 3.20 | 8.91% |
| r=5 | Layered | 34.97% | 3.20 | 8.91% |

**Conclusion:** Layered strategy reduces drawdown at high rebalance frequency but gives up too much return (r=1/2). At r=5 it converges to overwrite behavior. Recommend overwrite for production unless explicitly optimizing for lower drawdown.

## Daily inference and recommendation (2026-05-15)

`data/pipeline.py` `build_inference_sample(config, stock_universe=None, as_of_date=None)` constructs a label-free cross-section sample for the latest or specified-as-of trading date using the same feature construction logic as training/backtest. It does not require future returns (`valid_ret`), so the latest usable date is the most recent trading day with enough feature history, not bounded by `max_horizon`.

`run/recommend_daily.py` multiplexes V9/GAT checkpoints via the ensemble predictors and scores one inference sample.

```bash
# Top 20 recommended stocks (avg_score ensemble, latest available date)
F:/miniconda3/envs/pytorch/python run/recommend_daily.py --as-of latest --predictor avg_score --top-n 20

# Single-stock query
F:/miniconda3/envs/pytorch/python run/recommend_daily.py --as-of latest --predictor avg_score --code 000001.SZ

# Narrower fraction version
F:/miniconda3/envs/pytorch/python run/recommend_daily.py --as-of latest --predictor top_union_bottom_intersection --top-frac 0.03
```

Predictor choices: `v9`, `gat`, `avg_score`, `intersection`, `top_union_bottom_intersection`. Default output fields: date, rank, code, alpha, percentile, predictor, regime.

Key constraints:

- Alpha is a cross-sectional ranking score, not an absolute expected return.
- "Latest" means the most recent trading date in `data/raw` with valid features; this is not intraday live prediction.
- **This script does NOT auto-update `data/raw`.** For recommendations on a new trading day, manually run the data update scripts first.
- Inference reuses the same feature dimensions as training; do not change feature schema without retraining.

## Label semantics

- `sample['raw_y']`: unstandardized forward return for the main target horizon.
- `sample['y']`: cross-section standardized main target label for training.
- `sample['y_seq']`: multi-horizon forward return sequence, shape `(N, max_horizon)`.
- Backtests should use `raw_y` or recomputed realized returns, not standardized `y`.

## Known follow-up items

These are not safe to change casually because they alter data definitions or historical comparability:

1. Fundamental ROE currently uses available statement net income/equity and may mix cumulative report periods; a TTM or quarter-normalized redesign should be treated as a new factor schema and cache version.
2. `revenue_yoy` uses a 4-row quarterly shift; missing quarters can misalign comparisons. Fixing it requires stricter quarter calendar alignment and cache invalidation.
3. Consider adding availability masks for fundamental and macro cold-start fields so the model can distinguish true zero from unavailable data.
4. Network experiments should be isolated one at a time: remove model-level rank embedding, add head LayerNorm/Dropout, try FiLM/gated regime fusion, or reduce Transformer depth.
5. Loss/label experiments should be isolated: residualized labels, style exposure penalty, rank labels, top-bottom spread loss, and horizon consistency loss.

## Additional issues found (2026-05-15)

1. **`run/train_legacy.py` overwrites V9 checkpoint**: saves to `ultimate_v7_best.pt` (same path as `run/train_v9.py`). Running it accidentally destroys the V9 training result. Either change its save path to `ultimate_v7_legacy_best.pt` or retire the script.

2. **`data/raw/` contains stale `*_feat.parquet` cache files**: per-stock feature cache files mixed in with CSV data (5332 files). These consume disk space and are not cleaned up when feature schema changes. Consider adding cleanup logic or ignoring them.

3. **Thread safety in `safe_tushare_call()`**: `backtest/engine.py` uses a function attribute `safe_tushare_call.last_call` for API rate limiting without locks. This is not thread-safe. `data/update.py`'s `TushareProLite` correctly uses `threading.Lock()`. Recommend unifying the rate-limiting pattern.

4. **Dead config options in `core/config.py`**: `use_stacking` and `dynamic_risk_budget` are declared but never read by any code. Consider removing or explicitly documenting as reserved for future use.

5. **`_sys_check.ps1` nvidia-smi quoting**: line 30 uses comma-separated query fields without quotes. Works but may cause PowerShell parsing edge cases. Safer: `nvidia-smi --query-gpu="name,utilization.gpu,..."`.

## Recommendation system (2026-05-15)

Three recommendation scripts, all reading from `data/raw` without auto-updating.

### `run/recommend_daily.py` — single-day top-N + single-stock query

```bash
F:/miniconda3/envs/pytorch/python run/recommend_daily.py --as-of latest --predictor avg_score --top-n 20
F:/miniconda3/envs/pytorch/python run/recommend_daily.py --as-of latest --predictor avg_score --code 000001.SZ
```

### `run/recommend_persistent.py` — multi-date persistence/momentum/consistency

```bash
# Main-board only (default), last 5 trading days
F:/miniconda3/envs/pytorch/python run/recommend_persistent.py --test-stocks 5000 --ndates 5 --top-n 30

# All boards
F:/miniconda3/envs/pytorch/python run/recommend_persistent.py --all-boards

# Save CSV output
F:/miniconda3/envs/pytorch/python run/recommend_persistent.py --output recommendations/persistent
```

Outputs three ranked tables:
- **Persistent high scorers**: composite = avg_percentile × 0.5 + consistency × 0.3 + max(0, trend_slope) × 10 × 0.2
- **Rising stars**: sorted by momentum (recent − early percentile)
- **Most consistent**: sorted by consistency = 1/(1 + percentile_std)

Default `--main-board-only` excludes 688 (STAR Market, 500k min), 300/301 (ChiNext, 100k min).

### `run/rank_watchlist.py` — multi-date ranking for a watchlist file

```bash
F:/miniconda3/envs/pytorch/python run/rank_watchlist.py --test-stocks 5000 --ndates 5 --top-n 20
F:/miniconda3/envs/pytorch/python run/rank_watchlist.py --from-date 2026-05-09 --to-date 2026-05-15
```

Reads `watchlist.txt` (one code per line). Outputs per-date detail and summary (avg rank, best/worst, top-N rate).

### Reference results (2026-05-11 ~ 05-15, ~4816 main-board stocks, avg_score ensemble)

**Persistent top-10 main board:**

| Code | Avg %ile | Momentum | Trend | Days in Top30 |
|------|----------|----------|-------|---------------|
| 002384.SZ | 74.8% | +58 | +20.6%/d | 0 |
| 605389.SH | 76.1% | +54 | +18.8%/d | 0 |
| 002189.SZ | 81.7% | +40 | +15.6%/d | 2 |
| 603897.SH | 73.5% | +55 | +20.8%/d | 0 |
| 002636.SZ | 72.0% | +56 | +19.8%/d | 0 |
| 603657.SH | 64.2% | +74 | +24.3%/d | 0 |
| 000567.SZ | 78.9% | +41 | +15.9%/d | 0 |
| 600708.SH | 86.9% | +30 | +10.2%/d | 2 |
| 002921.SZ | 88.2% | +29 | +11.2%/d | 3 |
| 002631.SZ | 86.2% | +28 | +10.7%/d | 0 |

**Rising stars (largest momentum):**

| Code | Avg %ile | Momentum | Worst→Best |
|------|----------|----------|------------|
| 600790.SH | 62.9% | +80.6 | 17% → 99% |
| 603131.SH | 58.8% | +77.7 | 4% → 95% |
| 603657.SH | 64.2% | +74.3 | 4% → 98% |

**Most consistent (lowest rank volatility):**

| Code | Avg %ile | Consistency | Range |
|------|----------|-------------|-------|
| 603989.SH | 99.2% | 0.997 | 98.7–99.6% |
| 603075.SH | 94.2% | 0.998 | 93.8–94.5% |
| 002179.SZ | 98.2% | 0.995 | 97.2–98.5% |
| 002388.SZ | 98.8% | 0.995 | 98.1–99.3% |

### Key constraint

Alpha is a cross-sectional ranking score, not expected return. All recommendation scripts use local `data/raw` data only and do NOT auto-update it. For up-to-date recommendations, run `data/update_daily.py` first (optimized to ~107 batch API calls for 5332 stocks).

## Project refactoring plan

### Completed (2026-05-15 session)

- Fixed `.gitignore`: added `backtest_results_*/`, `models_multi_*/` directories
- Fixed garbled Chinese in `backtest/engine.py` (2 places) and `data/pipeline.py` (2 places)
- Updated `README.md`: corrected `core/train_utils.py` and `run/train_legacy.py` filenames
- Removed duplicated paragraphs in Architecture overview section above
- `run/train_gat_v2.py`: backup old checkpoint as `.pt.bak` before deletion
- `run/recommend_daily.py`: added `--from-date`/`--to-date` batch mode and `--exclude-prefix` for filtering stock codes (e.g., 300/301/688)
- `data/pipeline.py`: refactored `build_inference_sample` into reusable `_build_inference_matrices()`, `_sample_from_matrices()`, and added `build_inference_samples()` for batch multi-date inference with single CSV loading; added `cache/inference_matrices_cache.pkl` for disk caching

### Phase 1: Directory cleanup (low risk)

- Create `checkpoints/`, `logs/`, `recommendations/` directories under project root
- Move `.pt` files to `checkpoints/`, `.log` to `logs/`
- Move recommendation CSVs to `recommendations/`
- Update paths in `run/train_v9.py`, `run/train_gat.py`, `run/train_gat_v2.py`, `run/recommend_daily.py`

### Phase 2: Reduce `run/` entry scripts (medium risk, requires equivalence testing)

- 4 training scripts → 1 unified `run/train.py --model v9|gat|gat_v2|legacy`
- 5 backtest experiment scripts → 1 unified `run/backtest.py --experiment ensemble|intersection|concentrated|layered`
- Validate that CLI defaults and output labels are preserved

### Phase 3: Split large files (high risk, validate step by step)

- `data/pipeline.py` (886 lines) → extract `data/features.py` (feature computation) and `data/inference.py` (inference matrix building)
- `backtest/engine.py` (1461 lines) → extract `backtest/optimizer.py` (portfolio optimizers) and `backtest/execution.py` (order execution, ADV fill logic)

### Phase 4: Merge data update scripts (low risk)

- `data/update_index.py`, `data/update_index_ak.py`, `data/download_market_data.py` → fold into `data/update.py`

### Invariant for all phases

- Labels, return metric, `entry_day`/hold-window semantics, ADV execution constraints, portfolio weight construction, and the `data/raw` vs `data/tracking_raw` boundary must not change.
- Every phase should be a separate PR/commit so rolling back is cheap.
