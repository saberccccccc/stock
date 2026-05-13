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
F:/miniconda3/envs/pytorch/python -m py_compile core/config.py core/model.py core/train_utils.py data/pipeline.py data/fundamental_factors.py data/macro_factors.py data/update.py data/update_daily.py backtest/engine.py backtest/ensemble.py backtest/risk.py run/train_v9.py run/train_gat.py run/train_gat_v2.py run/train_legacy.py run/hyper_search.py
F:/miniconda3/envs/pytorch/python run/train_v9.py
F:/miniconda3/envs/pytorch/python run/train_gat.py
F:/miniconda3/envs/pytorch/python backtest/engine.py --model-type both
F:/miniconda3/envs/pytorch/python run/hyper_search.py
```

The environment currently has PyTorch `2.7.1+cu118` with CUDA available on an RTX 2060-class 6GB GPU.

If dependencies need to be installed:

```bash
F:/miniconda3/envs/pytorch/python -m pip install -r requirements.txt
```

Data update commands require a Tushare token supplied outside source code:

```bash
export TUSHARE_TOKEN="<token>"
F:/miniconda3/envs/pytorch/python data/update.py
F:/miniconda3/envs/pytorch/python data/update_daily.py
F:/miniconda3/envs/pytorch/python data/update.py --init
```

On Windows CMD use `set TUSHARE_TOKEN=<token>` instead of `export`.

## Architecture overview

- `run/` contains entry scripts. `run/train_v9.py` is the main Transformer training entry; `run/train_gat.py` and `run/train_gat_v2.py` train the GAT variant; `run/train_legacy.py` is a legacy backup entry; `run/hyper_search.py` runs LightGBM/backtest parameter search.
- `core/model.py` defines `UltimateV7Model`: `FeatureGrouper` + cross-stock Transformer + optional industry GAT branch + multi-horizon heads.
- `core/train_utils.py` defines `CrossSectionDataset`, collate functions, loss/evaluation, `get_regime_dim()`, and `train_model()`.
- `data/pipeline.py` builds cross-sectional samples from `data/raw/`, joins market/fundamental/macro factors, creates labels, and returns `(train_samples, val_samples)`.
- `data/fundamental_factors.py` and `data/macro_factors.py` must remain point-in-time (PIT).
- `backtest/engine.py` trains/loads multi-horizon LightGBM models, loads DL checkpoints, and runs production backtests through `DLPredictor` / `LGBPredictor`.
- `backtest/ensemble.py` handles stacking/blending experiments.

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
- `run/train_v9.py` defaults to `test_mode=False` for full training. For smoke tests, set `cfg.test_mode=True` and reduce `cfg.test_stocks`.
- GAT training uses a separate checkpoint `ultimate_v7_gat_best.pt`; V9 Transformer uses `ultimate_v7_best.pt`.
- If label logic, feature schema, PIT factor logic, market/macro/fundamental factor logic, or industry IDs change, old checkpoints should be considered invalid and models should be retrained.

## Data and cache rules

Runtime artifacts are gitignored: `data/raw/`, `cache/`, logs, model weights, LightGBM model folders, and backtest outputs.

Rebuild caches after changing data semantics:

```bash
rm -f cache/cross_section_*.pkl
rm -f cache/fundamental_features*.parquet
rm -f cache/north_flow.csv cache/margin_balance.csv cache/pmi_pit_v2.csv
```

Current cache behavior:

- `data/pipeline.py` uses descriptive cross-section cache names with `CACHE_VERSION`, stock count/universe digest, enabled feature flags, `seq_len`, `target_horizon`, and `max_horizon`.
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
- `run/train_gat.py` currently uses `keep_ratio=0.9`; `run/train_gat_v2.py` is a no-stdout variant using `keep_ratio=0.5` and `resume=False`.

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
F:/miniconda3/envs/pytorch/python backtest/engine.py --portfolio-mode simple_ls
F:/miniconda3/envs/pytorch/python backtest/engine.py --portfolio-mode simple_long
```

ADV modes:

- `execution` is the default and recommended mode. It constrains actual fills at execution.
- `both` is a conservative variant.
- `weight_cap` is retained for comparison only and is deprecated because it can understate execution liquidity risk in the optimizer while forcing full execution later.

Backtest return metric is `next_close_to_next_close`. Do not change the `entry_day` / hold window semantics casually; changing it alters historical performance comparability.

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
