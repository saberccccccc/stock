# Project Current Index - 2026-07-10

> 2026-07-18 执行治理更新：项目唯一有效的完整推进顺序已统一到
> `MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md`。本索引继续登记当前
> baseline、候选、报告和入口，但其中历史“next phase”文字不再决定执行
> 顺序。当前位于总计划 P2 正式训练主线收敛阶段。

Purpose: prevent repeated context loss, missed baselines, and mixed
proxy/realistic evidence. This file is intentionally ASCII so it can be read
reliably from PowerShell.

## 1. Current Research Rules

- Use only `2024 val` + `2025 test` for model/rule selection.
- Use `2026 forward` for observation only.
- Current execution protocol is `realistic open-price share-ledger`.
- Required stress set: `normal`, `lag1`, `cost2x`, `capacity_3pct`.
- Required capital set: CNY 500k and CNY 1m.
- Reports must include `signal_start`, `signal_end`, `backtest_start`,
  and `backtest_end`.

## 1A. Qlib-Inspired Research Framework

- ADR 0004 establishes a project-native research/manual-shadow framework.
- It preserves the existing PIT data, realistic open-price ledger, and
  registry governance; Qlib default data and close-price execution are not
  formal evidence.
- The first model experiment after the framework acceptance gate is fixed
  four-year Train, six-month Valid, monthly OOS walk-forward retraining.
- Automatic trading, automatic retraining, and automatic candidate promotion
  are out of scope. See `QLIB_ADOPTION_PLAN_20260712.md`.
- The first recorded rolling baseline comparison is complete: full v14,
  compact, and broad arms all have common Val/Test realistic-ledger evidence
  under the low-memory 100k/30k training cap. Compact leads the research
  comparison but is not formally promoted. See
  `reports/qlib_research_framework_20260712/factor_baseline_full_feature_comparison_20260715.md`.

## 2. Formal Baseline

Formal portfolio-layer baseline:

- `ledger_path_v3_t0001_nolookahead`

Baseline evidence is split across two directories:

- normal:
  - `reports/state_aware_policy_training_20260704/ledger_path_v3_t0001_nolookahead_open_ledger`
- stress:
  - `reports/state_aware_policy_training_20260704/stress_ledger_path_v3_t0001_nolookahead`

Important: the baseline stress directory was previously missed by one summary
path scan. `run/audit_recent_candidate_scorecard.py` now aliases
`stress_ledger_path_v3_t0001_nolookahead` back to the formal baseline.

## 3. Main Current Candidates

### 3.1 cond_pairrisk_volg001

Signal path:

- `reports/state_aware_policy_applied_20260704/multi_downside_e19_sa_p05_ledger_path_v3_nolookahead_cond_pairrisk_volg001`

Old sweep backtest:

- `reports/state_aware_policy_training_20260704/ledger_path_v3_nolookahead_riskguard_sweep/cond_pairrisk_volg001`

Warning: the old sweep is `proxy`, not `realistic`. Do not compare it directly
with latest realistic results.

Realistic rerun:

- `reports/sa_20260710_backtest/cond_pairrisk_volg001_realistic`

### 3.2 cond_pairrisk_v2_svol0_betag001_gate0

Goal: test a more conservative replacement rule that rejects specific-vol
worsening and lightly penalizes beta worsening.

Signal path:

- `reports/sa_20260710/v2_svol0_betag001_gate0`

Realistic backtest:

- `reports/sa_20260710_backtest/v2_svol0_betag001_gate0`

Current conclusion: this v2 is too conservative. It underperforms
`cond_pairrisk_volg001_realistic` on val/test and forward observation. Do not
promote.

### 3.3 capital-aware hybrid

Latest capped realistic evidence:

- `reports/state_aware_policy_training_20260710_capped/ledger_path_v3_capital_aware_hybrid_50_75no_100volg001`

Related manifests:

- `reports/recent_candidate_scorecard_20260710/hybrid_backtest_manifest_capped.json`
- `reports/recent_candidate_scorecard_20260710/capital_aware_manifests`

## 4. Key Reports

- Recent candidate audit:
  - `reports/recent_candidate_scorecard_20260710/recent_candidate_evidence_audit.md`
  - `reports/recent_candidate_scorecard_20260710/recent_candidate_selection_summary.csv`
  - `reports/recent_candidate_scorecard_20260710/recent_candidate_scorecard_long.csv`
- `cond_pairrisk_volg001` attribution:
  - `reports/cond_pairrisk_volg001_attribution_20260710/cond_pairrisk_volg001_attribution_report.md`
  - `reports/cond_pairrisk_volg001_attribution_20260710/attribution_aggregate_summary.csv`
- v2 comparison:
  - `reports/sa_20260710/v2_svol0_betag001_gate0_final_compare/cond_pairrisk_v2_realistic_audit.md`

## 5. Script Map

### 5.1 Single official open-ledger backtest

- `run/backtest_retention_open_ledger.py`

Use for one alpha / one stress validation. It starts a fresh Python process
and reloads OHLC, recomputes ADV, and rebuilds realistic masks each time.

### 5.2 Batch open-ledger backtest

- `run/sweep_open_price_ledger_params.py`

Prefer this for APM audit batches. It loads shared OHLC once, then loops over
alpha specs, stresses, capital values, and parameter grids in one process.

The formal wrapper is `run/official_backtest_from_registry.py`. It selects
`data/raw` for 2024 validation and 2025 test, and `data/forward_raw` for 2026
forward observation. Do not use the historical `run/sweep_open_ledger_params.py`
for new formal evidence; it is retained only to reproduce earlier research.

### 5.3 Manifest runner

- `run/run_backtest_command_manifest.py`

Good resume behavior, but slow for large batches because each command is a
separate Python process.

### 5.4 Ledger Path V3 signal generation

- `run/generate_ledger_path_v3_signal.py`

Recently added pass-through args:

- `--pair-risk-worsen-penalty`
- `--beta-worsen-penalty`
- `--gate-max-pair-risk-delta`
- `--gate-max-pair-downside-delta`
- `--gate-max-diff-specific-vol-60d`
- `--gate-min-diff-ret20`
- `--gate-condition`

It also creates `policy_features` and `pairwise_inference` subdirectories
before launching child scripts.

### 5.5 Recent candidate audit

- `run/audit_recent_candidate_scorecard.py`

Maintain aliases whenever a new report root uses a different directory name
for an existing candidate.

## 6. Runtime Performance Notes

OHLC matrix cache exists:

- `cache/open_ledger_ohlc_matrix`

Files:

- `open.dat`
- `high.dat`
- `low.dat`
- `close.dat`
- `money.dat`
- `volume.dat`

Each `.dat` file is about 170MB. Slow backtests are mainly caused by repeated
process startup and repeated conversion from memmap slices to pandas
DataFrames, not by a missing cache.

Manifest runner repeats these steps per command:

1. Load OHLC matrix metadata.
2. Slice OHLC into pandas DataFrames.
3. Recompute ADV.
4. Prepare realistic execution masks.
5. Run both capital values.

For future large audits, prefer `sweep_open_price_ledger_params.py`.
The realistic masks are additionally cached on disk under
`cache/open_ledger_execution_masks/`, so repeated compatible sweeps avoid
rebuilding limit, ST, listing-age, volume, and zero-trade masks.

## 7. Historical Documentation Conflict (Resolved)

This section records a former conflict. The README has since been corrected;
follow `PROJECT_RULES.md`, `RESEARCH_PROTOCOL.md`, and `registry/`.

Historical README wording was:

- V9 `avgw3` + `maxret095` + open-price share-ledger

Current plan document says the formal portfolio-layer baseline is:

- `ledger_path_v3_t0001_nolookahead`

For current research decisions, follow:

- `C:/Users/x/Documents/股票预测/主动投资组合管理导向详细目标_20260706.md`

Treat the wording below as historical only. Do not use the mojibake path or
this section to make research decisions.

## 8. Historical Remediation Checklist (Completed)

The items below were the pre-refactor checklist. They are retained for
historical traceability and must not be treated as pending work:

1. Fair scorecard coverage was regenerated.
2. Formal batches now use `sweep_open_price_ledger_params.py` via the registry
   wrapper.
3. README and current-baseline references were updated.
4. `registry/` now records candidate, report, attribution, split, and
   eligibility metadata.
5. Active signals and capped realistic evidence were retained during archive
   cleanup.

## 9. Refactor Status Update - 2026-07-10

The section above records the pre-refactor state. The current state is:

- `registry/` is now the source of truth for formal baselines, candidates,
  reports, attribution paths, and decision rules.
- `run/official_backtest_from_registry.py` is the official batch backtest
  entry point; it uses the shared-OHLC sweep runner.
- `run/scorecard_from_registry.py` produces a selection-only decision table;
  2026 forward remains observation-only.
- `run/attribution_from_registry.py` checks whether a challenger has complete
  attribution evidence before it can be formally promoted.

Current scorecard state: all four candidates have complete performance
coverage. `cond_pairrisk_volg001_realistic` now also has complete 2024/2025
attribution, including `cost2x` and `capacity_3pct`, and is marked
`selection_pass_pending_governance`. It is not the formal baseline yet:
2026 forward attribution is weak, and forward remains observation-only.
The capital-aware hybrid has no registered attribution. Pair-risk V2 remains
rejected on performance.

## 10. Archive Review

Run `python run/generate_reports_archive_review.py` to produce the current,
registry-aware non-destructive archive review. The latest review is in
`reports/archive_review_20260710/` and protects registered evidence,
official outputs, and current state-aware research automatically. On
2026-07-11, all 162 reviewed unregistered/unreferenced candidates were moved
to `archive/experiments_202606/`; the retained inventory is
`archive/experiments_202606/MIGRATION_MANIFEST_20260711.csv`. No formal
evidence, raw data, active cache, or current signal was deleted.

## 11. Qlib Adoption Status - 2026-07-15

The bounded compact LightGBM search and its independent-seed confirmation are
complete. `t03_minleaf160` is rejected for promotion because the apparent
2025 Test uplift did not reproduce under seed `20260713`; the compact baseline
remains the fixed research arm. The evidence and mandatory date fields are in
`reports/qlib_research_framework_20260712/compact_lgbm_t03_confirmation_20260715.md`.

This paragraph records the 2026-07-15 historical plan state and no longer sets
the active sequence. The canonical contract is 2024 Val plus 2025 Test for
selection and the full 2026 calendar year as observation-only Forward;
`2026-05-18` is only legacy artifact metadata. The active sequence is defined
by `MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md`.

The initial OOF blend pilot and 2018-2023 historical stability audit are now
complete. `compact_v14_eq_rank` is the only conditional research candidate;
its evidence is in `reports/qlib_research_framework_20260712/`. The active next
phase is state-aware portfolio construction and risk attribution, with no
forward-based selection.
