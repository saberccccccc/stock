# Experiment Output Ledger 2026-06-18

This ledger classifies non-reranker experiment outputs before any batch move to
`archive/experiments_202606`. It is intentionally conservative: important
strategy evidence stays in place until a narrower ledger or summary proves it
can be archived.

## Current Archive-Plan Context

For non-reranker `experiment_output` entries:

| Action | Count |
|---|---:|
| `archive_candidate` | 126 |
| `protect` | 3 |

Protected roots:

| Path | Reason |
|---|---|
| `forward_results` | Forward observation root; do not move as a whole |
| `v9_avgw3_open_ledger_20260617` | Current official open-price share-ledger baseline and candidate comparison root |
| `v9_avgw3_extend_to_20260518_20260616` | Frozen V9 extension to the 2026-05-18 research cutoff |

## Decision Summary

| Group | Decision |
|---|---|
| Candidate/loss validation directories | Keep until checkpoint/loss result ledger exists |
| Official V9 filter/open-ledger evidence | Keep while it supports the official baseline |
| Open-reranker/negfilter/edge candidates | Keep until forward observation and attack-candidate ledgers are complete |
| Breadth/state/conditional overlays | Keep for now; archive only after caveats and summaries are indexed |
| Legacy `backtest_results_*` groups | Archive candidates after representative summary sampling |
| Run scripts tied to validation outputs | Archive with their matching outputs, not separately |

## Active Strategy Evidence

| Path | Status | Reason |
|---|---|---|
| `v9_avgw3_filter095_validation_20260616` | Active strategy evidence | 9.5% filter validation evidence for V9 avgw3. |
| `unified_good_ops_validation_20260616` | Active strategy evidence | Unified validation comparing open-to-open/share-ledger variants. |
| `v9_avgw3_open_ledger_20260616` | Superseded strategy evidence | Earlier open-ledger sweep superseded by protected `v9_avgw3_open_ledger_20260617`; archive only after best rows are summarized. |
| `v9_avgw3_open_ledger_20260617` | Protected official baseline | Current official open-ledger candidate comparison root. |
| `v9_avgw3_extend_to_20260518_20260616` | Protected cutoff evidence | Frozen V9 extension to research cutoff. |

## Candidate And Loss Validation Evidence

| Path | Status | Reason |
|---|---|---|
| `candidate_model_validation_20260614` | Active training evidence | Strict A0/A4/A5/frozen-V9 candidate validation; source for A4-E6 decision and V9 caveat. |
| `locked_candidate_confirmation_20260614` | Active training evidence | Locked candidate confirmation from the same validation round. |
| `loss_ablation_portfolio_validation_20260614` | Active training evidence | Portfolio validation for A0-A5 loss ablation candidates. |
| `multi_loss_validation_20260614` | Active training evidence | Multi-loss validation outputs and portfolio summary. |
| `m0_topfocus_validation_20260614` | Active training evidence | M0 plus Top-focus validation evidence. |
| `downside_topfocus_validation_20260616` | Failed or unpromoted evidence | Downside/Top-focus validation did not establish a production improvement. |
| `lag1_checkpoint_sweep_m0_20260616` | Active training evidence | M0 checkpoint/lag1 sweep includes frozen-V9 and M0 epoch evidence referenced by unified plan. |

Related scripts:

| Path | Reason |
|---|---|
| `resume_downside_topfocus_remaining_20260616.ps1` | Resume script for downside/top-focus validation. |
| `validate_downside_topfocus_candidates_20260616.ps1` | Validation script for downside/top-focus candidates. |
| `run_lag1_loss_ablation_after_sweep_20260616.ps1` | Script used after lag1 sweep. |
| `run_unified_good_ops_validation_20260616.ps1` | Script for unified good-ops validation. |

## Open-Reranker And Attack Candidates

| Path | Status | Reason |
|---|---|---|
| `open_reranker_current_v9_20260617` | Attack candidate evidence | Current V9 open-reranker model and alpha outputs. |
| `open_reranker_current_v9_edge_20260617` | Attack candidate evidence | Edge candidate alpha outputs. |
| `open_reranker_current_v9_light_20260617` | Attack candidate evidence | Light open-reranker candidate alpha outputs. |
| `open_reranker_current_v9_negfilter_20260617` | First attack candidate evidence | `negfilter_r030_100_drop3` is first attack candidate but not official baseline. |
| `open_reranker_current_v9_market_switch_20260617` | Attack candidate evidence | Market-switch candidate outputs. |
| `open_reranker_current_v9_forward_20260617` | Forward candidate evidence | Forward open-reranker outputs; needed for observation comparison. |
| `diagnostics_negfilter_drop3_20260617` | Attack candidate diagnostics | Diagnostics explaining negfilter_drop3 source of improvement and stability. |

Interpretation:

- `negfilter_r030_100_drop3` is historically strong and is a first attack
  candidate.
- It is not the official baseline.
- It needs forward/live observation before any promotion.

## Market Overlay Experiments

| Path | Status | Reason |
|---|---|---|
| `conditional_negfilter_breadth_20260618` | Conditional overlay evidence | Conditional breadth negfilter overlay candidate, not promoted. |
| `state_triggered_target_20260617` | Unpromoted overlay evidence | State-triggered target experiment had implementation caveat and is not official. |
| `breadth_triggered_target_20260617` | Unpromoted overlay evidence | Breadth-triggered target experiment; research candidate only. |
| `breadth_triggered_market_20260617` | Unpromoted overlay evidence | Breadth-triggered market experiment; research candidate only. |

These directories should not be moved until the related reports are indexed:

```text
reports/state_triggered_target_20260617/
reports/breadth_triggered_target_20260617/
reports/breadth_triggered_market_20260617/
reports/forward_observation_plan_20260617.md
```

## Reports To Keep

| Path | Reason |
|---|---|
| `reports/open_ledger_candidate_summary_20260617` | Candidate summary for open-ledger comparison. |
| `reports/forward_observation_20260617` | Forward observation report directory. |
| `reports/candidate_leaderboard_20260617` | Generated registry-backed candidate leaderboard. |

## Legacy Backtest Output Groups

The following groups are archive candidates, but should be moved only after a
small sampling check confirms their conclusions are represented in reports:

| Pattern | Status | Sampling requirement |
|---|---|---|
| `backtest_results_exp_*` | Legacy experiment outputs | Check representative `summary.csv` / report files before batch move. |
| `backtest_results_test_plan_*` | Legacy test-plan outputs | Verify official conclusions are in `TEST_PLAN.md` and `official_baselines.md`. |
| `backtest_results_switch_value_*` | Legacy switch-value outputs | Verify switch-value reports are indexed. |
| `backtest_results_temporal_*` | Legacy temporal probe outputs | Verify temporal reports and snapshots are indexed. |
| `backtest_results_summary_*.txt` | Legacy summary files | Archive after source snapshots are indexed. |

## Cleanup Rules

1. Do not move protected roots.
2. Do not move official baseline, forward observation, first attack candidate,
   or active training evidence before a narrower ledger exists.
3. Move legacy `backtest_results_*` groups in small dry-run batches only after
   sampling summaries.
4. Keep run scripts with their matching outputs; do not scatter them into a
   different archive class.
5. Use `run/archive_from_plan.py --class experiment_output --target archive/experiments_202606`
   only after a class-specific batch list has been reviewed.

## Next Step

Run a representative sampling pass for legacy `backtest_results_*` directories,
then archive a first small batch of clearly superseded legacy backtest outputs.
