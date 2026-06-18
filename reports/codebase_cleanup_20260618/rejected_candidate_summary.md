# Rejected And Observation Candidate Summary 2026-06-19

This compact summary collects failed, unpromoted, and observation-only
candidates that still appear as top-level experiment outputs. It supports future
cleanup decisions without rereading every long report.

## Live Baselines And Shadow Candidates

| Item | Status | Decision |
|---|---|---|
| M0 | Current new-model training baseline | Keep as model baseline; still not promoted over official V9 live baseline. |
| V9 avgw3 + maxret095 + open-price share-ledger | Official strategy baseline | Keep official baseline evidence protected. |
| V3 reranker | Frozen research reference | Keep as shadow evidence. |
| V4 reranker | Preferred safe shadow candidate | Keep; can abstain to exact M0, but not promoted. |
| V4.1 reranker | Failed forward candidate | Preserve as documented failed experiment; do not retune from forward. |

## Rejected Training Losses

| Candidate | Decision | Reason |
|---|---|---|
| Separate multi-horizon auxiliary IC loss | Remove from M0 baseline | Trained horizon heads but hurt delay robustness. |
| M0 + Top-focus 0.005 | Rejected | Lower executable base, cost2x, and lag1 results versus M0. |
| Diversity loss | Rejected | Worse than A0 after execution filtering. |
| Spread loss | Rejected | Raw stability improved, but executable return collapsed. |
| Industry IC 0.10 | Not default | Mixed result; lower weight needs separate retest. |
| Lag1 auxiliary loss | Experimental only | Accept only if base, lag1, and cost2x all remain competitive with M0. |
| Downside loss | Experimental only | Must be judged by executable portfolio, not IC. |

Primary sources:

```text
reports/loss_ablation_decision_20260614.md
reports/lag1_training_selection_plan_20260616.md
reports/downside_topfocus_ablation_plan_20260615.md
reports/codebase_cleanup_20260618/training_validation_ledger.md
```

## Rejected Or Observation-Only Reranker Paths

| Candidate | Status | Decision | Reason |
|---|---|---|---|
| V1 LambdaRank reranker | Failed experiment | Keep until artifacts are split or explicitly archived | Not promoted. |
| V2 regression reranker | Failed experiment | Keep for audit trail | Mixed 2024 execution result; M0 retained. |
| V3 regression reranker | Shadow reference | Keep | Better Sharpe/drawdown in some historical checks, but not promoted. |
| V4 gated reranker | Safe shadow | Keep | Abstains to M0 when inactive; forward window matched M0 because it did not activate. |
| V4.1 meta-gate reranker | Failed forward | Keep as failed evidence | Historical improvement contradicted true forward underperformance. |

Primary sources:

```text
RERANKER_V4_PLAN_20260615.md
reports/codebase_cleanup_20260618/reranker_artifact_ledger.md
forward_results/m0_v41_20260615
```

## Open-Reranker Observation Candidates

| Candidate | Role | Decision |
|---|---|---|
| `main_candidate` | Official open-ledger baseline | Keep as baseline. |
| `negfilter_r030_100_drop3` | First attack candidate | Observe forward/live; no immediate replacement. |
| `edge_r030_100` | First stability candidate | Observe; not official. |
| `market_switch` | Conservative watch candidate | Watch only. |
| Light open-reranker variants | Unpromoted sensitivity checks | Archive only after a light-rerank rejection note exists. |

Primary sources:

```text
reports/open_ledger_candidate_summary_20260617/candidate_summary.md
reports/forward_observation_plan_20260617.md
reports/codebase_cleanup_20260618/open_reranker_market_overlay_ledger.md
```

## Market Overlay Decisions

| Candidate | Decision | Reason |
|---|---|---|
| Fixed lower target | Observation only | Helps validation/forward but hurts historical test. |
| `risk_target_r004` | Observation only | Improves validation and slightly improves forward, but weakens test. |
| Breadth-triggered target shrink | Rejected | Lowers validation/test quality despite lower drawdown in some cases. |
| State-triggered target shrink | Rejected | Hurts historical test; forward mixed. |
| Breadth-triggered market multiplier `ma3_035_m085` | Observation candidate | Reduces forward loss/drawdown and keeps 100w test Sharpe near flat, but not promoted. |
| Conditional breadth negfilter | Needs summary | Existing output should not move until its result is summarized. |

Primary sources:

```text
reports/target_fraction_validation_20260617/target_fraction_validation_report.md
reports/breadth_triggered_target_20260617/breadth_triggered_target_report.md
reports/state_triggered_target_20260617/state_triggered_target_report.md
reports/breadth_triggered_market_20260617/breadth_triggered_market_report.md
```

## Cleanup Implications

Do not move parent directories that mix active and failed artifacts. Future
archive candidates should be exact-name or subdirectory-level moves after the
following are true:

1. the failure/observation decision is represented in this summary or a more
   specific ledger;
2. no current script/config expects the path directly;
3. protected official evidence remains in place;
4. the archive move is dry-run first and regenerated reports/tests pass.

Potential later archive targets, after path-reference checks:

```text
open_reranker_current_v9_light_20260617
breadth_triggered_target_20260617
state_triggered_target_20260617
downside_topfocus_validation_20260616
```

Hold for now:

```text
open_reranker_current_v9_negfilter_20260617
open_reranker_current_v9_edge_20260617
open_reranker_current_v9_forward_20260617
breadth_triggered_market_20260617
reranker_models_20260615
reranker_validation_20260615
reranker_confirmation_20260615
```

## Path-Reference Check 2026-06-19

The first potential archive targets were checked with `rg` before any move.
They should still be held:

| Path | Reference status | Decision |
|---|---|---|
| `open_reranker_current_v9_light_20260617` | Referenced by protected `v9_avgw3_open_ledger_20260617/sweep_openrerank_w098` and `sweep_openrerank_w099` configs. | Hold. |
| `breadth_triggered_target_20260617` | Referenced by protected open-ledger result directories and summary reports. | Hold. |
| `state_triggered_target_20260617` | Referenced by protected open-ledger result directories and forward result summaries. | Hold. |
| `downside_topfocus_validation_20260616` | Referenced by validation script and result audit CSVs. | Hold. |

Do not archive these directories until either the referencing protected result
directories are also archived or the references are intentionally rewritten to
point at archived locations.
