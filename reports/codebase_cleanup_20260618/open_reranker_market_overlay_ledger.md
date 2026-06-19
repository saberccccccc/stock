# Open-Reranker And Market-Overlay Ledger 2026-06-19

This ledger covers the remaining open-reranker, negfilter, edge, and
market-overlay experiment-output candidates. It does not move files. Its purpose
is to distinguish active observation evidence from rejected research outputs.

## Source Reports

| Source | Role |
|---|---|
| `reports/open_ledger_candidate_summary_20260617/candidate_summary.md` | Official baseline, attack candidate, and stability candidate summary. |
| `reports/forward_observation_plan_20260617.md` | Forward/live observation rules and promotion gates. |
| `reports/breadth_triggered_market_20260617/breadth_triggered_market_report.md` | Breadth market-mult overlay decision. |
| `reports/breadth_triggered_target_20260617/breadth_triggered_target_report.md` | Breadth target-shrink rejection. |
| `reports/state_triggered_target_20260617/state_triggered_target_report.md` | State-triggered target-shrink rejection. |

## Current Open-Ledger Decision

| Role | Candidate | Decision |
|---|---|---|
| Official baseline | `main_candidate` | Keep as official baseline. |
| First attack candidate | `negfilter_r030_100_drop3` | Observe forward/live; do not replace baseline yet. |
| First stability candidate | `edge_r030_100` | Observe as stability candidate. |
| Conservative watch | `market_switch` | Watch only; not official. |

Promotion remains based on executable open-price share-ledger results, not IC:

1. forward/live Sharpe for 50w and 100w;
2. annualized return;
3. lag1 and cost2x robustness;
4. max drawdown;
5. turnover and blocked buys;
6. no one-month burst dependency.

## Open-Reranker / Attack Candidate Directories

| Path | Status | Decision | Reason |
|---|---|---|---|
| `open_reranker_current_v9_20260617` | Candidate generator evidence | Keep | Contains model, labels, and base reranked alpha outputs. |
| `open_reranker_current_v9_negfilter_20260617` | First attack candidate evidence | Keep | Contains `r030_100_drop3`, the first attack candidate under forward observation. |
| `diagnostics_negfilter_drop3_20260617` | Attack diagnostics | Keep | Explains stability/source of negfilter improvement. |
| `open_reranker_current_v9_edge_20260617` | Stability candidate evidence | Keep | Contains `edge_r030_100`, the first stability candidate. |
| `open_reranker_current_v9_market_switch_20260617` | Conservative watch evidence | Keep for now | Candidate is not official, but supports candidate-summary comparison. |
| `open_reranker_current_v9_forward_20260617` | Forward observation evidence | Keep | Needed for forward/live observation comparison. |
| `open_reranker_current_v9_light_20260617` | Unpromoted sensitivity evidence | Hold | Can archive later only after light-rerank rejection is summarized. |

## Market Overlay Directories

| Path | Status | Decision | Reason |
|---|---|---|---|
| `breadth_triggered_market_20260617` | Risk-control observation candidate | Keep | `ma3_035_m085` improved forward loss and slightly improved 50w test, but is not promoted. |
| `breadth_triggered_target_20260617` | Rejected overlay evidence | Hold | Target shrink lowered validation/test quality; archive only after rejected-overlay summary is indexed. |
| `state_triggered_target_20260617` | Rejected overlay evidence | Hold | State target shrink hurt historical test; archive only after rejected-overlay summary is indexed. |
| `conditional_negfilter_breadth_20260618` | Conditional overlay evidence | Hold | Needs a report/summary before cleanup decision. |

## Cleanup Decision

Do not archive any open-reranker or market-overlay parent directory yet.

## Source Reproducibility

The open-reranker training and forward-application entrypoints are maintained:

```text
run/train_open_reranker_current_v9.py
run/apply_open_reranker_forward.py
tests/test_open_reranker.py
```

Safety guarantees:

- training input is rejected after the 2023-12-31 OOF cutoff;
- open-to-open labels may not cross the end of their OOF year;
- price tails are bounds-checked before array access;
- validation/test scoring may not use post-2026-05-18 forward rows;
- the forward entrypoint rejects rows before 2026-05-19;
- Alpha JSONL I/O uses the shared `alpha.io` module.

These guarantees preserve the existing research branch; they do not promote
an open-reranker candidate over `main_candidate`.

The market-overlay source utilities are also maintained:

```text
run/make_breadth_triggered_market_alpha.py
run/make_breadth_triggered_target_alpha.py
run/make_state_triggered_target_alpha.py
run/switch_alpha_by_market_state.py
```

They use shared Alpha JSONL I/O, reject mismatched paired Alpha files, and
require explicit `--allow-forward` for forward-only inputs. Research mode
rejects signal dates after 2026-05-18; forward mode rejects dates before
2026-05-19. This source preservation does not reverse the target-shrink
rejections or promote a market overlay.

The next cleanup movement in this family should wait for one of:

1. a forward/live observation summary that freezes the attack/stability
   candidate conclusions;
2. a rejected-overlay summary that explicitly covers target shrink, state
   shrink, and conditional breadth negfilter;
3. a split of mixed parent directories so failed variants can be archived
   without moving active candidates.

## Safe Follow-Up

Rejected/unpromoted decisions are summarized in:

```text
reports/codebase_cleanup_20260618/rejected_candidate_summary.md
```

Before any archive move, run a path-reference check for:

```text
breadth_triggered_target_20260617
state_triggered_target_20260617
conditional_negfilter_breadth_20260618
open_reranker_current_v9_light_20260617
```

Only exact-name archive batches should be considered, and active candidates
such as `negfilter_r030_100_drop3`, `edge_r030_100`, and
`breadth_triggered_market_20260617` should remain in place.
