# Review Document Index 2026-06-18

This index covers root-level markdown files that remain in manual review.
It does not move files; it records the intended cleanup path for each document.

## Summary

| Cleanup decision | Count |
|---|---:|
| archive_after_consolidation | 1 |
| keep | 4 |
| keep_or_consolidate | 7 |

## Documents

| Name | Role | Decision | Consolidation target | Notes |
|---|---|---|---|---|
| CLAUDE.md | agent_guide | keep | CLAUDE.md | Operational notes, architecture overview, and experiment-branch history. |
| README.md | entrypoint | keep | README.md | General project overview and quick-start commands. |
| EXPERIMENTS.md | legacy_experiment_log | archive_after_consolidation | reports/codebase_cleanup_20260618/training_research_index.md | Short early experiment-branch log; keep until key points are merged. |
| RERANKER_IMPLEMENTATION_PLAN_20260614.md | reranker_research | keep_or_consolidate | reports/codebase_cleanup_20260618/reranker_research_index.md | M0, V2, V3, V4, V4.1 reranker history and decisions. |
| RERANKER_V4_PLAN_20260615.md | reranker_research | keep_or_consolidate | reports/codebase_cleanup_20260618/reranker_research_index.md | Confidence-gated V4 and V4.1 result summary. |
| RESEARCH_PROTOCOL.md | research_control | keep | RESEARCH_PROTOCOL.md | Research cutoff, data boundary, and model-selection protocol. |
| FROZEN_FORWARD_STRATEGY.md | strategy_manifest | keep | reports/codebase_cleanup_20260618/official_baselines.md | Frozen live/shadow strategy decision made before forward observation. |
| TEST_PLAN.md | strategy_plan | keep_or_consolidate | reports/codebase_cleanup_20260618/official_baselines.md | V9 small-account test plan and locked research result. |
| SHARPE_OPTIMIZATION_REPORT.md | strategy_report | keep_or_consolidate | reports/codebase_cleanup_20260618/official_baselines.md | Small-account Sharpe optimization results and decision evidence. |
| LOSS_ABLATION_PLAN.md | training_plan | keep_or_consolidate | reports/codebase_cleanup_20260618/training_research_index.md | Loss ablation protocol and promotion gates. |
| PURGED_ALPHA_OPTIMIZATION_PLAN.md | training_plan | keep_or_consolidate | reports/codebase_cleanup_20260618/training_research_index.md | Purged alpha optimization sequence and historical confirmation caveat. |
| CANDIDATE_MODEL_VALIDATION_PLAN_20260614.md | validation_plan | keep_or_consolidate | reports/codebase_cleanup_20260618/training_research_index.md | Purged V9 checkpoint validation plan and final decision. |

## Heading Preview

| Name | Title | Headings |
|---|---|---|
| CLAUDE.md | CLAUDE.md | PowerShell 编码注意事项; Environment and commands; Architecture overview; Training; Backtest; Recommendation system; Data and cache rules; Current feature structure |
| README.md | README.md | 项目结构; 快速开始; 每日推荐; 回测; 环境; 测试; 关键不变量 |
| EXPERIMENTS.md | EXPERIMENTS.md | Isolation; Quick start; Experiment log; Priority experiments |
| RERANKER_IMPLEMENTATION_PLAN_20260614.md | M0 Candidate Reranker Implementation Plan | Objective; Training Metrics versus Model Selection; Formal Model-Selection Priority; Non-negotiable Data Protocol; Responsibilities; Label Definition; Candidate Pool; Initial Features |
| RERANKER_V4_PLAN_20260615.md | Reranker V4: Confidence-Gated Marginal Fill Plan | Objective; Architecture; Leakage Controls; Gate Calibration; Comparative Results; Decision; Next Research Step; V4.1 Date-Level Meta Gate Result |
| RESEARCH_PROTOCOL.md | Research and Forward-Test Protocol | Frozen boundary; Data directories; Account size; Model-selection rule |
| FROZEN_FORWARD_STRATEGY.md | Frozen Forward Strategy | Primary; Fallback; Frozen Evidence; Forward Rules |
| TEST_PLAN.md | V9 Small-Account Test Plan | Fixed constraints; Execution table; Candidate order; Memory rules; Decision gates; Initial findings; Locked research result |
| SHARPE_OPTIMIZATION_REPORT.md | Small-Account Sharpe Optimization | Experiments; Results; Shadow Result; Decision; Evidence |
| LOSS_ABLATION_PLAN.md | V9 Loss Ablation and Trading-Objective Plan | Objective; Fixed Research Boundary; Fixed Hardware Configuration; Metrics Recorded Per Epoch; Phase 1: Baseline A; Phase 2: Minimal Loss Ablation; Phase 3: Component Decision Rules; Phase 4: Trading Objectives |
| PURGED_ALPHA_OPTIMIZATION_PLAN.md | Purged Alpha Optimization Plan | Objective; Fixed Boundaries; Baselines; Experiment Sequence; Pre-Registered Alpha Candidates; Promotion Gate; Completed Screen; Resource Policy |
| CANDIDATE_MODEL_VALIDATION_PLAN_20260614.md | Candidate Model Validation Plan | Objective; Research Boundary; Candidate Checkpoints; Fixed Alpha Construction; Fixed Portfolio Configuration; Execution Scenarios; Metrics; Promotion Gate |
