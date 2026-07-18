# Progress

## 2026-07-12

- Read research protocol, registry values, architecture, rules, current index,
  development log, ADRs, and OOF proxy result.
- Verified rolling prototype implementation, generated raw signals, and focused
  tests. Found incomplete artifact persistence and manifest-overwrite gap.
- Wrote the scoped Qlib adoption plan. No research/model/backtest behavior was
  changed.
- Expanded the plan with a complete Qlib component-to-project mapping and
  explicit work packages for recorder, factor baselines, tuner, ensemble
  lineage, risk-aware construction, and online lifecycle.
- Corrected one inspection command that used the project repo as the Qlib
  reference root; the actual reference root is under Documents and was read
  successfully on the second attempt.
- Reordered the plan after user clarification: complete the reusable research
  framework first, then run fixed-window monthly walk-forward training as the
  first formal experiment on that framework.
- Added an A-share-specific Stage A acceptance gate so generic Qlib patterns
  cannot bypass local information timing, universe, price-limit, execution, or
  capacity constraints.
- Completed Phase 0 governance and Phase 1a experiment provenance. Added ADR
  0004 and `experiments.recording`; focused recording/rolling/alpha tests pass
  (21 passed). No model, policy, data, or forward evidence was changed.
- Completed Phase 1b: added trading-calendar monthly window generation and
  unique-OOS-owner validation. Focused recording/rolling/alpha tests pass
  (23 passed). No model training or backtest was run.
- Completed Phase 2: generated the active v14 transform contract under
  `reports/qlib_research_framework_20260712/`; focused tests pass (24 passed).
  The report records rather than conceals source/publication/universe coverage
  gaps. No feature values or labels were changed.
- Added the Qlib TopK/dropout strategy pattern and separately timestamp-audited
  close/mixed execution research to the plan; neither changes the current
  formal T-close signal to T+1-open ledger protocol.
- Completed the Phase 3 alpha-to-ledger adapter. `run/evaluate_experiment_alpha.py`
  records a dated alpha, exact ledger command, dry-run/started/failed/completed
  events, and resulting sweep summary under an existing experiment manifest;
  it delegates all fills to `run/sweep_open_price_ledger_params.py` in
  realistic mode. Focused suite: 24 passed.
- Completed the A-share execution-coverage audit. The new machine-readable
  report confirms OHLC/listing coverage for 2024/2025 and records historical
  ST coverage as a named blocker rather than silently treating a 2026-04-27
  industry snapshot as historical status data. Focused suite: 25 passed.
- Started the project-native strategy contract. Extracted the current
  rank-retention target policy into `backtest.strategy`, added a pure
  TopK/dropout proposal, and made `open_ledger` call the extracted retention
  policy. The ledger remains the only executor; targeted tests: 63 passed.
- Completed the strategy contract. `--selection-policy topk_dropout --top-k N
  --n-drop M` now runs through the existing realistic ledger, while retention
  remains default. Added sweep-key identity protection for policy/TopK/drop
  settings. Focused suite: 116 passed.
- Started the independent factor-baseline work. Defined and audited compact and
  broad price/volume-only feature specifications from the existing v14
  contract; both are ready for a future shared training adapter. No model was
  trained. Focused suite: 118 passed.

## 2026-07-15

- Began the full-feature v14 comparison. Dry-run exposed that the 258 cache
  dimensions are expanded aggregate/rank/industry features rather than the 34
  raw column names; rolling provenance now records the expanded layout.
- Added optional omission of unused risk/industry payload copies and released
  sampled arrays after LightGBM Dataset construction. Focused rolling suite:
  13 passed.
- The first full v14 process with the 250k/75k cap was interrupted before any
  model artifact was written. Machine state was 16 GB total and about 5 GB
  free. No Python process remains. Next action is a common 100k/30k low-memory
  rerun for all three arms before any ledger comparison.
- The v14 low-memory two-window rerun completed successfully. Compact also
  generated both windows, but finalization collided with a prior dry-run
  `rolling_manifest.json`; it is recorded as failed provenance and will be
  rerun in a clean directory after the runner guard is addressed.
- Added the runner guard for dry-run/formal manifest collisions. Clean v14,
  Compact, and Broad low-memory runs then completed. All six Val/Test ledger
  evaluations completed with normal/lag1/cost2x/capacity_3pct, both capitals,
  and explicit date fields. The comparison report and 48-row filtered CSV are
  under `reports/qlib_research_framework_20260712/`.
- Full-feature comparison is now complete with a gate. Compact led 15/16
  Sharpe cells and is the next bounded-tuning research arm; no registry or
  formal production promotion was made.
- Evidence audit confirmed 48 rows, all four required date fields populated,
  and six experiment manifests ending in `ledger_completed`. Full pytest
  collection is blocked by missing `torch`; rolling/recording focused tests
  remain green.

- Completed the independent-seed confirmation for bounded-tuning candidate
  `t03_minleaf160`. The confirmation used seed `20260713` with the same
  compact features, `oo_lag1` label, rolling windows, sample caps, threads,
  realistic ledger, four stresses, and two capital levels. Both 2024 Val and
  2025 Test ledger evidence completed.
- The t03 Test uplift did not reproduce: at CNY 1m normal, the confirmation
  reached 45.84% annualized / Sharpe 1.633 versus the compact baseline's
  52.50% / 1.796. Its minimum stress Sharpe was 1.288 versus the baseline's
  1.422. Val improved, but the two-period gate failed.
- Recorded the rejection in
  `reports/qlib_research_framework_20260712/compact_lgbm_t03_confirmation_20260715.md`
  and its 16-row evidence CSV. No registry promotion or forward selection was
  performed. The next active phase is chronological OOF lineage and simple,
  predeclared rank-blend research.

- Completed the initial Phase 4 lineage pilot for the formal compact, broad,
  and v14 rolling runs. Added strict lineage validation, legacy-manifest
  completion evidence, per-row model/train/validation metadata, and three
  predeclared equal-weight rank blends. The six Val/Test ledger runs completed
  under the unchanged realistic contract; the complete 48-row comparison is
  in `reports/qlib_research_framework_20260712/oof_blend_ledger_comparison_20260715.csv`.
- `compact_v14_eq_rank` is the only conditional blend candidate: it wins all
  eight Test Sharpe/annualized-return cells and improves the Val aggregate,
  but worsens the Val maximum drawdown bound. The broad-containing blends are
  not retained for further tuning.
- Completed six historical OOF windows for compact and v14 covering
  2018-2023, then ran the historical blend ledger and yearly rank-diversity
  diagnostics. The six-year 1m normal result is 17.71% annualized / Sharpe
  0.846, with cost2x Sharpe 0.555; this is materially weaker than the 2025
  Test uplift and prevents automatic promotion.
- The next active phase is the state-aware portfolio-construction pilot using
  the fixed compact-v14 blend. No forward data was used for any decision.

- Completed the first Phase 5 proposal family on the fixed
  `compact_v14_eq_rank` alpha: execution-date `global_defensive_pressure`
  `risk_rank` with threshold `0.035`, width `0.055`, and rank penalty `0.10`.
  The sweep entry now accepts all state-aware parameters, loads the global
  feature frame when selection is enabled, and includes those parameters in
  the resumable sweep key.
- Fixed a formal-report gap in the shared sweep: every row now records
  `signal_start`, `signal_end`, `backtest_start`, and `backtest_end`. The
  focused sweep/open-ledger tests pass (`89 passed` for the Phase 5 subset).
- Re-ran baseline and `risk_rank_t035_p010` on 2024 Val and 2025 Test under
  realistic open-price execution, four stresses, and CNY 500k/1m. The
  candidate is Sharpe non-worse in 16/16 cells and annualized-return
  non-worse in 15/16, but maximum drawdown is non-worse in only 13/16 and
  executed turnover is not reduced. It remains a conditional research
  candidate; no registry or forward promotion was made.
- Recorded the comparison in
  `reports/qlib_research_framework_20260712/state_aware_portfolio_pilot_20260715.md`
  with the full delta CSV and frozen proposal config. The next action is
  trade-level changed-replacement attribution, followed by the second
  predeclared replacement-suppression family only if the attribution supports
  it. Forward data remains observation-only and was not used here.

- Completed the first proposal-family attribution. Direct pressure-triggered
  changes occurred on 43 Val and 51 Test days, while retention created 173-188
  later path-divergence days depending on capital. The candidate reduced beta
  and specific volatility on average, but the path effect is not uniformly
  positive; no threshold tuning was performed.
- Implemented the second planned family, `risk_suppress`: during an active
  global-pressure state, keep the incumbent when the proposed replacement's
  interpretable risk score exceeds the displaced incumbent by at least `0.15`.
  The rule does not alter gross exposure, target count, or execution
  constraints. Its suppression count and risk delta are included in ledger
  diagnostics and resumable sweep identity.
- Completed the exploratory `risk_suppress_d015` run on 2024 Val and 2025 Test with normal/lag1/
  cost2x/capacity_3pct, both capital levels, and the unchanged realistic
  open-price ledger. It is Sharpe non-worse in 13/16 cells, annualized-return
  non-worse in 12/16, and maximum-drawdown non-worse in 15/16. Turnover is
  lower in all 16 cells, but Val lag1 has negative return/Sharpe deltas. It is
  an exploratory risk candidate, not a clean confirmation or promotion: the
  exact `0.15` threshold was fixed after the first-family attribution.
- Completed normal-path attribution for `risk_suppress_d015`. The four
  capital/split paths show lower beta, specific volatility, turnover, and cost
  on average; direct state-change days are mostly positive, but path-only
  effects are mixed. The two Phase 5 proposal families are frozen as research
  artifacts, but only `risk_rank` was fixed before its own Val/Test run; the
  suppression result must remain explicitly exploratory. Phase 6 is the next
  active phase: manifest/date audit and forward-shadow preparation only; no
  forward data has participated in a selection decision. Any new research
  rule must be committed with a config hash before its evaluation window is
  read.

## Phase 6 Shadow Preparation

- Extended `experiments/recording.py` so each manifest records both the Git
  revision and a working-tree status fingerprint. This matters for the copied
  checkout: the current repository is dirty, so `HEAD` alone cannot identify
  the runnable source state.
- Added `run/create_phase6_shadow_manifest.py`, which freezes the formal
  baseline, the two conditional Phase 5 artifacts, the alpha source, the
  realistic ledger contract, required scorecard fields, and a manual rollback
  policy into one research-only experiment bundle.
- Generated
  `reports/experiments/phase6_shadow_bundle_20260715/`. The manifest records
  `working_tree_dirty=true`, `forward_selection_allowed=false`, and
  `current_forward_activation_allowed=false`. The activation log explicitly
  keeps `ledger_path_v3_t0001_nolookahead` as the manual fallback.
- The bundle is preparation evidence, not an active forward strategy. No
  forward alpha was generated, no registry candidate was promoted, and no
  rolling experiment was started.
- Added focused tests for source-state provenance and Phase 6 rollback/config
  contracts. The relevant suite is green: 75 passed.
- Added `run/validate_forward_shadow_scorecard.py`. It is read-only with
  respect to the registry and checks frozen source/artifact integrity, the
  2026 forward boundary, unique dated signals when supplied, and the four
  required model/execution/cost/portfolio-construction contribution fields.
  A signal on 2026-05-18 is allowed only as the final research-date signal;
  execution must begin on 2026-05-19.
- The validator reports `not_ready` when no forward scorecard is supplied and
  never turns that state into an active strategy. Phase 6 remains gated because
  there is no formally promoted Phase 5 candidate and the copied checkout is
  not a clean release snapshot.
- The expanded focused suite is green: 79 passed.
- After the final documentation and validator changes, regenerated the
  immutable bundle as
  `reports/experiments/phase6_shadow_bundle_20260715_v4/`; it supersedes the
  earlier preparation bundles. Its 14 artifact hashes and source-state
  fingerprint validate successfully.
- Ran the read-only validator against the v4 bundle. Result:
  `overall_status=not_ready`, with all integrity/control checks complete and
  only `forward_inputs` not ready. No forward alpha, registry change, or
  rolling training was performed.

## 2026-07-15 - Historical ST Contract

- Added `data/st_status.py` as the reusable historical ST event contract. It
  normalizes codes and dates, derives activation/removal state from event text,
  rejects undecidable rows, and exposes the dated event loader to the ledger.
- Added `run/download_historical_st_events.py`. It keeps the research copy
  capped at `2026-05-18`, stores raw/checkpoint material under
  `data/tracking_raw`, writes a manifest with coverage and SHA-256, and never
  prints the Tushare token.
- Updated `open_ledger` to prefer dated events and to include the event file
  and manifest in execution-mask cache identity. Updated the coverage audit to
  require a valid event manifest and matching hash before declaring historical
  ST coverage.
- The focused contract/execution suite passes: 56 tests. The CLI audit also
  runs successfully after fixing a `run/backtest.py` path-shadowing issue.
- Actual Tushare download remains externally gated: the approved token has no
  `st` endpoint permission, and no research event file was published. The
  latest audit remains `audited_with_declared_gaps`; Stage A and Stage B are
  not advanced.

- Because the historical-ST implementation changed the source fingerprint,
  regenerated the Phase 6 source-aligned bundle as
  `reports/experiments/phase6_shadow_bundle_20260715_v5/`. Its read-only
  validator passes source/artifact/control checks and remains `not_ready` only
  for the absent forward scorecard; activation is still disabled.

## 2026-07-16 - Namechange Fallback Adapter

- Rechecked the live Tushare token once with no retries; `st` still returns
  permission denied. Repeating the same bulk download would not add evidence.
- Added `derive_is_st_from_name` and `normalize_namechange_events` to
  `data/st_status.py`. They reconstruct state transitions from historical
  `start_date`/`end_date` name intervals and explicitly label the source as
  `tushare_namechange_intervals`.
- Generalized the downloader to `--endpoint st|namechange`, with separate
  default source/checkpoint paths so a fallback cannot overwrite an ST source
  cache accidentally.
- Strengthened event validation to check `ts_code`, `imp_date`,
  `event_date`, and `is_st`. Focused ST/execution tests now pass: 58.
- The adapter is implementation-complete but has not been given a full API
  download. No `data/raw/st_status_events.csv` was created, so the historical
  ST gate and Stage B rolling gate remain open.
- Regenerated the current-source Phase 6 bundle as
  `reports/experiments/phase6_shadow_bundle_20260716_v6/`; its validator is
  still `not_ready` with activation disabled.

## 2026-07-16 - Namechange Provenance Label

- Added a machine-readable `source_label` to the downloader manifest. A
  `namechange` output is now explicitly labelled `由历史股票名称区间重建`,
  while the direct `st` source is labelled separately.
- Exposed that label through `audit_execution_coverage` and added a regression
  test proving that `tushare_namechange_intervals` cannot be reported as a
  generic event history.
- Compilation passed and the focused ST/execution suite passed 11 tests; the
  broader focused suite passed 137 tests.
- A verification command initially assumed a nonexistent
  `validation_summary.json`; the actual validator artifact is
  `forward_shadow_validation.json`. This was a read-path mistake only; the
  actual artifact still reports `not_ready` and activation remains disabled.

## 2026-07-16 - Source-Aligned Bundle v7

- Regenerated the source-aligned Phase 6 bundle as
  `reports/experiments/phase6_shadow_bundle_20260716_v7/` after adding the
  provenance label. The read-only validator output is
  `reports/experiments/phase6_shadow_scorecard_validation_20260716_v7/forward_shadow_validation.json`.
- The validator reports `overall_status=not_ready`,
  `activation_allowed=false`; source-state and 14 artifact-integrity checks
  are complete, while `forward_inputs` is intentionally not ready.
- The current focused regression suite passes 138 tests. No forward data,
  model promotion, registry change, or rolling training was performed.

## 2026-07-16 - Tushare Access Policy Audit

- Checked the current official Tushare documentation. `st` requires 6000
  points and `stock_st` requires 3000 points; a new account receives 100
  points. `namechange` has no separate threshold displayed on its endpoint
  page, but that does not establish free bulk-download access.
- Recorded the official links, local token probe results, and the decision to
  keep the historical-ST gate open in
  `reports/qlib_research_framework_20260712/st_source_access_policy_20260716.md`.

## 2026-07-16 - Access Failure Fast Path

- Extended `SafeAPICaller` with opt-in non-retryable markers and enabled them
  for the historical `st`/`namechange` downloader. Permission, points,
  forbidden, and invalid-token errors now fail on the first attempt; transient
  network/frequency failures retain bounded retry behavior.
- Added two focused tests and reran the relevant suite: 140 tests passed.
- The official endpoint pages document code/date filters and row limits but do
  not promise generic `offset/limit` pagination. The current historical gate
  therefore remains closed until a real source download and coverage audit
  prove the fetch contract; no new API call was made in this step.

## 2026-07-16 - Source-Aligned Bundle v8

- Regenerated the current Phase 6 bundle as
  `reports/experiments/phase6_shadow_bundle_20260716_v8/` after the caller
  fail-fast change.
- Its validator reports `overall_status=not_ready`,
  `activation_allowed=false`, `forward_selection_allowed=false`; artifact
  integrity is complete and only the absent forward scorecard is not ready.
- The frozen protocol remains `research_end=2026-05-18`,
  `forward_start=2026-05-19`, and `backtest.open_ledger`. No API download,
  rolling training, forward selection, or registry promotion was performed.

## 2026-07-16 - Endpoint-Specific Historical Fetch

- The formal downloader now uses `namechange_date_range` for the documented
  `start_date`/`end_date` contract and `st_by_ts_code` for the documented
  per-code ST history contract. The old generic offset helper remains only for
  compatibility tests and is not used by the CLI.
- ST code checkpoints include a fingerprint of the requested code universe;
  changing the universe prevents accidental resume from an incompatible cache.
- Empty per-code responses are now valid resumable checkpoints. The focused
  historical-source suite passes 16 tests after this change.

## 2026-07-16 - Source-Aligned Bundle v9

- Regenerated the current Phase 6 bundle as
  `reports/experiments/phase6_shadow_bundle_20260716_v9/` after the
  endpoint-specific fetch changes.
- The validator reports `overall_status=not_ready`,
  `activation_allowed=false`, and `forward_selection_allowed=false`; artifact
  integrity is complete and only the absent forward scorecard is not ready.
- The full current focused suite passes 143 tests. No real API request,
  forward selection, rolling training, or registry promotion was performed.

## 2026-07-16 - Research/Forward Dataset Roles

- Unified the historical-event downloader and execution-coverage audit around
  an explicit `dataset_role` contract. Research defaults to `data/raw` and is
  capped at `2026-05-18`; forward defaults to `data/forward_raw` and must be
  after that cutoff.
- Forward tracking/checkpoint material is kept under
  `data/forward_tracking_raw`; research tracking remains under
  `data/tracking_raw`. Manifests now record the role, cutoff, output root, and
  whether the data can participate in selection.
- Fixed the coverage-audit CLI so `--dataset-role forward` no longer silently
  reads the frozen research cache when `--data-dir` is omitted. Explicit cache
  overrides remain supported for controlled audits.
- The focused suite passes 144 tests. No forward data was used for selection;
  the historical ST source file is still absent and remains an external-data
  gate.

## 2026-07-16 - Source-Aligned Bundle v10

- Regenerated the source-aligned Phase 6 bundle as
  `reports/experiments/phase6_shadow_bundle_20260716_v10/` after the dataset-
  role and audit-default fixes.
- The validator output is
  `reports/experiments/phase6_shadow_scorecard_validation_20260716_v10/forward_shadow_validation.json`.
  It validates all 14 artifact hashes and reports `overall_status=not_ready`,
  `activation_allowed=false`, and `forward_selection_allowed=false`; the only
  missing input is the intentionally absent forward scorecard.
- The focused suite passes 146 tests. No API download, forward selection,
  registry promotion, or rolling training was performed.

## 2026-07-16 - Final Source-Aligned Bundle Preparation

- All role-separation documentation and regression evidence is now recorded
  before the final Phase 6 snapshot. The final bundle will be
  `reports/experiments/phase6_shadow_bundle_20260716_v12/`; no forward result
  will be used for selection.

## 2026-07-16 - Planbook Corrected To Current Repository State

- Re-audited the live split wrapper, registry decision rules/reports,
  experiment recorder, transform contract, Phase 6 validator, and historical
  ST artifacts before changing the plan.
- Updated `QLIB_ADOPTION_PLAN_20260712.md` with a normative current-state
  contract: 2024 Val, 2025 Test, and full-year 2026 Forward through the latest
  complete date, currently 2026-06-30.
- Explicitly retired 2026-05-19 as a Forward start and required full-year 2026
  parent artifacts to freeze by 2025-12-31.
- Reclassified Qlib borrowing into implemented, partial, and missing pieces.
- Reordered the executable roadmap to protocol unification, mandatory manifest
  schema, one workflow configuration, provider/processor contracts, monthly
  walk-forward, risk-aware construction, and manual shadow lifecycle.
- Moved historical ST acquisition off the critical framework path while
  retaining it as a declared execution-coverage limitation.

## 2026-07-16 - Phase A Completed

- Added immutable `SplitSpec` definitions and centralized selection/forward
  roles in `core/research_protocol.py`.
- Migrated official backtest, registry scorecard, Phase 6, forward validation,
  OOF lineage, ledger evidence, update boundaries, forward generators, and the
  legacy APM adapter to the canonical split contract.
- Corrected all 32 Forward rows in `registry/reports.csv` to
  `selection_eligible=false`; all 96 rows now pass role validation.
- Rewrote `RESEARCH_PROTOCOL.md` and updated README/architecture wording.
- Focused Torch regression: 67 passed. Phase B is now in progress.
- Completed Phase B mandatory experiment schema. Formal runs now require the
  complete schema v2 scope, terminal completion, an artifact index rebuilt
  from append-only events, valid hashes, and actual result dates inside the
  canonical split. All 96 existing registry rows are explicitly legacy; 86
  Phase A+B focused tests pass. Phase C workflow controller is now in progress.
- Started Phase C with `experiments/workflow.py`, a config compiler, and a
  restricted resumable executor. The formal baseline replay config compiled
  and passed manifest validation without executing a ledger. New rolling alpha
  still needs automatic experiment-local candidate and attribution wiring.
- Completed Phase C after fixing two smoke-test findings: child scripts now
  bootstrap the project root, and workflow scorecards use isolated report
  registries plus explicit candidate/split filters. The v4 frozen-alpha replay
  produced all 16 Val/Test stress-capital cells with zero coverage gaps and no
  Forward rows. Rolling manifests now resolve directly to split alpha files
  through an experiment-local candidate adapter. No rolling training started.
  Focused Phase A-C regression: 97 passed. Phase D is in progress.
- Completed Phase D provider/processor contracts. A real five-provider audit
  passed for OHLC, v14, fundamentals, global markets, and execution coverage;
  historical ST is the sole declared blocker. The 2025 rolling dry-run reused
  the existing 2026 physical v14 superset, created no duplicate cache, and
  resolved 242/243 OOS days. Phase E is in progress; no monthly training began.

## 2026-07-16 - Phase E Completed With Rejection Gate

- Added and tested window-level rolling resume, stitched split alpha, and
  resumable workflow failure receipts. Focused workflow/rolling tests pass.
- Compiled and executed
  `workflow_monthly_rolling_compact_4y6m1m_v1_20260716`. All 24 model windows,
  485 OOS dates, Val/Test ledgers, and the isolated 32-row scorecard completed.
- Fixed the missing experiment-local reports registry discovered by the first
  scorecard attempt, then resumed from the scorecard stage without retraining
  or rerunning completed ledger stages.
- The Compact monthly candidate was rejected against the frozen baseline. No
  Forward evaluation, global registry append, or model promotion occurred.
- Phase E is closed as negative evidence. The next model task is a strong-model
  rolling adapter; Phase F must not optimize the rejected Compact alpha.
- Strong-model adapter prerequisites started: fixed train/valid starts and
  stateful-inference warm-up/output separation are implemented and tested. A
  full strong rolling run was not launched because the original e19 command is
  incomplete and the estimated runtime is tens of hours; one-window smoke and
  a reconstructed-profile provenance manifest come first.

## 2026-07-16 - Phase E1 Two-Layer Fairness Audit Started

- Confirmed the formal baseline uses `multi_downside_e19` raw scores only as
  its lowest-level input; `alpha_sa_p05` and Ledger Path V3 subsequently
  rewrite the ranking.
- Located complete 2024 Val and 2025 Test raw e19 signals and the stitched
  monthly Compact raw-rank signals.
- Started an isolated, non-registry fairness audit. The first gate reruns both
  raw signals under one realistic ledger contract; the second gate is the
  same-stack portfolio-policy comparison and runs only when justified by the
  first result.
- Completed 32 raw fairness cells and 32 frozen same-stack cells. The Compact
  raw signal averaged 10.97% annualized and 0.563 Sharpe versus e19 raw at
  64.94% and 1.895. Frozen sa_p05 + V3 improved Compact to 15.17% and 0.728,
  but e19 under the same stack reached 70.22% and 2.062.
- No Forward data, global registry write, parameter search, or candidate
  promotion occurred. The Chinese report is
  `reports/experiments/same_stack_monthly_compact_20260716/two_layer_fairness_report_zh.md`.

## 2026-07-17 - Long-Term Qlib Alignment Roadmap

- Re-audited the local Qlib references and the live project recorder,
  workflow, Provider/Processor, rolling, strategy, registry, and Shadow code.
- Classified capabilities into substantively aligned, partial, and deferred;
  the project is not represented as a complete Qlib-equivalent platform.
- Added `LONG_TERM_QUANT_PLATFORM_ROADMAP_20260717.md` with L1-L8 acceptance
  gates. The next executable milestone is a reconstructed e19 one-window
  strong-model rolling smoke, not further Compact tuning.
- No model training, Forward selection, registry mutation, or formal baseline
  change was performed.
# 2026-07-17 Qlib-first plan revision started

- Re-audited current workflow adapters, providers, processor provenance, rolling contracts, experiment recorder, and strategy boundary.
- Identified that the prior roadmap placed strong-model rolling before generic framework contracts.
- Began rewriting the long-term plan around Qlib component alignment and project-specific A-share acceptance gates.
- No model training, backtest, baseline change, or registry promotion was started.
- Rewrote the long-term roadmap into Q0-Q8 Qlib-first stages and added the superseding priority section to the adoption plan.
- Frozen Qlib reference checkout: `C:\Users\x\Documents\股票预测\references\qlib` at `d5379c520f66a39953bad76234a7019a72796fd0`.
- Immediate acceptance point is now Q0 alignment matrix plus workflow v2 schema draft; e19 rolling moved behind Dataset/Processor/Model/Record contracts.

## 2026-07-17 - Q0 artifacts implemented

- Added a 13-component machine-readable alignment matrix and Chinese audit under `reports/qlib_alignment_20260717/`.
- Added a project terminology table fixing Provider, DataHandler, Processor, Dataset, Model, Record, Strategy, Executor, Workflow, Rolling, OOF, Registry, and Online boundaries.
- Added draft-2020-12 `schemas/workflow_v2.schema.json` and `configs/workflow_v2_golden.json`.
- Kept runtime `WORKFLOW_SCHEMA_VERSION=1`; the v2 schema is intentionally a Q1 migration target, not a false claim of runtime support.
- Added `jsonschema>=4.26` to requirements and installed it into the documented Torch environment.
- New Q0 contract suite: `4 passed`.
- Q0 compatibility gate passed: 41 tests covering alignment, workflow,
  recorder, providers, rolling windows, and strategy.
- README and Architecture now link the Q0 artifacts and distinguish workflow
  schema v2 draft from the active experiment-manifest schema v2.
- Q0 is complete. Q1 declarative Task/Workflow migration is next.

## 2026-07-17 - Q1 declarative Workflow completed

- Added `experiments/workflow_schema.py` for v2 JSON Schema validation,
  cross-field date/governance checks, and explicit compatibility normalization.
- Preserved workflow v1 replay. V2 rolling LightGBM and frozen-artifact tasks
  compile to the proven stage graph; Q3-only adapters fail explicitly.
- Updated the CLI compiler to freeze v2 protocol roles and the original source
  config/hash.
- Added function and CLI-boundary tests for v2 compilation, Forward-selection
  rejection, and unsupported-adapter rejection.
- Q0/Q1 compatibility suite: 45 passed. Q2 is next.

## 2026-07-17 - Q2 Dataset/DataHandler/Processor completed

- Added `data/dataset_runtime.py` with streamed raw/infer/learn data keys,
  shared/infer/learn Processor chains, Train-only fitting, frozen state
  save/load and hash verification, and a DatasetH-like named-segment facade.
- Added stateless PIT/rank/clip/label processors and a streamed train-fitted
  feature standardizer.
- Added Workflow-v2-to-v14 Dataset construction over the existing memmap
  provider with no cache format change or full-date materialization.
- Added lazy-stream, state roundtrip, invalid refit, processor-kind, and real
  micro-v14 integration tests.
- Q0-Q2 focused regression and compile check: 54 passed. Q3 is next.

## 2026-07-17 - Q3 Model Adapter completed

- Added `experiments/model_adapters.py` with one lifecycle and governed factory
  for LightGBM, PyTorch strong alpha, frozen alpha artifacts, and legacy
  read-only signals.
- Added the long-form dated `PredictionFrame` contract plus alpha JSONL
  conversion, provenance manifests, checkpoint save/resume, and frozen-file
  hash verification.
- PyTorch training is injected explicitly; Q3 does not claim to have
  reconstructed the historical e19 trainer. That binding remains Q5.
- Q3 focused and Q0-Q2 compatibility regression: 30 passed. No training,
  backtest, Forward selection, registry mutation, or baseline change occurred.
- Q4 dependent Record templates are now active.

## 2026-07-17 - Q4 dependent Record templates completed

- Added an immutable six-record dependency chain with standard fields,
  artifact paths/hashes, and a complete bundle index.
- Added a JSON Schema and CLI for materializing existing analytics and official
  ledger artifacts without embedding a second implementation.
- Added hard gates for parent records, dated coverage, official open execution,
  50w/100w four-stress coverage, and Forward-selection separation.
- Existing summary-only workflows remain explicitly incomplete because they do
  not preserve every detailed ledger artifact required by the new contract.
- Q0-Q4 focused regression: 49 passed. Q5 strong-model reconstruction is now
  active; no training or backtest has started yet.

## 2026-07-17 - Q5 strong-model cache gate

- Reconstructed and froze the historical e19 architecture, labels, horizon weights, and exact auxiliary-loss coefficients in `configs/reconstructed_multi_downside_e19_profile_v1.json`.
- Built the one-window, one-epoch, raw-signal, nonselecting smoke controller.
- The first execution exposed an unsafe cache-key miss and started rebuilding 5,332 raw CSV files. Stopped it before training and recorded the failure immutably under `reports/experiments/strong_e19_rolling_smoke_20260717`.
- Added a validated explicit-cache loader to both V9 training and inference. Bound Q5 to the matching x_dim=250 v14 `funda` cache and removed future-label availability from the inference-universe filter.
- New cache/strong-rolling focused tests: 13 passed. A replacement smoke will use a new experiment directory; the failed evidence remains intact.

## 2026-07-17 - Q5 runtime smoke and pilot completed

- Replacement one-window smoke completed in about five minutes using the explicit x_dim=250 cache: 963 Train cross-sections, 115 Valid cross-sections, and 22 January-2024 OOS alpha dates.
- A predeclared three-window one-epoch runtime pilot completed for 2024-01, 2025-01, and 2025-12. It produced 63 unique OOS dates, three checkpoints, three raw-alpha files, hashes, append-only events, and a finalized artifact index.
- Added a system-memory guard for the 16 GiB host. No pilot window crossed the 0.75 GiB stop threshold.
- Corrected the next-gate definition: the historical model requires three training stages (1-6, 7-15, 16-19), not 19 epochs of the final learning rate/loss from random initialization.
- Added tested stage commands that resume exact epoch 6 and epoch 15 checkpoints with optimizer reset. The completed one-epoch pilot remains runtime evidence only; the full three-window reconstruction pilot is still pending.
- Chinese status report: `reports/qlib_alignment_20260717/Q5_STRONG_ROLLING_STATUS_ZH.md`.

## 2026-07-17 - Q5 resumable staged pilot controller

- Added `run/rolling_strong_staged_pilot.py` for the actual three-stage e19 reconstruction over the three predeclared pilot windows.
- The controller freezes commands and hashes, persists progress atomically after every stage, verifies target epoch and input dimension from each checkpoint, and resumes an interrupted stage from its latest same-stage checkpoint without resetting optimizer state.
- Completed stages verify both exact and selected checkpoint hashes on resume. Missing artifact events are repaired without retraining.
- Each window emits raw alpha only. The final stitch rejects duplicate OOS ownership and remains nonselecting/nonpromoting.
- Focused staged-controller and strong-rolling tests pass. The complete 19-epoch pilot is the next execution gate.

## 2026-07-17 - Q5 full staged pilot completed and rejected

- Completed all three predeclared windows through exact epochs 6, 15, and 19, with 63 unique OOS alpha dates and no ownership collision.
- Selected final checkpoints were epoch 16 for 2024-01, epoch 19 for 2025-01, and epoch 18 for 2025-12.
- The read-only OO-label audit failed: aggregate weighted-horizon Rank IC was -0.0388 for rolling versus 0.1144 for frozen e19; rolling Top0.6% weighted raw return averaged -1.63% versus 0.39%.
- The main failure is 2024-01. Stage 1 selected epoch 1, but the historical reconstruction forced exact epoch 6 into stage 2 and exact epoch 15 into stage 3, bypassing stage-level validation selection.
- The 24-window run remains prohibited. The next single structural test is selected-checkpoint stage transition on 2024-01 with all other settings frozen.
- Chinese audit: `reports/qlib_alignment_20260717/Q5_STAGED_PILOT_AUDIT_ZH.md`.

## 2026-07-17 - Q5 selected-transition ablation completed and rejected

- Added an explicit, frozen `exact|selected` stage-transition contract and a
  logical resume-epoch override that preserves selected weights while keeping
  the historical 6/15/19 stage boundaries.
- Added one-window selection and safe same-stage resume semantics. Nineteen
  focused tests passed before execution.
- Ran only the predeclared strict-OOS `oos_2024_01` window. No 2026 Forward
  rows, registry entries, formal baseline, or portfolio settings were touched.
- The final selected artifact remained the epoch-7 checkpoint because later
  stage candidates did not improve the frozen validation selection metric.
- The ablation worsened weighted Rank IC from -0.1302 to -0.1725 and Top0.6%
  weighted raw return from -6.95% to -7.84%. It is rejected.
- The 2025-01, 2025-12, and 24-window expansions remain prohibited. The next
  Q5 design step must revisit the strong-model checkpoint objective and
  historical stage schedule using Val-only evidence before another OOS run.

## 2026-07-17 - Qlib-first priority restored after Q5 pilot

- Profitability work is now explicitly deferred to Q5B.
- Live architecture audit found that the strong staged runner still bypasses
  Workflow v2 runtime compilation and does not emit the same standardized
  `rolling_manifest.json` plus split-alpha contract as the LightGBM runner.
- Q5A therefore owns framework binding only: Workflow v2 support,
  hash-checked resumable execution, and learner-neutral rolling artifacts.
- No additional training, OOS selection, Forward evaluation, registry
  promotion, or baseline change is permitted as part of Q5A.

## 2026-07-17 - Q5A strong-model framework binding completed

- Added `torch_strong_alpha` to Workflow v2 validation, normalization,
  compilation, allowlisted execution, and resume handling.
- Extracted learner-neutral split-alpha materialization so LightGBM and
  PyTorch strong rolling use one `rolling_manifest.json` contract.
- Strong staged runs now publish standard rolling artifacts in addition to
  their private resumable progress state.
- Compiled `configs/workflow_strong_e19_framework_pilot_v2.json` without
  execution under
  `reports/experiments/workflow_strong_e19_framework_compile_20260717`.
- Materialized the completed exploratory pilot into
  `reports/experiments/strong_e19_standard_rolling_adapter_20260717` without
  retraining, mutation, promotion, or formal reclassification.
- Focused Workflow/rolling/model compatibility suite: 37 passed.
- Runtime audit identified Q4B as the next framework gate: formal Workflow
  runs still lack genuine order/fill, rejection, position, cost, and equity
  artifacts needed for automatic standard Records. Aggregate diagnostics will
  not be relabeled as those artifacts.

## 2026-07-17 - Q4B native ledger evidence layer completed

- Added an optional stock-level execution trace directly to the existing
  A-share execution function. It records target/executed shares, price, status,
  rejection reason, commission, stamp tax, slippage, and total cost.
- Verified the trace is a side channel: enabling it preserves shares, cash,
  execution summaries, and portfolio returns.
- `run_open_ledger` can now emit dated order and position evidence and includes
  portfolio equity in its return path.
- Detailed sweeps persist equity, diagnostics, positions, orders, rejections,
  and costs under a collision-resistant cell key plus `path_artifact_index.csv`.
- Formal registry backtests now enable detailed evidence and add the index and
  every detail file to the immutable experiment artifact index.
- Final combined execution/Workflow/rolling regression: 120 passed. Automatic
  six-record materialization remains the Q4B task.

## 2026-07-17 - Q4B automatic standard Records completed

- Added `workflow_standard_records` after scorecard in Workflow v2. V1 replay
  stage graphs remain unchanged.
- Added a strict automatic builder that aligns main-candidate raw predictions
  with raw v14 labels, computes diagnostic-only daily Rank IC/Top0.6 evidence,
  references genuine official-ledger equity/position/order/rejection/cost
  artifacts, reuses existing APM attribution analytics, and separates
  Val/Test decisions from Forward observation.
- Missing rolling predictions, cache metadata, labels, ledger detail paths,
  stress coverage, or scorecard artifacts fail explicitly before a complete
  Record bundle can be claimed.
- Synthetic end-to-end evidence generated all six immutable records and their
  dependency bundle. Combined regression reached 123 passed.
- A real read-only v14 smoke aligned 63 exploratory e19 signal dates to raw OO
  horizon-5 labels with 99.879% stock-label coverage. It did not train,
  backtest, promote, or modify Registry/baseline state.
- A fresh compile-only strong Workflow emitted the five-stage graph ending in
  `standard_records`; `executed=false` is frozen in its compile evidence.

## 2026-07-17 - Q7A manual Shadow lifecycle completed

- Reused the standard Record bundle as the sole eligibility input instead of
  extending the hard-coded historical Phase 6 package.
- Added immutable lifecycle and state manifests plus an append-only SHA-256
  event chain for `prepared`, `shadow`, `paused`, and terminal `retired`.
- Every transition requires a named actor, reason, and explicit manual
  approval. There is no active/automatic-trading transition.
- Dated observation artifacts are unique, hash checked, and permanently marked
  nonselecting. Frozen Workflow/Record or observation mutation invalidates the
  lifecycle.
- Added ADR 0006 and a thin CLI for create/transition/observe/status.
- Final combined Q4B/Q7A/ledger/Workflow/rolling regression: 136 passed.
- No real candidate lifecycle was created because production eligibility must
  wait for the first newly executed complete formal Workflow Record bundle.

## 2026-07-17 - Alignment gap consolidation

- Reconciled the current plan with completed Q4B and Q7A implementation.
- Reclassified the remaining work into required framework closure, later
  research work, explicit external limitations, and optional deferred platform
  features.
- Fixed stale roadmap wording that still described Q1-Q4 as missing and Q7A as
  the next phase.
- The immediate order is now: frozen/legacy dated-prediction adapter, first
  full formal Workflow bundle, Q7B daily Shadow/replay, then Q5B/Q6 research.
- No training, backtest, Forward selection, Registry mutation, or baseline
  change was performed by this documentation update.

## 2026-07-17 - Frozen/legacy dated-prediction adapter completed

- Added one shared split-alpha resolver used by both official ledger and the
  new read-only frozen prediction adapter.
- Frozen Workflow model stages now materialize source registry/hash/range/row
  lineage instead of succeeding as a no-op.
- Automatic Records accept dated prediction manifests and recheck every source
  alpha hash; no alpha file is copied or rewritten.
- Added list- and code-mapped alpha normalization because the formal baseline
  contains both historical shapes.
- Real baseline smoke covered 242 Val dates / 1,197,376 stock rows and 243 Test
  dates / 1,213,948 stock rows. Training and promotion flags remain false.
- Full regression passed: 510 tests. The next gate is the first complete formal
  Workflow execution, not Q5B/Q6 tuning.

## 2026-07-17 - First full formal Workflow accepted

- Compiled and executed the frozen formal baseline through dated predictions,
  2024 Val and 2025 Test realistic ledgers, scorecard, and all six Records.
- The first attempt exposed a Windows path-length failure and missing
  subprocess-failure propagation. The second reached Records and exposed a
  zero-byte optional Forward CSV case. All three issues have regression tests.
- Resume reused completed model/ledger/scorecard receipts and ran only the
  failed Record stage; partial Record inputs were renamed and retained.
- Acceptance evidence contains 16 cells, both capitals, all four stresses,
  genuine position/order/rejection/cost artifacts, four completed stage
  receipts, and bundle SHA-256
  `f47404016c501084ff4e5e8ab54b2695b404528b570862074d5d6dd45d0c6234`.
- Created the first real lifecycle in `prepared` state. It is not shadow,
  active, promoted, or trading. Q7B is now the active alignment phase.
- Full regression after fixes: 514 passed.
