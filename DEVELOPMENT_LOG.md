# Development Log

## 2026-07-16 - Qlib Adoption Phase B

- Upgraded `experiments.recording` to formal schema v2 with mandatory source
  fingerprints, warm-up/train/valid/signal/backtest ranges, transform state,
  split roles, max-data date, and forward-parent lineage.
- Added terminal completion, artifact-index existence, and SHA-256 validation
  before formal leaderboard/scorecard use. The artifact index is rebuilt
  atomically from append-only events so later ledger stages are not omitted.
- Added actual result-date validation against canonical 2024 Val, 2025 Test,
  and 2026 Forward contracts. Formal registry ingestion rejects missing or
  out-of-range signal/backtest dates.
- Classified all 96 historical registry rows as `legacy_registered`; all split
  roles validate, and none is silently upgraded to formal evidence.
- Phase A+B focused regression: 86 tests passed in the Torch environment.

## 2026-07-16 - Qlib Adoption Phase C Started

- Added a versioned declarative workflow contract covering data, ranges,
  features, labels, windows, model, checkpoint, alpha, strategy, realistic
  ledger, evaluation roles, stresses, capital, and reports.
- Added a compiler that fingerprints physical inputs and freezes a formal
  manifest plus existing-runner stage graph before expensive execution.
- Added a restricted resumable executor with dependency checks and
  command-hashed stage receipts. Qlib/proxy execution is rejected; the graph
  delegates to project-native rolling, official open-ledger, and scorecard
  entrypoints.
- Compiled and formally validated the frozen registry baseline replay example.
  An initial smoke exposed missing script path bootstrapping and a global
  scorecard-isolation leak; both were fixed with script-level tests and
  workflow-local report registries.
- Completed the v4 frozen-alpha ledger replay: 16 expected 2024/2025 cells,
  all four stresses, both 50/100W capitals, zero coverage gaps, and no Forward
  rows. Mean selection Sharpe was 2.0617 for the existing formal baseline; this
  is a reproducibility replay, not a new performance claim.
- Added rolling-manifest alpha resolution and an experiment-local candidate
  adapter, avoiding alpha copies and global candidate promotion during
  research. Phase C is complete; no rolling training was started.
- Phase A-C focused regression: 97 tests passed.

## 2026-07-16 - Qlib Adoption Phase D

- Added project-native `DataView`, OHLC matrix, v14 memmap, fundamental PIT,
  external-market PIT, and execution-constraint provider contracts.
- Added processor classifications and train-fit boundary/hash validation.
- Enabled an explicit rolling-only physical-cache superset mode. It reused the
  existing `end20260518` v14 cache under a 2025-12-31 logical cutoff and created
  no duplicate 2025 cache files; legacy callers remain strict.
- Real rolling dry-run resolved 242 Val-2024 and 243 Test-2025 prediction days.
- Real provider audit covered five providers. OHLC has 5,332 codes; the
  fundamental source has 321,598 rows across 5,322 codes; global features use
  only completed prior sessions. The only blocker is missing complete
  historical ST events.
- Phase D completed with that declared external-data limitation. Phase E
  monthly walk-forward construction started; no monthly training was launched.

Append-only record of material engineering and governance changes. Experiment metrics remain in `reports/` and `registry/`.

## 2026-07-12 - Qlib-Inspired Research Framework Governance

- Added ADR 0004 to establish a project-native, manual-shadow research
  framework inspired by Qlib experiment, processor, rolling, and lifecycle
  patterns.
- Preserved existing A-share PIT inputs, `oo_lag1` label semantics, realistic
  open-price share-ledger execution, frozen research boundary, and registry
  authority.
- Declared the initial post-framework model experiment as fixed-window monthly
  walk-forward retraining: trailing four-year Train, six-month Valid, one-month
  OOS, with label-tail purge and one continuous ledger path.
- Explicitly excluded automatic trading, automatic retraining, and automatic
  promotion while profitability remains unproven.
- Added `experiments.recording`: immutable experiment manifest, append-only
  event history, artifact descriptors/checksums, and immutable final artifact
  index for research/manual-shadow runs. It does not write formal registry
  entries. Focused alpha/rolling/recording tests: 21 passed.
- Extended `experiments.rolling` with a fixed-length monthly task schedule that
  snaps Train, Valid, and OOS ranges to the supplied A-share trading calendar,
  rejects forward dates, and guarantees a unique OOS owner per signal date.
  It is a scheduling primitive only; no model retraining or performance claim
  has been run. Focused alpha/rolling/recording tests: 23 passed.
- Added a read-only v14 transform/PIT contract auditor and generated
  `reports/qlib_research_framework_20260712/v14_transform_contract.json` for
  the active 2010-01-04 through 2026-05-18 cache (5,178 codes, 3,972 dates,
  258 features). It confirms per-date cross-sectional feature normalization
  rather than a future-fitted global scaler, and explicitly records remaining
  source-hash/publication-time/universe-coverage gaps. Focused tests: 24
  passed.
- Added `run/evaluate_experiment_alpha.py`, a manifest-bound adapter from a
  dated research alpha to the existing realistic open-price ledger sweep. It
  records alpha and request checksums plus dry-run/started/failed/completed
  events, and does not add a second execution engine or alter registry state.
  Focused alpha/recording/rolling tests: 24 passed.
- Added `backtest.execution_coverage` and its CLI audit. The generated
  2024/2025 report confirms the OHLC matrix and listing-date coverage, while
  explicitly gating historical ST realism because the available industry data
  is only a 2026-04-27 snapshot. No execution behavior changed. Focused tests:
  25 passed.
- Began the strategy/execution separation by adding `backtest.strategy`.
  Existing rank-retention target selection now uses that pure module, while a
  separately tested TopK/dropout proposal remains non-runnable until it is
  explicitly wired through the same ledger CLI. No fill/cost/limit logic moved
  out of `open_ledger`. Focused strategy and ledger tests: 63 passed.
- Completed the first strategy interface: explicit TopK/dropout policy settings
  now flow through the same realistic ledger CLI as retention. Policy, TopK,
  and replacement count are included in sweep resume identity; retention
  remains the default. Focused framework/strategy/ledger tests: 116 passed.
- Added audited compact and broad Alpha158-inspired price/volume-only baseline
  specifications. They reuse project PIT technical columns and `oo_lag1`, not
  Qlib provider data or close-return labels; training remains deliberately
  pending a shared feature-subset adapter. Focused tests: 118 passed.
- Completed training and ledger evidence for both independent price/volume
  baselines. The shared realistic sweep covers 2024 val and 2025 test, both
  capitals, and normal/lag1/cost2x/capacity_3pct. Compact leads 2024 val while
  broad leads 2025 test, so neither is promoted; the comparison is recorded in
  `reports/qlib_research_framework_20260712/factor_baseline_ledger_comparison_20260713.md`.
- Completed the fair full-feature comparison on 2026-07-15. Added expanded v14
  feature-layout provenance, low-memory LightGBM dataset release, optional
  payload omission for rolling samples, and dry-run/formal manifest collision
  protection. Clean v14, compact, and broad runs use the same 100k/30k sample
  cap and four threads, then complete realistic Val/Test ledger evidence for
  both capitals and four stresses. The 48-row comparison is recorded in
  `reports/qlib_research_framework_20260712/factor_baseline_full_feature_comparison_20260715.md`;
  compact leads 15/16 Sharpe cells and is the next bounded-tuning research arm,
  not an automatic production promotion. Full repository pytest remains
  environment-blocked by missing `torch`; focused framework tests pass.

## 2026-07-11 - Engineering Governance Bootstrap

- Added `PROJECT_RULES.md`, `ARCHITECTURE.md`, and ADR records.
- Formalized current Torch/CUDA runtime: PyTorch 2.11.0+cu128, RTX 5070 Laptop GPU.
- Recorded precedence so historical documentation cannot override registry/protocol evidence.
- Added architecture-review template, module-completion gate, ADR schema,
  artifact placement/naming, and token-handling rules after governance audit.
- Moved 12 unregistered, unreferenced June experiment-output directories from
  `reports/` to `archive/experiments_202606/` (about 3.62 GB retained, not
  deleted); see that archive's README for the manifest.
- Moved the remaining 150 reviewed unregistered/unreferenced report folders
  into the same archive and added a generated manifest tool.
- Fixed the official backtest runner so `forward_2026` explicitly reads
  `data/forward_raw`, while validation and test continue to read frozen
  `data/raw`; added command-construction regression coverage.
- Replaced stale archive-only script/test inventory checks with checks against
  the active architecture and project-rule documents.
- Marked the historical two-file sweep as legacy at its CLI boundary and
  documented the registry-driven shared-OHLC path as the only formal route.
- Confirmed a warm two-capital realistic grid keeps mask/context preparation
  outside the grid loop: 6.45 seconds total, with 0.22/0.11 seconds for those
  two preparation stages.
- Extracted reusable alpha persistence transforms. M0 `raw,avg2,avg3,avg5`
  generation now derives rolling variants from ordered per-date score rows;
  the existing V9 score cache already avoided repeated GPU forwards, so this
  is a clarity and CPU-orchestration cleanup rather than a new GPU speed claim.
- Consolidated the three `DLPredictor` inference paths and switched them to
  `torch.inference_mode()` with output-head parity coverage.
- Removed hard-coded user home paths from the retained PowerShell training and
  comparison runners; they now honor `$env:PYTHON` and otherwise resolve the
  Torch environment from `$env:USERPROFILE`.

## 2026-07-10 - Registry And Performance Flow

- Established registry-driven baseline, backtest, attribution, and scorecard flow.
- Added shared open-ledger context and chunked sweep-result writing.
- Kept AMP disabled; added pinned-memory/non-blocking CUDA transfer support.
- Added realistic execution-mask disk cache and performance report output.
  Measured identical cold/warm 2024 ledger results at 31.95s/6.82s.

## 2026-07-15 - Bounded Tuning Confirmation

- Completed the independent-seed confirmation of the compact LightGBM
  `t03_minleaf160` candidate under the same `oo_lag1`, rolling, low-memory,
  realistic open-ledger, four-stress, and two-capital contract.
- The candidate improved 2024 Val but did not reproduce its 2025 Test uplift:
  CNY 1m normal Test was 45.84% annualized / Sharpe 1.633 versus the compact
  baseline's 52.50% / 1.796. The minimum stress Sharpe was 1.288 versus 1.422.
- Recorded a rejection-gate report and complete 16-row evidence matrix under
  `reports/qlib_research_framework_20260712/`. No registry or forward-shadow
  state changed. Phase 4 OOF lineage and simple blend research is now active.

## 2026-07-15 - OOF Lineage And Historical Stability

- Added strict chronological OOF lineage validation and experiment-level
  recording for component alpha artifacts and simple rank blends. The validator
  records component hashes, model ids, train/validation cutoffs, and each
  prediction date; legacy v14 manifests require completed experiment evidence.
- Completed three predeclared equal-weight blend ledger comparisons for 2024 Val
  and 2025 Test. `compact_v14_eq_rank` is the conditional candidate, not a
  production promotion, because its Val drawdown bound worsens.
- Completed compact/v14 historical OOF training for 2018-2023 and a six-year
  realistic ledger/annual rank-diversity audit. The historical 1m normal Sharpe
  is 0.846 and cost2x Sharpe is 0.555. The next work package is state-aware
  portfolio construction on the fixed candidate.

## 2026-07-15 - State-Aware Portfolio Pilot

- Added state-aware selection parameters to the shared open-price ledger sweep,
  including global-feature loading, resumable-key identity, and explicit
  parameter columns in summary output.
- Added the mandatory signal/backtest date fields to each shared sweep row and
  regression coverage for the new CLI/key contract.
- Ran the fixed `compact_v14_eq_rank` baseline and the predeclared
  `risk_rank_t035_p010` proposal on 2024 Val and 2025 Test with the realistic
  ledger, all four stresses, and both capital sizes.
- The proposal is Sharpe non-worse in 16/16 cells and annualized-return
  non-worse in 15/16, but maximum drawdown is non-worse in 13/16 and turnover
  is not uniformly lower. It remains conditional research evidence only.
- Evidence: `reports/qlib_research_framework_20260712/state_aware_portfolio_pilot_20260715.md`,
  its delta CSV, and `configs/state_aware_portfolio_pilot_20260715.json`.

### 2026-07-15 - Phase 5 closure and Phase 6 gate

- Closed the state-aware portfolio-construction pilot after checking both
  proposal families on 2024 Val and 2025 Test, four stress scenarios, and both
  capital levels under the realistic open-price ledger.
- `risk_rank_t035_p010` remains a conditional research candidate. It is
  Sharpe non-worse in 16/16 cells, annualized-return non-worse in 15/16, and
  MDD non-worse in 13/16.
- `risk_suppress_d015` remains exploratory. It is Sharpe non-worse in 13/16,
  annualized-return non-worse in 12/16, and MDD non-worse in 15/16, with lower
  turnover in all 16 cells. Its Val `lag1` weakness and post-attribution
  threshold choice prevent clean confirmation.
- Mandatory date fields are complete in all four full-stress summary files;
  no forward data was used. The closure report is
  `reports/qlib_research_framework_20260712/state_aware_phase5_closure_20260715.md`.
- Phase 6 now starts with frozen research-shadow/rollback artifacts. No new
  Phase 5 threshold or weight search is permitted on the same Val/Test data.

## 2026-07-15 - Phase 6 Shadow Manifest

- Extended the experiment recorder with a working-tree status fingerprint in
  addition to Git `HEAD`; a dirty copied checkout is now visible in every new
  experiment manifest.
- Added `run/create_phase6_shadow_manifest.py` and generated the research-only
  bundle under `reports/experiments/phase6_shadow_bundle_20260715/`.
- The bundle freezes the formal baseline, both conditional Phase 5 artifacts,
  the realistic open-price ledger contract, required scorecard fields, and a
  manual fallback policy. It does not activate a candidate, generate forward
  alpha, or start rolling training.
- Because the current worktree is dirty and no Phase 5 candidate is formally
  promoted, activation remains disabled. The relevant focused tests pass: 75.

- Added `run/validate_forward_shadow_scorecard.py` as a read-only Phase 6
  evidence gate. It checks the frozen source/artifact hashes, forward date
  boundaries, observation-only labels, and the four required attribution
  contributions. With no supplied forward scorecard and no promoted candidate,
  its correct outcome is `not_ready`; it never changes registry state.
- The expanded focused tests pass: 79.
- Regenerated the immutable Phase 6 bundle as
  `reports/experiments/phase6_shadow_bundle_20260715_v4/` after the validator
  entry-point fix. The read-only scorecard validation passes all integrity and
  activation checks and returns `not_ready` only because no forward scorecard
  was supplied; no forward selection, registry change, or rolling training
  occurred.

### 2026-07-15 - Historical ST Contract And Access Gate

- Added the reusable `data/st_status.py` contract for dated ST status events,
  including deterministic activation/removal parsing, effective-date loading,
  schema validation, manifest lookup, and file hashing.
- Added `run/download_historical_st_events.py` with cutoff filtering at
  `2026-05-18`, atomic output, source-cache reuse, and page-level checkpoints
  under `data/tracking_raw`. Tushare credentials are resolved in-process and
  are not printed.
- Updated `backtest.open_ledger` to prefer the historical event file and to
  avoid applying current ST names when an event source exists. Event files and
  manifests are now part of the execution-mask cache key.
- Extended `backtest.execution_coverage` to require valid event rows, a
  manifest coverage range, and a matching output hash. The new audit artifact
  is `reports/qlib_research_framework_20260712/execution_coverage_st_contract_20260715.json`.
- Added ADR 0005 and focused regression coverage. The relevant suite passes
  56 tests; direct CLI audit succeeds after fixing the `run/backtest.py`
  module-shadowing entry-point bug.
- A real download was attempted with the approved local token, but Tushare
  denied access to `st`; `stock_st` was also unavailable. No historical ST
  data was fabricated or written, so the Stage A historical-status gate stays
  open and no rolling experiment or registry change was made.
- The endpoint probe is recorded without credentials in
  `reports/qlib_research_framework_20260712/st_source_access_probe_20260715.json`;
  `namechange` remains a separately audited fallback because the API imposed a
  one-hour frequency limit during the probe.
- Regenerated the source-aligned Phase 6 shadow bundle as
  `reports/experiments/phase6_shadow_bundle_20260715_v5/` after these changes.
  Its validator passes integrity and control checks and correctly remains
  `not_ready` because there is no forward scorecard or promoted candidate.

### 2026-07-16 - Namechange Fallback Adapter

- Rechecked the approved Tushare token once without retries; the preferred
  `st` endpoint still returns permission denied. No bulk retry was performed.
- Added `derive_is_st_from_name` and `normalize_namechange_events` to
  `data/st_status.py`. Historical `namechange` intervals become explicit
  start/end state transitions and are labelled as
  `tushare_namechange_intervals`, not as direct ST events.
- Generalized `run/download_historical_st_events.py` with an explicit
  `--endpoint st|namechange` switch and endpoint-specific raw/checkpoint
  defaults. This prevents a fallback cache from overwriting the preferred
  source cache.
- Strengthened event-file validation for code, effective date, event date, and
  boolean state. Focused ST/execution tests pass: 58.
- The fallback has not been downloaded in full because the endpoint is rate
  limited; no `data/raw/st_status_events.csv` was created. Stage A remains
  gated and no rolling training, registry change, or forward selection ran.
- Regenerated the source-aligned Phase 6 bundle as
  `reports/experiments/phase6_shadow_bundle_20260716_v6/`; the read-only
  validator remains `not_ready` and activation-disabled.

### 2026-07-16 - Explicit Namechange Provenance Label

- Added manifest `source_label=由历史股票名称区间重建` for the explicit
  `--endpoint namechange` fallback. The direct `st` endpoint receives a
  separate label.
- Coverage reports now expose `source_label`; a focused regression test
  verifies the fallback remains distinguishable from direct ST history.
- `py_compile` passed, the ST/execution subset passed 11 tests, and the
  broader focused suite passed 137 tests.

### 2026-07-16 - Source-Aligned Bundle v7

- Regenerated the current Phase 6 source-aligned bundle as
  `reports/experiments/phase6_shadow_bundle_20260716_v7/` after the manifest
  provenance-label change.
- The read-only validator writes
  `reports/experiments/phase6_shadow_scorecard_validation_20260716_v7/forward_shadow_validation.json`
  and reports `overall_status=not_ready` with activation disabled. Source,
  control, and 14 artifact-integrity checks pass; no forward scorecard exists.
- The current focused regression suite passes 138 tests. No rolling run,
  forward selection, or registry promotion was performed.

### 2026-07-16 - Tushare Access Policy Audit

- Verified the current official endpoint documentation: `st` is 6000 points,
  `stock_st` is 3000 points, and a newly registered account receives 100
  points. `namechange` has no separate threshold displayed, but its single-
  probe success does not prove bulk free access.
- Recorded the source links and the unchanged Stage A gate decision in
  `reports/qlib_research_framework_20260712/st_source_access_policy_20260716.md`.

### 2026-07-16 - Downloader Access Failure Fast Path

- Added opt-in non-retryable error markers to `data/api_utils.py` and enabled
  them for `run/download_historical_st_events.py`. Permission/points and
  invalid-token failures now stop immediately; transient failures still use
  bounded retries.
- Added focused API-caller tests. The current relevant suite passes 140 tests.
- Recorded the remaining endpoint-contract caution: official `st` and
  `namechange` documentation does not itself establish generic `offset/limit`
  pagination, so no research cache was published from that assumption.

### 2026-07-16 - Source-Aligned Bundle v8

- Regenerated the current Phase 6 source-aligned bundle as
  `reports/experiments/phase6_shadow_bundle_20260716_v8/`.
- The read-only validator remains `not_ready` only because no forward
  scorecard is supplied; activation and forward selection are disabled. The
  frozen boundary remains 2026-05-18 with `backtest.open_ledger` execution.

### 2026-07-16 - Endpoint-Specific Historical Fetch

- Updated the formal historical downloader to use date-range fetching for
  `namechange` and per-code fetching for `st`, with code-universe fingerprints
  and resumable checkpoints.
- Empty per-code results now reload safely from checkpoints. The focused
  historical-source tests pass 16 tests.
- The code-universe completeness question remains an explicit gate for any
  future formal ST download; no external source was promoted.

### 2026-07-16 - Source-Aligned Bundle v9

- Regenerated the current Phase 6 source-aligned bundle as
  `reports/experiments/phase6_shadow_bundle_20260716_v9/` after the
  endpoint-specific fetch changes.
- Its validator remains `not_ready` only because no forward scorecard is
  supplied; activation and forward selection remain disabled. The frozen
  boundary and `backtest.open_ledger` contract are unchanged.
- The full current focused suite passes 143 tests. No external API request,
  rolling training, or registry promotion was performed.

### 2026-07-16 - Explicit Research/Forward Dataset Roles

- Formalized the distinction between the frozen research cache and later
  forward observations in the historical-event downloader. `research` writes
  to `data/raw` with the `2026-05-18` cutoff; `forward` writes to
  `data/forward_raw` and requires dates after the cutoff. Tracking/checkpoint
  roots follow the same separation.
- Added dataset-role and selection-eligibility fields to source manifests and
  required the execution-coverage audit to match the requested role.
- Fixed the audit CLI default so `--dataset-role forward` resolves to
  `data/forward_raw` unless an explicit `--data-dir` is supplied.
- Added CLI regression coverage. The focused suite passes 144 tests. The real
  historical ST source remains unavailable, so no coverage gate or rolling
  experiment was promoted.

### 2026-07-16 - Source-Aligned Bundle v10

- Regenerated the Phase 6 source-aligned bundle as
  `reports/experiments/phase6_shadow_bundle_20260716_v10/` after the explicit
  research/forward cache-role changes.
- The read-only validator output is
  `reports/experiments/phase6_shadow_scorecard_validation_20260716_v10/forward_shadow_validation.json`.
  Fourteen artifact hashes and source/control checks pass; the result remains
  `not_ready` with activation and forward selection disabled because no
  forward scorecard or promoted candidate exists.
- The current focused suite passes 146 tests. No real API download, rolling
  training, forward selection, or registry promotion was performed.

### 2026-07-16 - Final Source-Aligned Bundle Preparation

- Completed the role-separation documentation and regression evidence before
  the final Phase 6 snapshot. The final bundle is planned as
  `reports/experiments/phase6_shadow_bundle_20260716_v12/`; its source-state
  fingerprint will include the completed audit records.

### 2026-07-16 - Canonical Split Protocol (Phase A)

- Replaced the obsolete global 2026-05-18/2026-05-19 boundary with one
  canonical contract: 2024 Val, 2025 Test, and observation-only full-year 2026
  Forward, currently through 2026-06-30.
- Official backtest, registry writer/reader, Phase 6, Forward validator, OOF
  lineage, and ledger evidence now import the same `SplitSpec` definitions.
- Full-year 2026 parent artifacts must finish fitting and selection by
  2025-12-31. Forward registry rows are never selection-eligible.
- Corrected 32 existing Forward report rows; all 96 registry rows pass role
  validation. Focused Torch regression: 67 passed.
- Historical configs retaining 2026-05-18 remain immutable experiment
  metadata; it no longer defines the active Forward split.

### 2026-07-16 - Phase E Formal Monthly Walk-Forward

- Added hash-checked window-level resume and stitched split alpha output to
  `run/rolling_lgbm_alpha.py`. Added immutable failed-stage receipts and child
  rolling resume support to the declarative workflow executor.
- Generated and froze the 4-year Train, 6-month Valid, 1-month OOS schedule.
  All 24 windows completed and produced 485 unique OOS dates: 242 in 2024 Val
  and 243 in 2025 Test. No 2026 Forward data participated.
- Ran one continuous realistic open-price ledger per selection split under
  normal, lag1, cost2x, and capacity_3pct for 50 and 100 万 accounts. The
  isolated scorecard contains 32 rows with zero coverage gaps.
- The monthly Compact candidate was rejected: mean annualized return 10.97%,
  mean Sharpe 0.563, minimum Sharpe -0.001, and worst drawdown 26.10%, versus
  the frozen baseline's 70.22%, 2.062, 1.414, and 18.18% respectively.
- Fixed a workflow contract bug exposed by the smoke run: the scorecard's
  experiment-local reports CSV is mandatory even when global registry append
  is disabled. No global registry row or Forward result was added.
- Phase E is complete as durable negative evidence. Further Compact window or
  parameter searches are stopped; future rolling work must adapt an existing
  strong-model trainer to the proven controller.
- Began the strong-model adapter prerequisite without launching a long run.
  `run/train.py` now supports explicit fixed-window training and validation
  starts, while V9 inference can consume warm-up dates without emitting them.
  Torch tests cover fixed boundaries, invalid overlaps, and avgw3 warm-up
  filtering. The missing original e19 command is documented as a provenance
  gap; reconstructed weights cannot be labelled as the original config.

### 2026-07-16 - Two-Layer Rolling Fairness Audit

- Re-ran `multi_downside_e19` raw and monthly Compact raw through one fixed
  realistic open-ledger contract on 2024 Val and 2025 Test, all four stresses,
  and both capital values.
- Applied the frozen `sa_p05` transform and no-lookahead Ledger Path V3 model
  to Compact using its own previous-day 100w normal ledger diagnostics, then
  repeated the same 32-cell comparison against the formal e19 stack.
- Compact improved under the portfolio stack but remained decisively weaker:
  mean Sharpe 0.728 versus 2.062. This rejects further Compact-specific policy
  tuning and supports adapting the strong e19 training family to the proven
  monthly rolling controller.
- No Forward data, registry mutation, or parameter selection was performed.
  Evidence is recorded in
  `reports/experiments/same_stack_monthly_compact_20260716/two_layer_fairness_report_zh.md`.

### 2026-07-17 - Long-Term Qlib Alignment Roadmap

- Audited the project-native recorder/workflow, Provider/Processor contracts,
  rolling controller, strategy boundary, open ledger, registry, and manual
  Shadow artifacts against the local Qlib source references.
- Recorded that workflow/recording, purged rolling, unique OOS ownership, and
  signal/strategy/executor separation are substantively aligned, while generic
  Processor execution, a strong-model adapter, standardized records, rolling
  ensemble, and online lifecycle remain partial.
- Added `LONG_TERM_QUANT_PLATFORM_ROADMAP_20260717.md`. The next gate is a
  reconstructed e19 one-window rolling smoke; no training or formal governance
  state changed in this documentation-only step.

### 2026-07-17 - Qlib-First Priority Revision

- Changed the immediate priority from strong-model rolling to Qlib-inspired framework alignment.
- Reordered the roadmap as Q0 alignment audit, Q1 declarative workflow, Q2 Dataset/DataHandler/Processor, Q3 Model adapters, Q4 record templates, Q5 Rolling/OOF, Q6 portfolio construction, and Q7 manual Shadow lifecycle.
- Frozen the local Qlib reference at commit `d5379c520f66a39953bad76234a7019a72796fd0`.
- Preserved the project-native PIT data contracts, 2024/2025 selection discipline, 2026 observation-only role, and realistic A-share `open_ledger` execution authority.
- No training, backtest, baseline, or Registry state was changed.

### 2026-07-17 - Q0 Qlib Alignment Baseline Completed

- Frozen the external Qlib reference path and commit and created a
  13-component machine-readable alignment matrix plus Chinese audit and
  terminology documents.
- Added draft `schemas/workflow_v2.schema.json` and a golden A-share workflow
  config. Runtime remains workflow schema v1 until the Q1 compiler migration.
- The draft preserves 2024/2025 selection, observation-only 2026 Forward,
  Valid-only checkpoint selection, 50w/100w, four stresses, and realistic open
  execution.
- Added `jsonschema>=4.26` and contract tests. Q0 plus workflow/recorder/
  provider/rolling/strategy compatibility regression: 41 passed.
- No training, backtest, baseline, global Registry, or candidate state changed.

### 2026-07-17 - Q1 Declarative Workflow V2 Completed

- Added workflow-v2 JSON Schema and semantic validation plus explicit
  normalization to the existing safe stage graph.
- Preserved workflow-v1 replay and source-config hashing. Compiled artifacts
  record the source workflow version separately from stage-graph schema v1.
- V2 currently executes only adapters backed by existing safe runners:
  rolling LightGBM and frozen artifacts. PyTorch/Q3 adapters fail explicitly.
- Updated the compiler entrypoint for v2 selection/observation split roles and
  added CLI-boundary coverage.
- Q0/Q1 focused compatibility regression: 45 passed. No model training,
  backtest, baseline, Registry, or candidate state changed.

### 2026-07-17 - Q2 Dataset And Processor Runtime Completed

- Added a low-memory DatasetH/DataHandlerLP-inspired runtime over streamed
  daily cross sections with raw/infer/learn views and shared/infer/learn chains.
- Train-fitted processors fit only on the named Train segment, persist a
  hash-checked state, reject refit, and cannot silently fit on Valid/Test.
- Added Workflow-v2 construction over the existing V14 memmap provider. No
  physical cache format or large dataset was duplicated.
- Added real micro-memmap, lazy streaming, state roundtrip, Processor-kind,
  and learning-only filter tests.
- Q0-Q2 focused regression: 54 passed. No training, backtest, baseline,
  Registry, or candidate state changed.

### 2026-07-17 - Q4B Automatic Records And Q7A Manual Shadow Lifecycle

- Added a `standard_records` Workflow-v2 stage after the official scorecard.
- Automatic Record materialization aligns dated rolling predictions to raw v14
  labels, computes diagnostic-only signal evidence, and references genuine
  open-ledger equity, positions, orders, rejections, costs, stress cells, and
  decision outputs.
- A real read-only e19/v14 smoke aligned 63 signal dates with 99.879% raw-label
  coverage. It did not train, backtest, select, or promote anything.
- Added a project-native manual Shadow lifecycle with hash-frozen Workflow and
  Record artifacts, append-only hash-chained events, manual state transitions,
  unique dated observations, and terminal retirement.
- Automatic trading, retraining, promotion, and Forward selection remain
  disabled. No live lifecycle was created because no newly executed formal
  Workflow has yet produced the complete production Record bundle.
## 2026-07-17 - Consolidated remaining Qlib alignment gaps

- Reconciled the adoption plan, long-term roadmap, persistent planning files,
  and machine-readable alignment matrix with completed Q4B and Q7A work.
- Classified remaining items as required framework closure, later research,
  external data limitations, intentional exclusions, or optional extensions.
- Fixed the next order to frozen/legacy prediction compatibility, one complete
  formal Workflow bundle, and Q7B daily Shadow/replay before Q5B/Q6 research.
- Verified the alignment contract suite: 4 tests passed. No training, backtest,
  Forward selection, Registry mutation, or baseline change was performed.

## 2026-07-17 - Added frozen dated-prediction Workflow support

- Added a streaming, read-only adapter for registered frozen and legacy alpha
  with PredictionFrame validation, split bounds, source hashes, and provenance.
- Reused its split-alpha resolver in the official ledger and made frozen
  Workflow stages materialize evidence instead of acting as no-ops.
- Automatic standard Records now accept and hash-check either rolling or frozen
  dated prediction manifests.
- Normalized historical list-valued and code-mapped alpha scores without
  changing ranked codes or ledger behavior.
- Validated the formal baseline's complete 2024/2025 signal files and passed
  the full test suite: 510 passed. No training, backtest, Registry change, or
  promotion was performed.

## 2026-07-17 - Accepted first complete formal Workflow

- Ran the frozen formal baseline through prediction lineage, 16 realistic
  Val/Test ledger cells, scorecard, and six immutable Records.
- Fixed full-path Windows filename bounds, official child-process failure
  propagation, and empty optional Forward CSV handling.
- Preserved the first failed experiment, failed receipts, and the second run's
  partial Record inputs; resumed only the final failed stage.
- Verified formal manifest/index completeness and Record bundle hash
  `f47404016c501084ff4e5e8ab54b2695b404528b570862074d5d6dd45d0c6234`.
- Created a real manual lifecycle in `prepared` state for Q7B. No activation,
  Forward selection, global Registry mutation, promotion, or trading occurred.
- Full regression: 514 passed.

## 2026-07-17 - Q7B Daily Shadow And Deterministic Replay

- Added a project-native daily runner that consumes frozen dated predictions
  and delegates execution to the sole formal realistic open-price ledger.
- Added one immutable package per signal date with next-session proposal,
  equity, diagnostics, positions, orders, rejections, costs, drift, and
  silent-failure checks.
- Kept lifecycle controls strict: historical replay is allowed in `prepared`,
  while formal observation requires a prior manual transition to `shadow`.
- Added true replay by re-executing into a separate directory and comparing
  path-independent economic-semantic hashes.
- Completed a real 20-signal-day historical acceptance and independent replay;
  semantic SHA-256 is
  `7c82628f4e1f3e92b1667664f7a788b7a97ca041bd5598de96dc340f0d0dce8d`.
- Rebuilt the 5,332-code OHLC matrix cache on first run; cached 20-day run plus
  replay completed in about six seconds. No candidate, Registry, lifecycle
  state, model, or Forward-selection rule changed.

## 2026-07-18 - Qlib Reverse Alignment And Cleanup Audit

- Superseded the stale 2026-05-18 boundary wording with ADR 0007: 2024 Val and
  2025 Test select; all 2026 data is Forward-only; 2026-05-18 is legacy cache
  provenance.
- Corrected the alignment matrix: Dataset/DataHandler, Model adapters, and the
  generic Recorder are implemented interfaces but not yet the sole formal
  training/recording path.
- Distinguished executed rolling `run_mode=formal` from a governance-formal
  experiment and added regression coverage against invalid formal parents.
- Audited disk use without moving or deleting artifacts. Current v14 caches,
  multi_downside lineage, formal baseline evidence, Registry, raw data, and
  open-ledger caches remain protected; rebuildable inference caches and old
  experiment families are listed separately for review.
- No training, backtest, Registry mutation, lifecycle transition, candidate
  promotion, archive move, or deletion was performed.

## 2026-07-18 - Safe Disk Cleanup

- Rechecked runtime processes and text references before deletion; no active
  trainer/backtest used the project, and no config, Registry entry, or formal
  manifest named the old inference cache files.
- Deleted 13 rebuildable inference matrix caches older than the retained three
  `end20260630` caches, releasing 9.231 GiB.
- Deleted Python/pytest caches and 22 obsolete June root logs, releasing about
  7.72 MiB; retained the recent `train_v9.log`.
- C-drive free space increased from 132.65 GiB to 141.89 GiB.
- Preserved all v14 feature caches, model lineage, e22-e25 checkpoints, OHLC,
  raw/Forward data, Registry, formal baseline inputs, and Workflow evidence.
- Revalidated the formal Workflow manifest, all four multi_downside/e15
  evidence hashes, and the three retained inference caches. Full regression:
  518 passed with one existing pandas FutureWarning.

## 2026-07-18 - LightGBM Dataset Runtime Mainline Parity

- Added a rolling Dataset bridge that consumes the exact resolved/purged date
  indices and fails closed on unordered, duplicate, or non-contiguous slices.
- Added explicit `legacy_iter` and `project_dataset` sample-source selection to
  the formal LightGBM runner while retaining the existing sampler, model
  parameters, prediction ranking, and artifacts.
- Fixed formal scope recording when a configured calendar start precedes the
  first physical trading day; actual samples were unchanged.
- Added a zero-processor fast path that avoids redundant array copies while
  preserving mapping isolation and defensive copies for real processors.
- Completed three formal one-window runs. Legacy, initial Dataset, and
  optimized Dataset produced identical model, raw alpha, and stitched Val
  hashes; optimized Dataset took 307 seconds versus 275 seconds for legacy.
- No Registry, baseline, candidate, ledger, Forward decision, or lifecycle
  state changed. Legacy remains the default pending 2025 parity.

## 2026-07-18 - Single Master Execution Roadmap

- Consolidated the long-term roadmap, Qlib adoption plan, project index, and
  live migration state into `MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md`.
- Defined one P0-P9 sequence covering framework convergence, reproducibility,
  monthly OOF, controlled model research, portfolio construction, promotion,
  manual Shadow, and long-term extensions.
- Added explicit stage dependencies, G0-G6 gates, failure fallbacks, stop rules,
  prohibited shortcuts, and the final industrialization completion definition.
- Made the master plan required reading and marked older plan-specific next-step
  wording as non-normative. Specialized plans may expand only the active stage.
- Current position is P2 training-mainline convergence. No training, backtest,
  Registry, baseline, Forward, or lifecycle state changed.

## 2026-07-18 - P2A LightGBM Dataset Second-Window Acceptance

- Added immutable Legacy and Project Dataset configs for `predict_2025` with
  Train through 2023, Valid 2024, and Test prediction 2025.
- Both paths produced identical date/sample counts, best iteration 63, model
  bytes, raw alpha, and stitched Test alpha.
- Legacy took 322.6 seconds and Project Dataset 329.8 seconds. Together with
  the earlier 2024 result, P2A passes byte parity but does not prove a stable
  performance improvement.
- Retained `legacy_iter` as the default and the validated Project Dataset as an
  optional unified interface. No Registry, candidate, ledger, Forward, or
  lifecycle state changed.

## 2026-07-18 - P2B Model Adapter Acceptance And P2C Dataset Start

- Added the exact daily seeded quota sampler and complete LightGBM parameters
  to `LightGBMModelAdapter`; the rolling runner now supports explicit
  `model.runtime=project_adapter`.
- Fixed PredictionFrame compatibility for NumPy tie ordering and legacy date
  serialization after preserving the first immutable mismatch experiment.
- Completed 2024 and 2025 formal Adapter runs. Both model and raw alpha files
  are byte-identical to the legacy trainer in both windows.
- Added a strong-model Dataset view carrying all multi-label, raw-return, lag1,
  risk, industry, and mask fields. Synthetic and real-v14 parity against
  `PrecomputedMemmapDataset` passed field by field.
- ADR 0009 separates internal pilot checkpoint saving from formal OOF-ledger
  selection. No Registry, candidate, ledger, Forward, or lifecycle state
  changed.

## 2026-07-18 - P2C Strong Model Adapter Acceptance

- Added a concrete `ExistingStrongTrainerDelegate` that preserves the existing
  `run/train.py` training behavior while exposing fit/predict through
  `TorchStrongAlphaAdapter`.
- Added label-free v14 inference samples, explicit risk-width slicing, and the
  established `v9_rank` score transform to the unified Dataset/Adapter path.
- Completed one immutable CUDA smoke over `oos_2024_01`. Same-checkpoint legacy
  inference and Adapter inference are byte-identical for all 22 signal dates.
- Documented that independent CUDA reruns can have different tensor hashes even
  when validation metrics match; this is training nondeterminism, not Adapter
  inference drift.
- Replayed the completed e19 staged checkpoints for `oos_2024_01`,
  `oos_2025_01`, and `oos_2025_12`; all 63 Adapter alpha rows are byte-identical
  to the prior oracle.
- Closed the P2 engineering gate and moved the master plan to P3 reproducibility
  governance. P4 remains responsible for complete 2024/2025 OOF-ledger model
  selection. Registry, ledger, Forward, and lifecycle state were unchanged.

## 2026-07-18 - P3 Reproducibility Governance Acceptance

- Added `provenance_bundle_v1` with mandatory environment, participating-source,
  data-view, feature-transform, resolved-command, and runtime manifests.
- Dirty worktrees now retain hashes for the actual files involved in a run;
  large repositories and data caches are not duplicated or recursively hashed.
- Added low-overhead runtime sampling for wall time, process RSS, system memory,
  CPU time, and CUDA peak allocation/reservation.
- Added bundle and artifact-index validators requiring all six manifests plus
  the bundle to be hash-valid and explicitly indexed.
- Integrated provenance into a real three-window strong Adapter replay. The 63
  OOS alpha rows remained byte-identical to the staged oracle; the complete
  provenance bundle and terminal event passed validation.
- Moved the master plan to the P4 pre-audit. Existing Compact OOF evidence must
  be checked against the new provenance gate before any expensive rerun.

## 2026-07-18 - P4 Compact Pre-Audit And Strong Launch Planning

- Audited the existing full Compact monthly OOF experiment: 24 windows, 485
  uniquely owned OOS dates, hash-valid window models/alpha, 242 Val dates, 243
  Test dates, and all 16 Compact ledger cells are present.
- Kept Compact as historical controller evidence but not a promotion candidate.
  Its mean Sharpe delta is -1.499 and mean annual-return delta is -59.25 points;
  missing P3 runtime provenance cannot be honestly backfilled after the fact.
- Added P3 provenance generation to the resumable staged strong controller and
  extended runtime metrics to include child-process-tree peak RSS.
- Built the immutable 24-window strong launch plan from measured three-window
  pilot evidence: 456 target epochs, about 16.05 hours, and 14.96 GiB artifacts.
  The disk gate passes with 154.15 GiB free and a 30 GiB reserve.
- Did not launch the expensive full run. The next implementation boundary is
  binding staged full training to the accepted Torch Adapter delegate while
  preserving exact checkpoint transitions and resume behavior.

## 2026-07-18 - P4 Strong Staged Adapter Mainline

- Routed each e6/e15/e19 staged trainer invocation through
  `TorchStrongAlphaAdapter` while preserving the established `run/train.py`
  delegate, exact checkpoint transitions and internal selected artifacts.
- Added one purged ProjectDataset per rolling window, frozen schedule-count
  validation, Train-only processor fitting and persisted Dataset/Adapter
  runtime metadata.
- Added safe adoption of a valid exact checkpoint when training completed just
  before a progress write, plus a controlled stage-boundary pause for real
  resume acceptance and future long runs.
- Real v14 `oos_2024_01` Dataset validation matched 965/117/22 segment dates,
  input width 250 and risk width 59. Full regression: 536 passed with one
  existing pandas FutureWarning. No ledger, Registry, Forward, lifecycle or
  promotion state changed.

## 2026-07-18 - P4-C Controlled Resume Acceptance

- Preserved the first immutable acceptance failure after discovering that
  experiment status does not include `paused`; represented operational pause
  with a running experiment status and a dedicated pause event.
- Completed a clean e6 stage-boundary pause and resume in a new v2 experiment.
  The e6 hash and mtime did not change; resumed stages ran only epochs 7-15 and
  16-19 before producing 22 OOS signal dates.
- Validated 19 indexed artifacts, all six provenance manifests and a
  byte-identical selected-checkpoint Adapter replay.
- Added predeclared exact/selected inference profiles with isolated split Alpha
  and rolling manifests. The refreshed 24-window launch plan estimates 16.05
  hours and 15.04 GiB while retaining the 30 GiB disk reserve.
- Observed one transient system-memory reading below 0.75 GiB, but not the three
  consecutive readings required to stop. No ledger, Registry, Forward,
  lifecycle or promotion state changed.

## 2026-07-18 - Model Experiments Branch Baseline

- Audited 116 tracked changes and 420 collapsed untracked entries before
  committing; expanded local artifacts contained about 10,800 generated files.
- Fixed direct-entrypoint import precedence so `run/backtest.py` cannot shadow
  the project `backtest` package. Full Torch regression: 547 passed with one
  pandas FutureWarning.
- Committed the governed research/runtime framework, governance and Registry
  contracts, and archived-report cleanup as three reviewable commits.
- Added explicit ignore policy for reports, checkpoints, global downloads,
  legacy root launchers and named experiment outputs. Local artifacts were
  retained on disk and excluded from Git rather than deleted.
- Verified every tracked report deletion against its retained archive copy;
  47 are byte-identical and five archive versions are later supersets.

## 2026-07-19 - Standalone Repository Migration

- Preserved the dirty legacy optimized checkout as isolated branch
  `legacy/optimized-pre-model-exp-20260719` at `c91eb2a`; it is historical
  evidence and must not be merged into accepted branches.
- Copied its two unique research files to the ignored local archive
  `archive/legacy_optimized_20260719/` and verified their SHA-256 hashes.
- Fast-forwarded local `master` to `model-experiments`, then replaced the
  linked-worktree metadata with a no-hardlink standalone Git repository while
  retaining the original GitHub remote and branch history.
- Verified the independent repository with `git fsck` and the full suite:
  547 passed with one existing pandas FutureWarning.
- Removed the temporary migration clone and retired the obsolete 2.98 GiB
  `deepseek_optimized` checkout. `deepseek_model_exp` is now the sole active
  repository for this development line.

## 2026-07-19 - Non-Training Research Closure Plan

- Added a subordinate ten-stage implementation plan for repository durability,
  baseline and artifact freezing, data/execution audits, fixed 24-cell replay,
  existing-candidate review, attribution, backtest parity optimization,
  prepared-only Shadow replay, governance closure and a final decision review.
- Kept `ledger_path_v3_t0001_nolookahead` as the Registry-defined formal
  baseline and treated `multi_downside_e19` as an Alpha/candidate family rather
  than silently changing the baseline.
- Explicitly prohibited training, Forward-based selection, parameter sweeps,
  Qlib Executor substitution and automatic `prepared -> shadow` transition.
- Switched the active planning pointer to
  `.planning/2026-07-19-non-training-closure/`.

## 2026-07-19 - Plan Governance Consolidation

- Kept one authoritative execution sequence in the master plan and one current
  execution ledger through `.planning/.active_plan`.
- Added `.planning/README.md` as a status index and classified older Qlib,
  Rolling, reranker, loss, validation, cleanup and migration plans as paused,
  transferred, complete, closed or superseded evidence.
- Corrected the stale Qlib `in_progress` label and current-index P2 statement;
  no historical plan or report was deleted or moved, avoiding broken links.
- Added a rule that new plan files require a distinct stage boundary, protocol,
  owner, deliverable and acceptance gate; covered subtasks must update the
  active plan instead.

## 2026-07-19 - NT0 Repository Durability

- Audited tracked object sizes and sensitive-value patterns without printing
  matching values; only the explicitly permitted Tushare-token references were
  present.
- Fast-forwarded remote `master`, created remote `model-experiments`, preserved
  the isolated legacy snapshot under an archive branch, and published annotated
  tag `accepted-research-20260719` without force-pushing or changing `main`.
- Verified recovery from a fresh shallow clone at exact commit `83bba60`, with
  clean status, valid Git objects and all critical governance/runtime files.

## 2026-07-19 - NT1 Formal Baseline Freeze

- Froze `ledger_path_v3_t0001_nolookahead` as an explicit JSON contract with
  fixed Val 2024/Test 2025 selection, Forward 2026 observation, 50w/100w,
  four stresses and realistic open-ledger execution.
- Added canonical evidence and supersession lineage to Registry so formal v4
  Val/Test replay replaces duplicate legacy/v2 evidence without deleting audit
  history. Forward legacy evidence remains observation-only pending NT3 replay.
- Verified 23 required artifacts with no missing files and confirmed the eight
  common v2/v4 ledger summaries are byte-identical.
- The canonical scorecard contains 24 unique cells with complete coverage.
  Focused tests passed 17/17; full regression passed 550 tests with one existing
  pandas FutureWarning.

## 2026-07-19 - NT2 Data, PIT And Execution Audit

- Audited separate logical research and Forward views over 5,332 readable
  stock files. Research is capped at 2025-12-31 despite a physical 2026
  superset; Forward is covered through 2026-06-30.
- Confirmed complete OHLCV/money and listing-date coverage for all three
  evaluation splits while retaining `historical_st_status_not_covered` as a
  formal blocker. A current-name snapshot is not historical ST evidence.
- Changed provider-contract audit to be split-aware and non-mutating. Cache
  identity mismatch and unavailable v14 Forward coverage are now emitted as
  blockers rather than causing an implicit OHLC cache rebuild.
- Documented effective-date fundamental quality flags, strictly prior US/HK
  session alignment, missing corporate-action lineage, and actual A-share
  execution-rule samples. No training, backtest or lifecycle transition ran.
- Focused NT2 tests passed 16/16; full regression passed 553 tests with one
  existing pandas FutureWarning.

## 2026-07-19 - NT3 Formal Baseline Dry-Run

- Compiled the frozen baseline into Val 2024, Test 2025 and Forward 2026
  commands covering four stresses and 50w/100w, for exactly 24 cells.
- Verified role-specific data roots, dates, max-data dates and all three alpha
  inputs. No parameter search or Forward-based selection was introduced.
- Kept Registry unchanged and did not start the replay because free memory was
  2.66 GiB, below the predeclared 3 GiB safety threshold. NT3 remains in
  progress; the dry-run command manifest is persisted for resumption.

## 2026-07-30 - NT6 Market Data And Incremental Cache Plan

- Audited the current daily-market storage path: `data/raw` and
  `data/forward_raw` contain more than 10,000 files and about 3.0 GiB, while
  the global OHLC matrix cache is about 0.95 GiB and invalidates as a whole
  after source-file changes.
- Added `NT6_MARKET_DATA_PARQUET_INCREMENTAL_CACHE_PLAN_20260730.md` as the
  subordinate NT6 specification. It retains DataView, Provider and realistic
  open-ledger boundaries while planning one content-addressed Parquet authority,
  immutable manifests, daily transactional updates and month-sharded execution
  caches.
- Required CSV/Parquet/provider/cache and full 24-cell ledger parity before
  changing the formal backend. CSV remains the read-only oracle and immediate
  rollback path during migration.
- Confirmed `pyarrow` is available. DuckDB remains optional and may be added
  only if measured Arrow performance justifies another dependency.
- No data migration, cache rebuild, training, Registry mutation or lifecycle
  transition was started.

## 2026-07-30 - NT6 MD0 And MD1

- Added a read-only market-data profiler and froze the MD0 report under
  `reports/non_training_closure_20260719/nt6_market_data_baseline_20260730`.
  The full 2026 Forward CSV scan read 0.945 GiB across 5,332 files in
  22.91 seconds; the existing 0.954 GiB matrix ends at 2026-06-30 and is stale.
- Added the MD1 content-addressed Parquet store with strict one-date schema
  validation, immutable manifests, hash-verified CURRENT pointers, explicit
  revision permission, writer locking and failure-safe staging cleanup.
- Added ADR 0010 and retained CSV as the default parity/rollback backend.
- Focused MD0, MD1 and compatibility tests passed 23/23; the full suite passed
  576 tests with one pre-existing pandas FutureWarning. No full migration,
  matrix rebuild, ledger replay, training or lifecycle transition ran.

## 2026-07-30 - NT6 MD2 And MD3

- Migrated 13,313,700 legacy CSV rows from 2010-01-04 through 2026-07-29 into
  4,023 daily candidate Parquet partitions. All 17 annual exact comparisons and
  the complete CURRENT/root/month-index/partition hash audit passed.
- Added a resumable direct daily writer with strict prevalidation, progress
  locking, explicit revisions, partial-failure evidence and lightweight active
  state snapshots. The formal CSV backend remains unchanged.
- A real isolated 2026-07-29 pilot committed 5,524 Tushare equity rows and four
  AkShare broad indices in about six seconds. A fresh-progress replay was a
  content no-op and retained the same active manifest hash.
- Tushare `index_daily` is code-specific and rate-limited to one call per minute
  for the current account. AkShare is therefore the default broad-index client;
  its unavailable amount is explicitly recorded as a zero placeholder, not an
  observed value. Tushare index mode remains opt-in.
- MD4 Provider dual-read parity is the next gate. No cache switch, ledger replay,
  training, Registry mutation or lifecycle transition ran.
