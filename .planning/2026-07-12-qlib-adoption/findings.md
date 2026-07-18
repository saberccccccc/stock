# Findings

## Current Architecture

- Formal chain: frozen PIT data -> v14 cache -> alpha -> candidate policy ->
  realistic open-price ledger -> registry evidence -> attribution -> scorecard.
- Formal baseline: `ledger_path_v3_t0001_nolookahead`.
- Selection: 2024 validation and 2025 test. Forward 2026 is observation only.

## Qlib-Style Prototype Status

- Rolling window and label-tail purge code exists and has focused test coverage.
- LightGBM raw alpha was generated for 2024 and 2025.
- Existing output was generated before model-file persistence and its manifest
  was overwritten by a subsequent window invocation; it is diagnostic only.
- Full realistic ledger evidence, attribution, scorecard, and registry entry do
  not yet exist for this prototype.

## Full Qlib Borrowing Scope

- The complete plan now covers experiment recording, processor provenance,
  rolling/purge, OOF ensemble lineage, Alpha158/360-inspired factor baselines,
  bounded tuning, risk-aware construction, and online lifecycle.
- Each is mapped to a project owner and staged so no Qlib default execution or
  proxy evidence can bypass the formal ledger and registry.

## Walk-Forward Position

- The user requires framework work first. Monthly fixed-window retraining is
  not the first implementation priority; it is the first formal model
  experiment after framework completion.
- Planned initial controller protocol: trailing four-year Train, six-month
  Valid, label-tail purge, and one-month OOS step. It stitches unique OOS alpha
  into one continuous realistic-ledger path rather than resetting capital per
  month.

## A-Share Stage A Gate

- The plan now centralizes A-share-specific requirements: T-to-T+1 open timing,
  announcement/effective dates, PIT historical universe, adjusted-feature versus
  raw-execution prices, board/ST/new-listing limits, suspension, lots, costs,
  ADV, external-input lags, and frozen research governance.
- These are acceptance gates before any rolling performance result is valid.

## TopK Dropout And Timing Extension

- Qlib TopkDropout is adopted as a strategy-layer reference, not as an
  execution replacement. The project version must emit desired holdings only;
  realistic open ledger remains the single fill/cost/constraint authority.
- Open-only is the current formal family. Close-only and mixed open/close
  research require separately declared signal cutoff, order type, executable
  price, capacity, and costs; close mark-to-market is not close execution.

## Phase 1a Result

- `experiments.recording` now separates immutable static manifests from
  append-only status/artifact events and a final immutable index.
- This avoids overwriting a previous window manifest, the defect found in the
  initial rolling LightGBM prototype.

## Phase 1b Result

- `MonthlyRollingSpec` generates fixed-history Train/Valid/monthly OOS windows
  from the supplied trading calendar and rejects OOS windows beyond the frozen
  research cutoff.
- It is deliberately model-agnostic and has no training side effect. Per-window
  training, checkpoint selection, alpha stitching, and continuous-ledger
  execution remain later integration work.

## Phase 2 Result

- Active v14 cache: 2010-01-04 to 2026-05-18, 5,178 codes, 3,972 dates, and
  258 feature dimensions.
- Feature normalization is daily cross-sectional; no global train-fitted scaler
  is reused across later rolling windows.
- The audit deliberately leaves source file hashes, external publication times,
  and historical universe/ST/listing coverage as declared gaps for the next
  A-share execution coverage audit.

## Phase 3 Result

- `run/evaluate_experiment_alpha.py` is the project-native bridge from a dated
  alpha artifact to the existing realistic open-price ledger sweep. It creates
  a ledger request artifact and append-only experiment events before executing
  the existing runner; it does not introduce a second executor or modify the
  registry.
- It supports only the formal selection splits (`val_2024`, `test_2025`) and
  fixes realistic execution, four stresses, two capital levels, and the
  frozen split data boundary in the delegated command.
- A dry-run regression test proves manifest, alpha lineage, execution contract,
  and command recording without loading OHLC data. Focused suite: 24 passed.

## A-Share Execution Coverage Result

- `reports/qlib_research_framework_20260712/execution_coverage.json` verifies
  the OHLC matrix has the required open/high/low/close/volume/money fields and
  covers 2024-01-01 through 2025-12-31. `stable_stocks.csv` has 5,332 rows with
  5,332 listing dates.
- The only material gap is `historical_st_status_not_covered`: the currently
  discoverable `stock_industry.csv` is a 2026-04-27 snapshot, not a dated ST
  history over validation/test. Current-name fallback is conservative in some
  cases but cannot prove historical correctness. ST-dependent formal claims
  remain gated until a dated status-event source is downloaded and audited.
- The audit is read-only; no execution rule was weakened or silently changed.
  Focused suite: 25 passed.

## Strategy Contract Progress

- Added `backtest.strategy` as a pure score-to-target-holdings boundary. It
  contains the current rank-retention policy and a Qlib-inspired TopK/dropout
  proposal; neither sees price, cash, ADV, lots, limits, or fills.
- `open_ledger.build_desired_target` now delegates current retention target
  selection to that module, preserving the existing ledger's execution layer.
  TopK/dropout is exposed only as an explicit `selection_policy` on the shared
  realistic sweep CLI; retention remains the default. Strategy name, `top_k`,
  and `n_drop` are part of the resumable sweep key, so its output cannot be
  mistaken for retention evidence. Focused suite: 116 passed.

## Factor-Baseline Progress

- Added two Alpha158-inspired but project-native specifications: compact (15)
  and broad (23) price/volume-only features. They reuse only existing v14
  technical columns and retain the project's `oo_lag1` label/PIT contract.
- `factor_baseline_audit.json` confirms both specifications are available and
  exclude fundamental, macro, industry, restriction, and global-market fields.
  The next required item is a shared feature-subset training adapter; no model
  has been trained or compared yet. Focused suite: 118 passed.

## Factor-Baseline Ledger Result

- Both compact and broad rolling LightGBM experiments completed two windows and
  produced immutable model/alpha artifact records.
- The same realistic ledger sweep completed 2024 val and 2025 test for both
  models, both capitals, and all four required stress cases.
- At 1m normal, compact led 2024 val (38.50% annualized, Sharpe 0.920) while
  broad led 2025 test (70.56%, Sharpe 2.354). The cross-period reversal means
  neither is eligible for automatic promotion; full-feature rolling comparison
  is still required.

## Full-Feature Baseline Diagnostic

- The v14 cache has 258 expanded columns, not 258 raw feature names: five
  aggregate blocks, extra-feature last/qoq blocks, rank blocks, and industry
  relative blocks. The rolling manifest now records this expanded layout.
- A full-feature run using the initial 250k/75k sample cap terminated after
  `rolling_training_started` without model artifacts on a 16 GB machine with
  about 5 GB free memory. It is treated as an interrupted/failed diagnostic,
  not as training evidence.
- The runner now avoids copying unused risk/industry payloads for LightGBM and
  constructs/freezes LightGBM datasets before releasing sampled arrays.
- The next fair comparison uses the same 100k train / 30k validation cap and
  four LightGBM threads for compact, broad, and full v14. Prior 400k/120k
  compact/broad results remain historical evidence and are not mixed with the
  new low-memory comparison.
- A compact rerun generated both model/alpha pairs but could not finalize
  because its output directory already contained a dry-run manifest. The
  runner must distinguish a dry-run artifact from an immutable formal manifest;
  the affected run is retained as failed provenance and will not enter the
  comparison.

## Full-Feature Comparison Result

- A clean v14 full-feature run, clean compact run, and clean broad run all
  completed two rolling windows with the common 100k/30k low-memory cap and
  four LightGBM threads.
- Each arm completed realistic open-ledger evidence for 2024 Val and 2025
  Test across normal, lag1, cost2x, capacity_3pct, and CNY 500k/1m. The
  combined filtered evidence has 48 rows and explicit signal/backtest date
  fields.
- Compact won 15 of 16 Sharpe cells across split, stress, and capital; v14
  won only 2024 Val CNY 500k lag1. Broad did not lead Sharpe in any cell.
- At CNY 1m normal, compact was 42.58% annualized / Sharpe 1.049 on 2024 Val
  and 52.50% / 1.796 on 2025 Test. v14 was 36.14% / 0.948 and 46.19% /
  1.644. This supports compact as the next research arm, not as an automatic
  production promotion.
- Forward data remains unused for selection. Historical ST status remains an
  execution coverage gate.
- The complete repository pytest collection is not currently runnable in this
  Python environment because `torch` is missing. The focused rolling,
  recording, and feature-layout suite passes; this is an environment gate for
  the wider deep-learning tests, not evidence about the LightGBM comparison.

## Rejected Direction

- The historical OOF proxy selector did not transfer to 2024/2025 and did not
  use realistic ledger paths. It must not become the basis for formal selection.

## Bounded Tuning Confirmation Result

- The five-trial compact LightGBM search was completed under one fixed
  low-memory rolling/ledger contract. `t03_minleaf160` was the only apparent
  two-period candidate and therefore received an independent-seed confirmation.
- The confirmation used seed `20260713`; all other model, feature, label,
  window, sample-cap, execution, stress, and capital settings were unchanged.
- The original seed's 2025 Test uplift was not reproducible. At CNY 1m normal,
  confirmation Test Sharpe was `1.633` and annualized return `45.84%`, below
  the compact baseline's `1.796` and `52.50%`. The worst stress Sharpe was also
  lower (`1.288` versus `1.422`).
- The candidate is rejected for promotion and forward use. The compact
  baseline remains the fixed research arm. This is evidence of seed-sensitive
  tuning, not evidence that the feature baseline itself is invalid.
- Phase 4 is now active: build chronological OOF component lineage and test
  only simple, predeclared rank blends against the unchanged realistic ledger.
  Final decisions remain restricted to 2024 Val and 2025 Test.

## OOF Blend Result

- The formal compact, broad, and v14 rolling alpha artifacts pass chronological
  lineage checks. The old v14 rolling manifest is accepted only because its
  same-directory experiment manifest, completed event, and artifact index
  prove a formal run; a bare legacy file is rejected.
- Three equal-weight rank blends were tested under the same realistic ledger.
  `compact_v14_eq_rank` is strongest: at CNY 1m normal it reaches 45.61% /
  Sharpe 1.060 on Val and 61.29% / 2.288 on Test, versus compact's 42.58% /
  1.049 and 52.50% / 1.796. Its Val maximum drawdown is worse (20.86% normal;
  21.88% worst stress versus compact's 20.18%), so it remains conditional.
- Historical OOF 2018-2023 confirms component diversity but not a free return
  improvement. Compact-v14 rank correlation ranges from 0.638 to 0.813 and
  Top30 overlap from 17.3% to 35.8%; the blend-to-component rank correlation
  remains above 0.90. The six-year 1m normal ledger Sharpe is 0.846 and cost2x
  Sharpe is 0.555.
- Do not select weights or promote from the historical period. The correct next
  step is a small state-aware construction pilot on the fixed blend, with
  exposures and risk attribution first and no global hard defense switch.

## Phase 6 Findings

- A copied checkout can have a valid Git `HEAD` while still containing
  uncommitted source, configuration, or report changes. Experiment provenance
  now records a deterministic `git status --porcelain` fingerprint and dirty
  flag in addition to the revision.
- The Phase 6 bundle correctly refuses to imply activation: both conditional
  Phase 5 policies remain research-only, forward selection is disabled, and
  the formal baseline is the only manual fallback.
- The current working tree has many pre-existing changes, so it is not a
  release snapshot. This is an operational gate, not a reason to discard or
  revert user work. A future active shadow must use a clean commit or an
  explicitly archived source snapshot.
- The next implementation item is a read-only forward scorecard validator
  that consumes a frozen manifest and checks date fields, artifact hashes,
  execution reconciliation, and observation-only labeling. It must not tune
  from forward outcomes.

- The read-only validator now enforces the declared contribution fields rather
  than accepting a bare annualized return/Sharpe table as a complete shadow
  scorecard. With no formally promoted candidate, the correct current result
  is `not_ready`, not a fabricated forward comparison.
- The boundary check permits a final research-date signal on 2026-05-18 when
  its open execution starts on 2026-05-19, matching the project's T-close to
  T+1-open contract.
- The final v4 manifest passes source and artifact integrity validation, but
  remains activation-disabled and reports `not_ready` because there is no
  formally promoted candidate or forward scorecard. This is the correct gate;
  the next action is a release-snapshot/governance decision, not parameter
  tuning against forward data.

## State-Aware Portfolio Pilot Result

- The shared sweep previously exposed global overlay parameters but not the
  existing selection-layer `risk_rank` parameters. That gap is now closed;
  selection settings are part of the sweep identity and the global frame is
  loaded whenever either global overlay or state-aware selection is enabled.
- The current global feature cache provides `global_defensive_pressure` and
  `global_hk_risk_pressure` for the full 2024/2025 selection periods. It does
  not provide the old `global_us_hk_pressure` column used by earlier July
  reports, so the new pilot uses the actually available defensive-pressure
  column and records that choice in the config/report.
- Fixed candidate: `compact_v14_eq_rank`; fixed execution: realistic
  open-price share-ledger, target `0.006`, hold `0.10`, band `0.20`, five new
  names, CNY 500k/1m, and normal/lag1/cost2x/capacity_3pct.
- `risk_rank_t035_p010` is active on 44/242 Val signal days and 53/243 Test
  signal days. It changes at least one replacement on 43 and 51 of those
  days, respectively, so the observed improvement is not a no-op.
- Across 16 Val/Test cells, candidate Sharpe is non-worse in 16/16 and
  annualized return in 15/16. Maximum drawdown is non-worse in 13/16 and
  executed turnover is slightly higher in some cells. This passes a research
  continuation gate but fails a clean replacement/promotion gate.
- The correct next analysis is changed-replacement attribution using the same
  execution date and ledger path: compare candidate-vs-baseline industry
  concentration, beta, specific volatility, new-name count, cost, and realized
  open-to-open contribution. Do not tune the pressure threshold again before
  this attribution.

## Phase 5 Closure

- The four full-stress summary files for `risk_rank_t035_p010` and
  `risk_suppress_d015` each contain 8 cells (four stresses x two capitals),
  complete mandatory date fields, and no backtest date beyond the relevant
  2024/2025 selection period.
- `risk_rank_t035_p010` improves the selected normal-period metrics and is
  non-worse in 16/16 Sharpe cells, but its drawdown and turnover evidence is
  not uniformly better.
- `risk_suppress_d015` reduces turnover and is non-worse in 15/16 MDD cells,
  but it loses return/Sharpe in Val `lag1`. Its exact 0.15 threshold was fixed
  after the first-family attribution, so it is exploratory rather than a clean
  pre-registered confirmation.
- Normal path attribution is complete for both families. It distinguishes
  direct state-triggered changes from retention-induced path divergence; full
  stress attribution is summary-only. No forward attribution or selection was
  performed.
- Phase 5 is closed with a gate. The formal baseline remains unchanged and
  Phase 6 should focus on frozen shadow manifests, rollback rules, and forward
  scorecard infrastructure rather than further Phase 5 tuning.

## Historical ST Contract And External Access

- The current `stock_industry.csv` remains a one-date snapshot and cannot be
  used to prove historical ST status for 2024/2025.
- Added `data/st_status.py` with a normalized event schema, deterministic
  status-transition parsing, effective-date loading, validation, and SHA-256
  helpers. `open_ledger` now prefers this source and does not mix current names
  into history when an event file exists.
- Added a cutoff-aware, atomic, page-checkpointed downloader at
  `run/download_historical_st_events.py`. Raw/checkpoint material is kept in
  `data/tracking_raw`; only rows with `imp_date <= 2026-05-18` may enter
  `data/raw`.
- Extended the execution coverage audit to require event-contract validity,
  manifest coverage through the requested end date, and an output-hash match.
  The latest artifact is
  `reports/qlib_research_framework_20260712/execution_coverage_st_contract_20260715.json`.
- The real API attempt was not successful: the approved token has no access to
  Tushare `st`, and `stock_st` is also unavailable. No fallback snapshot was
  promoted and no historical ST claim was made. `namechange` is retained as a
  separately auditable fallback candidate, not silently substituted.
- The endpoint evidence is recorded in
  `reports/qlib_research_framework_20260712/st_source_access_probe_20260715.json`;
  it contains no credential material.
- Focused contract/execution tests pass: 56. Stage A remains gated, and the
  Qlib-style rolling experiment must not start until this external data gate
  is closed or a formally audited alternative is approved.
- The Phase 6 source-aligned bundle is now v5, regenerated after the ST
  contract changes. Its validator remains `not_ready` for the intended reason:
  no forward scorecard and no promoted candidate; source and artifact checks
  pass, and activation remains disabled.

## 2026-07-16 - Fallback Source Decision

- A second live permission probe confirmed that the preferred Tushare `st`
  endpoint is still unavailable. The correct response is to change the source
  strategy, not to repeat the same request.
- A `namechange` adapter is now implemented. It interprets displayed-name
  intervals as a separately labelled historical status source and emits the
  same audited event schema consumed by `open_ledger`.
- The adapter does not make the gate pass by itself. It must still download a
  complete source, be capped at `2026-05-18`, produce a manifest/hash, and pass
  the 2024/2025 coverage audit. Until then, current-name snapshot fallback is
  not historical evidence and Stage B remains prohibited.
- The current source-aligned Phase 6 snapshot is v6 under
  `reports/experiments/phase6_shadow_bundle_20260716_v6/`; it remains
  observation-only and activation-disabled.

## Namechange Provenance Label

- `source_kind=tushare_namechange_intervals` identifies the technical source;
  `source_label=由历史股票名称区间重建` makes the fallback interpretation
  visible to human readers and downstream reports.
- The label is metadata only. It does not close the historical-ST gate: a
  complete cutoff-filtered download, manifest coverage range, valid event
  contract, and matching SHA-256 are still required.
- The current validator artifact is named
  `forward_shadow_validation.json`, not `validation_summary.json`.

## Current Source-Aligned Snapshot

- v7 is the current Phase 6 source-aligned snapshot after the provenance-label
  change. It is observation-only and activation-disabled.
- The validator has complete source/artifact/control checks and is
  `not_ready` only because no forward scorecard was supplied. This is an
  expected gate state, not a failed integrity check.
- Current focused evidence is 138 passing tests; historical ST data is still
  absent, so the namechange fallback remains an audited implementation rather
  than a formally usable research source.

## Tushare Free-Account Access Decision

- The official current documentation lists 6000 points for `st` and 3000
  points for `stock_st`; a newly registered account receives 100 points.
- The `namechange` page does not list a separate endpoint threshold, but the
  platform-wide low-point rules do not guarantee bulk access to every API.
  One successful single-code probe followed by rate limiting is insufficient
  evidence for a complete download.
- Evidence and the resulting gate decision are recorded in
  `reports/qlib_research_framework_20260712/st_source_access_policy_20260716.md`.

## Downloader Contract Audit

- Permission/points failures were previously retried three times by the shared
  caller. This was wasteful and could turn a permission failure into a rate
  limit; the historical downloader now opts into fail-fast markers.
- Tushare's current `st` and `namechange` pages show `ts_code`/date filters,
  while the official text does not establish a generic `offset/limit` contract.
  Keep the existing pagination helper as unproven test scaffolding; do not
  publish a research cache until the actual endpoint fetch mode is verified or
  replaced with an endpoint-specific implementation.

## Endpoint-Specific Fetch Contract

- The formal CLI no longer relies on the unverified generic pagination helper.
  `namechange` requests a bounded date interval; `st` requests each code from
  an explicit code universe and records a universe fingerprint in checkpoints.
- An empty ST response is a legitimate result for a code, so empty checkpoint
  files must reload as empty frames rather than being treated as corruption.
- The `st` code universe remains an input contract: a future formal download
  must document why the supplied code file covers the 2024/2025 research
  universe, especially for delisted historical names.

## Current Download/Phase 6 State

- The source-aligned Phase 6 snapshot is now v9. It retains the frozen
  2026-05-18 research boundary and remains observation-only.
- Current focused evidence is 143 passing tests. Historical ST data is still
  absent, so no execution-coverage claim or rolling experiment is promoted.

## Research Versus Forward Cache Roles

- `2026-05-18` is a research-selection boundary, not a project-wide data
  availability limit. Training, validation, test, feature fitting, and model
  or rule selection must use the frozen research role only.
- Later observations belong to the separate forward role: `data/forward_raw`
  and `data/forward_tracking_raw`. They may be refreshed through a later
  complete date such as `2026-06-30`, but they remain observation-only and
  cannot alter the research record.
- The downloader enforces this split before any API call. The coverage-audit
  CLI now applies the same role-specific default directory, so a forward audit
  cannot accidentally inspect `data/raw` merely because that is the historical
  default.

## Current Source-Aligned Snapshot

- v10 is the current Phase 6 snapshot after the role-default correction. Its
  manifest records `data/raw` as the research root and `data/forward_raw` as
  the forward root, with forward selection disabled.
- The read-only validator confirms source/artifact integrity and remains
  `not_ready` only because no forward scorecard or formally promoted candidate
  exists. This is an intentional governance state, not a performance result.
- Focused evidence is 146 passing tests. Historical ST data is still absent;
  the endpoint-specific downloader and namechange adapter remain auditable
  implementations rather than evidence that historical coverage is complete.

- The final source-aligned snapshot is scheduled as v12 after this record is
  written, so its working-tree fingerprint will include the complete role
  separation audit and these findings.

## 2026-07-16 - Canonical Split And Qlib-Alignment Reassessment

- Live code evidence establishes the intended evaluation contract:
  `run/official_backtest_from_registry.py` defines 2024 Val, 2025 Test, and
  2026-01-01..2026-06-30 Forward; `registry/decision_rules.json` permits only
  Val/Test selection and marks Forward observation-only.
- Actual registry reports cover 2026 signals from 2026-01-05 through
  2026-06-30 and executions from 2026-01-06/07 through 2026-06-29. Therefore
  2026-05-19 is not the Forward start.
- `core/research_protocol.py`, `RESEARCH_PROTOCOL.md`, `README.md`, the Phase 6
  manifest builder, and forward-validator tests still encode the obsolete
  2026-05-18/2026-05-19 boundary. The current Phase 6 bundle is therefore not
  valid evidence of the canonical full-year Forward contract.
- Some `registry/reports.csv` Forward rows have `selection_eligible=true` even
  though `is_forward=true` and the decision rule forbids Forward selection.
  This must fail validation rather than be tolerated as redundant metadata.
- The experiment recorder is real but partial: immutable config/source/events/
  artifact metadata exist, while `data_scope` can still be inferred and
  incomplete. It is not yet mandatory across every formal entrypoint.
- The v14 transform contract correctly documents daily cross-sectional
  normalization and declared PIT gaps. It is not yet a generic DataHandlerLP-
  equivalent fit/apply state system for future train-fitted processors.
- Rolling windows, label-tail purge, OOF lineage, TopK/dropout strategy
  separation, realistic open-ledger execution, factor baselines, and bounded
  trial records are implemented foundations. A single declarative workflow,
  unified arbitrary-range provider, generic dataset/model interface, and full
  monthly walk-forward lifecycle remain incomplete.
- Historical ST adapters and downloader logic exist, but the normalized event
  file and manifest do not. Treat this as a deferred external-data limitation:
  it blocks a complete historical-ST execution claim, not framework work or a
  clearly caveated rolling audit.
- Correct full-year Forward lineage requires the model, train-fitted
  transforms, checkpoint rule, and portfolio policy to be frozen no later than
  2025-12-31. A model using any 2026 observation cannot claim full-year 2026
  Forward evidence.

## 2026-07-16 - Phase A Protocol Unification Result

- `core/research_protocol.py` is now the single official split source for 2024
  Val, 2025 Test, and 2026 Forward through 2026-06-30.
- Official backtest commands, registry writing, registry scorecard reads,
  Phase 6 manifests, forward validation, OOF lineage, and research ledger
  evidence use the canonical contract instead of local date dictionaries.
- Forward registry eligibility is derived from the split and cannot be passed
  as a permissive default. Existing `registry/reports.csv` was corrected: all
  96 rows validate; 32 Forward rows are observation-only.
- OOF audit must use actual window train/valid/predict dates. A legacy
  `research_end=2026-05-18` field can describe cache visibility and is not
  itself proof of model leakage.
- Full-year Forward parent fitting and selection must end by 2025-12-31.
- The Torch environment passed 67 focused tests. System Python still lacks
  Torch; framework tests should use the documented Torch interpreter.
- A formal manifest alone is insufficient: multi-stage experiments can append
  ledger artifacts after model completion. The manifest stays immutable and
  events append-only, but the artifact index must be atomically rebuildable;
  formal ranking requires the current terminal event to be `completed` and
  verifies every indexed file hash.
- Existing registry evidence is historically useful but lacks the new complete
  provenance contract. All 96 rows are marked `legacy_registered`, with no
  missing evidence class and no fabricated formal eligibility.
- Script-level smoke tests are necessary even when import-based unit tests pass:
  both official backtest and scorecard initially lacked project-root bootstrap
  when invoked as standalone scripts. The workflow smoke exposed and fixed it.
- A scorecard must be scoped by workflow candidate IDs and expected splits.
  Reading the global reports registry silently mixed unrelated candidates and
  Forward rows; the isolated v4 replay now reports exactly 16 selection rows,
  zero Forward rows, and zero coverage gaps.
- A physical v14 cache extending beyond the selection period is safe to reuse
  only for transforms proven date-local and only through an explicit logical
  Provider cutoff. The new rolling-only superset mode avoids a duplicate 20GB
  cache; legacy callers still reject future-dated cache metadata.
- The real Provider audit found no new timing defect. Fundamentals are keyed by
  effective date with quality flags, and global features use sessions strictly
  before the A-share date. Historical ST event coverage remains the only
  execution-source blocker and is not inferred from current stock names.

## Phase E Formal Monthly Walk-Forward Result

- The 4y Train/6m Valid/1m OOS controller completed 24 windows and 485 unique
  OOS dates. Window-level progress, model hashes, alpha hashes, stitched split
  signals, and continuous ledgers are reproducible and resumable.
- The Compact LightGBM arm failed the performance gate by a large margin. Its
  average Val/Test four-stress Sharpe was 0.563 versus 2.062 for the frozen
  baseline; its worst drawdown was 26.10% versus 18.18%.
- This is evidence against this base learner, not against monthly rolling as a
  method. Schedule-length sweeps would confound learner weakness with window
  tuning. The next legitimate comparison requires a strong-model adapter.
- The experiment read no Forward rows and wrote no global registry candidate.
  Missing trade-level attribution explains `incomplete_evidence`, but cannot
  reverse the already failed return, Sharpe, and drawdown gates.

## Phase E1 Two-Layer Fairness Audit

- The completed monthly Compact artifact is explicitly configured as
  `raw_cross_sectional_rank`; its split files are model-layer raw rankings even
  though the generic filename is `alpha_policy.jsonl`.
- The governed portfolio baseline is not raw. Its lineage is
  `multi_downside_e19 raw -> alpha_sa_p05 -> ledger_path_v3 -> realistic
  open-ledger`.
- The preserved e19 Val/Test raw inputs are
  `reports/stair_defense_val_test_20260703/signals/<split>/multi_downside_e19/raw/alpha_raw.jsonl`.
- A fair audit therefore needs two stages: identical-ledger raw versus raw;
  only if the rolling raw signal is competitive should both inputs be passed
  through the same state-aware and pairwise portfolio stack.
- The complete audit was run despite the raw rejection to measure transfer.
  Compact improved from mean Sharpe 0.563 / worst MDD 26.10% to mean Sharpe
  0.728 / worst MDD 19.86% under frozen sa_p05 + V3, but e19 under the same
  stack remained much stronger at mean Sharpe 2.062 / worst MDD 18.18%.
- The frozen V3 model was trained on e19-distribution data. Applying it to
  Compact tests transferability, not a Compact-specific optimum. Retraining or
  tuning V3 for a base alpha that already fails both Val and Test is not
  justified.

## 2026-07-17 Long-Term Qlib Alignment Audit

- The project has substantively aligned Qlib's recorder/workflow, purged
  rolling-task, unique OOS lineage, and signal/strategy/executor separation
  ideas. The project-native realistic A-share ledger remains intentionally
  authoritative rather than Qlib's executor.
- `ProcessorContract` currently validates fitted-state provenance but is not a
  full fit/apply runtime. The workflow allows only frozen candidates and
  rolling LightGBM, so the generic model layer is incomplete.
- The 24-window Compact run validates orchestration, resume, stitching, and
  continuous-ledger behavior. Its rejection does not test strong-model monthly
  rolling because `multi_downside_e19` has not been adapted.
- The immediate gate is a reconstructed-profile e19 one-window smoke. The
  longer sequence is recorded in
  `LONG_TERM_QUANT_PLATFORM_ROADMAP_20260717.md`.
# 2026-07-17 Qlib-first priority correction

- User changed the immediate priority from strong-model experimentation to completing the Qlib-inspired research framework first.
- The previous L1-before-L2 order was structurally wrong for that objective: a strong PyTorch rolling adapter should validate a stable Dataset/Processor/Model contract, not precede it.
- Live code confirms `experiments/workflow.py` currently accepts only `frozen_registry_candidate` and `rolling_lgbm_alpha`; a generic model-adapter lifecycle is still missing.
- Project-native providers, transform provenance, recorder, rolling windows, strategy proposals, and realistic ledger are useful foundations, but they are separate contracts rather than one declarative end-to-end research runtime.
- The external Qlib checkout is not under the code repository's `references/` path. The revised plan must freeze its actual path and revision rather than rely on relative-path assumptions.
- Revised architectural order should be: alignment audit and vocabulary -> declarative task/schema -> Dataset/DataHandler processor lifecycle -> Model adapter lifecycle -> standardized Records -> Rolling/OOF orchestration -> Strategy/portfolio construction -> manual Online/Shadow lifecycle. Strong `multi_downside_e19` rolling becomes an acceptance workload after the interfaces exist.
- Qlib's executor remains reference-only. The project's realistic A-share open-price share-ledger remains the sole formal execution authority.

## Q0 live contract findings

- `experiments/workflow.py` is a real schema-v1 validator/compiler, but it hard-codes two model adapters and emits commands for specialized runners. Workflow v2 must remain compatible with those runners while introducing explicit component descriptors rather than replacing working execution code immediately.
- Existing formal configs already freeze data sources, ranges, feature transform provenance, labels, windows, checkpoint rule, signal transform, strategy, realistic ledger, splits, and reports. V2 should normalize and tighten these fields, not invent an unrelated configuration language.
- `data/providers.py` has useful project-native `DataView`, five providers, and a provenance-only `ProcessorContract`; the missing Qlib-like piece is the executable shared/infer/learn processor chain and a Dataset `prepare` facade.
- `experiments/recording.py` already exceeds a minimal Qlib recorder in immutable source/config/scope provenance, append-only events, artifact hashes, and formal completeness gates. Q4 should add standardized record templates on top of it, not replace it with MLflow.
- `backtest/strategy.py` correctly keeps ranked target proposals independent from prices, cash, lots, ADV, and limits. This boundary is retained; Qlib strategy concepts can be adapted without importing its executor.
- Qlib `DataHandlerLP` explicitly distinguishes raw, inference, and learning data plus shared/infer/learn processors. `DatasetH` owns named time segments and delegates data retrieval to its handler. These are the minimum semantic contracts Q2 must reproduce.
- Qlib `SignalRecord`, `SigAnaRecord`, and `PortAnaRecord` form an explicit artifact dependency chain. The project needs equivalent project-native records whose portfolio record delegates to realistic `open_ledger`.
- Workflow v2 schema can be delivered in Q0 as a non-runtime draft with a golden config and structural tests; changing `WORKFLOW_SCHEMA_VERSION` belongs to Q1 after the compiler migration exists.

## Q1 implementation findings

- Backward-compatible normalization is the lowest-risk migration: source v2 remains immutable and hash-identifying, while the compiled stage graph uses the proven v1 runner vocabulary.
- Workflow version and compiled-stage version must be separate fields. Compiled outputs keep stage schema v1 and record `workflow_schema_version=2` for v2 sources.
- Cross-field checks catch errors JSON Schema alone cannot, including logical ranges beyond `max_data_date`, governance/evaluation split disagreement, and Forward parents fitted after 2025.
- Q1 supports only adapters with existing safe runners: rolling LightGBM and frozen artifacts. Declared PyTorch/Q3 adapters fail explicitly, preventing false framework-completion claims.
- The compile entrypoint now derives protocol splits from either v1 `evaluation.splits` or v2 selection/observation roles and freezes the original v2 config/hash rather than its normalized compatibility form.

## Q2 implementation findings

- The v14 cache already yields one date cross section at a time, so a Qlib-like handler can remain streaming and avoid a second materialized dataset or duplicated 20GB cache.
- The project runtime uses shared + infer for inference and shared + infer + learn for learning. Learning-only label filters are structurally prohibited from inference chains.
- Train-fitted processors consume a replayable Train sample factory, persist state and fit range, and reject refit or unfitted Valid/Test use.
- Workflow v2 can build the Dataset runtime directly from its `dataset` and `processors` sections over `V14MemmapProvider`.
- Processor `kind` is validated against its implementation so a config cannot claim an unfitted transform is train-fitted or vice versa.
- Q2 intentionally does not modify the physical v14 layout or current model trainers. Q3 adapters consume the common `ProjectDataset.prepare` interface.

## Q5 strong-model reconstruction and cache findings

- Historical `multi_downside_e19` is a 250-input V9 model. Its matching cache is the v14 `funda` bundle, not the newer 258-input `fundaq` bundle.
- The matching physical bundle covers 2010-01-04 through 2026-05-18, has risk dimension 59, 83 industries, and physical `oo` plus aliased `oo_lag1` labels.
- The first one-window smoke did not match the cache key and entered a full raw CSV rebuild. It was stopped before training when private committed memory reached about 21.7 GiB and free physical memory fell to about 1.04 GiB.
- Training and inference now accept one explicit cache metadata path, validate architecture and label compatibility, resolve every memmap, and refuse fallback rebuilding.
- Historical OOS inference must select stocks from feature/risk availability, not future-label availability. The explicit-cache inference path now enforces this and emits dummy labels only for compatibility with the legacy predictor interface.
- The original lineage is staged rather than a single 19-epoch run: e1-e6 uses `1e-4` with base OO+OO-lag1 losses, e7-e15 resumes exact e6 at `1e-5`, and e16-e19 resumes exact e15 at `5e-6` with multi/downside and the other solved auxiliary weights.
- A one-epoch three-window pilot validates runtime mechanics only. It must not be treated as the profile's declared 19-epoch pilot or used for performance selection.
- The complete exact-stage pilot failed its signal gate, driven by 2024-01 inversion. Historical fixed epoch boundaries bypassed the better stage-selected checkpoint and propagated degraded weights.
- The next justified ablation changes only stage transition ownership from exact checkpoint to validation-selected checkpoint. Launching all 24 windows before that test would be expensive negative repetition.
- The selected-checkpoint transition did not repair the 2024-01 inversion. It
  made strict-OOS weighted Rank IC and Top0.6% weighted return worse even
  though its validation metric preferred epoch 7. The deeper issue is therefore
  not only checkpoint handoff: the current `rawtopstable_h5_top0p6` validation
  objective is not transferring reliably from the six-month Valid segment to
  the next OOS month under this historical schedule.
- Qlib-style rolling orchestration is functioning correctly: purge, unique OOS
  ownership, immutable contract, stage resume, artifact hashes, and raw signal
  generation all passed. The rejection concerns the learner/selection recipe,
  not the research framework or realistic A-share executor.
- The strong-model pilot currently remains a specialized runner rather than a
  first-class Workflow v2 model stage. `torch_strong_alpha` is declared in the
  schema and implemented in the Q3 model factory, but the v2 normalizer still
  rejects it and the stage executor allowlist does not dispatch it.
- The LightGBM rolling runner emits `rolling_manifest.json` with
  `split_alpha_paths`; the strong staged runner emits only its private contract,
  progress, and one stitched alpha. This prevents learner-neutral downstream
  candidate/ledger/Record handling and is the immediate Qlib-alignment gap.

## Q5A framework binding and Q4B runtime evidence findings

- The strong-runner gap above is now closed: Workflow v2 validates and compiles
  `torch_strong_alpha`, the executor allowlists and resumes it, and both strong
  and LightGBM rolling publish one learner-neutral `rolling_manifest.json` plus
  canonical split-alpha contract.
- Compatibility materialization can standardize an already completed
  exploratory strong pilot by verifying source completion and artifact hashes;
  it deliberately preserves exploratory/nonpromotable status.
- Q4 Record contracts are stricter than current formal Workflow outputs. The
  templates require equity, positions, orders, rejections, and costs, while the
  current official ledger path normally persists summary CSVs and can
  optionally persist daily return/aggregate diagnostics only.
- `backtest.execution.apply_open_ledger_constraints` computes genuine desired
  and executed share deltas and rejection reasons internally, but returns only
  aggregate daily counts and costs. Q4B must expose a separate evidence trace
  from this same execution path; deriving fake order rows from aggregate
  diagnostics would violate the audit objective.
- Q4B runtime Record integration therefore precedes Q5B profitability work.
  It must not add a second executor, modify execution decisions, or use Forward
  for selection.

## Q4B automatic Record findings

- Workflow v2 previously validated the six Record names but ignored them at
  runtime. It now compiles an explicit `standard_records` stage after scorecard.
- Signal evidence can be reconstructed without a second prediction path by
  joining the learner-neutral split alpha with the v14 raw label memmap on
  signal date and stock code. `oo_lag1` applies the cache's one-trading-date
  view shift and is covered by a direct regression test.
- Portfolio and risk records reference the official open-ledger path index.
  The Record builder never decides fills and refuses summary-only historical
  runs that do not have genuine positions/orders/rejections/costs.
- The first real label smoke covered 63 dates from the prior exploratory strong
  pilot. Its low aggregate IC is consistent with earlier negative evidence but
  was not used for selection; the smoke proves data alignment only.
- Frozen candidates still require a dedicated dated-prediction adapter before
  they can use automatic Signal Records. The automatic stage currently accepts
  the learner-neutral rolling manifest contract and fails explicitly otherwise.
- With profitability deferred, the next Qlib-alignment target is the manual
  Shadow/Online lifecycle, not Q5B retraining or Q6 portfolio optimization.

## Current alignment closure findings

- Q7A is now implemented, so the statement above is historical rather than the
  current next step.
- The main remaining platform risk is acceptance evidence, not another missing
  abstraction: no newly executed formal Workflow has yet produced and passed a
  complete six-Record bundle from genuine prediction and ledger artifacts.
- Automatic Signal Records currently require a learner-neutral rolling
  manifest. Frozen and legacy candidates need one dedicated hash-checked dated
  PredictionFrame adapter; compatibility must not promote or rewrite old runs.
- Q7B must be built and accepted against a real complete bundle. A synthetic
  candidate or compile-only Workflow would hide the exact daily failure modes
  that replay and silent-failure detection are intended to catch.
- Q5B, dynamic Rolling Ensemble, and Q6 portfolio construction are research
  work after framework closure. Historical ST remains an external data gap,
  while Qlib Executor and automatic platform features are intentional
  exclusions rather than unfinished core work.

## Frozen prediction compatibility findings

- The formal baseline alpha uses two valid historical score encodings: numeric
  lists and code-to-score mappings. Both preserve the same ranked `codes`
  contract consumed by the ledger, so the common alpha reader now normalizes
  either representation into code order for PredictionFrame diagnostics.
- A frozen adapter must not copy 40-50 MB split alpha files into each Workflow.
  The accepted contract stores immutable source paths, SHA-256, signal range,
  date count, stock-row count, registry lineage, and explicit
  `training_executed=false` / `promotion_performed=false`.
- The actual baseline source now resolves through the automatic Record input
  path for all 2024 Val and 2025 Test dates. This closes compatibility, but a
  full ledger/scorecard/six-Record Workflow execution is still required for
  production acceptance.

## First formal Workflow acceptance findings

- Windows path safety must account for the full absolute parent path, not only
  a collision-resistant filename hash. Detailed ledger artifact tags are now
  bounded to a 240-character full path budget.
- A wrapper that writes a failed status but returns zero creates false-complete
  Workflow receipts. The official backtest wrapper now raises after preserving
  status/events/artifacts whenever a child sweep fails.
- Observation splits are optional. A zero-byte Forward summary therefore means
  `not_requested`, not a malformed required selection artifact.
- Hash-checked stage receipts enabled the intended recovery: completed ledger
  and scorecard work was reused, failed Record inputs were preserved under a
  failed name, and only the final stage reran.
- The real bundle now passes formal manifest, current artifact index, six
  Record hash, dual-capital, four-stress, and lifecycle eligibility checks.
