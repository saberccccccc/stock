# Qlib Borrowing And Integration Plan

> Execution-order notice (2026-07-18): this document remains the detailed Qlib
> adaptation design and historical acceptance record. The sole active project
> sequence, gates, fallbacks, and completion definition are now in
> `MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md`. Any older `next phase` or
> priority wording here is non-normative and cannot reorder the master plan.

Status: historical technical reference - execution order superseded by the master plan
Date: 2026-07-12; corrected 2026-07-16
Scope: research-layer improvements inspired by the local Qlib reference. This
plan does not replace the project's PIT data pipeline, realistic execution
engine, or registry governance.

## 0. Canonical Current-State Contract (2026-07-16)

This section is normative and supersedes older date-boundary wording elsewhere
in this document. Phase A resolved the live split-contract conflict on
2026-07-16. New formal experiments remain gated on Phase B's mandatory
manifest schema rather than on the obsolete May boundary.

### 中文执行摘要

- 当前唯一正确的评价划分是：2024 验证、2025 测试、2026 全年 Forward；
  Forward 当前数据更新到 2026-06-30，只观察，不参与模型或规则选择。
- 现有工程已经具备实验清单、滚动窗口、标签尾部 purge、OOF 血缘、
  TopK/dropout 策略接口、真实 `open_ledger`、Compact/Broad 基线和部分归因。
- 这些是可复用基础，但还没有形成 Qlib 式的统一工业流程：日期协议仍冲突，
  实验字段未全局强制，缺少一份配置驱动全链路，数据 Provider 与变换状态也未统一。
- 当前不是继续训练或调阈值的时候。下一步是 Phase A，先统一日期与选择协议；
  然后依次完成强制实验 schema、统一 workflow、Provider/Processor，最后才跑正式月度滚动实验。
- 不采用 Qlib 的简单执行器替代现有 A 股真实账本，不用 2026 Forward 调参，
  不继续无假设的防御阈值搜索，也不重复消耗时间批量轮询 `namechange`。
- 本次只完成计划书校准；Phase A 代码修复尚未开始。

### Evaluation Roles

| Split | Canonical interval | Role | May select or tune? |
|---|---|---|---|
| `val_2024` | 2024-01-01 through 2024-12-31 | validation evidence | yes |
| `test_2025` | 2025-01-01 through 2025-12-31 | test evidence | yes, under the predeclared joint decision rule |
| `forward_2026` | 2026-01-01 through the latest complete available 2026 market date; currently 2026-06-30 | forward observation | no |

The actual signal and execution dates may begin or end on adjacent trading
days. Every result must therefore record both the declared split and the actual
`signal_start`, `signal_end`, `backtest_start`, and `backtest_end`.

`2026-05-18` is not the start of `forward_2026`. It is a legacy research-cache
cutoff retained by older protocol code and some historical artifacts. It may be
recorded as a specific artifact's physical/effective data date, but it must not
reclassify January-May 2026 as selection data or truncate the full-year forward
split. A model claiming full 2026 forward evidence must freeze model fitting,
checkpoint selection, train-fitted transforms, and policy selection no later
than 2025-12-31, with the required label-tail purge. An artifact trained or
selected with any 2026 observation cannot be called full-year 2026 forward.

### Data-View Contract

The target design uses one physical historical market-data store and explicit
logical experiment views. It does not require a second set of CSV files merely
because a row belongs to forward evaluation.

```text
one physical PIT market-data store
  -> train/valid view defined by the experiment task
  -> val_2024 view
  -> test_2025 view
  -> forward_2026 view through max_data_date
```

The provider may read earlier history for feature warm-up. This does not make
that history part of the evaluated split. Each formal experiment must record:

- physical data root and immutable fingerprint;
- feature warm-up start/end;
- train, valid, signal, and backtest start/end;
- label family, horizon, label-tail purge, and transform fit range;
- `max_data_date` as the read ceiling, not as the split definition;
- split role, selection eligibility, and whether any forward row was used.

### Verified Repository State

| Area | Current status | Decision |
|---|---|---|
| Official split wrapper and registry decision rule | Already use 2024 Val, 2025 Test, and 2026-01-01..2026-06-30 Forward | Keep as canonical intent. |
| Canonical protocol, README, Phase 6 builder, and validators | Phase A now imports one SplitSpec contract using 2024 Val, 2025 Test, and full-year 2026 Forward | Completed; historical configs may retain 2026-05-18 only as immutable artifact metadata. |
| Experiment recorder | Schema v2 formal manifests require data fingerprints, all logical ranges, transform state, split roles, lineage, terminal completion, and a hash-verified artifact index | Phase B completed; exploratory and legacy evidence remain readable but cannot silently become formal evidence. |
| Transform/PIT contract | v14 daily cross-sectional transform audit exists | Partial DataHandlerLP alignment; no general train-fit/frozen-infer processor state contract yet. |
| Rolling and OOF | Window controller, label-tail purge, dated OOS signals, and lineage checks exist | Useful foundation, not yet an industrial monthly workflow covering all models and reports. |
| Strategy/execution separation | Retention and TopK/dropout feed project-native realistic `open_ledger` | Keep. Qlib's executor must not replace `open_ledger`. |
| Factor baselines | Compact/Broad/v14 comparisons and bounded trials exist | Durable evidence; do not resume broad tuning without a new hypothesis. |
| State-aware proposals | Pilot and attribution artifacts exist; no proposal is promoted | Archive as research evidence and stop threshold searching. |
| Phase 6 shadow bundle | Infrastructure exists but is built on the stale May boundary and has no governed candidate | Do not call ready; rebuild only after the split contract is unified. |
| Historical ST | Adapter/downloader exist, but complete historical source data is absent | Deferred external-data enhancement. It blocks a claim of complete historical-ST execution coverage, not core framework construction or a clearly caveated rolling audit. |

### Historical A-G Execution Order (completed foundation; superseded for current priority)

The following sequence records how the existing foundation was built. It is
not the current next-step order. Current work follows Section 10 and
`LONG_TERM_QUANT_PLATFORM_ROADMAP_20260717.md` Q0-Q8.

1. **Phase A - protocol unification (completed):** centralize split definitions, remove the
   stale May forward boundary, fix forward selection flags, and add regression
   tests proving that forward starts in January and is never selection-eligible.
2. **Phase B - mandatory experiment schema:** make actual data/transform/model/
   signal/ledger ranges and fingerprints mandatory at every formal entrypoint;
   incomplete manifests fail before leaderboard or registry ingestion.
3. **Phase C - one workflow configuration:** add one declarative experiment
   config that binds dataset view, features, labels, model, strategy, ledger,
   stresses, capital, and report bundle under one experiment ID.
4. **Phase D - provider and processor contracts:** implement one cached,
   arbitrary-range data provider plus train-fit/frozen-infer transform state.
5. **Phase E - monthly walk-forward experiment:** run the predeclared 4-year
   Train, 6-month Valid, 1-month OOS schedule, stitch unique OOS predictions,
   and execute one continuous realistic ledger. Compare schedules only after
   the first arm passes integrity checks.
6. **Phase F - portfolio construction:** only after rolling evidence exists,
   evaluate alpha-minus-risk/cost proposal logic with realized-ledger
   attribution. Do not return to blind threshold sweeps.
7. **Phase G - manual shadow lifecycle:** rebuild Phase 6 artifacts from the
   unified contract; 2026 remains observation-only and cannot trigger tuning.

ST source acquisition runs independently as a deferred data workstream. It is
reopened only when a usable historical source or sufficient Tushare permission
exists; bulk `namechange` polling is not part of the critical path.

### Current Work Packages And Acceptance Gates

#### Phase A - Protocol Unification

Tasks:

1. Create one importable split specification used by training, official
   backtest, registry ingestion, Phase 6, validators, and tests.
2. Remove `2026-05-19` as a semantic Forward boundary. Preserve it only in
   historical result notes that explicitly analyse the post-May regime.
3. Require `selection_eligible=false` whenever `is_forward=true` or the split
   role is observation-only. Reject contradictory registry rows.
4. Validate full-year Forward lineage: all fitting and selection timestamps
   must precede 2026-01-01.
5. Update protocol documentation and regression tests together.

Acceptance: all split consumers resolve the same dates and roles; a 2026-01
Forward row passes, a Forward selection row fails, and a model trained with
2026 data cannot claim full-year 2026 Forward.

Implementation status (2026-07-16): completed. The canonical contract is in
`core/research_protocol.py`; official backtest, scorecard, Phase 6, forward
validator, OOF checks, data-update bounds, and registry roles were migrated.
All 96 registry rows pass role validation and 67 focused tests pass.

#### Phase B - Mandatory Experiment Schema

Tasks:

1. Upgrade the experiment schema and remove silent date inference from formal
   runs. Incomplete exploratory runs may be recorded, but cannot enter the
   leaderboard or registry.
2. Record source revision/snapshot, physical data fingerprint, logical ranges,
   transform state hash, model hash, alpha hash, exact command/config, strategy,
   ledger contract, costs, capital, and report hashes.
3. Make `signal_start/signal_end/backtest_start/backtest_end` mandatory result
   fields and cross-check them against the requested split.
4. Add a compatibility importer for older artifacts that marks missing fields
   explicitly instead of fabricating them.

Acceptance: deleting terminal history still leaves enough immutable evidence
to rerun or reject the experiment, and an incomplete experiment cannot appear
in a formal ranking.

Implementation status (2026-07-16): completed. Formal entrypoints now use
schema v2 scopes; registry and scorecard ingestion reject missing manifests,
unfinished runs, missing/stale artifact indexes, hash mismatches, contradictory
split roles, and actual signal/backtest dates outside the declared split. The
96 existing registry rows remain explicitly classified as
`legacy_registered`; none is fabricated into formal evidence. The manifest and
event log are immutable/append-only, while `artifact_index.json` is an atomic,
rebuildable projection of all recorded artifact events so later ledger stages
cannot leave a stale index. Phase A+B focused regression: 86 tests passed.

#### Phase C - Declarative Workflow Controller

Tasks:

1. Define a versioned experiment YAML/JSON schema for data view, features,
   labels, windows, model, checkpoint rule, alpha transform, strategy,
   realistic ledger, stresses, capitals, and reports.
2. Add a dry-run compiler that resolves paths/dates and writes the frozen
   manifest before expensive data loading.
3. Execute the existing project modules through adapters. Do not duplicate the
   model trainer or `open_ledger` inside the controller.
4. Resume only when the config hash and completed artifacts match.

Acceptance: one config and experiment ID reproduce the complete chain from
logical dataset to scorecard without hand-assembled command sequences.

Implementation status (2026-07-16): completed. A versioned JSON contract,
compiler, and resumable executor now freeze all required sections, reject
non-project execution, fingerprint sources, emit existing-runner commands, and
write hash-checked stage receipts. The frozen-alpha baseline ledger replay ran
end to end through the realistic ledger and isolated scorecard: 16/16 expected
2024/2025 stress-capital rows, zero coverage gaps, and zero Forward rows. The
rolling path also automatically materializes a completed rolling manifest as
an experiment-local candidate without copying alpha files or editing the
global candidate registry. Missing candidate attribution remains an explicit
`incomplete_evidence` promotion gate; it is not fabricated by the workflow.
No new rolling model training has been started.

#### Phase D - Provider And Processor Contracts

Tasks:

1. Put OHLC, volume/ADV, tradability, listing, fundamentals, macro, and external
   markets behind date-sliced provider interfaces with reusable matrix/memmap
   caches and immutable source fingerprints.
2. Separate physical coverage, feature warm-up, task data, and evaluation
   intervals. Avoid duplicating the entire raw library by split.
3. Classify processors as PIT alignment, same-day cross-sectional, train-fitted,
   or inference-only. Persist learned fill/winsorize/normalization state for
   every train-fitted processor.
4. Apply fundamental values by actual effective date and retain missing,
   estimated-notice, freshness, and age flags.

Acceptance: arbitrary intervals load from the same provider; repeated ledger
runs reuse cached matrices; validation/test/forward never refit a train-fitted
processor.

Implementation status (2026-07-16): completed with the pre-existing external
historical-ST limitation. `DataView` separates physical coverage, warm-up,
task, evaluation, and max-read dates. OHLC/money, v14 memmaps, effective-date
fundamentals, completed-session global markets, and A-share execution coverage
now expose provider manifests. Processor kinds distinguish PIT, daily
cross-section, train-fitted, and inference-only state; train-fitted state cannot
cross train end. Rolling may explicitly reuse the existing 2026 physical v14
superset under a 2025 logical cutoff, while legacy callers remain strict. A
real dry-run reused the cache without creating a 2025 duplicate and resolved
242/243 prediction days. The five-provider audit passed with exactly one
declared blocker: missing complete historical ST events.

#### Phase E - Formal Monthly Walk-Forward

Tasks:

1. Freeze one model/label/feature/strategy/ledger config before generating
   windows.
2. Run trailing 4-year Train, trailing 6-month Valid, exact label-tail purge,
   and next-month unique OOS predictions.
3. Stitch OOS alpha in date order and run one continuous realistic ledger so
   cash and holdings are not reset each month.
4. Report per-window and aggregate return, Sharpe, drawdown, turnover, cost,
   capacity, rejection reasons, IC diagnostics, and regime stability.
5. Compare 3-year, 5-year, and expanding histories only after the 4-year arm
   passes integrity checks and only under identical OOS months.

Acceptance: every OOS date has exactly one earlier-trained owner, all parent
artifacts and processor states are traceable, and schedule choice is based on
2024 Val plus 2025 Test rather than 2026 Forward.

**2026-07-16 result:** the first formal arm completed all 24 monthly windows
and 485 unique OOS dates, then ran a continuous realistic ledger on 2024 Val
and 2025 Test. The Compact LightGBM arm was decisively weaker than
`ledger_path_v3_t0001_nolookahead` across return, Sharpe, and drawdown, so it
was rejected without registry promotion or Forward evaluation. The integrity
acceptance passed; the performance gate failed. The planned 3-year, 5-year,
and expanding schedule comparison is therefore suspended to avoid tuning a
weak base learner. The next rolling experiment must reuse the same controller
with an existing strong-model trainer adapter, not search Compact parameters.

#### Phase F - Portfolio Construction

Use the selected fixed alpha as expected return and test a small number of
predeclared risk/cost objectives covering industry/style concentration, beta,
specific volatility, turnover, new names, crowding, and execution risk. The
output is desired holdings; `open_ledger` remains the fill authority. Promotion
requires realized-trade attribution on both selection splits.

#### Phase G - Manual Shadow Lifecycle

Freeze the governed parent experiment, regenerate full-year 2026 scorecards,
record daily signal/order/fill lineage, and support manual `shadow`, `paused`,
and `retired` states. Forward observations can trigger an integrity incident or
new future research hypothesis, but cannot tune the active artifact.

## 1. Objective

Build an auditable, reproducible research framework around the existing A-share
pipeline, then use it to run a controlled monthly walk-forward experiment. The
framework must make data/feature contracts, training tasks, model artifacts,
signals, execution evidence, and governance traceable end to end.

The intended benefit is stronger experiment discipline before attempting to
improve return. The monthly walk-forward experiment is the first formal model
experiment to use the completed framework; it is not an immediate replacement
for the formal baseline.

## 1A. Current Operational Objective (corrected 2026-07-16)

The project objective is to complete the framework in the order below, using
the current repository's actual PIT data, labels, alpha artifacts, realistic
open-price share-ledger, registry, and A-share execution constraints:

> First make the research chain reproducible, auditable, and operationally
> governed; then run a separate Qlib-style monthly rolling experiment to test
> whether genuinely out-of-sample updating improves robustness. Do not claim a
> return improvement merely because a framework component, IC value, or short
> forward segment looks better.

The framework is considered ready for the rolling experiment only when each
candidate run can be traced from data/cache and transform provenance through
dated alpha, strategy proposal, realistic ledger fills, attribution, and a
complete scorecard. The formal baseline remains unchanged until a candidate
passes the registry decision rules on both `val_2024` and `test_2025` under all
required stresses and capital levels.

The full 2026 forward period starts on 2026-01-01 and remains observation-only.
It may expose operational or
integrity failures and may be reported in a shadow scorecard, but it cannot
select a model, tune a threshold, change a blend weight, or justify a rerun.
The first rolling experiment will therefore be a separately identified
research artifact with fixed windows, label-tail purge, unique OOS ownership,
and the same realistic ledger contract. It is not an automatic retraining,
promotion, or live-trading mechanism.

### Objective Acceptance Criteria

1. A clean or explicitly snapshotted source state, immutable experiment
   manifest, config hash, cache/transform contract, and artifact checksums are
   recorded for every formal research run.
2. The selection protocol uses only the predeclared 2024 validation and 2025
   test splits; all four required stress scenarios and both capital levels are
   present before a candidate can enter the registry decision path.
3. The official executor remains the project-native realistic open-price
   share-ledger. Qlib strategy ideas may generate desired holdings, but cannot
   replace A-share fills, costs, lots, ADV, suspension, listing, ST, or price
   limit handling.
4. Forward shadow artifacts are activation-disabled, manually reversible to
   the formal baseline, and explicitly marked observation-only.
5. A rolling experiment is started only after criteria 1-4 pass; its result is
   judged by cross-window stability, drawdown, turnover, cost, capacity, and
   ledger performance rather than IC alone.

This is a framework-and-evidence objective, not a promise that the next model
will produce a higher return.

## 1B. Qlib Workflow Alignment Review (2026-07-12)

This plan was reviewed against the project's Qlib workflow note. Its central
principle is adopted: a research result is the complete, reproducible chain
from data and feature definitions through model score, strategy, execution,
and recorded evidence. A model checkpoint or an IC value alone is not an
experiment result.

| Workflow-note component | Plan location | Status and project decision |
|---|---|---|
| Data / DataHandler / Dataset | Sections 3A.2, 4B, Phase 2 | Adopt transform/PIT provenance and explicit splits; retain the existing A-share cache rather than Qlib provider data. |
| Model -> dated score | Sections 3A.1, 3A.5, Phase 3 | Every alpha has model/config/cache/split lineage. It is a score, never an order. |
| Strategy -> target holdings | Sections 3A.6 and 3A.6A, Phase 5 | Build a project-native retention/TopK-dropout policy interface. It proposes desired holdings only. |
| Executor / account / fills | Section 2, Section 3A.6A, Phase 3 | Keep realistic `open_ledger` as the sole formal executor. Qlib's executor is not a replacement. |
| Signal, IC, portfolio records | Sections 3A.1, 3A.4, Phase 3 | Record alpha, diagnostics, ledger, positions, and attribution under one immutable experiment ID; IC is diagnostic, not the selection objective. |
| Rolling OOS workflow | Sections 3A.5 and 5A | Implement monthly fixed-window tasks with label-tail purge, unique OOS ownership, and one continuous ledger. |
| Online lifecycle | Section 3A.7, Phase 6 | Manual shadow/rollback evidence only. No automatic retraining or promotion. |
| Alpha158/360 baseline | Section 3A.3, Phase 3B | Recreate only an auditable subset from the project's PIT OHLC data; do not import Qlib labels or China data. |
| Tuner, optimizer, RL | Sections 3A.4 and 3A.6 | Bounded tuner and deterministic proposal research are later phases. RL and broad automatic search are deliberately out of scope. |

### Timing Clarification

`TopkDropoutStrategy` is a useful strategy-layer reference, but its precise
signal-to-order timing depends on Qlib's executor frequency and configuration.
This project therefore does not inherit an assumed "previous-bar signal"
behavior from Qlib. Every project policy must explicitly record `score_time`,
`decision_time`, `order_time`, and `execution_price_field`. The official family
is currently T-close information -> T+1-open execution.

### Priority Correction

The work is intentionally not a full Qlib port. The immediate sequence is:
immutable experiment record -> transform/PIT contract -> common
score/strategy/ledger adapter -> report bundle -> walk-forward audit. Factor
baselines, bounded tuning, ensembles, and risk-aware construction follow only
once that chain is demonstrably reproducible. This prevents framework work
from becoming another large, untestable model experiment.

## 2. Non-Negotiable Boundaries

1. Split roles are calendar-based: 2024 Val, 2025 Test, and full-year 2026
   Forward through the latest complete available date. Physical storage roots
   do not define these roles.
2. Only `val_2024` and `test_2025` select models, model blends, or portfolio
   rules. `forward_2026` starts 2026-01-01 and is observation-only.
3. Official execution remains the existing realistic open-price share-ledger,
   including cash, shares, lots, minimum commission, ADV, listing/ST,
   suspension, and price-limit rules.
4. Required official coverage remains `normal`, `lag1`, `cost2x`, and
   `capacity_3pct` for CNY 500k and CNY 1m.
5. `registry/` remains the only formal source for baselines, candidates,
   evidence, attribution, and decisions.
6. Qlib's default close-price execution, `TopkDropoutStrategy`, CSI300
   defaults, and CNY 100m account assumptions are diagnostic references only.
7. Qlib strategy backtest/executor must never replace the project's realistic
   open-price share-ledger. Qlib strategy ideas may produce desired holdings;
   only `open_ledger` determines A-share fills, cash, lots, costs, ADV, limits,
   suspensions, and realized performance.
8. Do not revive the rejected `topk_oof_proxy_selector` as a formal candidate.
   It learned proxy portfolio labels and failed to transfer to 2024/2025.

## 3. What Is Being Borrowed

| Qlib idea | Project adaptation | Owner | Not adopted |
|---|---|---|---|
| Rolling tasks | Date-safe expanding windows in `experiments/rolling.py` | `experiments/`, `run/` | Qlib workflow runtime |
| Label-tail truncation | Purge training/validation signal dates whose full label ends outside that segment | `experiments/`, `data/` | Any label change |
| OOF predictions | Dated signals made by models that did not train on the prediction period | `alpha/`, `experiments/` | Proxy selector as formal evidence |
| Experiment recorder | Immutable per-run manifest and artifacts before registry promotion | `experiments/`, `reports/` | MLflow as formal authority |
| Processor fit/infer split | Provenance manifest for existing feature transforms | `data/` | Blindly replacing v14 normalization |
| Risk-aware optimization | Thin candidate-weight/proposal layer before the ledger | `backtest/` | Qlib's optimizer and execution assumptions |
| Alpha158/Alpha360 | Independent, PIT-audited price-volume factor baselines | `data/`, `experiments/` | Direct use of Qlib China data/labels |
| Tuner | Bounded candidate generator with immutable trials and promotion gates | `experiments/`, `registry/` | Optimize on forward or a single proxy score |
| Online manager | Frozen-candidate shadow lifecycle and rollback records | `run/`, `registry/`, `forward_results/` | Automatic adaptive retraining in forward |

## 3A. Complete Borrowing Matrix And Priority

The following is the complete implementation map. "Adopt" means reproduce the
useful contract inside the project; it does not mean importing Qlib as a second
production framework.

| Qlib component | Why it is useful here | Minimum project implementation | When it may run | Formal decision role |
|---|---|---|---|---|
| MLflow recorder | Makes a trial traceable from config to output artifact | `experiment_manifest.json`, checksums, artifact index, status transitions | Immediately | Evidence provenance only; registry still decides |
| DataHandlerLP | Separates fitted preprocessing from inference-time use | `feature_transform_manifest.json` plus fit-boundary validator | Immediately, audit first | Leakage control |
| Rolling / label purge | Prevents a label from crossing the training or validation boundary | Existing rolling window contract, strengthened tests and manifests | Immediately | Generates research alpha only |
| RollingEnsemble | Ensures an ensemble date only uses earlier trained models | Component-alpha lineage and chronological blend builder | Only after a fair single-model baseline | Research; promotion needs full ledger evidence |
| Alpha158 / Alpha360 | Supplies a simple, external price-volume benchmark and factor-audit vocabulary | Recreate a selected, documented subset from existing A-share OHLC/PIT inputs | After transform audit | Independent model baseline, never direct replacement |
| Tuner | Makes small experiments comparable and stops ad-hoc parameter drift | Predeclared YAML/JSON search space, fixed budget, trial ledger, two-stage promotion | After one manual baseline is reproducible | Candidate generation only |
| EnhancedIndexingOptimizer | Separates expected return from risk, turnover and exposure limits | Proposal interface and constrained heuristic first; optional solver only later | After fixed-alpha evidence | Portfolio research only; ledger executes trades |
| TopkDropoutStrategy | Separates ranked score, retention/drop logic, and target holdings | Project-native TopK/dropout policy feeding the existing open ledger | After the common strategy interface exists | Strategy baseline only; ledger decides fills |
| OnlineManager | Records which frozen model and policy were active on each date | Operational manifest, activation log, shadow scorecard, manual rollback | Only after governed candidate exists | Forward observation and operations only |

### 3A.1 Experiment Recorder Work Package

Source idea: Qlib experiment/recorder lifecycle.

Implementation:

1. Add `experiments/recording.py` as the sole writer of a per-experiment
   manifest and append-only event log.
2. Give every run explicit states: `created`, `running`, `completed`,
   `failed`, `superseded`, and `archived`. A state change includes timestamp,
   command, and reason.
3. Persist config hash, code revision, Python/package versions, data/cache
   fingerprint, split bounds, model checksum, metrics, alpha path, log path,
   parent experiment IDs, and a working-tree state fingerprint. A Git `HEAD`
   hash alone is insufficient when the copied checkout contains uncommitted
   changes.
4. Add a read-only index builder for `reports/experiments/`; registry imports
   selected fields only after official ledger evidence exists.

Acceptance: deleting the terminal history must not prevent reproducing the
exact model, alpha, and evaluation inputs from the experiment directory. A
shadow or formal release must either use a clean source state or record an
explicit immutable source snapshot; a dirty working tree is never silently
treated as a release.

### 3A.2 Feature-Processing Work Package

Source idea: Qlib `DataHandlerLP` learn/infer processor separation.

Implementation:

1. Classify current v14 transformations as one of: source/PIT alignment,
   cross-sectional daily operation, train-fitted operation, or inference-only
   operation.
2. For every train-fitted operation, record `fit_start`, `fit_end`, learned
   statistic identifiers, input columns, missing-value policy, and application
   range. For cross-sectional daily operations, record the same-day universe
   rule instead of inventing a train fit.
3. Add a cache-reader guard: a model run must use a transform manifest whose
   fit range ends no later than its allowed training boundary.
4. Treat the present cache as an audited legacy cache until this sidecar
   exists; do not silently claim DataHandlerLP-equivalent guarantees before
   the audit is complete.

Acceptance: a reviewer can identify whether each feature transformation can
see the prediction period without reading model code.

### 3A.3 Factor-Baseline Work Package

Source idea: Qlib Alpha158 and Alpha360 handlers.

Implementation:

1. Create a factor specification that maps a selected Alpha158-like subset to
   columns available from this project's own OHLC/PIT inputs: returns, moving
   averages, volatility, volume/turnover, price position, and cross-sectional
   ranks.
2. Explicitly exclude any Qlib field not reproducible from project data with a
   known effective date. The label remains the project's `oo_lag1`/open
   execution-compatible label, not Qlib's default close label.
3. Build two reference variants: a compact price-volume set and a broader
   price-volume set. They are model baselines, not features automatically
   appended to v14.
4. Compare compact factor baseline, broad factor baseline, and v14 LightGBM
   using the same rolling windows and unchanged ledger policy.
5. Write a factor-audit report showing incremental value of v14 feature groups
   over the compact baseline, by selection split and market state.

Acceptance: the project can answer whether complexity beyond basic OHLC
factors improves realistic selection-period performance, rather than only IC.

Stop condition: do not import Qlib's provider data or labels merely to make
Alpha158 run; that would create a second, mismatched PIT contract.

### 3A.4 Bounded Tuning Work Package

Source idea: Qlib Tuner's declarative search space and persisted trial results.

Implementation:

1. Define a versioned search specification for one component at a time:
   model, alpha transform, or portfolio proposal. Never tune all three at
   once.
2. Cap the first model search at 12 trials and use predeclared parameters only
   (for example learning rate, leaves, minimum leaf size, regularization, and
   selected factor set).
3. Each trial creates an immutable experiment manifest and is screened first
   for data/protocol validity, then for 2024 validation. Only a small fixed
   number of survivors can run 2025 test.
4. No candidate is selected by IC alone. The final survivor uses the registry
   scorecard rules and full realistic stress coverage.
5. Never use 2026 forward to rank, prune, or rerun a tuning trial.

Acceptance: a failed search has a complete trial table, search-space hash,
budget, survivor rationale, and no unlogged manual parameter changes.

### 3A.5 Rolling Ensemble Work Package

Source idea: Qlib `RollingEnsemble`.

Implementation:

1. Implement `alpha/lineage.py` metadata containing prediction date, component
   experiment ID, component model train end, valid end, transform, and blend
   weight.
2. Require `component_train_end < signal_date` for every component. A blend
   builder rejects dates that violate this rule or have missing components.
3. Begin only with rank-average and equal-weight rank-average across named
   raw alpha families. Do not learn weights from proxy portfolio labels.
4. Assess diversity with rank correlation, Top-k overlap, sector overlap, and
   realistic-ledger marginal contribution, not just IC correlation.

Acceptance: every ensemble score can be traced to dated component artifacts;
no date is accidentally influenced by a later-trained component.

### 3A.6 Risk-Aware Construction Work Package

Source idea: Qlib `EnhancedIndexingOptimizer` objective of return minus risk
under turnover and exposure constraints.

Implementation:

1. Define an explicit proposal objective in project terms:
   expected alpha benefit minus penalties for active beta, specific volatility,
   industry concentration, turnover/new names, crowding, and execution risk.
2. Start with a deterministic constrained heuristic, because the current
   ledger contains non-convex tradability, lot, cash, and price-limit rules.
3. Add an optional optimizer only after the heuristic has a stable contract;
   it must emit desired target weights, never assumed fills.
4. The open ledger converts desired targets into feasible trades and returns
   realized holdings/exposure. Optimization is evaluated on realized rather
   than requested trades.
5. Treat benchmark weights as an optional risk-reference input, not an
   assumption that the strategy must become a CSI300 tracker.

Acceptance: the report decomposes alpha, risk penalty, desired turnover,
realized turnover, and blocked orders for every proposal.

### 3A.6A TopK Dropout And Execution-Timing Work Package

Source idea: Qlib `TopkDropoutStrategy` and the separation of Strategy from
Executor/Exchange.

Implementation:

1. Define one project-native strategy interface with inputs of dated alpha,
   current realized holdings, and policy parameters; outputs are desired target
   weights and an explicit replacement rationale.
2. Implement a simple equal-weight TopK/dropout policy beside the current
   retention policy. Parameters are `top_k`, `n_drop`, retention threshold,
   and optional maximum new names. It must not contain fill, cost, limit, ADV,
   or lot logic.
3. Feed every policy output to the existing realistic open ledger, which alone
   determines T+1 open orders, fill price, blocked orders, cash, lots, costs,
   and participation caps.
4. Compare TopK/dropout, current retention, and later risk-aware policies with
   identical alpha, selection dates, stress set, capitals, and ledger settings.
5. Treat close execution as a separate research family. A close-order policy
   must declare signal cutoff, order type (closing auction or VWAP), executable
   price source, capacity, and cost assumptions before it can run. Complete-T
   close features may never generate a same-T close order.
6. An open/close mixed policy must identify each order leg separately. Existing
   close mark-to-market is valuation, not proof that a close order can fill.

Acceptance: each policy report names score date, decision timestamp, order
timestamp, execution price field, and whether it is open-only, close-only, or
mixed. The formal baseline remains T-close signal to T+1-open execution until a
separate execution family completes its own realistic evidence.

Implementation status (2026-07-12): the project-native retention and
TopK/dropout policies share the existing realistic open-ledger execution path.
`selection_policy`, `top_k`, and `n_drop` are explicit sweep identity fields;
retention remains the default. Close-only and mixed execution remain unbuilt.

### 3A.7 Online Lifecycle Work Package

Source idea: Qlib `OnlineManager` model activation history.

Implementation:

1. Create an activation manifest containing candidate ID, parent experiment,
   model checksum, cache contract, alpha transform, policy version, start date,
   and explicit end/supersede reason.
2. Produce a daily shadow record with data-as-of time, signal generation time,
   eligible universe, proposed orders, ledger rejections, and realized open
   fills.
3. Define manual operational states: `shadow`, `active`, `paused`, and
   `retired`. Changing state never retrains or changes parameters automatically.
4. Define rollback as an operational decision with a logged reason, not an
   optimization response to a few forward days.

Acceptance: for any forward date the active signal and its frozen parent
artifact can be reconstructed exactly.

## 4. Architecture Target

```text
one versioned PIT market-data provider
  -> experiment-defined logical date view and warm-up range
  -> feature cache plus fitted/cross-sectional transform provenance
  -> model task (train / valid / predict, label-tail purge)
  -> model artifact + experiment manifest + dated alpha JSONL
  -> project-native strategy proposal (retention / TopK-dropout / risk-aware)
  -> existing realistic open-price share-ledger
  -> registry evidence -> attribution -> scorecard -> decision
```

The rolling layer may write research artifacts under `reports/`, but it may
not write a formal candidate into `registry/` by itself.

## 4A. Framework Completion Before Any Rolling Experiment

The work is intentionally split into two stages.

### Stage A - Research Framework

Stage A creates reusable contracts: experiment recording, feature-transform
provenance, task/window specification, alpha lineage, official-ledger adapters,
aggregate reporting, bounded-tuning records, portfolio-proposal interfaces, and
shadow lifecycle records. It is complete only when an experiment can be
reproduced and audited without terminal history or hand-written commands.

### Stage B - First Formal Walk-Forward Experiment

Only after Stage A is complete, run the fixed-length monthly rolling experiment
in Section 5A. It is the first serious user of the framework. It must not begin
from the current incomplete LightGBM prototype artifacts.

## 4B. A-Share Production Constraints For Stage A

Stage A is not a generic machine-learning platform. Its contracts must reflect
the project's A-share trading and data reality before any walk-forward result
is considered valid.

| Area | Required project rule | Stage A implementation / audit |
|---|---|---|
| Information timing | A signal dated T may use only data known by the configured signal cutoff; an open trade is submitted for T+1 open. | Record feature as-of time, signal timestamp, order timestamp, and execution date in the task and daily shadow manifests. |
| Fundamental data | Financial values become usable on their actual `ann_date` / `effective_date`, never merely on reporting-period end. Missing announcement dates remain explicitly marked and use the documented conservative estimate. | Transform manifest records effective-date policy, imputation flag, age/freshness, and estimated-notice fields. |
| Universe survivorship | The eligible universe must be historical, not today's stock list. Delisted, suspended, ST, newly listed, and board-specific status must be evaluated as of each trading date. | Add a PIT universe contract and coverage audit; a missing historical status is a flagged data-quality issue, not silent eligibility. |
| Price data | Features may use adjusted series only under an explicit adjustment contract; execution must use raw tradable OHLC and actual corporate-action-consistent share accounting. | Data manifest names feature-price adjustment, execution-price source, and corporate-action treatment separately. |
| Price limits | Main-board, ChiNext/STAR, Beijing Exchange, ST/*ST, and newly listed exception periods have different limits. | Ledger adapter must use date-specific actual limit/tradability masks; no universal 9.5 percent proxy in official evidence. |
| Tradability | Open-price orders can fail because of open limit, suspension, zero volume, insufficient ADV, or lot/cash constraints. Intraday touch rules are diagnostic unless the declared order type requires them. | Continuous ledger reports proposed orders, blocked orders, fill price, realized quantity, and reason code. |
| Capacity and costs | The target account is CNY 500k/CNY 1m, with A-share round lots, minimum commission, taxes/fees, and ADV participation limits. | Every official run covers both capitals and `normal`, `lag1`, `cost2x`, and `capacity_3pct`. |
| Calendar and external inputs | Monthly window boundaries use the A-share trading calendar. US/HK/commodity/macro inputs must have known publication/close timestamps before the A-share signal cutoff. | Data manifest records source date, effective trading date, timezone, and availability lag for each external series. |
| Labels and execution | Training labels must match the declared executable horizon, including the T+1 open delay for `oo_lag1`; tail labels are purged at every task boundary. | Task validator computes `label_end_offset` from label family/horizon and rejects inconsistent train/valid/OOS bounds. |
| Research governance | 2024 Val and 2025 Test select; full-year 2026 observes. A 2026-forward parent model must freeze by 2025-12-31. | Experiment recorder validates task ranges and lineage rather than inferring role from a directory name. |

### Stage A A-Share Acceptance Gate

Stage A is not complete until the following tests and artifacts exist:

1. A date-level PIT audit for features, labels, universe, and external inputs.
2. A price/adjustment audit that proves feature prices and executable prices are
   intentionally different where appropriate, never accidentally mixed.
3. Realistic-ledger regression tests for board/ST/new-listing limits,
   suspension/zero-volume, round lots, costs, ADV, and blocked-order reasons.
4. One end-to-end dry run that records T signal -> T+1 open order -> realized
   ledger outcome without touching forward data.
5. A coverage report that identifies missing rather than silently substitutes
   historical ST, listing, financial-notice, or external-market availability.

Failure of timing, transform, label, split, or ledger-integrity checks blocks
Stage B performance evaluation. Missing historical ST remains a mandatory
declared limitation and blocks a claim of complete historical-ST execution
coverage, but it does not block framework construction or a rolling audit that
is explicitly labelled with that limitation.

### Stage A Status Update - 2026-07-15

The historical-ST implementation contract is now present in
`data/st_status.py`. `open_ledger` prefers dated `st_status_events.csv`, the
execution-mask cache includes the event source and manifest in its key, and
`run/download_historical_st_events.py` writes a cutoff-filtered research copy
with page checkpoints under `data/tracking_raw`. The new contract and ledger
regressions pass 58 focused tests. The same downloader now exposes an explicit
`--endpoint namechange` fallback that reconstructs state from historical name
intervals and labels its manifest as `tushare_namechange_intervals`.

The actual data gate is still open. The current approved Tushare token was
tested without printing it and returned no access to the `st` endpoint; no
historical event file was created. The new audit artifact
`reports/qlib_research_framework_20260712/execution_coverage_st_contract_20260715.json`
therefore remains `audited_with_declared_gaps`. The fallback adapter is code-
complete but its full source download and coverage audit are still pending;
it is not an implicit substitution for the unavailable `st` feed. When used,
its manifest must expose `source_kind=tushare_namechange_intervals` and
`source_label=由历史股票名称区间重建`; the coverage audit reports both fields.

## 5. Phased Plan

### Phase 0 - Governance And Baseline Lock

Purpose: make the Qlib-style work a clearly bounded research decision.

Tasks:

1. Add ADR 0004 describing the research-only rolling-model layer, frozen data
   boundary, formal execution reuse, and registry promotion gate.
2. Add this plan to the current project index and development log.
3. Record the formal comparison target: `ledger_path_v3_t0001_nolookahead`.
4. Define an experiment ID convention:
   `rolling_<model>_<label>_<feature-cache-id>_<YYYYMMDD>`.

Acceptance:

- No Qlib-style output is described as formal, candidate, or promoted.
- The plan has no conflict with `PROJECT_RULES.md`, `RESEARCH_PROTOCOL.md`,
  or `registry/decision_rules.json`.

Stop condition: stop and revise the ADR if the design requires forward data,
new execution semantics, or a registry exception.

### Phase 1 - Complete The Experiment And Task Contract

Purpose: build reusable experiment/task contracts. This phase may implement the
controller and its artifacts, but does not yet make a rolling performance claim.

Current status:

- `experiments/rolling.py` implements ordered train/valid/predict windows and
  label-tail purge.
- `data/rolling_samples.py` streams the v14 cache and supports `oo_lag1`.
- `run/rolling_lgbm_alpha.py` trains LightGBM and writes raw alpha JSONL.
- Existing `predict_2024` and `predict_2025` raw signals were generated before
  model-file persistence was added. They are diagnostic only and must be
  regenerated.

Tasks:

1. Make `rolling_manifest.json` append/merge by experiment and window instead
   of overwriting earlier windows.
2. Persist one model artifact per window and store its SHA-256 checksum.
3. Store config checksum, git revision when available, cache metadata path and
   fingerprint, label family/horizon, label-end offset, feature flags/dimension,
   seed, sampled-row counts, date bounds, and command line.
4. Add per-window validation metrics to the manifest: daily rank IC summary,
   coverage, prediction row counts, and early-stopping iteration. These are
   diagnostic metrics, not selection criteria.
5. Make output directories immutable by default: a conflicting experiment ID
   must fail unless an explicit `--resume` policy validates matching metadata.
6. Add tests for manifest merge, checksum fields, date-bound validation,
   `oo_lag1` shift, label-tail purge, and deterministic sampling.

Acceptance:

- A rerun produces `model.txt`, dated raw alpha, and one complete manifest for
  both 2024 and 2025 windows.
- Focused tests pass using the Torch environment.
- Repeating a run with the same inputs either reproduces the same artifact
  identity or fails clearly instead of silently overwriting it.

Stop condition: stop before model comparison if cache metadata cannot prove
the same frozen PIT inputs and label semantics were used.

### Phase 2 - Feature Transform Provenance, Not Feature Rework

Purpose: borrow Qlib's fit-versus-infer discipline without changing proven
v14 feature values.

Tasks:

1. Audit how each v14 transform is fitted and applied: filling, clipping,
   normalization, cross-sectional operations, and PIT availability.
2. Write a sidecar transform manifest for new cache builds containing fit date
   bounds, source columns, fitted statistics/version identifiers, missing-value
   policy, and effective-date policy for fundamental inputs.
3. Add a validator that rejects a rolling run when its cache manifest is
   missing or when a recorded fit end exceeds the allowed training boundary.
4. Do not rebuild v14 merely to change documentation. Rebuild only if the
   audit finds an actual time-leakage or semantic defect, with a separate ADR.

Acceptance:

- Every new rolling experiment can state exactly which feature-cache contract
  it used.
- No numerical feature change occurs in this phase without explicit evidence
  and a separate decision.

Stop condition: if the audit discovers a leakage risk, suspend all new model
comparisons until the affected cache and labels are repaired.

### Phase 3 - Rolling LightGBM Baseline And Selection Evidence

Purpose: test one constrained tabular baseline fairly against the existing
formal portfolio baseline.

Tasks:

1. Regenerate two complete alpha artifacts:
   - `predict_2024`: train 2010-2022, validate 2023, predict 2024.
   - `predict_2025`: train 2010-2023, validate 2024, predict 2025.
2. Use the existing signal schema and explicitly label the output as raw alpha.
   Do not apply a Qlib portfolio strategy.
3. Run the registry-compatible official ledger route for each selection split:
   `normal`, `lag1`, `cost2x`, and `capacity_3pct`, at both capital levels.
4. Produce attribution and a scorecard versus
   `ledger_path_v3_t0001_nolookahead`.
5. Record IC, Top-k return, turnover, capacity binding, rejection reasons,
   annualized return, Sharpe, maximum drawdown, and cost. IC is a diagnostic
   floor only, not the promotion objective.
6. Do not change target fraction, holding, replacement, maxret095, or any
   portfolio rule while measuring this first model baseline.

Acceptance:

- Complete, comparable 2024/2025 evidence exists under exactly the same
  realistic ledger settings as the formal baseline.
- The report carries signal and backtest date bounds and names the cache/model
  artifact used.

Decision gate:

- Promote to a registered shadow candidate only if it meets all formal
  decision rules on selection evidence and has complete attribution.
- Otherwise retain it as a documented negative baseline and do not tune it on
  2026 forward.

### Phase 3B - Alpha158/Alpha360-Inspired Factor Baselines

Purpose: test whether the project benefits from its rich feature set relative
to simple price-volume factor baselines, using the Factor-Baseline Work Package
in Section 3A.3.

Precondition: Phase 2 has documented transform provenance.

Acceptance: compact and broad factor baselines have the same rolling dates,
label family, raw alpha schema, and formal ledger settings as the v14 baseline.

Stop condition: do not compare a Qlib close-label Alpha158 result against the
project's open-execution result; it is a different prediction problem.

### Phase 3C - Bounded Tuning

Purpose: run a small, auditable candidate search only after manual baselines
are reproducible, using the Bounded Tuning Work Package in Section 3A.4.

Precondition: both Phase 3 and Phase 3B have complete manifests and at least
one official selection-period ledger report.

Acceptance: no more than the declared trial budget runs, and only predeclared
survivors are allowed to touch the 2025 test split.

### Phase 4 - Historical OOF Coverage And Ensemble Research

Purpose: only after Phase 3 is valid, create historical model-diversity
evidence without leaking future labels.

Tasks:

1. Generate chronological expanding or fixed-length rolling windows for an
   agreed pre-selection period, for example predictions spanning 2018-2023.
2. For every date, retain only predictions from models trained before that
   date. Store `train_end`, `valid_end`, `model_id`, and prediction date in the
   alpha metadata.
3. Build OOF diagnostics: stability by year/market state, correlation with
   existing alpha families, coverage, turnover tendency, and overlap of top
   names.
4. Test only predeclared simple blends, such as rank average or clipped rank
   average, against the unchanged ledger. No learned portfolio selector in
   this phase.
5. Use 2024 val and 2025 test only for the final blend decision. Historical
   OOF is for robustness/diagnosis, not a substitute for selection evidence.

Acceptance:

- No signal date is scored by a model trained on that date or later.
- Each blend is reproducible from named component alpha artifacts.
- No result is based on the old proxy portfolio-label dataset.

Stop condition: if a blend improves only 2026 forward or only one selection
split, it remains research-only and receives no further parameter tuning.

### Phase 5 - State-Aware Portfolio Construction Pilot

Purpose: borrow the risk-budget idea while retaining the proven ledger as the
single execution engine.

Precondition: at least one raw alpha or simple blend passes Phase 3 evidence,
or a documented existing alpha is selected as the fixed input.

Tasks:

1. Build an exposure report from actual ledger holdings: industry HHI, top
   industry weight, beta, specific volatility, active drawdown, turnover,
   new names, crowding/rapid-rise diagnostics, and global-pressure variables.
2. Define a small, interpretable proposal interface before execution:
   target weights or replacement priorities plus explicit constraints.
3. Implement no more than two predeclared proposal families:
   - exposure-aware reweighting with industry/beta/turnover penalties;
   - replacement suppression when a candidate worsens predeclared risk
     measures.
4. Feed each proposal into the existing realistic ledger. The ledger remains
   responsible for tradability, lots, cash, ADV, price limits, and fills.
5. Evaluate 2024/2025 with the full stress suite and attribution. Treat 2026
   as observation only.
6. Before risk-aware proposals, implement and evaluate the plain TopK/dropout
   policy in Section 3A.6A as the interpretable strategy-layer baseline.

Acceptance:

- Each changed trade can be traced to an alpha or an explicit risk constraint.
- No generic QP optimizer or proxy utility label bypasses real execution.
- A proposal cannot enter registry without complete formal evidence.

Stop condition: if the proposal improves proxy metrics but not realistic
ledger utility on both selection splits, archive it as negative evidence.

### Phase 6 - Operationalization And Forward Observation

Purpose: adopt the useful part of Qlib's lifecycle management only after a
candidate has passed selection.

Tasks:

1. Freeze model, feature-cache contract, signal transform, and ledger policy
   into an operational manifest.
2. Generate dated forward alpha from `data/forward_raw` without retraining or
   rule tuning during the observation campaign.
3. Maintain a forward scorecard that distinguishes delayed data availability,
   execution rejection, model alpha, and portfolio-constraint effects.
4. Define a manual rollback condition and report-only monitoring thresholds;
   do not automatically alter model parameters from forward results.

Acceptance:

- Forward artifacts have an explicit frozen parent experiment and registry
  candidate ID.
- Forward results cannot silently modify the research record.

Current status (2026-07-16): the provenance/rollback bundle and a read-only
scorecard validator are implemented, but the bundle is not current-valid. It
inherits the stale `2026-05-18`/`2026-05-19` protocol and has no formally
promoted candidate or complete forward scorecard. Phase A must correct and
retest the split contract before Phase 6 is regenerated. No conditional
forward alpha is activated. A clean commit or explicit source snapshot remains
required before any active shadow claim.

The corrected target is one physical PIT data source with explicit logical
views. `forward_2026` covers 2026-01-01 through the latest complete available
date and remains observation-only. Directory names are storage implementation
details, not evidence of selection eligibility.

## 5A. First Formal Monthly Walk-Forward Experiment

Purpose: determine whether retraining on a recent fixed-length history captures
regime change better than an expanding history, using many truly
out-of-sample months rather than one favorable backtest interval.

This experiment starts after Phases A-D in Section 0 have supplied the required
protocol, manifest, workflow, provider, and processor contracts. Operational
shadow activation is not a prerequisite for a historical rolling audit. The
experiment reuses the realistic ledger; it is not a new execution mode or a
new selection protocol.

### Initial Schedule

Start with one frozen base strategy configuration from the existing M0 family:

| Segment | Initial setting | Role |
|---|---|---|
| Train | trailing 4 years | fit model and train-fitted transforms |
| Valid | trailing 6 months after Train | choose checkpoint within that task only |
| Label-tail purge | exact `label_end_offset` at Train and Valid tails | remove labels crossing segment end |
| OOS | next calendar month | generate alpha only; no task-local retuning |
| Step | one calendar month | create the next independent task |

Fixed 3-year, 5-year, and expanding-history schedules are comparison arms only
after the initial 4-year controller passes correctness and parity checks. All
arms must use the same model family, feature/label contract, checkpoint rule,
OOS months, alpha transform, and ledger policy.

### Boundary And Embargo Rule

For the current delayed open label, do not mechanically discard the first five
OOS days. Instead, calculate the exact `label_end_offset` for the chosen label
and horizon, then purge the last offset trading days of Train and Valid when
their complete labels cross the segment boundary. OOS alpha can begin on the
first eligible trading day of the next month because no OOS label is used in
that window's fitting or checkpoint selection. A stricter explicit embargo is
allowed only when documented in the task manifest; omitted OOS days then remain
cash days in the stitched ledger path.

### Per-Window Artifact Contract

```text
reports/experiments/<experiment_id>/
  experiment_manifest.json
  windows/<YYYY-MM>/
    task.json
    data_manifest.json
    selected_checkpoint.json
    alpha.jsonl
    ledger/
    metrics.json
  oos_alpha.jsonl
  oos_equity_curve.csv
  aggregate_metrics.json
  stability_report.md
```

`task.json` freezes train/valid/purge/OOS bounds, model config, seed, cache and
transform fingerprints, checkpoint-selection metric, and declared policy.

### Controller Invariants

1. Every OOS date has exactly one owner window and exactly one alpha source.
2. The owner model's Train end and Valid label end precede the OOS date.
3. Standardizers, imputers, feature selection, and learned transforms fit only
   inside that window's Train segment.
4. Checkpoint selection may examine only that task's Valid segment.
5. The controller never selects a better window after observing OOS returns.
6. OOS alpha is stitched chronologically, then sent through one continuous
   realistic ledger path. Capital, holdings, cash, costs, blocked orders, and
   ADV constraints carry across month boundaries; monthly account resets are
   prohibited.
7. Window failures are retained and reported, never silently skipped.

### Staged Execution And Decision

1. Dry-run task generation for 2024/2025: verify date boundaries, purge
   counts, cache contracts, and exactly one OOS owner per date.
2. Pilot three consecutive OOS months with frozen M0: verify checkpoint
   isolation, alpha stitching, and continuous-ledger parity.
3. Run the complete 2024 validation schedule.
4. Freeze the implementation and run 2025 test once for surviving schedules.
5. Compare 4-year fixed, then 3-year/5-year fixed and expanding-history arms
   only under the same OOS months and formal ledger settings.
6. Generate 2026 forward only after selecting from 2024/2025; it cannot choose
   train length, valid length, step, checkpoint rule, blend, or policy.

Evaluate both CNY 500k and CNY 1m under `normal`, `lag1`, `cost2x`, and
`capacity_3pct`: annualized return, Sharpe, maximum drawdown, turnover, cost,
capacity binds, monthly win rate, worst-month behavior, and market-state
stability. A schedule improving only one selection split remains research-only.

## 6. Explicit Sequence

The old Phases 0-5 produced reusable foundations and negative/conditional
research evidence. They are not instructions to resume parameter searching.
From 2026-07-16 onward the execution order is:

1. Phase A: unify split semantics and repair the full-year Forward contract.
2. Phase B: make experiment/data/date/lineage fields mandatory for formal use.
3. Phase C: compile one declarative workflow into the existing project modules.
4. Phase D: unify date-sliced provider and transform fit/apply state.
5. Phase E: execute the first formal monthly walk-forward experiment.
6. Phase F: test hypothesis-driven portfolio construction only after Phase E.
7. Phase G: rebuild manual shadow lifecycle under the corrected contract.
8. Revisit historical ST only when the external source situation changes;
   never conceal its absence in formal execution coverage.

## 7. Deliverables

- ADR 0004 for the rolling research layer.
- Reproducible experiment directory per run: config, model, alpha JSONL,
  complete manifest, diagnostics, and log.
- Feature-transform sidecar/validator for future cache builds.
- Official 2024/2025 ledger, attribution, and scorecard reports for each
  serious candidate.
- OOF coverage report and blend manifest, if Phase 4 is reached.
- Proposal/constraint artifact and trade-level rationale, if Phase 5 is
  reached.
- TopK/dropout policy artifacts and separate, timestamp-audited execution
  research reports for any close-only or mixed open/close family.
- Monthly task manifests, stitched OOS alpha, one continuous realistic-ledger
  path, and a stability report after Stage A is complete.

## 8. What Success Looks Like

Success is not "Qlib installed" or a higher IC. It is one of these outcomes:

1. A reproducible rolling baseline proves stronger or diversifying under the
   existing formal ledger and becomes a governed shadow candidate; or
2. It fails fairly and becomes durable negative evidence, preventing future
   cycles of the same unverified experiment.

Either outcome improves the project because it strengthens model evaluation
without diluting its PIT, realistic-execution, and forward-data discipline.

## 9. Long-Term Roadmap Update - 2026-07-17

The project has aligned the core research workflow with Qlib's useful ideas,
but it is not a complete Qlib-equivalent platform. Declarative Workflow v2,
streamed Dataset/DataHandler/Processor execution, common model adapters,
purged rolling tasks, unique OOS ownership, standardized dependent Records,
signal/strategy/executor separation, continuous realistic-ledger evaluation,
and manual lifecycle governance are implemented. Full formal-run acceptance,
frozen/legacy prediction compatibility, strong-model Rolling performance,
dynamic rolling ensemble, unified portfolio construction, and the daily
Shadow runner/replay remain open.

The completed monthly Compact experiment validates the controller but rejects
that base learner. Reconstructed strong-model pilots also failed their
predeclared signal gates. These are learner results rather than framework
failures; Section 16 defines the current closure order.

The authoritative long-term sequence and acceptance gates are documented in
`LONG_TERM_QUANT_PLATFORM_ROADMAP_20260717.md`. This update does not alter the
formal baseline, use Forward for selection, or start training.

## 10. Qlib-First Priority Revision - 2026-07-17

The immediate priority is now framework alignment, not strong-model rolling.
The previous order that placed the reconstructed e19 rolling smoke before the
generic Dataset/Processor/Model contracts is superseded.

The canonical implementation order is:

1. Q0: freeze the Qlib source revision, component mapping, vocabulary, and
   workflow schema draft;
2. Q1: make one declarative Task/Workflow the source of all formal runtime
   parameters;
3. Q2: implement the project-native Dataset/DataHandler/Processor fit/apply
   runtime on top of the existing low-memory PIT caches;
4. Q3: implement interchangeable LightGBM, PyTorch strong-alpha, frozen, and
   legacy-read-only model adapters;
5. Q4: standardize Signal, SignalAnalysis, Portfolio, RiskAttribution, Stress,
   and Decision records with explicit dependencies;
6. Q5: run the strong-model monthly Rolling/OOF experiment through those
   interfaces;
7. Q6: continue Strategy and portfolio-construction research;
8. Q7: build the manual Shadow/Online lifecycle;
9. Q8: consider optional model-zoo, distributed, MLflow, RL, and meta-learning
   capabilities only after the core platform is stable.

The reference checkout is frozen at
`C:\Users\x\Documents\股票预测\references\qlib`, commit
`d5379c520f66a39953bad76234a7019a72796fd0`. Alignment means adapting Qlib's
interfaces and lifecycle discipline to this project. It does not mean using
Qlib's China data, labels, default close-price backtester, Exchange/Executor,
or automatic online promotion.

The detailed authoritative order and acceptance gates are in
`LONG_TERM_QUANT_PLATFORM_ROADMAP_20260717.md`. If an earlier phase or priority
statement conflicts with Section 10 or that roadmap, the newer Qlib-first
sequence governs.

## 11. Q0-Q1 Implementation Update - 2026-07-17

Q0 is complete: the frozen Qlib source reference, 13-component mapping,
terminology, exclusions, workflow-v2 JSON Schema, and golden A-share config are
machine-tested. Q1 is also complete for the currently implemented runtime:
workflow v1 remains replayable, while workflow v2 is schema-validated,
semantically checked, frozen in the experiment manifest, and explicitly
normalized to the proven stage graph for rolling LightGBM and frozen-artifact
models. Q3-only adapters fail explicitly until implemented.

The next phase is Q2. It must implement executable shared/infer/learn
Processor chains and a Dataset `prepare(segment, col_set, data_key)` facade on
top of the existing low-memory PIT providers. No strong-model training begins
in Q2.

## 12. Q2 Implementation Update - 2026-07-17

Q2 is complete. `data/dataset_runtime.py` provides streamed shared/infer/learn
Processor chains, Train-only fitting, frozen state persistence and hashing,
named-segment `prepare`, and Workflow-v2 construction over the real v14 memmap
provider. It does not duplicate or rewrite the physical cache.

Q3 is now the active phase. Existing model trainers remain operational during
the migration; complete Model alignment requires LightGBM, PyTorch strong
alpha, frozen artifacts, and legacy read-only models to emit one dated
PredictionFrame contract without importing portfolio execution.

## 13. Q3 Implementation Update - 2026-07-17

Q3 is complete at the framework-contract level. `experiments/model_adapters.py`
provides one lifecycle and factory for LightGBM, PyTorch strong alpha, frozen
artifacts, and legacy read-only signals. All emit the same dated
`PredictionFrame`; learnable adapters save and resume checkpoints, frozen
artifacts verify hashes, and the model layer has no ledger dependency.

The PyTorch adapter deliberately requires an explicit existing-trainer
delegate. Reconstructing and binding `multi_downside_e19` to monthly Rolling is
the Q5 acceptance workload, not an unrecorded assumption inside Q3. Q4 is now
active and must standardize dependent evidence records before any strong-model
training begins.

## 14. Q4 Implementation Update - 2026-07-17

Q4 is complete at the record-contract level. Six project-native templates now
form an immutable dependency chain with required fields, artifact hashes,
standard layout, and a schema-validated materialization CLI. Portfolio records
can only reference the official open-price ledger; the record layer contains
no second executor. Decision records enforce Val/Test selection and separate
Forward observation.

Existing summary-only experiments are not relabeled as complete because they
do not contain all positions, orders, rejections, and cost artifacts. Q5 must
produce those real artifacts for the new strong-model workflow. Q5 is now
active, beginning with reconstruction and audit of the historical
`multi_downside_e19` trainer configuration before a one-window smoke.

## 15. Q5A Framework Binding And Q4B Priority - 2026-07-17

Q5A is complete at the framework level. `torch_strong_alpha` is accepted by
Workflow v2, compiles to the hash-checked resumable staged runner, and emits
the same learner-neutral `rolling_manifest.json` and split-alpha contract used
by the LightGBM workflow. A compatibility materializer can wrap completed
exploratory strong runs without retraining or changing their nonpromotable
status. This acceptance is about interfaces and provenance, not profitability.

Q4B is now complete. The project-native `open_ledger` can preserve genuine
order/fill, rejection, position, cost, equity, and diagnostics evidence, and
Workflow v2 can materialize the six dependent Records without introducing a
second executor. The remaining production gate is a newly executed complete
formal Workflow, preceded by frozen/legacy dated-prediction compatibility and
followed by Q7B daily Shadow/replay. Strong-model rolling performance,
checkpoint redesign, and portfolio optimization remain deferred until this
framework closure is accepted.

## 16. Current Alignment Closure Plan - 2026-07-17

This section supersedes stale status wording in Sections 9 and 15. Q0-Q3,
Q4A/Q4B, Q5A, and Q7A are implemented. The project now has declarative
Workflow v2, streamed Dataset/DataHandler/Processor semantics, common model
adapters, learner-neutral rolling artifacts, genuine realistic-ledger detail,
automatic six-Record materialization, and a manual Shadow lifecycle.

Implementation is not yet the same as production acceptance. The remaining
alignment work is classified as follows.

### 16.1 Required framework closure

1. Completed: a hash-checked dated-prediction adapter now supports frozen and
   legacy candidates that do not own a learner-neutral `rolling_manifest.json`.
2. Completed: the frozen formal baseline executed end to end through dated
   prediction, realistic ledger, scorecard, and all six dependent Records.
3. Completed: the resulting bundle has exact signal/backtest ranges, parent hashes,
   500k/1m capital coverage, normal/lag1/cost2x/capacity_3pct evidence, genuine
   positions/orders/rejections/costs, and no Forward selection.
4. Completed at framework level: the complete bundle is the frozen acceptance
   input for the Q7B daily runner. A 20-signal-day historical run produced
   data-ready checks, frozen dated-prediction proposals, realistic accounting,
   daily evidence, close/active-return attribution, drift reports, and immutable
   hashes. An independent second execution matched all path-independent daily
   semantic hashes.
5. Operational gate remains manual: the bound lifecycle is still `prepared`.
   Formal observations can be appended only after explicit manual transition to
   `shadow`; historical replay cannot activate, promote, retrain, trade, or use
   Forward results for selection.

### 16.2 Research work after framework closure

1. Q5B: redesign the rejected strong-model checkpoint/selection recipe before
   any complete 24-window e19 Rolling run.
2. Build a reusable as-of Rolling Ensemble that rejects unavailable models,
   duplicate date ownership, Forward-selected weights, and nondeterministic
   failure fallback.
3. Q6: unify Retention, TopK/dropout, and alpha-risk-cost portfolio proposals
   behind one Strategy/Portfolio Constructor contract with trade-level reasons.
4. Evaluate profitability only with 2024 Val and 2025 Test selection evidence;
   keep full-year 2026 Forward observation-only.

### 16.3 Explicit external or optional gaps

- Historical ST source coverage remains an external-data limitation. The
  adapter and disclosure contract exist; missing events must never be silently
  approximated as complete.
- MLflow service, distributed task scheduling, broad model zoo, automatic
  tuner, RL, meta-learning, automatic retraining, automatic promotion, and
  automatic trading remain deferred. They are not required for the current
  16-GB single-machine research plus manual Shadow target.
- Qlib Exchange/Executor remains intentionally not adopted. The project-native
  realistic open-price `open_ledger` is the sole formal execution authority.

### 16.4 Completion rule

Qlib alignment may be called framework-complete only after the first full
formal Record bundle and Q7B replay acceptance exist. Compile-only workflows,
synthetic bundles, unit tests, and the 63-date v14 signal/label smoke are useful
evidence but do not satisfy that completion rule.

The frozen-adapter acceptance smoke used the formal
`ledger_path_v3_t0001_nolookahead` registered signal without training or
promotion. It validated 242 Val dates / 1,197,376 stock rows and 243 Test dates /
1,213,948 stock rows, including both list-valued and code-mapped alpha rows.

The first full formal acceptance is
`reports/experiments/workflow_frozen_baseline_formal_acceptance_v2b_20260717`.
It completed four Workflow stages, 16 Val/Test capital-stress cells, and all
six Records; bundle SHA-256 is
`f47404016c501084ff4e5e8ab54b2695b404528b570862074d5d6dd45d0c6234`.
The first attempt exposed Windows path-length and subprocess-failure
propagation bugs. The second attempt exposed empty optional Forward CSV
handling. All were fixed with regression coverage; failed receipts and partial
inputs remain preserved. A real lifecycle is now frozen in `prepared` state
and has not been activated.

Q7B framework replay acceptance is recorded in
`reports/shadow_replays/Q7B_DAILY_SHADOW_REPLAY_ACCEPTANCE_20260717.md`. The
accepted source and replay each contain 20 daily packets for signals from
2026-01-05 through 2026-01-30 and next-session execution through 2026-02-02.
Their path-independent semantic SHA-256 is
`7c82628f4e1f3e92b1667664f7a788b7a97ca041bd5598de96dc340f0d0dce8d`.
This closes the code-and-replay portion of Q7B, not the manual operational
activation gate.

## 17. Reverse Alignment Audit - 2026-07-18

The first-pass completion wording was too broad. Q2 and Q3 are complete as
tested interfaces, but they are not yet the exclusive formal training path.
`run/rolling_lgbm_alpha.py` and the staged strong-model runners still own
legacy sample loading and fit/predict orchestration. They must remain available
until an end-to-end migration proves parity through the formal ledger.

The alignment matrix now marks DataHandlerLP, DatasetH, Model adapters, and the
generic Recorder as partial. The accepted frozen-baseline Workflow and Q7B
replay remain valid; no baseline, candidate, or lifecycle state changed.

Rolling `run_mode=formal` historically means an executed non-dry run. It is no
longer described as governance-formal evidence unless the parent experiment
also passes the immutable formal-manifest and artifact-index validator.

ADR 0007 supersedes the stale boundary wording in ADR 0001: selection ends on
2025-12-31 using 2024 Val and 2025 Test, while all 2026 observations are
Forward-only. The date 2026-05-18 remains legacy cache provenance only.

## 18. LightGBM Dataset Mainline Migration - 2026-07-18

The formal LightGBM rolling runner now accepts an explicit project-native
Dataset source. It constructs each named segment from already purged rolling
indices, rejects non-contiguous reconstruction, and keeps the historical
daily-quota seeded sampler and trainer unchanged.

A real Compact `predict_2024` comparison produced identical date counts,
sampled rows, best iteration, model bytes, raw alpha bytes, and stitched Val
alpha bytes. The optimized Dataset path remains about 11.6% slower than the
legacy iterator, so `legacy_iter` stays the default pending 2025 parity and
performance analysis. This closes the first LightGBM integration proof, not
the strong-model or common Model-adapter migration.
