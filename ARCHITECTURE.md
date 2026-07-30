# Architecture

## Flow

```text
data/raw (compatibility) -> data/ PIT features and labels -> v14 memmap cache
-> core/ PyTorch model -> alpha/ JSONL signal -> portfolio candidate
-> backtest/ realistic open-price ledger -> registry evidence
-> attribution -> scorecard -> governance decision

data/forward_raw (compatibility) -> forward signal and ledger observation only

target: data/market_daily content-addressed Parquet
-> logical selection/Forward DataView
-> MarketDailyProvider
-> month-sharded execution cache
-> existing realistic open-price ledger
```

## Module Ownership

| Area | Owner | Responsibility |
|---|---|---|
| Model/config | `core/` | datasets, model, losses, train/eval |
| Data | `data/` | PIT features, labels, raw/cache lifecycle |
| Signal | `alpha/` | Alpha I/O and signal transforms |
| Research experiments | `experiments/` | task contracts, immutable manifests, artifact lineage, research-only utilities |
| Execution | `backtest/` | strategy proposals, ledger, costs, lots, ADV, limits, metrics |
| CLI | `run/` | thin orchestration only |
| Governance | `registry/` | baseline, candidate, report, attribution, decision metadata |
| Evidence | `reports/official/` | reproducible official outputs |

## Qlib-Aligned Target Interfaces

The accepted target flow is Provider -> DataHandler/Processor -> Dataset ->
Model Adapter -> dependent Records -> Strategy -> project-native Executor.
`reports/qlib_alignment_20260717/qlib_alignment_matrix.json` is the Q0
component map, and `schemas/workflow_v2.schema.json` is the active Q1
declarative contract. Runtime preserves schema v1 replay and compiles the
implemented v2 LightGBM/frozen/strong-PyTorch stage graph through an explicit
compatibility normalizer. The Q3 model lifecycle is implemented separately;
unsupported concrete trainers fail rather than silently falling back.

Qlib's DataHandler, Dataset, Model, Record, Rolling, and Online contracts are
interface references. Qlib's default data, labels, Exchange, and Executor are
not production dependencies. `backtest/open_ledger.py` remains the sole formal
A-share execution authority.

`data/dataset_runtime.py` is the Q2 streamed runtime. It owns named segments,
raw/infer/learn data keys, shared/infer/learn processor chains, Train-only
fitting, frozen state hashes, and the V14 provider adapter. It never owns
checkpoint selection or execution. `experiments/model_adapters.py` is the Q3
boundary: four model families share fit/resume/checkpoint/predict/save and the
dated `PredictionFrame`; the module cannot import the ledger. Q5A binds the
existing staged `multi_downside_e19` trainer to Workflow v2 and the common
rolling-manifest/split-alpha artifacts without importing execution code.

Q2 and Q3 are implemented contracts, not yet the sole production path. The
formal LightGBM and strong-model rolling CLIs still read samples and perform
fit/predict through legacy trainer code. Migration is complete only when those
CLIs consume the Dataset runtime and Model adapters end to end; until then the
legacy paths remain protected dependencies rather than cleanup candidates.

The LightGBM rolling CLI now supports an explicit `data.dataset_runtime` of
`legacy_iter` or `project_dataset`. The project Dataset path consumes resolved,
label-tail-purged indices and delegates daily reads to `V14MemmapProvider`; it
retains the established quota/seed sampler and LightGBM trainer. A real 2024
Compact window produced byte-identical model and alpha artifacts. Legacy
remains the default until a second-window parity and remaining performance
overhead audit complete; the Model adapter migration is a separate step.

`experiments/record_templates.py` is the Q4 evidence boundary. Signal,
SignalAnalysis, Portfolio, RiskAttribution, Stress, and Decision records form
one immutable dependency chain with standardized manifests and hashes.
`run/materialize_standard_records.py` references outputs from existing
analytics and the official ledger; it never implements a second executor.
Legacy summaries that lack positions/orders/rejections/cost details cannot be
upgraded silently into a complete formal bundle.

Q4B closed the Record integration gap. Formal registry runs now persist
and hash genuine equity, position, order/fill, rejection, and cost artifacts
through an optional side-channel of the project-native ledger. Daily aggregate
diagnostics remain distinct from the order ledger.

Q4B is now implemented: Workflow v2 compiles `workflow_standard_records` after
scorecard, aligns learner-neutral predictions to raw v14 labels, and references
the official ledger evidence when materializing the six immutable Records.
Summary-only historical experiments remain ineligible.

`experiments/prediction_artifacts.py` closes the frozen/legacy signal gap.
It resolves the same registered split-alpha shapes used by the official ledger,
validates list- and code-mapped alpha one date at a time against the
`PredictionFrame` contract, and freezes source paths, hashes, ranges, and row
counts in `dated_prediction_manifest.json` without copying the alpha. Frozen
Workflow stages now materialize this read-only lineage before ledger execution;
automatic Records accept either this manifest or a learner-neutral
`rolling_manifest.json` and recheck source hashes.

The first production-shaped acceptance bundle is
`reports/experiments/workflow_frozen_baseline_formal_acceptance_v2b_20260717`.
It proves the frozen prediction, isolated official ledger, scorecard, and six
Record stages can complete with hash-checked resume. Detailed ledger filenames
are bounded against the full Windows path, child sweep failures propagate to
Workflow failure, and an absent observation split is represented as
`not_requested`. The bound lifecycle remains `prepared`; Q7B owns daily
execution and replay.

`experiments/shadow_lifecycle.py` is the Q7A lifecycle boundary. It accepts
only a hash-valid complete Record bundle and manages manual `prepared`,
`shadow`, `paused`, and `retired` states through a hash-chained event log.
There is deliberately no automatic active/trading state. Dated observations
are immutable, unique, and observation-only. The legacy Phase 6 manifest is
retained as historical evidence, not used as the generic lifecycle engine.

`experiments/shadow_daily.py` and `run/run_daily_shadow.py` implement Q7B. The
runner consumes frozen dated predictions, delegates all accounting to the
project-native realistic open ledger, and packages each signal date with its
next-session execution, proposal, equity, diagnostics, positions, orders,
rejections, costs, drift, and immutable hashes. `historical_replay` is allowed
while a lifecycle is `prepared`, but cannot append lifecycle observations.
`shadow_observation` requires an explicit prior manual transition to `shadow`.
Replay re-executes the ledger in a separate directory and compares a
path-independent semantic digest; output paths remain part of artifact
identity but not economic equivalence.

## Official Entry Points

- Training: `run/train.py`
- Signal/policy generation: `run/generate_ledger_path_v3_signal.py`
- Official batch backtest: `run/official_backtest_from_registry.py`
- Formal baseline freeze: `run/freeze_formal_baseline.py`
- Daily Shadow and deterministic replay: `run/run_daily_shadow.py`
- Experiment alpha evaluation: `run/evaluate_experiment_alpha.py` (records a
  dated research artifact, then delegates to the same realistic ledger; it is
  not a registry-promotion path)
- Execution evidence audit: `run/audit_execution_coverage.py` (reports source
  coverage and gates claims where historical execution inputs are incomplete)
- Provider/PIT audit: `run/audit_provider_contracts.py` (read-only split-aware
  audit; cache identity mismatches are reported and never trigger a rebuild)
- Historical ST download: `run/download_historical_st_events.py` (Tushare
  source, filtered at the research cutoff, with page checkpoints; no registry
  or forward-data writes)
- Shared-OHLC sweep: `run/sweep_open_price_ledger_params.py`
- Attribution: `run/attribution_from_registry.py`
- Scorecard: `run/scorecard_from_registry.py`

Use single-run ledger scripts for diagnosis only. Official evidence uses registry-driven batch execution.

## Data Contracts

- Canonical logical views: 2024 Val, 2025 Test, and observation-only 2026
  Forward, physically verified through 2026-07-29. `core/research_protocol.py` is the
  single split source.
- `data/raw` and `data/forward_raw` are compatibility storage roots during the
  provider migration. Directory names do not define selection eligibility;
  manifests record physical coverage and logical ranges separately.
- `data/market_daily_store.py` is the MD1 transaction boundary for the target
  daily store. It validates one-date partitions, writes deterministic Parquet
  payloads, preserves revisions by content hash, verifies immutable manifests
  and changes the active generation only through `CURRENT`.
- `data/market_daily_migration.py` performs resumable year/month migration and
  exact CSV parity audits. `data/market_daily_update.py` is the MD3 candidate
  writer: Tushare supplies the A-share cross-section, an interchangeable index
  client supplies four frozen broad indices, and each run records source,
  coverage and amount semantics. Neither module changes the formal backend.
- `CsvMarketDailyBackend` and `ParquetMarketDailyBackend` now implement the same
  long-form storage contract behind `MarketDailyProvider`. The provider enforces
  DataView bounds and returns field-keyed date-by-code matrices compatible with
  the existing OHLC facade. CSV remains the configured oracle until monthly
  cache and ledger parity are complete.
- `backtest/monthly_ohlcv_cache.py` is the MD5 execution-read layer. Each month
  is bound to one immutable source month-index hash, written through a staged
  content-addressed generation and selected by an atomic CURRENT pointer.
  Cross-month derived fields are computed after stitching. Basic OHLC/volume
  masks remain distinct from full ST, listing-age and price-limit eligibility.
- MD6 keeps execution storage selection explicit: `legacy` is the unchanged
  formal default, `csv` is the direct parity oracle, and `monthly` is the
  candidate. Both ledger CLIs record the selected backend; monthly realistic
  masks bind their cache key to each active immutable monthly generation.
- `run/run_open_ledger_backend_parity_matrix.py` freezes the one-candidate,
  three-split, four-stress, two-capital 24-cell contract and resumes only
  incomplete sweep roots. `run/audit_open_ledger_backend_parity.py` compares
  summary economics and equity, diagnostics, positions, orders, rejections and
  costs by sweep key. A 3 GiB free-memory gate runs before every subprocess.
- ADR 0010 requires one physical market store with separate logical DataViews.
  CSV remains the default parity oracle until Provider, monthly-cache and full
  24-cell ledger equivalence gates pass.
- v14 memmap caches: rebuild only for data/feature/label semantic changes.
- OHLC matrix cache: immutable daily execution inputs.
- Historical ST contract: `data/raw/st_status_events.csv` plus its manifest;
  transitions are effective on `imp_date` and must cover the requested period.
  `data/stock_industry.csv` is a compatibility snapshot only and cannot close
  the historical-ST audit gate.
- Alpha JSONL: dated ranking signal; execution timing is resolved by the ledger.
- Strategy proposal: ranked alpha/current holdings -> desired names or weights;
  it contains no assumed fill, price, cash, lot, ADV, or limit behavior.
- Factor baseline specification: a versioned subset of PIT technical columns;
  it is research-only until trained through the same experiment and ledger path.
- Experiment manifest: schema v2 immutable task/config/cache contract, Git
  revision, working-tree state fingerprint, and
  mandatory formal scope covering physical source fingerprints, logical date
  ranges, transform fit state, split roles, and forward-parent lineage.
- Experiment events are append-only. `artifact_index.json` is an atomic,
  rebuildable projection of artifact events; formal consumption also requires
  a terminal `completed` event and verifies every indexed SHA-256.
- Registry evidence is classified as `formal_experiment` or
  `legacy_registered`. Only the former passes the formal manifest, artifact,
  and actual-result-date gates; legacy evidence remains available for audit.
  `canonical_evidence=false` explicitly excludes a superseded row from
  scorecards without deleting its audit history. The formal baseline identity,
  replay contract, artifact hashes, and experiment lineage are frozen in
  `registry/baseline_contract.json` and `registry/evidence_lineage.json` by
  `run/freeze_formal_baseline.py`.
- A dirty checkout is explicitly recorded and cannot be treated as a release
  snapshot. Experiment provenance does not replace the formal `registry/`.
- Declarative workflows use `experiments/workflow.py`. The compiler freezes a
  versioned config and emits a stage graph that delegates to the existing
  rolling trainer, official realistic ledger, and registry scorecard. The
  executor permits only named adapters and resumes from immutable,
  command-hashed stage receipts; it does not embed a second trainer or ledger.
- Workflow schema v2 is separate from the experiment-manifest schema v2. It
  adds explicit governance, Dataset, Processor, Model, Record, timing,
  resource, and observation-only contracts. Q1 compiles its currently
  implemented model stages to the proven stage graph. Q2/Q3 provide native
  Dataset/Processor/Model runtimes; Q5 binds the strong trainer to that graph.
- Monthly rolling training writes a hash-checked `rolling_progress.json` after
  every completed window. Resume verifies the frozen config, model, and alpha
  hashes before skipping a window. Completed monthly alpha is stitched once
  per canonical split and then passed to one continuous realistic ledger, so
  cash and holdings are not reset at month boundaries.
- Strong-model staged rolling now builds each purged window through the project
  Dataset runtime and delegates each exact e6/e15/e19 trainer stage through
  `TorchStrongAlphaAdapter`. The established `run/train.py` remains the parity
  oracle inside the delegate; the rolling CLI owns only orchestration, immutable
  progress and a controlled stage-boundary pause/resume operation.
- Workflow stage failures retain timestamped immutable receipts. A resumed
  rolling stage forwards `--resume`; successful stage receipts keep their
  canonical names. Every workflow builds an experiment-local reports registry
  for its scorecard and never depends on writing the global registry.

Canonical protocol values are in `registry/baselines.yaml` and `registry/decision_rules.json`.
