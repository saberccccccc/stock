# Qlib Adoption Plan - Task Tracking

## Goal

Complete an auditable, reproducible research framework adapted to this
repository's actual A-share PIT data, labels, alpha artifacts, realistic
open-price share-ledger, registry, and execution constraints. Use 2024 Val and
2025 Test for selection. Treat 2026-01-01 through the latest complete available
2026 date as full-year Forward observation only. Then run a separately
identified Qlib-style monthly rolling experiment with label-tail purge and
unique OOS ownership. Never promote a candidate or claim improvement from IC
or forward performance alone.

## Phases

| Phase | Status | Deliverable |
|---|---|---|
| Read governing protocol and current architecture | completed | Constraints captured in findings |
| Audit existing rolling prototype status | completed | Actual versus planned status recorded |
| Write formal Qlib adoption plan | completed | `QLIB_ADOPTION_PLAN_20260712.md` |
| Reorder plan: framework first, rolling experiment after | completed | Stage A and Section 5A added |
| Phase 0 governance and ADR | completed | ADR 0004, current index, development log |
| Phase 1a experiment provenance | completed | Immutable manifest and append-only events, 21 focused tests passed |
| Phase 1b rolling task controller | completed | Trading-calendar monthly windows and unique-OOS-owner checks, 23 focused tests passed |
| Phase 2 feature/PIT audit | completed | Read-only v14 transform contract; declared source/availability coverage gaps, 24 focused tests passed |
| Phase 3 alpha-to-ledger experiment adapter | completed | `run/evaluate_experiment_alpha.py` records alpha/command/events and delegates to the realistic sweep; 24 focused tests passed |
| A-share execution/universe coverage audit | completed with limitation | 2024/2025 OHLC and listing coverage verified; historical ST remains explicitly missing rather than silently approximated |
| Historical ST event contract and downloader | completed with external-data gate | `data/st_status.py`, cutoff-filtered downloader, page checkpoints, explicit `st`/`namechange` sources, ledger preference, cache-key and coverage-audit integration; actual historical source coverage is still missing |
| Project-native strategy contract | completed | Retention and TopK/dropout use the same realistic ledger path and isolated sweep identity |
| Alpha158-inspired independent factor baseline | completed with comparison gate | Compact/broad trained and realistic val/test evidence generated; registry comparison remains |
| Full-feature rolling baseline comparison | completed with comparison gate | All three arms completed common low-memory rolling and realistic Val/Test four-stress evidence; Compact leads research comparison, no automatic registry promotion |
| Bounded single-component tuning | completed with rejection gate | Five predeclared trials completed under the common low-memory rolling/ledger protocol; `t03_minleaf160` was rejected after independent-seed confirmation |
| Independent-seed confirmation | completed with rejection gate | Val uplift did not reproduce on 2025 Test; compact baseline remains the fixed research arm |
| Historical OOF lineage and simple ensemble | completed with evidence gate | 2024/2025 blend matrix and 2018-2023 historical diversity audit completed; `compact_v14_eq_rank` remains conditional only |
| State-aware portfolio-construction pilot | completed with gate | `risk_rank` and `risk_suppress` families completed on fixed `compact_v14_eq_rank` with full Val/Test stress evidence and normal-path attribution; `risk_suppress_d015` is exploratory because its numeric threshold was fixed after the first-family attribution; neither is promoted |
| Current-state split audit | completed | Verified official wrapper/registry use 2024 Val, 2025 Test, 2026-01-01..06-30 Forward; found stale May boundary in protocol/Phase 6/tests |
| Phase A protocol unification | completed | One canonical SplitSpec; full-year 2026 Forward; role-derived registry flags; parent-freeze validation; 67 focused tests passed |
| Phase B mandatory experiment schema | completed | Schema v2 formal scope, terminal/artifact/hash/date gates, formal-versus-legacy registry evidence; 86 focused tests pass |
| Phase C declarative workflow controller | completed | Frozen-alpha workflow ran 16/16 Val/Test cells with zero gaps; rolling manifest resolves to local candidate; 97 focused tests pass |
| Phase D provider and processor contracts | completed with external limitation | Five real providers audited; physical v14 superset safely serves 2025 logical view; historical ST remains missing |
| Phase E formal monthly walk-forward | completed with rejection gate | 24 monthly 4y Train/6m Valid/1m OOS windows, 485 unique OOS days, continuous Val/Test ledger and 32-cell scorecard completed; Compact LightGBM was rejected and no schedule sweep or promotion followed |
| Phase E1 two-layer fairness audit | completed with rejection gate | Raw-to-raw and frozen sa_p05+V3 same-stack comparisons completed across 32 cells each; Compact improved under the stack but remained far below e19 |
| Phase Q0 Qlib alignment baseline | completed | Frozen source revision, 13-component matrix, vocabulary, non-adopted scope, workflow v2 schema/golden config; 41 compatibility tests pass |
| Phase Q1 declarative Task/Workflow | completed | V1 replay retained; v2 schema/semantic validation, explicit normalization, CLI freeze and safe stage compilation; 45 focused tests pass |
| Phase Q2 Dataset/DataHandler/Processor runtime | completed | Streaming shared/infer/learn chains, Train-only fit, frozen state hash, named prepare facade and real v14 adapter; 54 focused tests pass |
| Phase Q3 unified Model Adapter | completed | One factory/lifecycle for LightGBM, PyTorch strong alpha, frozen artifact, and read-only legacy signals; Q5 owns concrete e19 trainer binding |
| Phase Q4 standard Record templates | completed | Immutable six-record dependency chain, schema/spec materializer, official-ledger and Forward-selection gates |
| Phase Q4B Record runtime integration | completed with first formal-run gate | Formal Workflow now persists genuine ledger evidence and compiles an automatic six-Record stage; synthetic full-bundle and real v14 signal/label smoke pass, while the first new full formal Workflow will provide production acceptance evidence |
| Phase Q5A strong-model framework binding | completed | `torch_strong_alpha` is bound to Workflow v2, the resumable executor, and the learner-neutral rolling-manifest/split-alpha contract; no performance search was launched |
| Phase Q5B strong-model Rolling performance | deferred | Both staged pilot variants were rejected; redesign and 24-window performance work wait until Q5A framework acceptance is complete |
| Phase F risk-aware portfolio construction | pending | Hypothesis-driven proposal layer and realized-ledger attribution; no blind threshold sweep |
| Phase G / Q7A manual Shadow lifecycle framework | completed with first live-bundle gate | Complete Record bundle freeze, manual prepared/shadow/paused/retired state machine, hash-chained events and dated nonselecting observations; no automatic promotion or trading |
| Phase G / Q7B daily Shadow runner and replay | in progress | A real prepared lifecycle is bound to the first complete formal bundle; daily data-ready/inference/proposal/ledger/Record/replay remains |
| Frozen/legacy dated-prediction adapter | completed | Hash-checked streaming adapter accepts registered split alpha, list/code-mapped scores and automatic six-Record input without copying or reclassifying old evidence |
| First full formal Workflow acceptance | completed | Frozen baseline completed prediction, 16 realistic ledger cells, scorecard and six hash-valid Records without Forward or global Registry mutation |
| Long-term platform roadmap | documented | `LONG_TERM_QUANT_PLATFORM_ROADMAP_20260717.md` defines Q0-Q8 and industrial acceptance gates |
| Historical ST source acquisition | deferred external | Adapters exist, source data absent; limitation must be declared but does not block Phases A-E |

Phase F must not optimize the rejected monthly Compact alpha. Portfolio work
waits until Q0-Q5 establish common framework interfaces and strong-model OOS
evidence. Window-length searches on the rejected Compact arm are prohibited.

Q0-Q4 contract alignment, Q5A strong-model Workflow binding, Q4B automatic
Records, and Q7A lifecycle governance are complete. Profitability and
checkpoint redesign remain deferred in Q5B. The immediate framework sequence
is Q7B daily Shadow/replay acceptance on the real prepared lifecycle. Q6 portfolio
optimization waits until framework closure and profitability work resume.

## Guardrails

- No formal candidate decision from proxy or forward evidence.
- No Qlib default execution in official reports.
- Preserve formal baseline and registry as source of truth.
- `2026-05-18` is not the forward start; it is only legacy artifact metadata
  unless an individual experiment explicitly used it as an as-of date.
- A full-year 2026 forward parent must freeze training, transform fitting,
  checkpoint selection, and policy selection by 2025-12-31.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| No project graph found at repo root | 1 | Used governing documents and source inspection instead. |
| Full v14 rolling process terminated before producing model artifacts | 1 | Added expanded feature-layout provenance and Dataset raw-array release; rerun all three arms with a common 100k/30k low-memory cap |
| Formal run collided with a dry-run `rolling_manifest.json` | 1 | Use clean experiment directories and add a runner guard/cleanup rule before formalizing the adapter |
| Full pytest collection blocked on missing `torch` | 1 | Relevant rolling/recording tests pass; install or select the project's PyTorch environment before claiming the full suite green |
| PowerShell regex ended with an unescaped backslash while locating raw ledgers | 1 | Switched to exact literal artifact paths; no files or experiment outputs were changed |
| PowerShell parsed `$p:$start` as an invalid drive-qualified variable while reading Qlib source snippets | 1 | Use `${p}:$start` in the output label; no project or Qlib source files were changed |
| `jsonschema` was absent from both system and Torch Python environments | 1 | Install the lightweight validator into the documented Torch environment and record it as a Q0 contract-test dependency |
| Three new Workflow v2 tests failed before validation because `Path` was not imported | 1 | Added the missing standard-library import; production code was not implicated |
| V2 test fixture lowered `max_data_date` to 2025 but left `feature_warmup.end` in 2026 | 1 | The semantic validator correctly rejected it; synchronized the test fixture's logical range |
| First patch for the synchronized v2 fixture contained a malformed multi-file hunk | 1 | Reissued a valid scoped patch; no file content was partially applied |
| Processor-kind patch placed the rank processor declaration after a return statement | 1 | Moved it into `CrossSectionRankProcessor` and added direct golden-kind factory coverage |
| First Q3 factory patch expected a nonexistent `import json` line in the test file | 1 | Re-read the file header and reapplied one correctly anchored atomic patch; no partial changes were left |
| Q4 bundle test found JSON key sorting had alphabetized the canonical record order | 1 | Preserve insertion order when writing the bundle; canonical hashing remains independently sorted |
| First Q5 smoke missed the existing v14 cache and began a full raw rebuild | 1 | Memory guard stopped the run at 1.04 GiB free physical memory; preserved the failed experiment and added an explicit validated cache contract for both training and label-free inference |
| PowerShell path search used a regex ending in an unescaped backslash | 1 | Replaced the regex with a literal wildcard match; no files were changed by the failed search |
| Broad recursive search for the execution function timed out | 1 | Restricted the search to `backtest`, `run`, and `tests`, then inspected `backtest/execution.py` directly |
| First evidence-summary script assumed `window_id` and top-level `formal_eligibility` fields | 1 | Inspected the actual manifest schema; windows use `name` and exploratory eligibility is frozen in the source reference |
| A PowerShell here-string used for manifest inspection missed its closing terminator | 1 | Replaced it with native `ConvertFrom-Json`; no artifacts were modified |
