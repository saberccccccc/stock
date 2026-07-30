# MD6 Ledger Backend Parity Status

Status: `passed`

## Scope

- Candidate: `ledger_path_v3_t0001_nolookahead`
- Selection: Val 2024 and Test 2025
- Observation only: Forward 2026-01-01 through 2026-06-30
- Stresses: normal, lag1, cost2x, capacity_3pct
- Capital: CNY 500,000 and CNY 1,000,000
- Cells per backend: 24
- Oracle backend: direct legacy CSV
- Candidate backend: month-sharded OHLCV cache
- Python: `C:\Users\x\miniconda3\envs\torch\python.exe`

## Completed

1. Added explicit backend selection without changing the formal `legacy`
   default.
2. Reused monthly high, low, volume and money for realistic execution masks.
3. Bound mask cache identity to immutable monthly generations.
4. Added exact summary and six-artifact parity auditing.
5. Added a resumable two-backend, three-split matrix runner.
6. Compiled all commands through a successful dry-run.
7. Passed 79 focused tests under the project Torch environment.
8. Passed the full 609-test suite with one pre-existing Pandas FutureWarning.

## Result

- CSV cells completed: 24/24
- Monthly cells completed: 24/24
- Split parity reports passed: 3/3
- Summary rows compared: 24
- Detailed artifact files compared: 144
- Exact DataFrame matches: 144/144
- Exact file-byte hash matches: 144/144

The comparison covers equity curves, diagnostics, positions, orders,
rejections and costs for every stress and capital cell. Monthly execution is
therefore behaviorally identical to the direct CSV oracle for this fixed
baseline contract.

## Runtime

| Split | CSV total | Monthly total | Total speedup | CSV OHLC load | Monthly OHLC load |
|---|---:|---:|---:|---:|---:|
| Test 2025 | 61.053s | 34.629s | 1.76x | 28.988s | 4.502s |
| Forward 2026 | 108.114s | 22.937s | 4.71x | 88.298s | 2.760s |

Val timing is excluded from the speed comparison because an initial stdout
pipe interruption left resumable artifacts; the successful run reused part of
that work. Its parity result is valid, but its timing is not a clean cold-run
benchmark.

Peak process RSS stayed below 527 MiB in the recorded subprocess reports.

## Recovery Evidence

The first attempt stopped at the 3 GiB memory gate. A later attempt was
interrupted when the outer command closed its stdout pipe, returning code 120.
The matrix runner did not treat partial files as complete; `--resume` completed
the missing cells, and exact parity passed afterward.

## Promotion State

- CSV remains the rollback path.
- Monthly Parquet-backed execution passed MD6 but remains non-authoritative
  until MD7-MD9 call-site, performance and rollback gates complete.
- Registry and lifecycle remain unchanged.
