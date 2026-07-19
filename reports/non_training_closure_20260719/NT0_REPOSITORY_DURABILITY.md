# NT0 Repository Durability Acceptance

Date: 2026-07-19  
Result: PASS

## Accepted Refs

| Ref | Object |
|---|---|
| `origin/master` | `83bba6001aa4041df997c74e3fcf8241d0a4211b` |
| `origin/model-experiments` | `83bba6001aa4041df997c74e3fcf8241d0a4211b` |
| `origin/archive/optimized-pre-model-exp-20260719` | `c91eb2a3c249113b90142129fffe7529d7d12dba` |
| `accepted-research-20260719^{}` | `83bba6001aa4041df997c74e3fcf8241d0a4211b` |
| untouched `origin/main` | `84b795a2c7d57078f1acdde8bf41e5d9c79639d2` |

No force push was used. `origin/main` has an independent commit and was
intentionally left unchanged.

## Safety Audit

- Tracked maximum object size: 1,936,931 bytes; no oversized tracked artifact.
- Private-key headers: 0 matched files.
- GitHub PAT patterns: 0 matched files.
- AWS access-key patterns: 0 matched files.
- Generic plaintext-password assignments: 0 matched files.
- Tushare token references: 4 matched files, retained under the user's explicit
  project exception; values were not printed into logs or this report.

## Recovery Acceptance

The following recovery shape was tested from a new directory:

```powershell
git clone --depth 1 --branch model-experiments `
  https://github.com/saberccccccc/stock.git <new-directory>
```

The recovered checkout resolved to
`83bba6001aa4041df997c74e3fcf8241d0a4211b`, was clean, passed `git fsck`,
and contained the project rules, master plan, active technical plan, Registry
baseline and official backtest entrypoint. The temporary checkout was removed
after verification.

## Gate Decision

NT-G0 passes. The accepted repository state is recoverable from the remote,
the historical legacy snapshot is preserved without merging, and no remote
history was overwritten.
