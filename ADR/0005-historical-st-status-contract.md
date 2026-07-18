# ADR 0005: Historical ST Status Contract

- Status: accepted
- Date: 2026-07-15

## Context

The realistic open-price ledger applies different price-limit and
tradability rules to ordinary shares and ST/*ST shares. The available
`data/stock_industry.csv` is a current snapshot dated 2026-04-27; using its
current names to explain a 2024 or 2025 trade is not point-in-time evidence.
The Stage A acceptance gate therefore needs a dated status source with a
declared coverage range and integrity metadata.

## Decision

1. Store the normalized research input at
   `data/raw/st_status_events.csv`, with its provenance at
   `data/raw/st_status_events_manifest.json`. The event schema preserves
   `pub_date` and `imp_date`, and uses `imp_date`/`event_date` for the
   execution state transition.
2. Derive `is_st` deterministically from the event type/reason/explanation;
   missing or contradictory transitions are invalid and are never guessed.
3. `open_ledger` prefers the dated event file. When it exists, the current
   stock-name snapshot is not used as a historical fallback. If the event
   file is absent, the old snapshot remains available only for compatibility,
   while the coverage audit keeps the historical-ST gate open.
4. A historical-ST coverage claim requires a valid event contract, a manifest
   coverage range spanning the requested interval, and a matching output
   SHA-256. The downloader filters the research copy at the declared
   `as_of_date`; later events remain outside `data/raw`.
5. Tushare `st` event history is the preferred source. The downloader also
   exposes a separately labelled `namechange` reconstruction, whose
    `source_kind` is `tushare_namechange_intervals` and whose human-readable
    `source_label` is `由历史股票名称区间重建`; it cannot silently replace the
    preferred event feed and still needs its own reconstruction and coverage
    audit before formal use.

## Consequences

The execution mask cache now includes the event file and manifest in its
identity, so newly downloaded status data cannot reuse stale masks. The new
downloader writes page-level checkpoints under `data/tracking_raw` and never
prints the API token. Current credentials do not have access to Tushare's
`st` endpoint, so no historical event file was published in this change and
the Stage A gate remains explicitly open.

## Supersedes

None.
