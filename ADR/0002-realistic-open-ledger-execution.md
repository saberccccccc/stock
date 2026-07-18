# ADR 0002: Realistic Open-Price Ledger Execution

- Status: accepted
- Date: 2026-07-11

## Context

Proxy and close-based execution can overstate implementable portfolio returns.

## Decision

Official evidence uses realistic open-price share-ledger execution with cash, whole shares, costs, minimum commission, ADV, listing, suspension, and price-limit constraints. Required stress coverage is normal, lag1, cost2x, and capacity_3pct for CNY 500k and CNY 1m. Proxy and close-based runs are diagnostic only.

## Consequences

Official scorecards reject mixed execution modes. Any exception requires a new
ADR and separate, clearly labelled evidence.
