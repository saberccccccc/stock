# Lag1 Training and Selection Plan

Date: 2026-06-16

## Goal

Test whether accounting for one-day execution delay improves model selection or
training, without using any data after the 2024 validation period.

## Stage L0: Lag1 Checkpoint Selection

Do not retrain. Validate every M0 epoch checkpoint under the same executable
2024 protocol:

- 9.5% signal-day return filter
- CNY 500k and CNY 1m
- base, doubled cost, one-day lag and 3% ADV scenarios

Decision: if an earlier epoch materially improves lag1 without hurting base and
cost2x, prefer checkpoint selection over changing loss.

## Stage L1: Add Lag1 Validation Metrics During Training

Record lag1-oriented metrics per epoch so future training does not select by IC
or raw top stability alone.

This stage changes checkpoint selection only, not gradients.

## Stage L2: Lag1 Auxiliary Loss

Train M0 plus a small lag1 IC auxiliary target:

- LAG005: `lag1_loss_weight=0.05`
- LAG010: `lag1_loss_weight=0.10`

The lag1 target uses the same h1/h3/h5/h7 weighted target, but shifted one
trading day forward. The loss starts at epoch 2.

Decision: accept only if executable base, lag1 and cost2x are all competitive
with M0. Do not accept a model that only improves lag1 while damaging base.

## Stage L3: Lag1 Top-book Loss

Only if L2 is inconclusive, test a tiny long-only lag1 top-book loss. This is
higher risk because prior Top-focus variants hurt the executable portfolio.
