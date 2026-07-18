# Findings

- The long-term roadmap is technically broad enough, but accumulated status
  appendices and an older Qlib plan created multiple apparent next-step lists.
- The live project is at P2 training-mainline convergence: the 2024 LightGBM
  Dataset path is byte-identical, but 2025 parity, Model Adapter substitution,
  and strong-model mainline acceptance remain open.
- Framework completion, return research, and Shadow operation must be separate
  gated phases. Mixing them caused prior local optimization drift.
- Historical ST data remains an explicit data-quality gap, not a reason to
  block the core framework migration.
- P4 contract review found semantically misleading strong-stage names:
  `base_e6` already contains OO-lag1 supervision and `lag1_low_lr_e15` only
  lowers the learning rate. The current monthly run remains paused until P4-R
  resolves provenance and checkpoint-selection interpretation.
