# Checkpoint Reference Audit 2026-06-19

This audit searches lightweight text sources for references to checkpoint/model
archive candidates. It is a cleanup guide, not permission for broad checkpoint moves.

## Summary

| Decision | Count |
|---|---:|
| `hold` | 30 |
| `keep` | 3 |

## Rows

| Name | Action | Group | References | Decision | Note |
|---|---|---|---:|---|---|
| `checkpoints` | `archive_candidate` | `alpha_checkpoint` | 232 | `hold` | high-risk model/checkpoint family; audit manually before moving |
| `checkpoints_exp` | `archive_candidate` | `alpha_checkpoint` | 78 | `hold` | high-risk model/checkpoint family; audit manually before moving |
| `checkpoints_exp_pairwise_w003_20260530_011759` | `archive_candidate` | `alpha_checkpoint` | 0 | `hold` | high-risk model/checkpoint family; audit manually before moving |
| `checkpoints_exp_purged_rawmetric_A_20260613` | `archive_candidate` | `alpha_checkpoint` | 2 | `hold` | high-risk model/checkpoint family; audit manually before moving |
| `checkpoints_exp_top06_stable_20260612` | `archive_candidate` | `alpha_checkpoint` | 1 | `hold` | high-risk model/checkpoint family; audit manually before moving |
| `checkpoints_exp_topfocus` | `archive_candidate` | `legacy_topfocus` | 45 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_exp_topfocus_w005_pairwise_w001_20260530_053300` | `archive_candidate` | `legacy_topfocus` | 2 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_exp_topfocus_w005_topic` | `protect` | `legacy_topfocus` | 40 | `keep` | protected by archive plan |
| `checkpoints_exp_topfocus_w005_topret` | `archive_candidate` | `legacy_topfocus` | 2 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_A0` | `archive_candidate` | `loss_ablation_a_series` | 17 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_A0_low_lr_e10` | `archive_candidate` | `loss_ablation_a_series` | 3 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_A1` | `archive_candidate` | `loss_ablation_a_series` | 3 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_A2` | `archive_candidate` | `loss_ablation_a_series` | 3 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_A3` | `archive_candidate` | `loss_ablation_a_series` | 3 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_A4` | `archive_candidate` | `loss_ablation_a_series` | 7 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_A4_low_lr_e10` | `archive_candidate` | `loss_ablation_a_series` | 1 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_A5` | `archive_candidate` | `loss_ablation_a_series` | 3 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_D001` | `archive_candidate` | `downside_lag_topfocus_series` | 3 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_D003` | `archive_candidate` | `downside_lag_topfocus_series` | 6 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_D003_T001` | `archive_candidate` | `downside_lag_topfocus_series` | 4 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_D005` | `archive_candidate` | `downside_lag_topfocus_series` | 2 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_LAG005` | `archive_candidate` | `downside_lag_topfocus_series` | 1 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_LAG010` | `archive_candidate` | `downside_lag_topfocus_series` | 1 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_loss_ablation_M0_nomulti` | `protect` | `protected_m_baseline` | 8 | `keep` | protected by archive plan |
| `checkpoints_loss_ablation_M1_nomulti_topfocus_w005` | `protect` | `protected_m_baseline` | 7 | `keep` | protected by archive plan |
| `checkpoints_loss_ablation_T001` | `archive_candidate` | `downside_lag_topfocus_series` | 3 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_reranker_oof_F1_train2017_val2018` | `archive_candidate` | `reranker_oof` | 1 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_reranker_oof_F2_train2018_val2019` | `archive_candidate` | `reranker_oof` | 1 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_reranker_oof_F3_train2019_val2020` | `archive_candidate` | `reranker_oof` | 1 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_reranker_oof_F4_train2020_val2021` | `archive_candidate` | `reranker_oof` | 1 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_reranker_oof_F5_train2021_val2022` | `archive_candidate` | `reranker_oof` | 1 | `hold` | referenced by text sources; inspect references before moving |
| `checkpoints_reranker_oof_F6_train2022_val2023` | `archive_candidate` | `reranker_oof` | 1 | `hold` | referenced by text sources; inspect references before moving |
| `models_multi_v9_tech_macro` | `archive_candidate` | `legacy_v9_model` | 1 | `hold` | high-risk model/checkpoint family; audit manually before moving |
