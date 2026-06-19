# Open-Ledger Candidate Summary 2026-06-17

## Normal Scenario

| candidate | role | capital | val ann | val sharpe | val mdd | test ann | test sharpe | test mdd |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| edge_r030_100 | first_stability_candidate | 50w | 80.89% | 1.789 | 20.67% | 84.30% | 2.953 | 14.28% |
| edge_r030_100 | first_stability_candidate | 100w | 84.78% | 1.818 | 20.36% | 88.21% | 2.887 | 15.24% |
| edge_r030_120 | stability_watch_candidate | 50w | 81.91% | 1.805 | 20.25% | 84.27% | 2.953 | 14.28% |
| edge_r030_120 | stability_watch_candidate | 100w | 84.95% | 1.821 | 20.25% | 87.56% | 2.873 | 15.24% |
| main_candidate | official_baseline | 50w | 79.79% | 1.771 | 21.07% | 84.33% | 2.954 | 14.28% |
| main_candidate | official_baseline | 100w | 84.85% | 1.817 | 20.79% | 88.23% | 2.888 | 15.24% |
| market_switch | conservative_watch_candidate | 50w | 80.48% | 1.781 | 20.92% | 84.33% | 2.954 | 14.28% |
| market_switch | conservative_watch_candidate | 100w | 85.22% | 1.822 | 20.79% | 88.23% | 2.888 | 15.24% |
| negfilter_r030_100_drop3 | first_attack_candidate | 50w | 80.51% | 1.777 | 21.05% | 87.03% | 3.009 | 13.94% |
| negfilter_r030_100_drop3 | first_attack_candidate | 100w | 84.87% | 1.812 | 21.09% | 91.08% | 2.927 | 15.54% |

## Current Decision

- official baseline: `main_candidate`
- first attack candidate: `negfilter_r030_100_drop3`
- first stability candidate: `edge_r030_100`
- use `negfilter_r030_100_drop3` for forward/live observation, not immediate replacement.