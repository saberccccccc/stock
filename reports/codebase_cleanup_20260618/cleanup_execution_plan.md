# 项目整理执行计划 2026-06-18

## 目标

这次整理不是为了把目录变漂亮，而是为了让后续训练、回测、候选比较和实盘观察不再互相污染。

核心目标：

- 固定当前可复现的正式回测口径；
- 整理目前结果和策略逻辑，区分正式基线、候选、废弃分支；
- 先抽公共模块，再冻结入口，最后清理结果目录；
- 不改变现有回测参数和结果含义；
- 保留旧脚本的可运行性，避免一次性大迁移导致结果失真。

第一阶段只做低风险整理：文档、清单、公共函数、薄入口。暂时不移动大结果目录，不删除旧结果，不重训模型。

## 当前目录诊断

已生成附件：

```text
reports/codebase_cleanup_20260618/source_inventory.csv
reports/codebase_cleanup_20260618/source_inventory.md
```

顶层目录当前大致情况：

| 类型 | 数量 | 处理原则 |
|---|---:|---|
| 源码/文档目录 | 8 | 保留，逐步模块化 |
| 实验输出目录 | 146 | 暂不移动，先登记和分层 |
| 实验输出文件 | 29 | 暂不删除，后续归档 |
| checkpoint/model 目录 | 13 | 暂不移动，先标记正式/候选/历史 |
| log/pid 文件 | 34 | 最后清理，先确认无进程依赖 |
| archive/cache 目录 | 3 | 保留 |

主要问题：

1. `run/` 同时包含正式入口、实验入口、一次性诊断脚本。
2. `backtest_results_*`、`checkpoints_*`、`reranker_*`、`open_ledger_*` 等实验输出混在项目根目录。
3. Alpha JSONL 读取、写入、rank、过滤、合并逻辑在多个脚本中重复。
4. open-price share-ledger 的参数和压力测试命令分散，容易造成“看起来一样，实际口径不同”。
5. 候选排行榜还依赖手工拼 CSV/报告，不利于复盘。
6. 一些历史高收益结果不是同一回测引擎，不能直接和当前 share-ledger 结果混排。

## 当前结果与逻辑分层

### A. 正式基线，必须冻结

当前正式基线：

```text
V9 avgw3 + maxret095 + open-price share-ledger
```

当前正式参数：

```text
target_frac=0.006
hold_frac=0.10
max_new_names=5
rebalance_band=0.20
market_timing_mode=legacy
min_adv_cny=3_000_000
adv_participation_cap=0.05
limit_threshold=0.095
portfolio_values=500000,1000000
```

执行含义：

- T 日收盘后生成信号；
- 对 T 日 close-to-close 涨幅 >= 9.5% 的股票降权或降到末尾，避免追入接近涨停；
- T+1 日用开盘价成交；
- 使用现金、股数、整手、最低佣金、印花税、滑点、ADV、涨跌停约束；
- 使用 `--max-data-date 2026-05-18` 时，不能访问 2026-05-19 之后数据。

正式基线的用途：

- 作为之后所有模型、loss、reranker、overlay 的主比较对象；
- 作为 2026-05-19 后前向观察的参照；
- 作为代码整理时的回归测试对象。

禁止事项：

- 不在整理过程中改默认参数；
- 不把 close-based 结果和 open-price share-ledger 结果直接混排；
- 不用 Alpha IC 单独决定模型优劣；
- 不用 2026-05-19 后真实前向数据回头调参。

### B. 当前候选，保留但不替换正式基线

候选分支：

```text
breadth_m085
negfilter_drop3
risk_target_r004
edge_r030_100
conditional_breadth_negfilter
V3/V4 conservative reranker
```

当前判断：

| 候选 | 当前角色 | 判断 |
|---|---|---|
| `breadth_m085` | 风险控制候选 | 压力和 forward 损失控制较好，可继续观察 |
| `negfilter_drop3` | 历史 test 增强候选 | 历史 test 强，但 forward 不稳，不能直接升正式 |
| `risk_target_r004` | 市场弱势缩仓候选 | validation/forward 减亏明显，但 test 损伤，疑似状态过拟合 |
| `edge_r030_100` | 精排边缘候选 | 大多接近 official，暂时价值不高 |
| `conditional_breadth_negfilter` | 组合 overlay 候选 | 只作为后续观察，不作为当前正式替换 |
| `V3/V4 reranker` | shadow 候选 | 思路可保留，但不能用短期 forward 反向调参 |

这些分支后续要通过统一入口跑同一套口径，不能各自用不同脚本手工比较。

### C. 历史有用逻辑，后续要保留

从旧报告中确认过、应该保留的逻辑：

1. `V9 average_w3` 仍是当前最可靠主信号。
2. `maxret095` 对当前 open-price share-ledger 口径有用。
3. 20% rebalance band 是有效的换手控制。
4. 点时财报因子不要随意去掉。
5. market timing/dynamic overlay 是风险控制方向，不是简单收益增强。
6. open-to-open 宽持仓研究框架仍有参考价值，但不能和 Top30 share-ledger 混成一个排行榜。
7. V10 是研究分支，不是当前主线。
8. trade_policy_v1、部分早期 fixed-window 结果不可作为正式决策依据。

### D. 已知不能混用的结果类别

后续排行榜必须分三类：

1. `official_open_price_share_ledger`
   当前正式小资金实盘近似口径。
2. `legacy_close_based_or_constrained`
   用于历史连续性和 loss 诊断。
3. `research_open_to_open_wide_book`
   旧 V9 宽持仓 open-to-open 研究框架。

同一张榜单里只能在同一类别内部排序。跨类别只能写解释，不能直接排名。

## 整理总原则

1. 先文档化，再抽模块，再冻结入口，最后移动结果。
2. 每一步都保留旧脚本入口，旧命令尽量还能跑。
3. 公共模块只能先抽重复逻辑，不能顺手改策略逻辑。
4. 每次抽模块都要有小测试或小 fixture 对齐。
5. 结果目录清理只移动已经登记过的历史输出。
6. 正式基线相关路径先不移动，避免报告和命令断链。

## 目标结构

建议最终形成：

```text
alpha/
  io.py
  transforms.py
  market_overlays.py
  diagnostics.py

backtest/
  runtime.py
  engine.py
  reports.py
  execution.py
  open_ledger.py
  presets.py
  stress.py

experiments/
  registry.py
  leaderboard.py
  checkpoint_selection.py

run/
  backtest_open_ledger.py
  apply_alpha_transform.py
  generate_v9_alpha.py
  train.py
  train_temporal.py
  legacy/

reports/
  codebase_cleanup_20260618/
  candidate_leaderboard_*/
  ...

archive/
  experiments_202606/
  logs_202606/
  pids_202606/
  old_backtest_results/
```

注意：这是目标形态，不是一次性迁移清单。

## 阶段计划

### Phase 0：冻结清单和结果逻辑

状态：已开始。

已完成：

- 扫描顶层目录；
- 生成 `source_inventory.csv`；
- 生成 `source_inventory.md`；
- 确认当前没有活跃 Python 训练/回测进程。

继续要做：

- 建一个 `official_baselines.md`，记录正式基线、候选、废弃分支；
- 建一个 `entrypoint_freeze.md`，记录哪些脚本当前不能改行为；
- 给所有主要结果目录打标签：正式、候选、历史、废弃、仅诊断。

验收：

- 后续任何人能看懂当前项目的“主线是什么”；
- 不需要翻几十个目录猜哪个结果有效。

### Phase 1：抽 Alpha JSONL 公共模块

风险：低。

新建：

```text
alpha/io.py
alpha/transforms.py
```

抽取内容：

- JSONL 读取/写入；
- date/code/alpha 字段校验；
- 按日期分组；
- rank 重排；
- maxret095 过滤；
- edge rerank；
- negative filter；
- 条件触发 filter。

先改造成薄入口的脚本：

```text
run/make_edge_rerank_from_full.py
run/make_negative_filter_from_full.py
run/make_conditional_negfilter_alpha.py
run/transform_alpha_for_execution.py
```

验收：

- 原脚本命令仍可运行；
- 小样本输出 row-equivalent；
- `py_compile` 通过；
- 新增 `tests/test_alpha_io.py` 和 `tests/test_alpha_transforms.py`。

### Phase 2：冻结正式回测入口

风险：低到中。

新建：

```text
backtest/presets.py
backtest/stress.py
```

写入命名 preset：

```text
official_open_price_share_ledger
official_lag1
official_cost2x
official_capacity_3pct
research_open_to_open_wide_book
legacy_close_based_top30
```

这一步的重点不是重写引擎，而是让所有命令统一从 preset 取参数。

验收：

- 正式参数只在一个地方定义；
- `run/backtest_retention_open_ledger.py` 仍能接受旧参数；
- 新入口和旧入口在小样本上结果一致；
- 不改变现有 summary CSV 字段。

### Phase 3：抽 open-price share-ledger 引擎

风险：中。

从：

```text
run/backtest_retention_open_ledger.py
```

抽到：

```text
backtest/open_ledger.py
backtest/execution.py
```

抽取内容：

- OHLC/ADV 加载；
- T 信号到 T+1 open 执行；
- 涨跌停阻塞；
- 整手股数；
- 现金账本；
- 最低佣金；
- 印花税和滑点；
- 持仓市值 mark-to-market；
- diagnostics 和 summary 输出。

保留：

```text
run/backtest_retention_open_ledger.py
```

作为兼容 wrapper。

验收：

- 对同一小样本，旧脚本和新模块 summary 一致；
- 正式基线不出现指标漂移；
- 旧报告中的命令不失效。

### Phase 4：候选注册表和统一排行榜

风险：低。

新建：

```text
experiments/registry.py
experiments/leaderboard.py
```

registry 记录：

```text
candidate_name
candidate_class
alpha_path_val
alpha_path_test
alpha_path_forward
result_dirs
execution_family
uses_maxret095
uses_market_overlay
notes
status
```

候选状态：

```text
official
shadow
research
rejected
diagnostic_only
```

验收：

- 可以重新生成 `reports/candidate_leaderboard_20260617/candidate_leaderboard.csv`；
- 榜单按执行口径分组，不混排；
- 每个候选能追溯到 alpha 文件和结果目录。

### Phase 5：训练和 checkpoint 选择逻辑整理

风险：中。

这一步不急着做，因为训练部分一动就容易影响可复现性。

后续新建：

```text
core/training_presets.py
experiments/checkpoint_selection.py
```

目标：

- 明确 V9/M0/V10 的网络、数据、loss、验证方式；
- checkpoint 选择不只看 Alpha IC；
- 选择标准包括正式 share-ledger、lag1、cost2x、drawdown、turnover；
- Alpha IC 只作为最低门槛。

验收：

- 能解释每个 checkpoint 为什么被选中；
- 训练脚本参数不再散落在日志和临时命令里。

### Phase 6：清理结果目录

风险：中。

只有 Phase 1-5 完成后再做。

先移动：

```text
*.pid
*.log
*_stderr.log
*_stdout.log
```

到：

```text
archive/logs_202606/
archive/pids_202606/
```

再移动历史实验输出：

```text
backtest_results_exp_*
checkpoints_loss_ablation_*
reranker_*_202606*
```

到：

```text
archive/experiments_202606/
archive/old_backtest_results/
```

暂不移动：

```text
data/
cache/
reports/
forward_results/
v9_avgw3_open_ledger_20260617/
v9_avgw3_extend_to_20260518_20260616/
checkpoints_exp_topfocus_w005_topic/
```

验收：

- 正式回测命令仍能跑；
- 关键报告链接不失效；
- 旧结果仍可恢复；
- 根目录显著变清楚。

## 入口冻结清单

以下入口先冻结行为，只允许内部调用公共模块，不允许改默认含义：

```text
run/backtest_retention_open_ledger.py
run/transform_alpha_for_execution.py
run/generate_v9_inference_alpha.py
run/train.py
run/train_temporal.py
run/validate_candidate_models.py
```

冻结含义：

- CLI 参数名尽量不变；
- 输出 CSV 字段名尽量不变；
- 默认参数不变；
- 如果必须新增参数，只能新增显式 flag，不能悄悄改默认。

## 第一轮具体执行清单

建议现在只做这些：

1. 完成 `official_baselines.md`。
2. 完成 `entrypoint_freeze.md`。
3. 新建 `alpha/io.py`。
4. 新建 `alpha/transforms.py`。
5. 先改 `run/transform_alpha_for_execution.py` 为薄入口。
6. 再改 `run/make_negative_filter_from_full.py` 和 `run/make_edge_rerank_from_full.py`。
7. 加小测试。
8. 跑 `py_compile` 和相关单测。

不做这些：

- 不移动大结果目录；
- 不删旧脚本；
- 不改正式基线参数；
- 不跑大型回测；
- 不重训模型；
- 不用 forward 结果反向调参。

## 成功标准

第一轮整理成功的标准：

1. 项目主线一眼能看懂。
2. 正式基线不会因为整理而变。
3. 新实验必须通过统一 preset 和 registry 记录。
4. 重复的 alpha transform 逻辑消失。
5. `run/` 逐步变成 CLI 入口，而不是逻辑堆放区。
6. 根目录清理有明确顺序，不会误删/误移关键结果。

## 当前建议

下一步不要继续训练，也不要继续堆新回测。

先执行 Phase 0 剩余部分和 Phase 1：

```text
冻结正式口径 -> 抽 alpha 公共模块 -> 保留旧入口 -> 小样本验证
```

这一步做完以后，再整理 open-ledger 引擎。这样最稳，也最不容易把已经确认过的回测口径弄坏。
