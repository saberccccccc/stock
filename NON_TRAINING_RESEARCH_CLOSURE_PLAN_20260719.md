# 非训练研究闭环实施计划

状态：当前活动阶段的详细技术说明，实施中
日期：2026-07-19  
上位计划：`MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md`

## 1. 文档定位

本计划回答“暂停新增训练时，项目接下来完整做什么”。它是总计划的专项实施包，
不能改变 P0-P9 顺序，也不能绕过训练证据、Registry gate 或人工批准直接晋级候选。

本计划不承诺提高收益；目标是把已有模型、信号、组合策略、真实执行和实验治理
整理为可复现、可比较、可审计、可做人工 Shadow 准备的正式闭环，并查明现有收益
究竟来自预测、组合构建还是执行假设。

## 2. 当前事实基线

- 活跃仓库是独立的 `deepseek_model_exp`；`master` 与
  `model-experiments` 已在本地对齐。
- 正式基线是 `ledger_path_v3_t0001_nolookahead`，不是
  `multi_downside_e19`。
- `multi_downside_e19` 是底层强 Alpha/候选体系；只有绑定明确组合策略和正式证据后，
  才能成为可与正式基线比较的候选。
- 选择区间固定为 `val_2024` 和 `test_2025`；`forward_2026` 只观察。
- 正式账户规模固定为 50 万和 100 万元。
- 正式执行固定为 realistic open-price share-ledger。
- 固定压力场景为 normal、lag1、cost2x、capacity_3pct。
- 当前 Registry 同一基线存在历史注册与正式 replay 的重复证据，需要选择唯一正式
  lineage，但不能删除历史记录。
- 历史 ST 事件适配器和审计接口已存在，完整历史 ST 数据仍是显式缺口。
- 当前 lifecycle 只能保持 `prepared`；进入 `shadow` 必须经过 P7 和人工批准。

## 3. 完成目标

完成本计划后，项目必须具备：

1. 一个有远端备份、可恢复的正式代码基线；
2. 一个不可歧义的正式 baseline contract；
3. 一套按数据、信号、策略、执行、记录逐层冻结的 artifact inventory；
4. 一份覆盖 Val/Test/Forward 的统一 24-cell 基线证据；
5. 一份真实交易约束与数据覆盖审计；
6. 一套对现有候选公平复跑和淘汰的流程；
7. 一份按主动管理框架拆解的收益与风险归因；
8. 一个经过等价性验证的高性能回测入口；
9. 一套只做 replay、不自动下单的 Shadow 准备与故障演练；
10. 一个明确回答“下一轮训练是否值得做、应该回答什么假设”的决策包。

## 4. 不可改变的协议

- 2026 数据不得用于模型、规则、参数、组合权重或候选选择。
- 不按 Forward 表现反向选择 `raw`、`maxret095`、防御阈值或 ensemble 权重。
- 不将 IC 作为核心晋级指标；它只用于诊断预测方向和漂移。
- 不改变正式成本、涨跌停、停牌、零成交、ADV、整手和现金约束来美化结果。
- 不用 Qlib Executor 替换 `open_ledger`；只借鉴 Workflow、Record、Rolling 和
  Strategy/Executor 分层。
- 不把旧报告文件名当作证据；正式结论必须来自 Registry、manifest 和 artifact hash。
- 不新建第二套回测器；性能优化必须保持逐日净值、订单、成交和阻塞原因等价。
- 不在本计划内训练、微调、续训、搜索 loss、搜索模型超参数或恢复 24 窗强模型训练。

## 5. 标准数据流

```text
physical market data
-> declared research/forward data view
-> frozen alpha artifact
-> candidate strategy proposal
-> realistic open-price share-ledger
-> standard Records and attribution
-> Registry scorecard and governance decision
-> prepared Shadow replay
```

每一层都必须记录输入范围、输出范围、配置、代码版本、数据版本和内容 hash。下游不得
反向修改上游产物。

## 6. 实施阶段

### NT0：仓库耐久化与远端备份

目标：消除“只有本地提交”的单点故障。

动作：

1. 检查 tracked 文件、提交历史、远端 URL 和大文件；Tushare token 可以按用户授权
   保留，但不得出现在日志、测试输出或提交信息中。
2. `git fetch --prune` 后比较远端 `main/master` 与本地分支拓扑。
3. 禁止 force push；若远端历史不适合直接快进，先推送新的备份分支和带日期 tag，
   再单独决定远端默认分支迁移。
4. 推送 `model-experiments`、接受后的 `master` 以及必要 tag。隔离 legacy 分支只作
   archive，不合并。
5. 从远端新目录做一次浅克隆可恢复性检查。

产物：远端 refs 清单、恢复命令、源码 commit/tag、推送日志摘要。

验收门 NT-G0：无 force push；远端可重新克隆；本地工作区保持干净；敏感信息审计有记录。

### NT1：正式基线与 artifact inventory 冻结

目标：让“baseline”“e19”“raw”等词只有一个可机器读取的含义。

动作：

1. 冻结正式基线 `ledger_path_v3_t0001_nolookahead` 的 candidate、父 Alpha、策略、
   执行参数和证据 lineage。
2. 把 `multi_downside_e19` 标记为底层 Alpha 家族，不冒充正式组合基线。
3. 为 checkpoint、alpha、配置、数据 view、OHLC cache、策略 proposal 和 ledger
   分别记录路径、日期范围、大小、SHA-256 和生成命令。
4. 审核 Registry 重复行，选择正式 experiment manifest 为 canonical evidence；旧行保留
   historical/superseded 状态，不物理删除。
5. 强制记录 `signal_start/end`、`backtest_start/end`、`data_as_of`、
   `fit_end`、`selection_end` 和 `forward_only`。

产物：`artifact_inventory.json`、`baseline_contract.json`、重复证据审计报告。

验收门 NT-G1：任何正式指标都能追溯到唯一 candidate、alpha、manifest 和 ledger 文件。

### NT2：数据边界、PIT 与真实执行覆盖审计

目标：明确回测使用了什么数据，以及哪些 A 股约束仍不完整。

动作：

1. 用 `audit_data_boundary.py` 分别审计 research 与 forward data view。
2. 审计行情、复权、财报 effective date、公告日估计标记、缺失沿用、海外时区和交易日
   对齐。
3. 用 `audit_execution_coverage.py` 检查 2024、2025、2026 的开高低收、成交量、ADV、
   停牌、零成交、真实涨跌停、新股上市天数和板块规则覆盖率。
4. 单独报告历史 ST 覆盖；完整数据缺失时必须输出
   `historical_st_status_not_covered`，不得自动视为非 ST。
5. 抽样人工核对主板、创业板/科创板、北交所、ST、新股和停牌案例。
6. 检查 OHLC matrix cache 的 key 是否包含数据目录、日期区间、字段版本和规则版本。

产物：三段日期审计、执行覆盖矩阵、缺口列表、人工抽样记录、data quality 总报告。

验收门 NT-G2：所有已知缺口显式可见；不存在日期越界和静默 fallback；未覆盖约束不会被
表述为“真实执行已完整覆盖”。

### NT3：正式基线的统一 24-cell replay

目标：用一个入口重建正式基线，不进行参数搜索。

固定矩阵：

- Val 2024、Test 2025、Forward 2026；
- normal、lag1、cost2x、capacity_3pct；
- 50 万、100 万；
- 总计 24 cells，其中前 16 cells 用于选择证据，后 8 cells 只观察。

动作：

1. 先对 `official_backtest_from_registry.py` 做 `--dry-run`，冻结完整命令。
2. 只使用 Registry 中固定参数复跑，不扫描 target/hold/rebalance/max-new-names。
3. 每个 split 复用一次只读 OHLC/cache context，并持续维护同一条 share ledger。
4. 将新 replay 与当前 canonical 历史证据比较净值、成交、成本、阻塞原因和指标。
5. 差异超过预声明容差时停止下游工作，先定位数据版本、执行器或旧证据问题。

产物：24-cell ledger、命令 manifest、parity report、标准 scorecard。

验收门 NT-G3：24 cells 完整、日期正确、执行模式统一、hash 完整；Forward 未进入决策列。

### NT4：现有候选的资格筛选与公平复核

目标：只评估已经存在且证据可恢复的候选，不创造新训练或新调参任务。

候选进入条件：

- 已有完整 Alpha 或可由冻结 checkpoint 纯推理重建；
- 能覆盖完整 Val/Test，Forward 缺失只影响观察，不影响资格；
- 策略参数在看 Forward 前已经冻结；
- 使用与正式基线相同的 realistic ledger contract；
- lineage、日期和 hash 可补齐。

优先审计对象：

1. `cond_pairrisk_volg001_realistic`；
2. `ledger_path_v3_capital_aware_hybrid_50_75no_100volg001`；
3. 基于 `multi_downside_e19` 的已冻结 raw 策略版本；
4. e22-e25 ensemble 仅在其日期覆盖和当时可用性可证明时进入。

动作：

1. 先做证据资格表：eligible、historical_only、incomplete_evidence、rejected。
2. 对 eligible 候选跑同一 24-cell 矩阵，不修改策略。
3. `maxret095` 作为一种已声明信号变换单列，不再解释成执行器本身；其参数若曾看过
   Forward，则只能 historical/shadow，不可晋级。
4. 不因 Forward 较好提升名次，也不因 Forward 较差改变 Val/Test 选择结论。

产物：候选资格表、统一 ledger、candidate-vs-baseline 差异表、拒绝原因。

验收门 NT-G4：没有混合执行模式、日期口径或资本规模；每个候选都有明确治理结论。

### NT5：主动管理收益、风险和执行归因

目标：回答“为什么赚、为什么亏、收益能否执行”，而不是只解释 Sharpe 高低。

归因维度：

1. 预测能力：Rank IC、Top30/Top0.6% 收益、信号衰减、lag1 保留率；
2. 广度：有效独立持仓数、行业 HHI、Top 行业占比、候选重合度；
3. 主动风险：相对沪深300的 beta、active return、active drawdown、specific vol；
4. 风格与状态：市值、动量、波动、急涨后平台、全球/港股压力；
5. 组合构建：保留、卖出、替换、新开仓分别贡献多少收益和风险；
6. 交易成本：佣金、印花税、滑点、换手、ADV 容量和最低佣金；
7. 执行损失：涨跌停、停牌、零成交、现金、整手、ADV 导致的 blocked orders；
8. 稳定性：年度、半年、月份、市场状态、50/100 万和四压力场景。

要求：所有组合层结论都能回到逐笔 proposal/fill/position；不能只用汇总净值反推原因。

产物：中文 APM attribution、逐笔替换归因、风险暴露时间序列、执行损失瀑布图数据。

验收门 NT-G5：每个候选都能回答新增收益、减少风险和付出成本；无法归因的提升不得晋级。

### NT6：回测性能与等价性优化

目标：降低反复读取硬盘和重复构建矩阵的时间，不改变回测语义。

动作：

1. 对官方 24-cell replay 分段计时：文件发现、CSV/Parquet 读取、矩阵构建、策略、ledger、
   报告写入。
2. 复用 immutable OHLC/limit/ADV matrix cache；同 split 的候选共享只读市场 context。
3. 只缓存输入派生数据，不缓存候选排名、持仓状态、压力覆盖或未来数据。
4. 控制并行度和内存峰值，16 GB 机器保留至少 3 GB 系统余量。
5. 优化前后比较逐日净值、订单、成交数量、成本、阻塞原因及最终指标。
6. 建立小样本快速测试和完整年度 benchmark，防止以后回测变慢却无人发现。

产物：性能 profile、cache contract、等价性报告、运行预算和 benchmark 基线。

验收门 NT-G6：结果逐项等价；峰值内存受控；完整矩阵耗时有可重复基准。任何结果差异都
视为行为变更，必须另立 ADR，而不是性能优化。

### NT7：Prepared Shadow replay 与故障演练

目标：在不进入正式 Shadow、不自动交易的前提下验证每日运行可靠性。

动作：

1. 使用冻结正式基线建立或复核 `prepared` lifecycle。
2. 用历史日期 replay 连续运行每日包：data as-of、信号、目标持仓、订单 proposal、
   baseline 并行结果、警告和 hash。
3. 演练数据迟到、行情缺列、Alpha 缺失、重复日期、任务中断、磁盘不足、缓存损坏、
   执行失败和 baseline fallback。
4. 验证 resume 不重复写账、不重放订单；验证 pause/retire 事件链不可篡改。
5. 输出每日人工检查清单，但不调用券商接口、不自动下单。

产物：replay 日包、故障注入记录、恢复报告、人工 runbook、回退策略。

验收门 NT-G7：所有故障要么 fail closed，要么明确回退到正式基线；当前状态仍为
`prepared`。只有 P7 通过且用户人工批准后才能切换 `shadow`。

### NT8：Registry、报告与文档收口

目标：让新对话只读规则、总计划、Registry 和本计划即可恢复状态。

动作：

1. 更新 Registry candidate/evidence 状态，不覆盖历史记录。
2. 生成唯一中文总报告，区分 formal baseline、eligible challenger、research only、
   rejected 和 incomplete evidence。
3. 更新 `PROJECT_CURRENT_INDEX`、`DEVELOPMENT_LOG`、`EXPERIMENTS` 和必要 ADR。
4. 对本轮中间产物生成非破坏 archive review；只清理可重建缓存和明确 superseded 副本。
5. 运行完整测试、Registry schema、artifact hash、链接和复现命令检查。

产物：中文闭环报告、Registry scorecard、文档索引、归档审查和测试记录。

验收门 NT-G8：没有两个“正式 baseline”；没有未注册正式报告；没有仅凭文件名进入排行榜
的结果。

### NT9：非训练阶段决策评审

目标：基于闭环证据决定下一项训练研究，而不是直接开始跑 epoch。

必须回答：

1. 当前正式基线的收益主要来自 Alpha、组合策略还是执行假设？
2. 哪些候选在 Val/Test 的最差场景仍优于或不劣于基线？
3. 2026 的问题是 Alpha 漂移、行业集中、风险预算、换手成本还是信号覆盖？
4. 下一次训练应回答哪个单一假设：静态 e19、月度 Rolling、标签/loss 消融，还是暂缓？
5. 预期增益、算力预算、停止规则和失败回退是什么？

产物：`GO / HOLD / STOP` 决策单和下一训练实验的预注册草案。该草案不在本计划内执行。

验收门 NT-G9：只有一个下一研究问题，且不使用 Forward 选参数；否则保持 HOLD。

## 7. 固定执行顺序与并行关系

```text
NT0 repository durability
 -> NT1 baseline/inventory freeze
 -> NT2 data/execution audit
 -> NT3 formal baseline replay
 -> NT4 existing candidate review
 -> NT5 attribution
 -> NT8 registry/document closure
 -> NT9 decision review

NT3 -> NT6 performance optimization -> NT3 parity rerun
NT1 + NT2 + NT3 -> NT7 prepared Shadow replay
```

NT6 可以与 NT4 的证据资格审查并行，但性能改动必须在候选正式复跑前通过等价性门。
NT7 只能做 prepared replay，不能抢跑 P7/P8 的正式生命周期晋级。

## 8. 资源和时间预算

| 阶段 | 人工/工程预算 | 机器预算 | GPU |
|---|---:|---:|---:|
| NT0 | 0.5 天 | 远端传输 0.5-2 小时 | 不需要 |
| NT1 | 0.5-1 天 | hash/盘点 0.5-2 小时 | 不需要 |
| NT2 | 1-2 天 | 审计 1-3 小时 | 不需要 |
| NT3 | 0.5-1 天 | 24-cell 约 1-4 小时 | 不需要 |
| NT4 | 1-2 天 | 依候选数约 2-8 小时 | 仅缺 Alpha 时做推理 |
| NT5 | 1-2 天 | 归因 1-4 小时 | 不需要 |
| NT6 | 1-2 天 | benchmark 2-6 小时 | 不需要 |
| NT7 | 1 天 | replay/故障演练 1-3 小时 | 通常不需要 |
| NT8-NT9 | 1 天 | 测试与报告 1-2 小时 | 不需要 |

总体预计 7-12 个工程工作日；现有工具复用良好时可压缩到 4-7 天。机器任务以可恢复、
分阶段方式运行，不长时间占满 16 GB 内存。

## 9. 停止和回退规则

- 发现后视、日期错位或 Forward 参与选择：停止所有下游阶段，回到 NT1/NT2。
- 新 replay 与 canonical 基线不一致：停止候选比较，先完成 parity 调查。
- 历史 ST 或执行覆盖不足：明确降级结论，不伪造填充，不阻塞其他已披露研究工作。
- 回测优化产生任何订单或净值差异：回退旧实现，另立行为变更提案。
- 候选缺少完整 Val/Test：标记 incomplete，不用 Forward 或局部月份补足。
- 同一方向连续两次在固定合同下失败：停止该候选，不扩大参数搜索。
- 内存可用量低于 3 GB 或发现持续 swap：暂停任务，降低并行度后恢复。
- 远端分支存在分叉：禁止 force push，先创建备份 ref 并人工复核。

## 10. 最终完成定义

只有以下条件全部满足，本计划才算完成：

- 代码和计划有远端可恢复备份；
- baseline contract 唯一且 Registry 无歧义；
- 数据/PIT/执行覆盖报告完成，历史 ST 缺口显式披露；
- 正式基线 24 cells 全部由统一入口生成并通过 parity；
- 所有现有候选完成资格判断，eligible 候选完成公平比较；
- 收益、风险、组合和执行归因完整；
- 回测性能优化通过逐项等价验证；
- prepared Shadow replay 和故障演练通过，但未越权进入 shadow；
- 中文总报告、Registry、开发日志和测试记录一致；
- NT9 给出唯一的 GO/HOLD/STOP 结论和下一实验预注册草案。

## 11. 本计划明确不做

- 不恢复或启动任何模型训练；
- 不增加 epoch；
- 不重做 loss、标签或模型超参数搜索；
- 不根据 2026 Forward 调整策略；
- 不扩大无理论依据的阈值网格；
- 不切换正式 baseline；
- 不进入真实交易；
- 不自动把 lifecycle 从 prepared 改为 shadow。
