# A股主动量化研究平台长期路线图

> 执行顺序状态：自 2026-07-18 起，本文件仅作为架构设计和历史依据。
> 唯一有效的阶段顺序、门禁、回退与完成标准见
> `MASTER_QUANT_RESEARCH_EXECUTION_PLAN_20260718.md`。本文件中所有“当前执行
> 顺序”或“下一步”表述均不得覆盖该总计划。

日期：2026-07-17
当前优先级：Qlib 式研究框架对齐优先，收益优化与强模型滚动随后进行。

## 1. 长期目标

建设一个项目原生、可复现、可审计、适合 50 万至 100 万资金规模的 A 股主动量化研究平台：

```text
版本化 PIT 数据
-> Dataset / DataHandler / Processor
-> 可替换 Model Adapter
-> dated raw score
-> Signal / Portfolio Record
-> Strategy / Portfolio Construction
-> realistic open-price share-ledger
-> 归因、压力测试与治理
-> 人工 Shadow、监控和回滚
```

借鉴 Qlib 的模块边界、声明式任务、实验记录、Rolling 和 Online 生命周期；不复制 Qlib 的默认数据、收盘成交假设、默认 Exchange/Executor 或大资金配置。正式成交始终由本项目的 A 股 `open_ledger` 决定。

本轮对齐参考源码固定为：

- 路径：`C:\Users\x\Documents\股票预测\references\qlib`
- Git commit：`d5379c520f66a39953bad76234a7019a72796fd0`
- 重点参考：`DataHandlerLP`、`DatasetH`、Model `fit/predict` 契约、`SignalRecord`、`SigAnaRecord`、`PortAnaRecord`、`Rolling`、`TopkDropoutStrategy`、`OnlineManager`

## 2. 不可改变的项目事实

1. 2024 Val 与 2025 Test 用于研究选择；2026 Forward 只观察，不参与模型、参数、组合权重或策略选择。
2. 模型只输出带日期和股票代码的 raw score，不直接输出订单或假设成交。
3. 正式回测使用 realistic open-price share-ledger，并保留现金、股数、整手、最低佣金、印花税、ADV、停牌、零成交、涨跌停、上市天数和可交易性约束。
4. 正式压力场景至少包括 `normal`、`lag1`、`cost2x`、`capacity_3pct`，资金规模为 50 万与 100 万。
5. 财报、ST、海外市场和宏观数据必须按可用日进入特征；缺失覆盖率必须显式记录。
6. `ledger_path_v3_t0001_nolookahead` 仍是正式组合基线，除非通过 Registry 晋级规则替换。
7. V9 不再作为正式模型选择依据；`avgw3` 不是默认信号处理。

## 3. 当前 Qlib 对齐度

### 3.1 已完成并通过代码级验收

| Qlib 思想 | 本项目现状 | 判断 |
|---|---|---|
| Experiment / Recorder | schema v2 manifest、append-only events、artifact hash/index、完成状态 | 核心治理已对齐 |
| 声明式 Task/Workflow | Workflow v2 校验、冻结、编译、恢复与阶段 receipt | 已对齐项目运行方式 |
| Dataset/DataHandler/Processor | 流式 raw/infer/learn、Train-only fit、状态 hash、命名 segment | 已完成项目原生实现 |
| Model Adapter | LightGBM、PyTorch strong alpha、frozen、legacy read-only 共用 PredictionFrame | 已完成统一契约 |
| 时间切分与 Rolling | Train/Valid/OOS、label-tail purge、唯一 OOS owner、断点恢复、连续账本 | 控制器已验证 |
| 标准 Records | 六类依赖 Record、真实 ledger 明细、选择与 Forward 分离 | Q4B 已完成代码接入 |
| Signal、Strategy、Executor 分离 | dated score -> proposal -> `open_ledger` 成交 | 已对齐且更符合 A 股 |
| 人工 Shadow 生命周期 | prepared/shadow/paused/retired、人工批准、事件 hash 链 | Q7A 已完成 |

### 3.2 当前真实缺口

| Qlib 抽象 | 当前缺口 | 后果 |
|---|---|---|
| 首个完整 formal Workflow 验收 | 已完成冻结正式基线的 prediction -> 16-cell ledger -> scorecard -> 六类 Records | 生产证据链已通过首轮验收 |
| Frozen/legacy dated prediction 适配 | 已完成只读 Registry 解析、逐日 PredictionFrame 校验、源 hash 和分段覆盖冻结 | 等待随首个 formal Workflow 一并做生产验收 |
| Strong-model Rolling | e19 已绑定 Workflow；重建 pilot 被拒绝，完整 24 窗口 Q5B 未执行 | 尚未证明强模型月度更新可提升或稳定收益 |
| Rolling Ensemble | 已有 OOF lineage 和简单组合，缺少可复用的动态 as-of 集成器 | 组合可用性和失败降级仍依赖专用脚本 |
| Portfolio constructor | Retention、TopK/dropout、风险试验已存在，缺少统一 alpha-risk-cost 接口 | 组合研究仍可能退化为散乱阈值搜索 |
| Daily Shadow runner/replay | Q7B runner、真实 ledger、逐日证据、漂移监控和确定性重放已通过 20 信号日历史验收 | lifecycle 仍为 prepared，正式日常观察等待人工激活 |
| 底层 Provider 一致性 | 模型侧已有统一 Dataset，原始行情、财报、海外和交易状态仍有专用加载器 | 数据源覆盖和运行契约仍需持续收敛 |
| 历史 ST 覆盖 | 适配器和显式缺失声明已有，真实历史事件源仍不完整 | A 股可交易性审计保留外部数据限制 |

工程判断：核心接口和治理骨架已经完成，剩余工作从“设计接口”转为“用真实
formal 实验验收接口、补齐历史候选兼容、完成日级运行”，随后才恢复强模型
Rolling 和组合收益研究。不能再把单元测试、compile-only 或 63 日标签 smoke
描述为完整生产闭环。

## 4. Qlib-First 实施阶段

## Q0：对齐基线与一致性审计

目标：明确“借鉴什么、项目如何实现、什么不采用”，消除计划与代码之间的歧义。

交付物：

- `qlib_alignment_matrix.json/md`：Qlib 组件、源码位置、项目模块、状态、缺口、验收测试；
- 外部 Qlib 路径与 commit 指纹；
- 项目术语表：Provider、Handler、Dataset、Processor、Model、Record、Strategy、Executor、Workflow、Rolling、Online；
- 现有入口和重复实现清单；
- 一份 golden workflow 配置及其 JSON Schema。

验收门：所有后续阶段都能指向明确接口、负责人模块和测试；计划中不再同时存在冲突的优先顺序。

## Q1：统一声明式 Task / Workflow

状态：已完成第一版运行时迁移（2026-07-17）。v1 保持重放兼容；v2 已支持
Schema、跨字段治理校验、正式冻结与现有 LightGBM/冻结模型 stage graph。
Q3 模型 adapter 在实现前会明确拒绝。

目标：一份配置描述完整研究任务，不再依赖散落命令行默认值。

配置至少冻结：

- 数据版本、股票池、特征、标签、horizon、Train/Valid/Test/Forward；
- Processor 链及 fit 区间；
- Model adapter、超参数、seed、checkpoint 规则；
- Signal policy、Strategy、Executor、资金、成本和压力场景；
- Record 模板、Registry 角色和产物目录。

实现：

- 将现有 `experiments/workflow.py` 升级为 schema 驱动编译器；
- 区分静态验证、编译、执行、恢复和终止状态；
- 禁止正式实验使用未写入 manifest 的隐式参数；
- config hash、source hash、data hash、processor hash 与 artifact hash 全链绑定。

验收门：同一配置可 dry-run、执行、恢复；修改任意关键字段会产生新实验身份；旧专用脚本仅作为 adapter 被调用。

## Q2：Dataset / DataHandler / Processor 运行时

状态：已完成第一版（2026-07-17）。已实现流式 shared/infer/learn 链、
Train-only fit、冻结状态哈希、命名 segment `prepare` 和真实 v14 Provider
适配，不改变物理缓存格式。

目标：借鉴 `DatasetH + DataHandlerLP`，统一数据切片和预处理状态，同时保持 v14 memmap 的低内存优势。

接口：

```text
Provider.load(view)
Handler.fit(train_view)
Handler.transform(view, data_key=raw|infer|learn)
Dataset.prepare(segment, col_set=feature|label, data_key=infer|learn)
```

要求：

- shared、infer、learn processor 明确分离；
- 标准化、截尾、填充、特征筛选只在 Train 拟合；
- Valid/Test/Forward 只加载冻结状态；
- 无状态 trailing/PIT 变换与有状态 train-fitted 变换分开；
- processor state、fit range、列顺序、dtype、缺失率、覆盖率和 hash 持久化；
- 支持日期切片和按需加载，16GB RAM 下不能复制整个 v14 缓存。

验收门：同一 Dataset 可供 LightGBM 与 PyTorch 使用；重复 transform 结果 hash 一致；故意在 Valid 重新 fit 必须失败；内存峰值有自动测试。

## Q3：统一 Model Adapter

状态：已完成第一版统一接口（2026-07-17）。Q5 仍需把真实
`multi_downside_e19` 训练器绑定到月度 Rolling 任务并完成窗口验收。

目标：让模型只关心 `fit(dataset)`、`predict(dataset, segment)`、保存和恢复，不自行解释日期、缓存或账本。

首批 adapter：

1. `LightGBMModelAdapter`；
2. `TorchStrongAlphaAdapter`；
3. `FrozenArtifactAdapter`；
4. 只读 `LegacyReadOnlyAdapter`。

统一契约：

- `fit`、`resume`、`select_checkpoint`、`predict`、`save_state`；
- checkpoint 选择只能读取该任务 Valid；
- 输出统一 PredictionFrame：`trade_date/code/score/model_id/asof_time`；
- 记录模型结构、loss、标签、seed、optimizer、资源峰值和训练耗时；
- 不把 reconstructed 配置冒充历史原始配置。

验收门：同一 golden workflow 仅替换 adapter 即可运行 LightGBM 与 PyTorch；raw score schema 完全一致；模型层不导入 `open_ledger`。

当前验收结果：统一工厂、四类 adapter、checkpoint 保存/恢复、冻结资产
hash 校验、共同 `PredictionFrame` 和执行层隔离均已通过。PyTorch adapter
要求显式注入现有训练器，不会把重建配置冒充历史 e19 配置。

## Q4：标准化 Record 依赖链

状态：Q4A 记录契约和 Q4B Workflow 运行时接入均已完成（2026-07-17）；
等待首个新执行的完整 formal bundle 做生产验收。
旧实验缺少的持仓、订单、拒单或成本明细不会被汇总 CSV 冒充。

目标：借鉴 `SignalRecord -> SigAnaRecord -> PortAnaRecord`，让每次正式实验自动产生同构证据。

项目记录模板：

- `SignalRecord`：raw prediction、label、覆盖率、日期与 as-of 审计；
- `SignalAnalysisRecord`：IC、RankIC、ICIR、分位收益、Top30/Top0.6%、行业和风格暴露；
- `PortfolioRecord`：连续 realistic ledger、持仓、订单、拒单、成本和净值；
- `RiskAttributionRecord`：beta、specific vol、行业集中、主动回撤、换手与执行损失；
- `StressRecord`：双资金与四压力场景；
- `DecisionRecord`：Val/Test 选择结果和 Forward 观察结果分离。

记录之间声明依赖，缺少父产物时不得静默生成不完整报告。

验收门：任意正式实验都生成相同目录结构和必需指标；缺少日期范围、hash、压力场景或父产物时不能进入排行榜。

当前验收结果：`experiments/record_templates.py` 固定六类记录、父依赖、
必需字段、标准目录和 SHA-256；`run/materialize_standard_records.py` 按
schema 化清单装入现有分析与账本产物。Forward 参与选择、Qlib/close
执行、缺少双资金四压力或缺父记录都会失败。

Q4B 已完成实现：项目原生 `open_ledger` 已增加不改变成交结果的可选旁路 trace；
正式 registry 回测会保存并哈希净值、每日持仓、逐股目标/成交、拒单原因和
分项成本。Workflow v2 已在 ledger/scorecard 后自动生成六类 Record；现有
`diagnostics.csv` 仍只代表每日聚合诊断，不得复制第二套执行逻辑。首个新完整
formal Workflow 仍需提供生产验收证据，但不阻塞后续框架对齐。

## Q5：Rolling / OOF / Ensemble

状态：Q5A 框架绑定已完成，Q5B 收益验证暂缓。`torch_strong_alpha` 已可由
Workflow v2 编译到可恢复的 staged runner，并输出与 LightGBM 相同的
`rolling_manifest.json` 和按 split 拼接的 alpha。历史 exploratory pilot
也可在不重训、不晋级的前提下生成标准兼容包装。

目标：在 Q1-Q4 稳定后，使用统一接口运行真正的月度样本外研究。

顺序：

1. 一个窗口 dry-run；
2. 一个窗口 LightGBM smoke；
3. 一个窗口 reconstructed `multi_downside_e19` smoke；
4. 三个跨市场状态窗口 pilot；
5. 完整 2024 Val 滚动；
6. 冻结选择后运行 2025 Test；
7. 最后生成 2026 Forward 观察。

Rolling invariants：每个 OOS 日期唯一归属；标签尾部 purge；Processor 只在窗口 Train 拟合；checkpoint 只看窗口 Valid；资本、持仓和成本跨月连续；失败窗口保留并报告。

Ensemble 只能组合预测日当时已完成训练且可用的模型，先做等权/rank-average 基线，再考虑状态条件权重。2026 Forward 不得选择窗口长度或组合权重。

验收门：24 窗口可恢复、无泄漏、无日期缺口、无月底资金重置；e19 raw 与静态 e19 在相同组合和账本下公平比较。

## Q6：Strategy 与组合构建

目标：在统一 raw score 上研究 Retention、TopK/dropout 和 alpha-risk-cost 组合构建，而不是继续散乱调阈值。

边界：

- Strategy 输出目标持仓/权重及理由；
- Portfolio constructor 处理 alpha、行业、beta、specific vol、换手和预期执行成本；
- `open_ledger` 独立决定可成交数量和实际成本；
- open、close 或混合时点研究必须单独声明 `score_time/order_time/fill_price`。

验收门：2024 Val + 2025 Test 的平均与最差 Sharpe、MDD、换手、成本和容量通过 Registry gate，并提供逐笔替换归因；Forward 只展示。

## Q7：人工 Shadow / Online 生命周期

状态：Q7A 生命周期治理和首个真实 `prepared` 实例已完成（2026-07-17）；
Q7B 日级 runner/replay 已绑定完整 formal Record bundle，并通过 20 信号日
真实历史执行与独立确定性重放。正式 lifecycle 观察仍等待人工激活。

目标：借鉴 `OnlineManager` 的模型历史和信号准备思想，但保持人工批准，不自动交易或自动晋级。

每日状态机：

```text
数据就绪 -> PIT/覆盖率检查 -> 冻结模型推理 -> proposal
-> 预交易约束 -> 人工批准 -> Shadow ledger
-> 收盘归因 -> 漂移监控 -> 保留/回滚
```

必须记录模型版本、数据 as-of、运行状态、失败重试、信号缺失、候选/基线并行、漂移告警、月度复核、人工切换和回滚原因。

验收门：连续模拟运行不少于 20 个交易日，无 silent failure；任意一天可重放；模型切换需人工批准并可回滚。

## Q8：可选扩展

仅在 Q0-Q7 稳定后考虑：模型 zoo、并发任务调度、MLflow 服务、自动因子建议、多强模型动态集成、图关系、元学习或强化学习。每项都必须先有简单基线、预注册假设、独立 OOS 和停止规则。

## 5. 当前执行顺序

```text
P0 框架对齐
Q0 对齐矩阵
-> Q1 Workflow schema
-> Q2 Dataset/DataHandler/Processor
-> Q3 Model Adapter
-> Q4A Record templates
-> Q5A strong-model Workflow binding
-> Q4B Record runtime integration
-> Q7A manual Shadow lifecycle governance

P1 研究验证
Q5B e19 月度 Rolling / OOF

P2 收益与风险优化
Q6 Strategy / Portfolio Construction

P3 运行治理
Q7B Daily Shadow runner / replay（框架验收完成，正式运行等待人工激活）

P4 远期扩展
Q8 Optional extensions
```

强模型收益训练不再是当前第一步。`multi_downside_e19` 的 Workflow 绑定、
Q4B 自动 Records 与 Q7A 人工生命周期治理均已完成。Q5B/Q6 继续暂缓；
Q7B 已使用首个完整 formal bundle 和真实冻结基线完成框架验收，没有使用
虚构候选。lifecycle 仍为 `prepared`，不能把历史重放称为正式 Shadow 激活。

## 6. 明确不做

- 不用 Qlib Executor 替换 `open_ledger`；
- 不直接使用 Qlib 默认 China 数据、标签或收盘成交回测；
- 不因“对齐 Qlib”而复制第二套数据缓存和第二套正式 Registry；
- 不在 Q0-Q4 完成前启动完整 24 窗口强模型训练；
- 不使用 2026 Forward 选择模型、loss、组合权重或防御条件；
- 不用 IC 单独选择 checkpoint；
- 不默认加入 `avgw3`、`maxret095`、V3 或状态覆盖；
- 不自动交易、自动月度再训练或自动晋级。

## 7. 下一验收点

Q7A、frozen/legacy dated-prediction 适配、首个完整 formal Workflow 和 Q7B
历史 replay 框架验收均已完成。下一步按以下顺序推进：

## 17. 2026-07-18 反向对齐审计修正

本节修正“接口验收通过”等同于“正式主线已迁移”的过度表述：

- Q2 Dataset/DataHandler/Processor：接口、低内存运行时和测试已完成，
  但正式 LightGBM 与强模型训练脚本仍直接读取旧 sample path，状态为
  `partial`。
- Q3 Model Adapter：统一契约与 Workflow 编译接入已完成，但正式 trainer
  仍自行执行 fit/predict，尚未做到端到端 adapter substitution，状态为
  `partial`。
- Recorder：正式 Workflow bundle 的 hash、artifact index 与完成门槛有效；
  通用实验尚未完整冻结 Python/package/hardware 环境，脏工作树也不能仅凭
  Git commit 完整复现，状态为 `partial`。
- Rolling manifest 的 `run_mode=formal` 只表示真实执行而非 dry-run，不等于
  Registry/governance 中的 formal experiment。代码现已明确区分这两类证据。
- Q6 统一 alpha-risk-cost 组合构建器、动态 as-of Rolling Ensemble、历史 ST
  完整数据仍未完成。Qlib Executor 继续不采用，正式执行器仍为 A 股
  realistic open-price `open_ledger`。

因此当前结论是“治理骨架和主要接口已建立，首个 frozen baseline 正式链路
通过；训练主线迁移和组合层仍未完全对齐”，不能表述为整个项目已经完成
Qlib 式工业化迁移。

1. 明确人工批准后，才把 `prepared` lifecycle 切换到 `shadow` 并积累正式日常观察；
2. 恢复 Q5B e19 Rolling 与动态 as-of Ensemble 研究；
3. 实现 Q6 统一 Strategy / Portfolio Constructor 并做 Val/Test 选择评估。

以上步骤均不得自动晋级、自动交易或使用 2026 Forward 调参。
