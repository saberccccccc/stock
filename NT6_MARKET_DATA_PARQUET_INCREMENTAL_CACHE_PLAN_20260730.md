# NT6 行情 Parquet 与增量执行缓存实施计划

状态：`in_progress`。本计划是
`NON_TRAINING_RESEARCH_CLOSURE_PLAN_20260719.md` 中 NT6 的技术展开，不改变
总计划顺序，不启动训练，不改变正式回测语义。

当前实施状态：

| 阶段 | 状态 | 证据 |
|---|---|---|
| MD0 | complete | `reports/non_training_closure_20260719/nt6_market_data_baseline_20260730` |
| MD1 | complete | ADR 0010、`data/market_daily_store.py`、事务/锁/回滚测试 |
| MD2 | complete | 17 个年度精确审计、全局活动链审计、`md2_migration_summary_20260730.md` |
| MD3 | complete | Tushare 股票 + AkShare 宽基真实试跑、真实重放 no-op、全量哈希审计 |
| MD4 | complete | 48 只固定样本 + 5,332 只全量，Val/Test/Forward 七字段精确 parity |
| MD5 | complete | 月索引绑定、按月失效、跨月派生、基础执行 mask、真实 2026-07 全量 parity |
| MD6 | complete | 24 CSV + 24 monthly cells；Val/Test/Forward 共 144 个逐路径工件精确值与字节哈希零差异 |
| MD7 | complete | 统一后端契约、Workflow/Shadow 透传、静态调用点治理 |
| MD8 | in progress | 真实 5,299 行增量提交/单月刷新已通过；clean matrix 等待 3.75 GiB 启动门 |
| MD9 | in progress | 双读、晋升门、追加式转换历史和原子回退已实现；全量观察未完成 |

## 1. 目标

将当前“每只股票一个 CSV + 全局 OHLC memmap 失效后完整重建”的行情链路，
迁移为：

```text
Tushare 单日原始行情
-> 经过校验的按交易日 Parquet 分区
-> 统一 MarketDailyProvider
-> 按月分片的增量 OHLC/ADV/涨跌停输入缓存
-> 现有 realistic open-price share-ledger
```

目标不是为了更换文件格式而更换文件格式，而是同时解决：

1. 每日更新需要打开约 5,300 个 CSV；
2. 研究与 Forward 维护两套约 3 GiB 的重复行情目录；
3. 任一 CSV 更新都会使约 0.95 GiB 的全局 OHLC 缓存整体失效；
4. 任意区间读取仍有脚本绕过 Provider 直接扫描 CSV；
5. 数据更新、缓存刷新、回测之间缺少完整事务和并发锁；
6. Windows 小文件、杀毒扫描和随机 I/O 带来的额外开销。

在正式切换前必须证明：

- 原始行情字段、日期、股票代码和缺失状态一致；
- 逐日候选、订单、成交、阻塞原因、成本、持仓和净值一致；
- Val 2024、Test 2025、Forward 2026 的逻辑边界不变；
- 16 GiB 内存机器始终保留至少 3 GiB 系统余量；
- 旧 CSV 路径可以立即回退。

## 2. 当前基线

截至 2026-07-30 的现场量级：

| 对象 | 文件数 | 大小 | 当前问题 |
|---|---:|---:|---|
| `data/raw` | 5,371 | 约 1.44 GiB | 每股 CSV，物理数据与逻辑研究 view 混合 |
| `data/forward_raw` | 5,373 | 约 1.56 GiB | 每股 CSV，每日更新触碰数千文件 |
| `cache/open_ledger_ohlc_matrix` | 7 | 约 0.95 GiB | 任一源文件变化后整体失效 |

现有可复用基础：

- `DataView` 已能在同一物理数据上声明逻辑日期范围；
- `OhlcvMatrixProvider` 已提供按股票、字段和任意日期区间读取；
- `open_ledger` 已是唯一正式 A 股执行权威；
- `pyarrow` 已安装；
- CSV 增量更新器已有进程锁、进度文件、幂等追加和尾部修复能力。

明确不复用或不引入：

- 不用 Qlib 数据和 Executor 替换项目执行层；
- 不建立第二套回测器；
- 第一阶段不强制安装 DuckDB；
- 不把所有数据合并成一个巨大 CSV 或一个不可局部替换的巨大 Parquet；
- 不在此次迁移中改变特征、标签、模型、Alpha 或组合规则。

## 3. 架构评审

### 项目理解

项目正式链路是 PIT 数据与逻辑 view 产生 Alpha，策略层产生 proposal，最终由
realistic open-price share-ledger 维护现金、股数、整手、成本、ADV、涨跌停、停牌和
上市天数约束。行情存储优化只能改变数据到 Provider 的实现，不能改变执行含义。

### 受影响模块与所有权

| 模块 | 所有权 | 计划变化 |
|---|---|---|
| `data/market_daily_store.py` | `data/` | 新增 Parquet schema、事务写入、manifest |
| `data/providers.py` | `data/` | 新增 `MarketDailyProvider`，保留现有接口 |
| `backtest/ohlc_matrix_cache.py` | `backtest/` | 从全局缓存升级为按月分片缓存 |
| `run/update_market_daily.py` | `run/` | 新增薄 CLI，下载、校验、提交单日分区 |
| `run/migrate_csv_market_daily.py` | `run/` | 一次性、可恢复的 CSV 迁移工具 |
| `run/audit_market_daily_parity.py` | `run/` | CSV/Parquet/缓存/ledger 等价审计 |
| `registry` 与正式回测 | 治理/执行 | 不改经济参数，只记录数据版本与 backend |

### 依赖和数据流

```text
Tushare
-> staging DataFrame
-> schema/invariant validation
-> staging Parquet
-> content-addressed partition promotion
-> immutable manifest + atomic CURRENT pointer
-> update event
-> monthly cache invalidation
-> MarketDailyProvider
-> OhlcvMatrixProvider-compatible facade
-> open_ledger
```

### PIT、执行和 Forward 风险

1. 一个物理行情库可覆盖最新日期，但研究 view 必须继续截止
   `2025-12-31`；2026 只允许 Forward 观察。
2. Parquet 中存储的是未复权真实 OHLC，不能把复权价格带入成交层。
3. Tushare `vol`、`amount` 单位必须冻结，不能在迁移中隐式换算。
4. 财报、ST、上市日期、全球市场等 PIT 数据继续由各自 Provider 管理，不混入
   日线价格分区。
5. 分区修订必须留下旧 hash 和新 hash，不能静默覆盖历史数据。

### 测试和证据计划

- schema、幂等、锁、崩溃恢复和坏分区测试；
- CSV/Parquet 字段与缺失值逐单元 parity；
- 月度缓存增量与完整重建 parity；
- open-ledger 订单、成交和净值 parity；
- Val/Test/Forward 24-cell parity；
- 冷启动、热缓存、单日更新和年度回测 benchmark；
- 峰值 RSS、读取字节、写入字节、文件打开数和缓存命中率。

### 推荐实施顺序

先冻结基准并建立只读审计，再实现 Parquet 权威层；随后实现 Provider 双读，
最后替换全局缓存。所有调用方完成迁移前不得删除 CSV。

### 是否发现架构问题

是。问题不在 CSV 能否工作，而在存储格式泄漏到大量脚本，以及全局缓存身份绑定到
数千个 CSV 的文件大小和修改时间。任何更新都会扩大为全局失效，无法满足可靠的每日
增量运行。

## 4. 权威数据契约

### 4.1 目录

```text
data/market_daily/
  schema.json
  CURRENT
  manifests/
    manifest-<sha256>.json
  update_events.jsonl
  locks/
  staging/
  equity/
    year=2026/
      month=07/
        day=29/
          part-<sha256>.parquet
  index/
    year=2026/
      month=07/
        day=29/
          part-<sha256>.parquet
```

每个交易日一个分区文件，约 5,500 行。这样每年约 250 个文件，而不是每年更新
5,000 多个股票文件。文件名包含内容 hash，历史 revision 写新文件，不覆盖已登记
文件。`CURRENT` 只指向一个不可变 manifest。月末可选做 compaction，但初版不依赖
compaction。

第一阶段的 `equity` 覆盖 A 股日线，`index` 覆盖沪深300、上证50、中证500、创业板
等正式市场状态输入。申万行业指数随后按同一 `index` schema 迁移。美股、港股、商品、
财报和 ST 状态仍是独立 PIT 数据集，不混入 A 股日线表。

### 4.2 字段

| 字段 | 类型 | 约束 |
|---|---|---|
| `trade_date` | `date32` | 分区日期，非空 |
| `code` | string | `000001.SZ` 等规范代码，非空 |
| `open/high/low/close` | float64 | 未复权真实价格 |
| `volume` | float64 | 保持现有 Tushare `vol` 单位 |
| `money` | float64 | 保持现有 Tushare `amount` 单位 |
| `factor` | float64 | 兼容保留；执行行情固定为 1.0 |
| `source` | dictionary string | 如 `tushare_daily` |

主键为 `(trade_date, code)`。分区内必须按 `code` 排序，保证确定性 hash 和可压缩性。
下载时间、运行命令和提交时间只写入 manifest/event，不进入内容 hash 对应的行情表，
因此相同市场数据重复运行可以稳定 no-op。

### 4.3 校验

提交前必须通过：

- 主键唯一；
- 行数不低于配置阈值，并与市场响应摘要一致；
- `high >= max(open, close)`、`low <= min(open, close)`；
- OHLC 为正，成交量和成交额非负；
- 代码后缀属于 `SH/SZ/BJ`；
- 分区日期与请求交易日一致；
- 列集合、dtype、单位和 schema version 一致；
- 重跑同一日期时，内容相同则 no-op，内容变化则生成 revision 事件。

## 5. 原子更新与故障恢复

每日更新固定为：

```text
获取全局进程锁
-> 查询交易日历
-> 下载当天全市场
-> 内存校验
-> 写 staging/<run_id>.parquet
-> 回读 staging 并校验 row count/schema/hash
-> 将内容寻址文件移动到正式分区
-> 写新的不可变 manifest
-> 原子切换 CURRENT 指针
-> 追加 update event
-> 释放锁
```

规则：

1. `CURRENT` 最后更新，因此活动 manifest 指向的分区一定已经存在；
2. 崩溃后 staging 可清理，正式分区不受影响；
3. 同一日期只允许一个 writer；
4. Reader 只读取 `CURRENT` 所指 manifest 登记的 complete 分区；
5. 历史修订默认拒绝，只有显式 `--allow-revision` 才能登记新 revision；
6. revision 保存旧/新 SHA-256、行数、原因、命令和时间；
7. 不再通过逐个文件尾日期决定需要下载哪些天。

## 6. 逻辑 DataView

迁移后只维护一个物理行情库：

```text
data/market_daily
├─ selection view: max_data_date=2025-12-31
└─ forward view:   max_data_date=当前已验收交易日
```

逻辑约束：

- Val 2024、Test 2025 可以参与选择；
- 2026 全年为 Forward，只观察；
- 请求超过 view 的 `max_data_date` 必须报错；
- manifest 必须记录物理最大日期和逻辑最大日期；
- 不能通过文件夹名称推断研究边界。

## 7. Provider 迁移

新增统一接口：

```python
provider.load(
    codes=[...],
    fields=["open", "close", "money"],
    start_date="2025-01-01",
    end_date="2025-12-31",
)
```

实现阶段：

1. `CsvMarketDailyBackend`：包装现有 CSV，作为 parity oracle；
2. `ParquetMarketDailyBackend`：用 `pyarrow.dataset` 做分区裁剪和列裁剪；
3. `MarketDailyProvider`：验证 DataView、规范代码和字段，屏蔽 backend；
4. `OhlcvMatrixProvider` 保留外部 API，内部委托新的行情 Provider 或月度缓存；
5. 逐步禁止 `run/`、`backtest/` 直接扫描每股 CSV。

`pyarrow.dataset` 是正式第一实现，因为它已经安装并支持分区、列和谓词裁剪。DuckDB
只作为交互审计和复杂聚合的候选层；只有 MD8 profile 证明 Arrow 查询仍是瓶颈，才新增
依赖和 ADR，不能为了“看起来工业化”提前引入。

切换由显式配置控制：

```yaml
data:
  market_daily_backend: csv  # csv | parquet
  market_daily_root: data/market_daily
```

正式默认值在所有 parity 门通过前保持 `csv`。

## 8. 增量 OHLC 缓存

不继续维护一个覆盖 2010 至今、任一变更整体失效的全局矩阵。新缓存按月分片：

```text
cache/ohlcv_monthly_v3/
  year=2026/
    month=07/
      meta.json
      open.dat
      high.dat
      low.dat
      close.dat
      volume.dat
      money.dat
```

每个分片记录：

- schema/cache version；
- 源 Parquet 分区及其 SHA-256；
- 日期列表、股票列表和 shape；
- dtype、单位、缺失值约定；
- 构建命令和时间；
- 原始字段与派生字段版本。

更新规则：

- 新增一天只重建或扩展当月约数 MiB 的分片；
- 历史某日修订只失效对应月份；
- 跨月回测按日期拼接所需分片；
- 不缓存候选排名、压力场景、持仓或 Forward 选择结果；
- `pre_close`、`pct_chg`、ADV 和涨跌停输入属于确定性派生缓存，单独版本化；
- ST、上市日期等时间状态继续作为独立 Provider 输入，不烘焙进原始 OHLC。

## 9. 分阶段实施

### MD0：冻结现场和基准

产物：

- 当前 CSV 与 OHLC cache inventory；
- 2024/2025/2026 固定样本读取 hash；
- 正式 baseline 的逐日订单、成交、阻塞和净值基准；
- 冷/热运行 profile。

停止条件：若当前数据仍有重复、乱序或未解释缺口，先修数据，不进入迁移。

### MD1：Schema、ADR 和 manifest

实现 schema、分区命名、单位、revision、锁和事务规则。新增 ADR，明确单物理库和
逻辑 DataView。

验收：schema/invariant/manifest/lock/crash-recovery 单元测试全部通过。

### MD2：CSV 到 Parquet 的可恢复迁移

按交易日分批迁移，不一次把所有 CSV 载入内存：

1. 每次处理一个年份或月份；
2. 从 CSV 流式读取该区间必要列；
3. 写 staging 分区；
4. 回读并与 CSV 比较；
5. 完成后记录 checkpoint，可中断续跑。

内存预算：目标峰值不超过 4 GiB，系统可用内存不少于 3 GiB。

验收：所有分区主键唯一，覆盖和字段 parity 通过；原 CSV 不删除。

### MD3：每日 Parquet 增量更新

将 Tushare 每日下载直接提交一个分区；CSV 更新器仅作为兼容镜像工具，不再是
权威写入入口。

同时将四个正式宽基指数接入 `index` 数据集；申万行业指数可以在不阻塞 equity
主线的后续批次迁移。

验收：重复运行 no-op；中断后可恢复；并发第二进程被锁拒绝；历史修订可审计；
股票与宽基指数的活动 manifest 日期一致或明确记录源端缺口。

### MD4：Provider 双读 parity

对固定股票、停牌股票、新股、ST/非 ST、不同板块和随机样本，比较 CSV 与 Parquet：

- 日期和代码完全一致；
- 缺失位置完全一致；
- 原始数值逐元素一致；
- 任意区间裁剪一致；
- DataView 越界行为一致。

验收：固定样本和全年度 parity 均通过。

### MD5：月度执行缓存

实现月度 OHLC cache、分片 manifest、按月失效和跨月拼接。先保留旧全局缓存作为
oracle。

验收：原始字段、派生字段和执行 mask 一致；更新一天不重建历史月份。

### MD6：open-ledger 严格等价

先跑小样本，再跑正式区间：

1. 10 个交易日快速样本；
2. Val 2024 normal；
3. Test 2025 normal；
4. Forward 2026 normal；
5. 完整 24-cell。

必须逐项比较：

- proposal 和订单顺序；
- 成交/未成交及原因；
- 股数、现金、持仓；
- 佣金、印花税和滑点；
- ADV/容量；
- 每日收益和净值；
- 年化、Sharpe、最大回撤、换手。

任何差异都阻止默认切换。若差异来自预先存在的 CSV 缺陷，必须单独 ADR，不得伪装成
性能优化。

### MD7：调用方收敛

用静态扫描列出直接读取股票 CSV 的入口，按正式程度分批迁移：

1. official backtest 和 Provider audit；
2. Shadow replay；
3. 市场状态和组合归因；
4. 训练/推理数据准备；
5. 历史研究脚本。

每迁移一批都保留 parity 测试。历史只读脚本可保留 CSV adapter，不强制重写。

### MD8：性能验收

记录：

- 网络下载时间与本地提交时间分开；
- 文件打开数、读取/写入字节；
- cold/warm 任意区间读取；
- 单日 cache refresh；
- 24-cell 总耗时；
- 峰值 RSS 和系统剩余内存；
- 分区和缓存命中率。

初始性能目标：

| 场景 | 门槛 |
|---|---|
| 单日约 5,500 行本地提交 | 不超过 10 秒，不打开 5,000 个股票文件 |
| 单日 cache refresh | 不重建历史月份，目标不超过 30 秒 |
| 年度 OHLC materialization | 相比 CSV cold path 至少快 3 倍 |
| 24-cell replay | 相比 MD0 至少快 2 倍，或 profile 证明 ledger 本身已成为主瓶颈 |
| 内存 | 峰值受控，始终保留至少 3 GiB 系统余量 |
| 正确性 | ledger 行为零差异 |

若基准证明某个目标不合理，只能基于 profile 调整目标并记录原因，不能降低 parity 门。

优化采用逐层晋级，不一次叠加所有技术：

```text
Arrow 分区裁剪
-> 月度 cache
-> 调整 batch/并发和读取列
-> profile 仍有查询瓶颈时才评估 DuckDB
```

每层单独记录收益、内存和复杂度；没有可测收益的层不进入正式默认。

### MD9：切换、观察和回退

切换顺序：

1. 默认和回退 backend 仍为 `legacy`；
2. `monthly` 作为主读、`csv` 作为六字段 shadow oracle 进入双读观察；
3. MD8 clean acceptance 和三段 full dual-read 全部通过后，控制器停在人工批准门；
4. 人工批准后，正式 backend 原子切为 `monthly`；
5. `legacy` 保留为回退后端，CSV 保留为只读逐字段 oracle；
6. 至少观察一个完整月后，是否取消每日 CSV 镜像或归档旧缓存另做非破坏性空间审计。

回退只允许从 `monthly` 改回 `legacy`，不修改 Registry、Alpha 或 ledger 参数。
晋级和回退都记录追加式 from/to/actor/reason/time 历史；晋级额外冻结所有准入
证据 SHA-256。`run/close_nt6_market_backend.py` 只生成和审计证据，永不自动晋级。

## 10. 测试矩阵

### 单元测试

- schema 和 dtype；
- 主键重复；
- OHLC invariant；
- 原子分区提交；
- lock 冲突；
- staging 恢复；
- revision/no-op；
- manifest hash；
- 分区裁剪和列裁剪；
- 月度 cache 命中/失效。

### 集成测试

- Tushare mock -> Parquet -> Provider；
- CSV -> Parquet migration resume；
- Provider 跨月和跨年；
- 新上市、停牌、零成交、北交所和 ST 输入；
- Parquet 修订只失效一个月；
- 旧/新 cache 跨月拼接一致。

### 正式证据

- Val 2024；
- Test 2025；
- Forward 2026；
- 50 万、100 万；
- normal、lag1、cost2x、capacity_3pct；
- 共 24 cells。

## 11. 空间预算

预期：

- Parquet 权威层通常小于两套 CSV 的合计体积；
- 迁移期间 CSV、Parquet 和新旧缓存并存，需要临时空间；
- 开始 MD2 前必须计算实际压缩率和剩余磁盘；
- 若预计峰值剩余空间低于 20 GiB，迁移按年份进行并暂停，不删除原数据腾空间；
- 旧缓存只有在新缓存 parity 和回退演练通过后才可列入删除候选。

## 12. 风险与控制

| 风险 | 控制 |
|---|---|
| 两个 updater 并发 | OS 级锁，第二进程立即失败 |
| 分区写一半 | staging + 回读 + 原子 rename |
| manifest 指向坏文件 | 数据文件先提交，manifest 最后提交 |
| 历史行情静默修订 | 默认拒绝，显式 revision 和双 hash |
| 研究使用 2026 | DataView 强制 max date，manifest 记录逻辑 view |
| 价格单位改变 | schema 和 parity 固定单位 |
| 月度缓存脏读 | 分区 hash 与 cache manifest 一一绑定 |
| 迁移占满内存 | 月/年分批，峰值 RSS 门禁 |
| 迁移占满磁盘 | MD2 前空间预算，不提前删除 CSV |
| 优化改变收益 | 逐订单和逐净值 parity，差异即停止 |

## 13. 完成定义

以下条件全部满足才算完成：

1. `execution_market_backend_policy_v2` 已将 `monthly` 设为 active，并冻结唯一权威 store/cache 路径和活动 manifest；
2. selection 和 Forward 使用同一物理库、不同逻辑 DataView；
3. 每日更新只提交当天分区，不打开 5,000 个股票文件；
4. OHLC cache 只刷新受影响月份；
5. 正式默认入口不再直接依赖每股 CSV；CSV 只保留为显式 shadow oracle；
6. 24-cell ledger 行为严格等价；
7. 16 GiB 机器资源门通过；
8. manifest、benchmark、ADR、架构和开发日志齐全；
9. `monthly -> legacy -> monthly` 回退与恢复经过演练，转换历史和证据哈希完整；
10. 旧 CSV 未经单独空间审计和用户批准不得删除。

## 14. 时间预算

| 阶段 | 预计工程时间 | 预计机器时间 |
|---|---:|---:|
| MD0-MD1 | 0.5-1 天 | 0.5-2 小时 |
| MD2 | 0.5-1 天 | 1-4 小时 |
| MD3-MD4 | 0.5-1 天 | 0.5-2 小时 |
| MD5 | 1-2 天 | 1-3 小时 |
| MD6 | 0.5-1 天 | 2-6 小时 |
| MD7-MD9 | 1-2 天 | 1-4 小时 |

总计约 4-8 个工程工作日。第一天只完成基准、契约和小规模迁移，不直接切换正式
backend。

## 15. 与当前计划的关系

- 当前 active plan 仍是 `2026-07-19-non-training-closure`；
- NT3 先提供正式 baseline profile 和 parity oracle；
- 本计划作为 NT6 的唯一行情/缓存技术规格；
- NT6 通过后回到 NT3 做相同 24-cell parity replay；
- 不改变 NT4-NT9 的治理顺序；
- 不恢复模型训练。
