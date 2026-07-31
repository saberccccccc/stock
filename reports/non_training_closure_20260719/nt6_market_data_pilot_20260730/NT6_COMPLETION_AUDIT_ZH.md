# NT6 行情后端与回测性能迁移完成审计

日期：2026-07-31

结论：**NT6 已完成**。正式执行行情后端已由策略原子切换为
`monthly`，`legacy` 保留为回滚后端，CSV 仅保留为只读等价性 oracle。
本结论只代表 NT6 完成，不代表 NT3-NT9 已全部完成。

## 完成定义逐项核对

| 项目 | 结果 | 主要证据 |
|---|---|---|
| monthly 正式激活并冻结身份 | 通过 | policy v2 为 `monthly_active`；候选 manifest SHA-256 为 `b9596f...692a` |
| Selection/Forward 共用物理库、逻辑视图分离 | 通过 | `MarketDailyProvider`、`DataView` 和三段双读身份一致 |
| 单日增量不扫描 5,000 个股票文件 | 通过 | 5,299 行提交耗时 0.066 秒，仅写当日 Parquet 分区 |
| 月缓存仅刷新受影响月份 | 通过 | 2026-07 月缓存刷新耗时 0.161 秒，随后 warm 状态为 `already_current` |
| 正式入口不再依赖逐股票 CSV | 通过 | 调用点审计通过，正式默认声明为 `monthly`；CSV 为显式 oracle |
| 24-cell ledger 严格等价 | 通过 | CSV 与 monthly 各 24 cells，Val/Test/Forward parity 均通过 |
| 16 GiB 机器资源门 | 通过 | MD8 acceptance 的 memory、process I/O 和 runtime gates 全部通过 |
| manifest、benchmark、ADR、架构、日志齐全 | 通过 | policy v2、incremental benchmark、ADR 0011、ARCHITECTURE 和 DEVELOPMENT_LOG |
| monthly → legacy → monthly 恢复演练 | 通过 | append-only 历史保留 5 次转换；历史演练身份一致，最终提升绑定当前证据 |
| 旧 CSV 未擅自删除 | 通过 | `data/raw` 仍保留 5,332 个股票 CSV；删除仍需独立审计和用户批准 |

## 性能与正确性摘要

- Val 2024：monthly 34.016 秒，CSV 117.803 秒，约快 3.46 倍。
- Test 2025：monthly 32.738 秒，CSV 60.166 秒；执行账本已成为主要瓶颈。
- Forward 2026：monthly 21.650 秒，CSV 101.429 秒，约快 4.69 倍。
- 全量双读：Val 2024、Test 2025、Forward 2026 六字段逐元素严格一致。
- 回归测试：651 项全部通过；仅保留 1 个既有 Pandas FutureWarning。

## 后续边界

1. 不删除 CSV；至少观察一个完整月后再单独做空间审计。
2. NT6 产物作为冻结基础设施，不再成为收益调参面。
3. 按总计划回到 NT3，使用相同 24-cell 契约重放正式 baseline。
4. 本阶段没有启动、恢复或调优任何模型训练。
