# MD2 CSV 到 Parquet 迁移验收摘要

状态：通过。候选库尚未替换正式 CSV backend。

## 范围与结果

| 项目 | 结果 |
|---|---:|
| CSV 来源 | `data/forward_raw` |
| 候选库 | `data/market_daily_candidate_v2` |
| 覆盖区间 | 2010-01-04 至 2026-07-29 |
| 年度批次 | 17 |
| 活动月份 | 199 |
| 交易日/活动分区 | 4,023 |
| 行数 | 13,313,700 |
| 活动 Parquet 大小 | 398,301,688 bytes |
| 候选库完整大小 | 561,081,706 bytes |
| 年度扫描总耗时 | 491.48 秒 |
| 年度提交总耗时 | 390.33 秒 |
| 观察到的峰值 RSS | 370,483,200 bytes |

## 正确性证据

1. 2010 至 2026 的 17 个年度批次均完成 CSV 与 Parquet 的主键、值和 dtype 精确比较。
2. 源 CSV 在每个批次扫描前后均校验签名，迁移期间变化会立即终止。
3. 全局活动链审计依次验证 `CURRENT`、根清单、199 个月索引和 4,023 个 Parquet 分区。
4. 所有活动 Parquet 的 SHA-256、schema 和 metadata 行数均与清单一致。
5. 单元测试覆盖幂等、恢复、源变化、CURRENT 损坏、月索引损坏和 Parquet 损坏。

完整活动链机器可读证据见 `md2_active_store_audit.json`。年度迁移进度保存在候选库本地，
不随 Git 提交行情数据本体。

## 决策

- MD2 完成，进入 MD3 每日 Parquet 增量更新。
- CSV 继续作为正式读取 backend、parity oracle 和回退路径。
- 在 Provider、月度缓存和 24-cell ledger 行为等价门全部通过前，不切换正式 backend，
  不删除 CSV，也不删除旧执行缓存。
