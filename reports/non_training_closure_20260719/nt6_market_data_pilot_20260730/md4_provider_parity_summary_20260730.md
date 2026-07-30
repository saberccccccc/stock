# MD4 Provider 双读等价验收摘要

状态：通过。正式 backend 仍为 CSV。

## 实现

- `CsvMarketDailyBackend`：只读取请求代码的旧 CSV，作为 parity oracle。
- `ParquetMarketDailyBackend`：从活动月索引解析日分区，使用 Arrow 列裁剪与代码过滤。
- `MarketDailyProvider`：统一代码、字段、日期和 DataView 越界检查，返回
  `field -> 交易日 x 股票` 矩阵。
- `pre_close`、`pct_chg` 与现有矩阵 Provider 使用相同的请求区间内派生语义。

## 验收结果

| 范围 | 代码数 | 字段 | 结果 |
|---|---:|---:|---|
| 固定跨板块样本 | 48 | 7 | Val/Test/Forward 全部精确一致 |
| 全量旧股票池 | 5,332 | 7 | Val/Test/Forward 全部精确一致 |

全量比较覆盖：

- Val：2024-01-01 至 2024-12-31，242 个交易日；
- Test：2025-01-01 至 2025-12-31，243 个交易日；
- Forward：2026-01-01 至 2026-07-29，137 个交易日；
- 比较内容：索引、列、数值、dtype、缺失位置；
- CSV 单次全范围读取：418.36 秒；
- 直接 Parquet 单次全范围读取：395.83 秒。

机器可读证据：`md4_provider_parity_48codes.json` 与
`md4_provider_parity_all_codes.json`。

## 决策

MD4 证明存储和 Provider 行为等价，但直接 Parquet 仍不适合反复构造全市场宽矩阵。
进入 MD5，建立按月、按字段的密集执行缓存；只有 MD5-MD9 和 24-cell ledger parity
全部通过后，才允许切换正式 backend。
