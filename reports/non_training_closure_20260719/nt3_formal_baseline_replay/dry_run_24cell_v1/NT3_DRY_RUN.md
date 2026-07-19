# NT3 正式基线 24-cell Dry-Run

## 状态

命令编译通过，正式回放尚未启动，Registry 未修改。

## 固定矩阵

- candidate：`ledger_path_v3_t0001_nolookahead`
- split：Val 2024、Test 2025、Forward 2026
- stress：normal、lag1、cost2x、capacity_3pct
- capital：50 万、100 万
- 总数：3 × 4 × 2 = 24 cells
- 执行：realistic open-price share-ledger
- 固定参数：target 0.006、hold 0.10、rebalance band 0.20、max new 5、
  exit hold 0、switch gap 0

## 日期与输入

| split | data root | start | end/max data | alpha |
|---|---|---|---|---|
| val_2024 | data/raw | 2024-01-01 | 2024-12-31 | 已找到 |
| test_2025 | data/raw | 2025-01-01 | 2025-12-31 | 已找到 |
| forward_2026 | data/forward_raw | 2026-01-01 | 2026-06-30 | 已找到 |

Forward 只作为 observation split，不能进入选择。

## 暂停原因

启动前检查得到 C 盘剩余 150.81GB、可用内存 2.66GiB。活动计划规定
可用内存低于 3GiB 时暂停机器任务，因此没有启动正式回放，也没有触发
research/Forward OHLC matrix 重建。待资源门恢复后使用同一冻结命令继续。

历史 ST 仍是已披露的数据降级项，最终 NT3 报告必须保留该声明。
