# MD5 月度执行缓存验收摘要

状态：通过。正式回测链路尚未切换。

## 设计

- 每个月绑定活动 `month-index SHA-256`，而不是绑定整个根 manifest；
- 每个 generation 先写 staging，再以内容哈希命名并原子切换 `CURRENT`；
- 六个密集字段：`open/high/low/close/volume/money`；
- 跨月拼接后派生 `pre_close/pct_chg`，与现有请求区间语义一致；
- 基础 mask：`valid_ohlc_mask/zero_volume_mask/basic_open_tradable_mask`；
- 基础 mask 不包含 ST、上市天数和板块涨跌停，完整约束仍由执行 Provider 叠加。

## 真实 2026-07 验收

| 项目 | 结果 |
|---|---:|
| 交易日 | 21 |
| 股票 | 5,313 |
| shape | 21 x 5,313 |
| 密集字段大小 | 5,355,504 bytes |
| 初次构建 | 0.155 秒 |
| 热缓存有效性检查 | 0.003 秒 |
| 直接 Parquet 读取 | 13.965 秒 |
| 热月缓存读取 | 0.027 秒 |
| 原始/派生/mask 字段 | 11 |
| 精确 parity | 通过 |
| 文件 SHA-256 审计 | 通过 |

## 失效与恢复

- 同一源月重复构建返回 `already_current`；
- 历史修订只使对应月份失效；
- 其他月份的 CURRENT 字节不变；
- metadata 或字段损坏会被审计拒绝；
- 第二个并发 writer 会被 OS 锁拒绝。

机器可读证据见 `md5_monthly_cache_202607_final.json`。进入 MD6 前不删除旧全局缓存，
不改变正式 backend，也不更改 Registry。
