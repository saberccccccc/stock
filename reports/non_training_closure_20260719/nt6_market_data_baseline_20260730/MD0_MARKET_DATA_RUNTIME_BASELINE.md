# MD0 行情与运行时基准

该报告只读冻结 CSV、OHLC cache 和正式 ledger parity oracle；未重建缓存或运行回测。

## 数据库存

| root | 股票文件 | GiB | 尾部扫描秒 |
|---|---:|---:|---:|
| `C:\Users\x\code\stock_prediction\deepseek_model_exp\data\raw` | 5332 | 0.827 | 51.377 |
| `C:\Users\x\code\stock_prediction\deepseek_model_exp\data\forward_raw` | 5332 | 0.945 | 1.843 |

## CSV 基准

- 区间：2026-01-01 至 2026-07-29
- 文件：5332
- 扫描源字节：0.945 GiB
- 区间行数：725577
- 耗时：22.910 秒
- 吞吐：42.221 MiB/s
- 失败：0

## OHLC Cache

- 路径：`C:\Users\x\code\stock_prediction\deepseek_model_exp\cache\open_ledger_ohlc_matrix`
- 大小：0.954 GiB
- 日期：2010-01-04 至 2026-06-30
- 与当前数据根一致：False
- identity 检查：0.091 秒

## Parity Oracle

- baseline contract：`C:\Users\x\code\stock_prediction\deepseek_model_exp\registry\baseline_contract.json`
- artifact inventory：`C:\Users\x\code\stock_prediction\deepseek_model_exp\reports\non_training_closure_20260719\nt1_baseline_freeze\artifact_inventory.json`
- MD1-MD6 不得改变现有正式 ledger 的订单、成交、阻塞、成本、持仓和净值。
