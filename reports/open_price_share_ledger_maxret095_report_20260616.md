# Open-Price Share-Ledger + MaxRet095 回测记录

日期：2026-06-16

项目目录：仓库根目录

## 结论摘要

当前最接近小资金实盘执行的候选版本是：

`V9 avgw3 信号 + 9.5% 信号日涨幅过滤 + open-price share-ledger`

核心结果在 `2025-01-03 ~ 2026-05-18` 的冻结研究区间内表现较强，并且通过了延迟一天、双倍成本压力测试。

| 场景 | 资金 | 收益天数 | 年化 | Sharpe | 最大回撤 | 平均执行换手 | 总成本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 50w | 328 | 72.65% | 2.750 | 13.34% | 0.357 | 11.9% |
| base | 100w | 328 | 77.32% | 2.728 | 13.98% | 0.368 | 10.6% |
| lag1 | 50w | 327 | 65.96% | 2.520 | 12.52% | 0.355 | 12.0% |
| lag1 | 100w | 327 | 77.23% | 2.681 | 13.11% | 0.368 | 10.6% |
| cost2x | 50w | 328 | 61.87% | 2.449 | 13.52% | 0.355 | 20.7% |
| cost2x | 100w | 328 | 64.42% | 2.390 | 14.24% | 0.368 | 20.6% |

对比只到 `2026-04-29` 信号的旧 test 版本：

| 版本 | 资金 | 年化 | Sharpe | 最大回撤 |
|---|---:|---:|---:|---:|
| 旧版，到 2026-04-29 信号 | 50w | 66.81% | 2.609 | 13.34% |
| 新版，到 2026-05-15 信号/2026-05-18 收益 | 50w | 72.65% | 2.750 | 13.34% |
| 旧版，到 2026-04-29 信号 | 100w | 70.85% | 2.580 | 13.98% |
| 新版，到 2026-05-15 信号/2026-05-18 收益 | 100w | 77.32% | 2.728 | 13.98% |

## 策略定义

### 信号

- 基础模型：旧 V9 `checkpoints_exp_topfocus_w005_topic\ultimate_v7_best.pt`
- 信号模式：`avgw3`
- 含义：V9 raw alpha 经过 3 日 average persistence 平滑
- 排名文件：按每日 alpha 从高到低保存全市场股票排序

### MaxRet095 过滤

过滤规则：

- 若股票在信号日的 close-to-close 涨幅 `>= 9.5%`
- 则将该股票降到当日排名末尾
- 目的：避免 open-to-open 或 share-ledger 中追入接近涨停、次日难买或高开风险较高的股票

本次合并后的过滤统计：

- 原始信号文件：`v9_avgw3_extend_to_20260518_20260616\v9_avgw3_test_to_20260515_raw.jsonl`
- 过滤后信号文件：`v9_avgw3_extend_to_20260518_20260616\v9_avgw3_test_to_20260515_maxret095.jsonl`
- 信号日期：`2025-01-02 ~ 2026-05-15`
- 信号天数：328
- demoted 总数：25002

### Open-Price Share-Ledger

这是当前更接近真实小资金实盘的回测方式。

执行逻辑：

- T 日收盘后产生信号
- T+1 日开盘价成交
- 用真实现金、股数、整手约束记账
- 买入卖出使用 open price
- 持仓市值使用 open price mark-to-market
- 保留最低佣金、印花税、滑点、ADV、涨跌停约束
- 保留最低 ADV、单股最大权重、市场状态仓位缩放

核心约束：

- 资金：50w / 100w
- `target_frac=0.006`
- `hold_frac=0.10`
- `max_weight=0.05`
- `min_adv_cny=3,000,000`
- `adv_participation_cap=0.05`
- `rebalance_band=0.20`
- `market_timing_mode=legacy`
- `limit_threshold=0.095`
- 交易单位：100 股
- 最低佣金：5 元

交易成本：

- commission：0.01%
- stamp tax：0.05%，卖出收取
- slippage：0.05%

压力测试：

- `lag1`：额外延迟一天执行
- `cost2x`：佣金、印花税、滑点均翻倍

## 数据边界

冻结研究边界：

- 研究数据截止：`2026-05-18`
- `2026-05-19` 之后保留给真正前向观察

注意：

- 如果使用 `2026-05-18` 信号，则 T+1 开盘成交会落到 `2026-05-19`
- 因此为了让收益序列严格截止 `2026-05-18`，本次只生成到 `2026-05-15` 的信号
- `2026-05-15` 信号对应 `2026-05-18` 开盘执行

核对结果：

- 回测收益序列最后一天为 `2026-05-18`
- open-ledger 使用 `--max-data-date 2026-05-18` 明确截断行情
- 截断版与未截断版在此前核对中结果一致，说明没有使用 `2026-05-19` 之后行情

收益序列：

- 50w：`2025-01-03 ~ 2026-05-18`
- 100w：`2025-01-03 ~ 2026-05-18`

## 文件与命令记录

### 新增脚本

`run\generate_v9_inference_alpha.py`

用途：

- 为没有未来标签的尾部日期生成 V9 推理信号
- 直接构造无标签 inference sample
- 加载同一个 V9 checkpoint
- 输出每日 alpha 排名 JSONL

生成尾段信号命令：

```powershell
python run\generate_v9_inference_alpha.py `
  --checkpoint checkpoints_exp_topfocus_w005_topic\ultimate_v7_best.pt `
  --start-date 2026-04-30 `
  --end-date 2026-05-15 `
  --output v9_avgw3_extend_to_20260518_20260616\v9_tail_20260430_20260515_raw.jsonl `
  --predictor-mode average `
  --window 3 `
  --device cuda `
  --progress-every 3
```

### 合并信号

旧 test 信号：

`backtest_results_test_plan_v9_avgw3_test\v9_daily_alpha_top_order.jsonl`

尾段补充信号：

`v9_avgw3_extend_to_20260518_20260616\v9_tail_20260430_20260515_raw.jsonl`

合并后：

`v9_avgw3_extend_to_20260518_20260616\v9_avgw3_test_to_20260515_raw.jsonl`

### 9.5% 过滤

```powershell
python run\transform_alpha_for_execution.py `
  --input v9_avgw3_extend_to_20260518_20260616\v9_avgw3_test_to_20260515_raw.jsonl `
  --output v9_avgw3_extend_to_20260518_20260616\v9_avgw3_test_to_20260515_maxret095.jsonl `
  --data-dir data\raw `
  --max-signal-return 0.095
```

输出：

```json
{
  "dates": 328,
  "codes": 5178,
  "demoted_total": 25002,
  "stall_demoted_total": 0
}
```

### Open-Price Share-Ledger 回测

基础版本：

```powershell
python run\backtest_retention_open_ledger.py `
  --alpha-jsonl v9_avgw3_extend_to_20260518_20260616\v9_avgw3_test_to_20260515_maxret095.jsonl `
  --output-dir v9_avgw3_open_ledger_20260616\test_maxret095_to_20260518 `
  --target-fracs 0.006 `
  --hold-fracs 0.10 `
  --portfolio-values 500000,1000000 `
  --adv-participation-cap 0.05 `
  --min-adv-cny 3000000 `
  --rebalance-band 0.20 `
  --market-timing-mode legacy `
  --limit-threshold 0.095 `
  --max-data-date 2026-05-18 `
  --progress-every 2500
```

延迟一天：

```powershell
python run\backtest_retention_open_ledger.py `
  --alpha-jsonl v9_avgw3_extend_to_20260518_20260616\v9_avgw3_test_to_20260515_maxret095.jsonl `
  --output-dir v9_avgw3_open_ledger_20260616\test_maxret095_to_20260518_lag1 `
  --target-fracs 0.006 `
  --hold-fracs 0.10 `
  --portfolio-values 500000,1000000 `
  --adv-participation-cap 0.05 `
  --min-adv-cny 3000000 `
  --rebalance-band 0.20 `
  --market-timing-mode legacy `
  --limit-threshold 0.095 `
  --max-data-date 2026-05-18 `
  --execution-lag 1 `
  --progress-every 2500
```

双倍成本：

```powershell
python run\backtest_retention_open_ledger.py `
  --alpha-jsonl v9_avgw3_extend_to_20260518_20260616\v9_avgw3_test_to_20260515_maxret095.jsonl `
  --output-dir v9_avgw3_open_ledger_20260616\test_maxret095_to_20260518_cost2x `
  --target-fracs 0.006 `
  --hold-fracs 0.10 `
  --portfolio-values 500000,1000000 `
  --adv-participation-cap 0.05 `
  --min-adv-cny 3000000 `
  --rebalance-band 0.20 `
  --market-timing-mode legacy `
  --limit-threshold 0.095 `
  --max-data-date 2026-05-18 `
  --commission-rate 0.0002 `
  --stamp-tax-rate 0.001 `
  --slippage-rate 0.001 `
  --progress-every 2500
```

## 结果文件

基础版本：

`v9_avgw3_open_ledger_20260616\test_maxret095_to_20260518\open_ledger_summary.csv`

延迟一天：

`v9_avgw3_open_ledger_20260616\test_maxret095_to_20260518_lag1\open_ledger_summary.csv`

双倍成本：

`v9_avgw3_open_ledger_20260616\test_maxret095_to_20260518_cost2x\open_ledger_summary.csv`

收益序列：

`v9_avgw3_open_ledger_20260616\test_maxret095_to_20260518\returns_pv0050w_target006_hold100.csv`

`v9_avgw3_open_ledger_20260616\test_maxret095_to_20260518\returns_pv0100w_target006_hold100.csv`

诊断文件：

`v9_avgw3_open_ledger_20260616\test_maxret095_to_20260518\diagnostics_pv0050w_target006_hold100.csv`

`v9_avgw3_open_ledger_20260616\test_maxret095_to_20260518\diagnostics_pv0100w_target006_hold100.csv`

## 核对事项

已核对：

- T 日信号对应 T+1 开盘成交
- 首日收益主要体现交易成本，不存在首日白吃涨幅
- 每日 return 与 equity 曲线逐日一致
- `--max-data-date 2026-05-18` 截断后结果可复现
- 5/18 收益来自 5/15 信号在 5/18 开盘执行，不使用 5/18 信号

仍需注意：

- `maxret095` 在 share-ledger 中明显有效，但在宽持仓 open-to-open 中曾经伤害收益，所以它不是所有执行模式通用
- `lag1` 压力测试仍强，但收益有所下降，说明信号有时效性
- 总成本不低，换手约 35% 到 37%，后续应继续关注真实成交滑点和交易频率
- 这仍是历史冻结研究回测，不等于 2026-05-19 之后真实前向收益

## 当前判断

`open-price share-ledger + maxret095` 是目前最值得保留的实盘近似回测版本。

它比旧 close-price share-ledger 更接近真实执行，也比纯 open-to-open 更完整地考虑了小资金账户中的现金、股数、整手、最低佣金、ADV 和涨跌停限制。

下一步建议：

1. 固定这套作为“小资金实盘候选主回测口径”。
2. 后续所有模型和精排器都必须在这套口径下评估。
3. 不再只用 Alpha IC 或 close-based 回测决定模型优劣。
4. 2026-05-19 之后只做前向观察，不再回头调参。
