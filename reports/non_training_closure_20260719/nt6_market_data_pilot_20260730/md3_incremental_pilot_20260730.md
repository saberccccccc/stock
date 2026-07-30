# MD3 每日 Parquet 增量更新验收摘要

状态：通过。试运行使用隔离临时库，正式 CSV backend 与 MD2 全历史候选库均未修改。

## 真实试运行

| 项目 | 结果 |
|---|---|
| 交易日 | 2026-07-29 |
| 股票来源 | `tushare.daily` |
| 股票行数 | 5,524 |
| 宽基来源 | `akshare.stock_zh_index_daily` |
| 宽基代码 | `000016.SH, 000300.SH, 000905.SH, 399006.SZ` |
| 宽基行数 | 4 |
| 首次提交 generation | 2 |
| 首次提交 manifest SHA-256 | `14294f842388bf096e06b726b85cf3eeb399d73d6d0c907ba3e7b8f914124689` |
| 独立进度重放 | 股票与指数均为 `already_present` |
| 重放后 generation/hash | 未变化 |
| 全量物理哈希审计 | 通过 |

## 接口决策

Tushare `index_daily` 必须逐个传入指数代码。当前账号实测为每分钟一次，四个指数逐个调用
不是高效的默认路径。因此正式候选 updater 默认使用 Tushare 下载全市场股票日线，使用项目
既有的 AkShare 宽基接口下载四个指数；高权限账号仍可显式选择 Tushare 指数模式。

AkShare 宽基接口不提供成交额。为保持统一 schema，`money` 写入 0，同时进度 manifest 强制
记录 `index_money_semantics=source_unavailable_filled_zero`。该字段不得解释为真实零成交，也不应
进入依赖指数成交额的特征。

## 门禁结论

- 重复运行 no-op：通过。
- 进度锁与恢复：通过。
- 历史修订必须显式允许：通过。
- 股票与宽基最新日期对齐或记录缺口：通过。
- 活动 Parquet 哈希、schema、行数：通过。
- 正式 backend 切换：未执行，等待 MD4-MD9。
