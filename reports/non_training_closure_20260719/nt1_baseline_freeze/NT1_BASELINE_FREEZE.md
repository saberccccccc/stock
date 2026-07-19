# NT1 正式基线冻结验收

## 结论

NT1 已将 `ledger_path_v3_t0001_nolookahead` 冻结为唯一正式基线，并将
Val 2024、Test 2025 的重复历史证据从排行榜输入中排除。该操作只治理
已有产物，没有训练模型或重新选择参数。

## 冻结合同

- 选择区间：Val 2024、Test 2025。
- 观察区间：Forward 2026，仅观察，不参与选择。
- 资金：50 万、100 万。
- 压力场景：normal、lag1、cost2x、capacity_3pct。
- 执行：realistic open-price share-ledger。
- 信号：registered raw signal。
- 正式合同：`registry/baseline_contract.json`。
- 证据血缘：`registry/evidence_lineage.json`。

## 证据去重

- v4 回放是 Val/Test 的规范实验，共 16 个 Registry 单元格。
- v2、v3 保留作审计历史，但不再进入排行榜。
- v2 与 v4 共有的 8 个 `open_ledger_summary.csv` 哈希全部一致，经济
  结果没有因切换规范血缘而改变。
- Forward 暂时保留 8 条旧 Registry 记录作为观察证据；它们不能参与
  模型选择，将在 NT3 正式 Forward 回放后被替换。
- 最终规范证据共 24 条，`split + stress + capital` 重复键为 0。

## 产物审计

- 冻结清单包含 23 个产物，缺失 0 个。
- baseline contract SHA-256：
  `015adbae751d2f00b3d2fe25abdb40ff3eefabec07ccdcfe655f0fa6c9f4acbf`
- artifact inventory SHA-256：
  `c2c8e7964becd88b6d7e162fd26a909e832dc88690a016271421c0c9377cbbd2`
- evidence lineage SHA-256：
  `087c20f10a205c9d8125e7a7175f6c78e1e8c5649721e21633092243ab1746fd`

## 已知限制

基线底层模型的训练 checkpoint 血缘仍为 `legacy_unresolved`。这不会
阻止当前基于冻结 alpha 的确定性回放，但不能把这份基线描述成可从
原始训练数据完整重建。后续候选必须具备完整模型和数据血缘，不能
继承这一历史豁免。
