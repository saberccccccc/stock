# deepseek_optimized - 股票多因子Alpha预测系统

## 项目结构

```
鈹溾攢鈹€ requirements.txt
鈹溾攢鈹€ .gitignore
鈹溾攢鈹€ README.md
鈹溾攢鈹€ __init__.py
鈹溾攢鈹€ _sys_check.ps1                 # GPU/CPU/内存诊断
鈹溾攢鈹€ ultimate_v7_best.pt            # [gitignored] 璁粌浜у嚭鐨勬渶浼樻潈閲?鈹?鈹溾攢鈹€ core/                          # 核心模型
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ config.py                  #   DataConfig 全局配置
鈹?  鈹溾攢鈹€ model.py                   #   UltimateV7Model 涓绘ā鍨?鈹?  鈹溾攢鈹€ model_gat.py               #   GATPredictor 图注意力分支
鈹?  鈹斺攢鈹€ train.py                   #   训练循环 + CrossSectionDataset + collate
鈹?鈹溾攢鈹€ data/                          # 数据
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ raw/                       #   [gitignored] 股票日线CSV
鈹?  鈹溾攢鈹€ stock_industry.csv         #   行业分类 (462KB)
鈹?  鈹溾攢鈹€ pipeline.py                #   build_cross_section_dataset 截面构建
鈹?  鈹溾攢鈹€ market_features.py         #   市场宽度/绂绘暎搴﹁绠?鈹?  鈹溾攢鈹€ fundamental_factors.py     #   鍩烘湰闈㈠洜瀛?(tushare)
鈹?  鈹溾攢鈹€ macro_factors.py           #   宏观因子 (北向资金/两融/PMI)
鈹?  鈹溾攢鈹€ update.py                  #   全量数据更新 (tushare/akshare/baostock)
鈹?  鈹斺攢鈹€ update_daily.py            #   日频增量更新
鈹?鈹溾攢鈹€ backtest/                      # 回测
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ engine.py                  #   澶氬懆鏈熷洖娴?+ 风险预算优化
鈹?  鈹溾攢鈹€ ensemble.py                #   Stacking/Blending 集成
鈹?  鈹斺攢鈹€ risk.py                    #   DynamicRiskBudget 鍔ㄦ€侀鎺?鈹?鈹溾攢鈹€ run/                           # 入口脚本
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ train_v9.py                #   [主入口] V9训练启动
鈹?  鈹溾攢鈹€ train.py                   #   [备用] 鏃х増璁粌鍏ュ彛
鈹?  鈹斺攢鈹€ hyper_search.py            #   瓒呭弬鏁版悳绱?鈹?鈹斺攢鈹€ cache/                         # [gitignored] 杩愯鏃剁紦瀛樼洰褰曪紙鑷姩鐢熸垚锛?```

## 环境要求

- Python 鈮?3.9（torch鈮?.0 瑕佹眰锛?- 推荐 CUDA 鐜锛堣繍琛?`_sys_check.ps1` 妫€鏌?GPU/CPU/鍐呭瓨锛?- Windows / Linux / macOS 均可，但 `_sys_check.ps1` 浠?Windows PowerShell

## 蹇€熷紑濮?
```bash
# 0. 安装依赖
pip install -r requirements.txt

# 1. 鍒濆鍖栵紙棣栨浣跨敤锛屼笅杞芥寚鏁?琛屼笟鏁版嵁锛?python data/update.py --init

# 2. 全量更新行情数据
python data/update.py

# 3. 日频增量更新（日常维护）
python data/update_daily.py

# 4. 璁粌涓绘ā鍨?python run/train_v9.py

# 5. 瓒呭弬鏁版悳绱?python run/hyper_search.py
```

## 关键文件说明

- `ultimate_v7_best.pt`锛氳缁冭繃绋嬩腑淇濆瓨鐨勬渶浼樻潈閲嶄骇鐗╋紝浣撶Н较大（~44MB锛夛紝宸?gitignored锛?*涓嶈鎻愪氦杩涗粨搴?*
- `data/raw/`锛氳偂绁ㄦ棩绾?CSV 原始数据，由 `data/update.py` 写入，已 gitignored
- `cache/`锛氳缁?回测过程中的中间缓存，已 gitignored
- `core/train_utils.py` vs `run/train_legacy.py`：前者是训练循环模块（`Dataset`/`collate`/璁粌鍑芥暟锛夛紝鍚庤€呮槸鏃х増鍏ュ彛鑴氭湰锛涙柊浠ｇ爜璇风敤 `run/train_v9.py`

## 依赖

```
numpy>=1.24, pandas>=2.0, torch>=2.0, lightgbm>=4.0,
scipy>=1.10, akshare>=1.12, baostock>=0.8, tushare>=1.4,
torch-geometric>=2.5, tqdm>=4.65
```
