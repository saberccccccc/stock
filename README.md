# deepseek_optimized - 股票多因子Alpha预测系统

## 项目结构

```
├── requirements.txt
├── .gitignore
├── README.md
├── __init__.py
├── _sys_check.ps1                 # GPU/CPU/内存诊断
├── ultimate_v7_best.pt            # [gitignored] 训练产出的最优权重
├── core/                          # 核心模型
│   ├── __init__.py
│   ├── config.py                  #   DataConfig 全局配置
│   ├── model.py                   #   UltimateV7Model 主模型
│   ├── model_gat.py               #   GATPredictor 图注意力分支
│   └── train.py                   #   训练循环 + CrossSectionDataset + collate
├── data/                          # 数据
│   ├── __init__.py
│   ├── raw/                       #   [gitignored] 股票日线CSV
│   ├── stock_industry.csv         #   行业分类 (462KB)
│   ├── pipeline.py                #   build_cross_section_dataset 截面构建
│   ├── market_features.py         #   市场宽度/离散度计算
│   ├── fundamental_factors.py     #   基本面因子(tushare)
│   ├── macro_factors.py           #   宏观因子 (北向资金/两融/PMI)
│   ├── update.py                  #   全量数据更新 (tushare/akshare/baostock)
│   └── update_daily.py            #   日频增量更新
├── backtest/                      # 回测
│   ├── __init__.py
│   ├── engine.py                  #   多周期回测+风险预算优化
│   ├── ensemble.py                #   Stacking/Blending 集成
│   └── risk.py                    #   DynamicRiskBudget 动态风控
├── run/                           # 入口脚本
│   ├── __init__.py
│   ├── train_v9.py                #   [主入口] V9训练启动
│   ├── train.py                   #   [备用] 旧版训练入口
│   └── hyper_search.py            #   超参数搜索
└── cache/                         # [gitignored] 运行时缓存目录（自动生成）
```

## 环境要求

- Python >= 3.9（torch>=2.0 要求）
- 推荐 CUDA 环境（运行 `_sys_check.ps1` 检查 GPU/CPU/内存）
- Windows / Linux / macOS 均可，但 `_sys_check.ps1` 仅 Windows PowerShell

## 快速开始
```bash
# 0. 安装依赖
pip install -r requirements.txt

# 1. 初始化（首次使用，下载指数/行业数据）
python data/update.py --init

# 2. 全量更新行情数据
python data/update.py

# 3. 日频增量更新（日常维护）
python data/update_daily.py

# 4. 训练主模型
python run/train_v9.py

# 5. 超参数搜索
python run/hyper_search.py
```

## 关键文件说明

- `ultimate_v7_best.pt`：训练过程中保存的最优权重产物，体积较大（~44MB），已 gitignored，**不要提交进仓库**
- `data/raw/`：股票日线 CSV 原始数据，由 `data/update.py` 写入，已 gitignored
- `cache/`：训练/回测过程中的中间缓存，已 gitignored
- `core/train_utils.py` vs `run/train_legacy.py`：前者是训练循环模块（`Dataset`/`collate`/训练函数），后者是旧版入口脚本；新代码请用 `run/train_v9.py`

## 依赖

```
numpy>=1.24, pandas>=2.0, torch>=2.0, lightgbm>=4.0,
scipy>=1.10, akshare>=1.12, baostock>=0.8, tushare>=1.4,
torch-geometric>=2.5, tqdm>=4.65
```
