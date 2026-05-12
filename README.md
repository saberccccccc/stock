# deepseek_optimized - 鑲＄エ澶氬洜瀛怉lpha棰勬祴绯荤粺

## 椤圭洰缁撴瀯

```
鈹溾攢鈹€ requirements.txt
鈹溾攢鈹€ .gitignore
鈹溾攢鈹€ README.md
鈹溾攢鈹€ __init__.py
鈹溾攢鈹€ _sys_check.ps1                 # GPU/CPU/鍐呭瓨璇婃柇
鈹溾攢鈹€ ultimate_v7_best.pt            # [gitignored] 璁粌浜у嚭鐨勬渶浼樻潈閲?鈹?鈹溾攢鈹€ core/                          # 鏍稿績妯″瀷
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ config.py                  #   DataConfig 鍏ㄥ眬閰嶇疆
鈹?  鈹溾攢鈹€ model.py                   #   UltimateV7Model 涓绘ā鍨?鈹?  鈹溾攢鈹€ model_gat.py               #   GATPredictor 鍥炬敞鎰忓姏鍒嗘敮
鈹?  鈹斺攢鈹€ train.py                   #   璁粌寰幆 + CrossSectionDataset + collate
鈹?鈹溾攢鈹€ data/                          # 鏁版嵁
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ raw/                       #   [gitignored] 鑲＄エ鏃ョ嚎CSV
鈹?  鈹溾攢鈹€ stock_industry.csv         #   琛屼笟鍒嗙被 (462KB)
鈹?  鈹溾攢鈹€ pipeline.py                #   build_cross_section_dataset 鎴潰鏋勫缓
鈹?  鈹溾攢鈹€ market_features.py         #   甯傚満瀹藉害/绂绘暎搴﹁绠?鈹?  鈹溾攢鈹€ fundamental_factors.py     #   鍩烘湰闈㈠洜瀛?(tushare)
鈹?  鈹溾攢鈹€ macro_factors.py           #   瀹忚鍥犲瓙 (鍖楀悜璧勯噾/涓よ瀺/PMI)
鈹?  鈹溾攢鈹€ update.py                  #   鍏ㄩ噺鏁版嵁鏇存柊 (tushare/akshare/baostock)
鈹?  鈹斺攢鈹€ update_daily.py            #   鏃ラ澧為噺鏇存柊
鈹?鈹溾攢鈹€ backtest/                      # 鍥炴祴
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ engine.py                  #   澶氬懆鏈熷洖娴?+ 椋庨櫓棰勭畻浼樺寲
鈹?  鈹溾攢鈹€ ensemble.py                #   Stacking/Blending 闆嗘垚
鈹?  鈹斺攢鈹€ risk.py                    #   DynamicRiskBudget 鍔ㄦ€侀鎺?鈹?鈹溾攢鈹€ run/                           # 鍏ュ彛鑴氭湰
鈹?  鈹溾攢鈹€ __init__.py
鈹?  鈹溾攢鈹€ train_v9.py                #   [涓诲叆鍙 V9璁粌鍚姩
鈹?  鈹溾攢鈹€ train.py                   #   [澶囩敤] 鏃х増璁粌鍏ュ彛
鈹?  鈹斺攢鈹€ hyper_search.py            #   瓒呭弬鏁版悳绱?鈹?鈹斺攢鈹€ cache/                         # [gitignored] 杩愯鏃剁紦瀛樼洰褰曪紙鑷姩鐢熸垚锛?```

## 鐜瑕佹眰

- Python 鈮?3.9锛坱orch鈮?.0 瑕佹眰锛?- 鎺ㄨ崘 CUDA 鐜锛堣繍琛?`_sys_check.ps1` 妫€鏌?GPU/CPU/鍐呭瓨锛?- Windows / Linux / macOS 鍧囧彲锛屼絾 `_sys_check.ps1` 浠?Windows PowerShell

## 蹇€熷紑濮?
```bash
# 0. 瀹夎渚濊禆
pip install -r requirements.txt

# 1. 鍒濆鍖栵紙棣栨浣跨敤锛屼笅杞芥寚鏁?琛屼笟鏁版嵁锛?python data/update.py --init

# 2. 鍏ㄩ噺鏇存柊琛屾儏鏁版嵁
python data/update.py

# 3. 鏃ラ澧為噺鏇存柊锛堟棩甯哥淮鎶わ級
python data/update_daily.py

# 4. 璁粌涓绘ā鍨?python run/train_v9.py

# 5. 瓒呭弬鏁版悳绱?python run/hyper_search.py
```

## 鍏抽敭鏂囦欢璇存槑

- `ultimate_v7_best.pt`锛氳缁冭繃绋嬩腑淇濆瓨鐨勬渶浼樻潈閲嶄骇鐗╋紝浣撶Н杈冨ぇ锛垀44MB锛夛紝宸?gitignored锛?*涓嶈鎻愪氦杩涗粨搴?*
- `data/raw/`锛氳偂绁ㄦ棩绾?CSV 鍘熷鏁版嵁锛岀敱 `data/update.py` 鍐欏叆锛屽凡 gitignored
- `cache/`锛氳缁?鍥炴祴杩囩▼涓殑涓棿缂撳瓨锛屽凡 gitignored
- `core/train_utils.py` vs `run/train_legacy.py`锛氬墠鑰呮槸璁粌寰幆妯″潡锛坄Dataset`/`collate`/璁粌鍑芥暟锛夛紝鍚庤€呮槸鏃х増鍏ュ彛鑴氭湰锛涙柊浠ｇ爜璇风敤 `run/train_v9.py`

## 渚濊禆

```
numpy>=1.24, pandas>=2.0, torch>=2.0, lightgbm>=4.0,
scipy>=1.10, akshare>=1.12, baostock>=0.8, tushare>=1.4,
torch-geometric>=2.5, tqdm>=4.65
```
