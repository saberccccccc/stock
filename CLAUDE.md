# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## PowerShell 编码注意事项

**所有 PowerShell 中读写 .py 文件的操作必须加 `-Encoding UTF8`：**

```powershell
# 正确
Get-Content -LiteralPath "file.py" -Raw -Encoding UTF8 | Set-Content -LiteralPath "file.py" -Encoding UTF8

# 错误（中文 Windows 默认用 GBK，会乱码）
Get-Content -LiteralPath "file.py" -Raw | Set-Content -LiteralPath "file.py" -Encoding UTF8
```

缺失 `-Encoding UTF8` 会导致 UTF-8 中文文件被当作 GBK 读取，产生不可逆的乱码损坏。Python 3 本身默认 UTF-8 无此问题。

## Environment and commands

Use the existing Miniconda environment when running project code on this machine:

```bash
F:/miniconda3/envs/pytorch/python -m py_compile core/config.py core/model.py data/pipeline.py data/fundamental_factors.py data/macro_factors.py data/update.py data/update_daily.py backtest/engine.py backtest/ensemble.py run/hyper_search.py
F:/miniconda3/envs/pytorch/python run/train_v9.py
F:/miniconda3/envs/pytorch/python run/train_gat.py
F:/miniconda3/envs/pytorch/python run/hyper_search.py
```

The environment currently has PyTorch `2.7.1+cu118` with CUDA available on an RTX 2060-class 6GB GPU. If dependencies need to be installed, use:

```bash
F:/miniconda3/envs/pytorch/python -m pip install -r requirements.txt
```

Data update commands require a Tushare token supplied outside source code:

```bash
export TUSHARE_TOKEN="<token>"
F:/miniconda3/envs/pytorch/python data/update.py
F:/miniconda3/envs/pytorch/python data/update_daily.py
F:/miniconda3/envs/pytorch/python data/update.py --init
```

## Architecture overview

- `run/` contains entry scripts. `run/train_v9.py` is the main training entry; `run/train_legacy.py` is legacy/backup; `run/hyper_search.py` runs LightGBM/backtest parameter search.
- `core/` contains the deep model and training loop. `core/model.py` defines `UltimateV7Model`; `core/train_utils.py` defines `CrossSectionDataset`, collate functions, loss/evaluation, and `train_model`.
- `data/` builds the cross-sectional dataset. `data/pipeline.py` loads stock CSVs from `data/raw/`, builds multi-scale features, joins market/fundamental/macro factors, creates per-date cross-section samples, and returns `(train_samples, val_samples)`.
- `data/fundamental_factors.py` must remain point-in-time: financial statement factors are keyed by announcement/effective date, not report `end_date`, and are aligned to daily dates using only already-published data.
- `data/macro_factors.py` must keep macro features point-in-time. PMI is treated as available from the following month, not the month start, and z-scores use historical expanding/rolling statistics rather than full-sample statistics.
- `backtest/engine.py` trains/loads multi-horizon LightGBM models and runs production backtests. Portfolio state is tracked as current weights; costs must be deducted from daily returns.
- `backtest/ensemble.py` handles stacking/blending. Stacking features should be deterministic per-stock LightGBM predictions, not random placeholders.

## Data and artifact notes

- `data/raw/`, `cache/`, logs, and model weights are gitignored runtime artifacts.
- After changing label alignment, feature flags, PIT factor logic, or market/fundamental/macro factor logic, rebuild cross-section caches. Old `cache/cross_section_*.pkl` and `cache/fundamental_features*.parquet` may be incompatible or biased.
- `ultimate_v7_best.pt` is a training artifact. If labels or PIT features change, consider the old checkpoint invalid and retrain from scratch.
- `run/train_v9.py` currently defaults to `test_mode=True` and `test_stocks=1000` for smoke training before full-universe training.

## Training configuration

**Mixed precision (AMP) must be disabled.** PyTorch AMP causes `Train Loss: nan` in this project due to numerical instability in the correlation-based loss function. Always use `use_amp=False` when calling `train_model()` in `core/train_utils.py`. The current `run/train_v9.py` correctly disables AMP.

**GPU memory management:** For RTX 2060 (6GB), use `batch_size=2` with `accum_steps=8` (effective batch size 16) to prevent OOM crashes during training. Test mode with 100 stocks is recommended for development before full-universe training.

## Training workflow (CRITICAL)

**ALWAYS check for existing training processes before starting a new one.** Multiple concurrent training processes will compete for GPU memory and cause crashes or inconsistent results.

Before starting any training:
1. Check for running training processes: `ps aux | grep -E "python.*(train|run)" | grep -v grep`
2. Stop any existing training processes if found
3. Delete old log files if starting fresh: `rm -f *.log`
4. Only then start the new training process

**Why this matters:** Training processes can run for hours in the background. Starting a new training without stopping the old one leads to:
- GPU OOM errors
- Multiple log files being written simultaneously
- Inconsistent or corrupted training results
- Wasted compute resources

## Industry embedding (2026-05-10)

**Implementation:** The model now includes industry embedding to capture industry-specific patterns. Each stock's industry ID (from 璇佺洃浼氳涓氬垎绫? 82 industries + 1 unknown) is mapped to a 16-dimensional learnable embedding vector, concatenated with the feature encoding, then projected back to hidden_dim (256).

**Key changes:**
- `data/pipeline.py`: Samples now include `industry_ids` field (int16 array, -1 for unknown industry)
- `core/train_utils.py`: Dataset and collate functions handle `industry_ids`; model calls pass `industry_ids` parameter
- `core/model.py`: Added `industry_embed` (nn.Embedding) and `industry_proj` (Linear 272鈫?56); forward maps -1 to index 82 (unknown industry)
- `run/train_v9.py`: Passes `num_industries=82` to `train_model()`

**Cache invalidation:** After adding industry embedding, old cache files (`cache/cross_section_*.pkl`) are incompatible and must be deleted. The pipeline will rebuild caches with `industry_ids` included.

**Model parameters:** Industry embedding adds ~71K parameters (83 industries 脳 16 dims + projection layer 272脳256).

## Bug淇璁板綍 (2026-05-10)

**鏁版嵁璐ㄩ噺淇锛堥渶瑕佸垹闄ょ紦瀛橀噸寤猴級锛?*
1. **market_features.py (L174, L188)**: 淇鏀剁泭鐜囧～鍏呴€昏緫锛岀Щ闄ら敊璇殑ffill()鍓嶅悜濉厖锛岀己澶辨棩鏈熺洿鎺ュ～0
2. **fundamental_factors.py (L117-126)**: 淇鍩烘湰闈erge閲嶅闂锛屾坊鍔犲幓閲嶉€昏緫淇濈暀姣忎釜(ts_code, end_date)鐨勬渶鏂癮nn_date

**璁粌閰嶇疆淇锛?*
3. **train.py (L249)**: 淇weight_decay榛樿鍙傛暟涓嶄竴鑷达紝浠?e-4鏀逛负2e-3
4. **train_v9.py (L131)**: 淇澶囦唤鏂囦欢瑕嗙洊闂锛屾坊鍔犳椂闂存埑閬垮厤瑕嗙洊鍘嗗彶澶囦唤

**鍥炴祴閫昏緫淇锛?*
5. **engine.py (L398-402)**: 淇鎸佷粨鏈湪exit_day骞充粨bug锛屾坊鍔犲钩浠撻€昏緫纭繚鎸佷粨鏈熺鍚坒uture_len鍙傛暟

**闆嗘垚妯″瀷淇锛?*
6. **ensemble.py (L108)**: 淇None鍊兼鏌ワ紝閬垮厤瀵筃one璋冪敤len()鎶ypeError
7. **ensemble.py (L200)**: 淇妯″瀷绫诲悕閿欒锛屼粠涓嶅瓨鍦ㄧ殑UltimateV5Model鏀逛负UltimateV7Model
8. **ensemble.py (L97-153, L217, L246)**: 娣诲姞industry_ids鏀寔锛屽湪StackingDataset銆乧ollate鍜屾ā鍨嬭皟鐢ㄤ腑瀹屾暣瀹炵幇

**鍥炴祴椋庢帶淇锛?*
9. **risk.py (L98-100)**: 淇IC琛板噺鍒ゆ柇閫昏緫锛屾墍鏈塈C鈮?.5鏃朵笉鍐嶈鍒や负蹇€熻“鍑?
**鏃х増璁粌鑴氭湰淇锛?*
10. **train.py (L98)**: 淇AMP鍚敤閿欒锛屼粠use_amp=True鏀逛负False閬垮厤Train Loss: nan
11. **train.py (L98)**: 娣诲姞缂哄け鐨刵um_industries=82鍙傛暟
12. **train.py (L26)**: 淇鏃ュ織鏂囦欢鍚嶄粠train_v9.log鏀逛负train.log

**缂撳瓨澶辨晥璇存槑锛?* Bug 1鍜?褰卞搷鏁版嵁璐ㄩ噺锛屽繀椤诲垹闄cache/cross_section_*.pkl`鍜宍cache/fundamental_features*.parquet`鍚庨噸鏂拌缁冦€?
## 鐗瑰緛宸ョ▼鏀硅繘璁板綍

### 宸插疄鏂芥敼杩?
1. **鍩虹鐗瑰緛鍘诲啑浣?*: 灏嗗師鏉ュ涓珮搴︾浉鍏崇殑鏀剁泭鐜囩壒寰佺簿绠€涓篳ret_5d`銆乣ret_20d`锛屼繚鐣欑煭涓湡鍔ㄩ噺浣嗛檷浣庨噸澶嶄俊鎭?2. **鏂板寰缁撴瀯鐗瑰緛**: 娣诲姞`upper_shadow`銆乣lower_shadow`銆乣body_size`銆乣gap`銆乣amplitude`锛屾崟鎹変笂涓嬪奖绾裤€佸疄浣撱€佽烦绌哄拰鎸箙
3. **鏂板褰掍竴鍖栨妧鏈寚鏍?*: 鍚敤`cfg.use_technical_features=True`锛屽姞鍏sma*_gap`銆乣ema*_gap`銆乣rsi_norm`銆乣macd*_pct`銆乣atr_pct`銆乣volume_ratio`锛屼笉鍔犲叆鍘熷SMA/EMA/MACD浠锋牸灏哄害
4. **淇濈暀鎴潰rank鐗瑰緛**: 姣忎釜鑱氬悎鐗瑰緛鍚屾椂鐢熸垚鎴潰鎺掑悕鐗瑰緛锛屼綔涓虹浉瀵瑰己搴﹁ˉ鍏咃紝涓嶆浛浠ｅ師濮嬬壒寰?5. **淇濈暀琛屼笟鐩稿鐗瑰緛**: 浠呭鏍稿績last鑱氬悎鍥犲瓙璁＄畻琛屼笟鍘诲潎鍊硷紝閬垮厤缁村害杩囬珮
6. **淇濈暀MAD绋冲仴鏍囧噯鍖?*: 瀵规嫾鎺ュ悗鐨刋鐗瑰緛鍋氭埅闈AD鏍囧噯鍖栵紝闄嶄綆鏋佺鍊煎奖鍝?7. **绉婚櫎纭紪鐮佺淮搴?*: `run/train_v9.py`銆乣run/train_legacy.py`銆乣core/train_utils.py`鍧囦娇鐢╜INDUSTRY_REL_FEATURES`鍔ㄦ€佽绠楄涓氱浉瀵圭淮搴?
### 褰撳墠鐗瑰緛缁撴瀯

鍩虹鐗瑰緛鍏?3涓細
- 鍔ㄩ噺/娉㈠姩: `ret_5d`, `ret_20d`, `vol_10d`, `vol_60d`, `price_momentum`
- 鎴愪氦閲? `log_volume`, `volume_spike`, `volume_ratio`
- 寰缁撴瀯: `upper_shadow`, `lower_shadow`, `body_size`, `gap`, `amplitude`
- 褰掍竴鍖栨妧鏈寚鏍? `sma5_gap`, `sma10_gap`, `sma20_gap`, `ema12_gap`, `ema26_gap`, `rsi_norm`, `macd_pct`, `macd_signal_pct`, `macd_diff_pct`, `atr_pct`

X缁村害鍏紡锛?- 鑱氬悎鐗瑰緛: `base_feat_dim 脳 N_AGGS = 23 脳 5 = 115`
- Rank鐗瑰緛: `115`
- 琛屼笟鐩稿鐗瑰緛: `len(INDUSTRY_REL_FEATURES) = 6`
- 鎬荤淮搴? `115 + 115 + 6 = 236`

### 鎴潰鎺掑悕鐗瑰緛璁捐鍘熷垯

**鍏抽敭鍐崇瓥**: 鍚屾椂淇濈暀鍘熷鐗瑰緛鍜屾帓鍚嶇壒寰侊紝鑰岄潪鏇夸唬
- 鍘熷鐗瑰緛淇濈暀缁濆閲忕骇淇℃伅
- 鎺掑悕鐗瑰緛鎻愪緵鐩稿寮哄害淇℃伅
- 琛屼笟鐩稿鐗瑰緛鎻愪緵鍚岃涓氬唴姣旇緝
- 璁㏕ransformer鑷姩瀛︿範涓嶅悓鐗瑰緛鐨勯噸瑕佹€?
### 瀹忚/璧勯噾娴佺壒寰佹帴鍏?
**褰撳墠瀹炵幇:** `run/train_v9.py`鍚敤`cfg.use_macro_features=True`锛屽畯瑙傜壒寰佷笉杩涘叆鑲＄エX鐗瑰緛锛岃€屾槸杩涘叆`risk/regime`甯傚満鐘舵€侀€氶亾銆?
**鍘熷洜:** 鍖楀悜璧勯噾銆佽瀺璧勮瀺鍒搞€丳MI鏄叏甯傚満鍚屽€肩壒寰侊紝濡傛灉鍔犲叆姣忓彧鑲＄エ鐨刋鐗瑰緛锛屼細鍦ㄦ埅闈AD鏍囧噯鍖栦腑琚姽鎴?锛屽苟涓攔ank鐗瑰緛浼氬鐩稿悓鍊间骇鐢熸棤鎰忎箟鎺掑悕銆?
**缁村害:**
- risk杩炵画鐗瑰緛: `3鑲＄エ绾?+ 81甯傚満鏁翠綋 + 3瀹忚 = 87`
- 81缁村競鍦烘暣浣撳寘鍚細16涓鍩烘寚鏁扮壒寰?+ 3涓競鍦哄搴︾壒寰?+ 31涓敵涓囪涓氭敹鐩?+ 31涓敵涓囪涓氬彲鐢ㄦ€ask
- 琛屼笟one-hot: `83`
- risk鎬荤淮搴? `170`
- 妯″瀷浣跨敤鍓峘get_regime_dim(cfg)=87`缁翠綔涓哄競鍦虹姸鎬佽緭鍏?
**鏃ユ湡瀵归綈鍘熷垯:**
- 鍖楀悜璧勯噾浠?014-11-17寮€濮嬫槸鍒跺害/鏁版嵁杈圭晫锛?010-2014涓嶈兘鍚戝悗鍥炲～鏈潵鏁版嵁锛屽簲濉?琛ㄧず涓嶅彲鐢?涓€?- 铻嶈祫铻嶅埜鏁版嵁瑕嗙洊鍒?010-03-31锛屽彲瑕嗙洊鑲＄エ鍘嗗彶涓讳綋鍖洪棿
- PMI浣跨敤`pmi_pit_v2.csv`锛屾寜淇濆畧鐢熸晥鏃ュ榻愶紝鍓嶆湡zscore鍐峰惎鍔ㄧ己澶卞～0
- 淇敼瀹忚閫昏緫鍚庡繀椤诲垹闄cache/cross_section_*.pkl`閲嶅缓缂撳瓨

### 琛屼笟embedding鍘熷垯

**base琛屼笟涓嶇Щ闄ゃ€?* 绉婚櫎base琛屼笟鍙€傚悎绾挎€фā鍨?one-hot閬垮厤铏氭嫙鍙橀噺闄烽槺锛沞mbedding鏄煡琛ㄥ涔狅紝涓嶅瓨鍦ㄨ櫄鎷熷彉閲忓叡绾挎€ч棶棰樸€?
**褰撳墠瀹炵幇:**
- `data/pipeline.py`淇濈暀鍏ㄩ儴83涓湡瀹炶涓氾紝琛屼笟ID鑼冨洿涓篳0..82`
- 鏈煡琛屼笟涓篳-1`锛岃繘鍏ユā鍨嬫椂鏄犲皠鍒癭num_industries`绱㈠紩
- `UltimateV7Model`浣跨敤`nn.Embedding(num_industries + 1, 16)`锛屽嵆83涓湡瀹炶涓?+ 1涓湭鐭ヨ涓?- `run/train_v9.py`鐢╮isk閲岀殑琛屼笟one-hot缁村害鎺ㄦ柇`num_industries`锛岄伩鍏嶅皬鏍锋湰娌¤鐩栧叏琛屼笟鏃秂mbedding杩囧皬

### 棰濆Bug淇璁板綍

1. **backtest/ensemble.py**: 淇`regime_dim=53`銆乣num_industries=82`纭紪鐮侊紝鏀逛负鍔ㄦ€佷娇鐢╜get_regime_dim(cfg)`鍜宺isk缁村害
2. **backtest/ensemble.py**: 淇楠岃瘉闆哠tackingDataset璇诲彇LGB CV棰勬祴鏃剁己灏憈rain offset鐨勯棶棰?3. **backtest/ensemble.py**: 淇Stacking妯″瀷`base_feat_dim=stacking_dim//2`閿欒锛屾敼涓烘寜鍘熷X缁村害鍜宍N_AGGS`璁＄畻
4. **backtest/engine.py / run/hyper_search.py**: 鍚敤鎶€鏈寚鏍囧拰瀹忚鐗瑰緛锛屽苟鏀圭敤`models_multi_v9_tech_macro`閬垮厤璇姞杞芥棫LightGBM妯″瀷
5. **data/pipeline.py**: 缂撳瓨鏂囦欢鍚嶄娇鐢╜CACHE_VERSION`锛岄伩鍏嶇増鏈父閲忓拰瀹為檯鏂囦欢鍚嶈劚鑺?6. **core/train_utils.py**: 淇alpha澶粹€滃鏍锋€ф鍒欌€濇柟鍚戦敊璇紝鏀逛负鎯╃綒alpha澶翠箣闂寸浉鍏虫€э紝閬垮厤澶氬ご濉岀缉
7. **core/train_utils.py**: 淇楠岃瘉闃舵璇悶鎵€鏈塕untimeError鐨勯棶棰橈紝浠匫OM鏃惰烦杩囷紝鍚﹀垯鎶涘嚭鐪熷疄閿欒
8. **core/train_utils.py**: 淇resume鍔犺浇鏃heckpoint缁撴瀯涓嶅吋瀹规椂纭穿锛屾敼涓哄拷鐣ユ棫checkpoint骞朵粠澶磋缁?9. **run/train_v9.py / run/train_legacy.py**: 淇Tee鏃ュ織鍏抽棴鍚庢湭鎭㈠stdout/stderr瀵艰嚧閫€鍑烘椂鍙兘鍐欏叆宸插叧闂枃浠剁殑闂
10. **core/config.py / data/pipeline.py**: 娣诲姞`tushare_token`閰嶇疆骞舵敮鎸佺幆澧冨彉閲廯TUSHARE_TOKEN`锛屼慨澶嶅熀鏈潰鐗瑰緛寮€鍏虫棤娉曟甯稿惎鐢ㄧ殑闂
11. **data/pipeline.py**: 鎴潰缂撳瓨key鍔犲叆`stock_universe`鎽樿锛屽苟鍥哄畾CSV鏂囦欢鎺掑簭锛岄伩鍏嶅悓鍚嶇紦瀛樺搴斾笉鍚岃偂绁ㄩ泦鍚?12. **data/update.py**: 淇Tushare澶氱嚎绋嬮檺娴侀潪绾跨▼瀹夊叏闂锛屽苟灏哷--init`鎸囨暟/琛屼笟鏂囦欢杈撳嚭鍒皃ipeline瀹為檯璇诲彇璺緞
13. **data/update.py**: 鑲＄エ姹犳瀯寤虹撼鍏ヤ笂甯傘€侀€€甯傘€佹殏鍋滀笂甯傝偂绁紝缂撹В浠呬娇鐢ㄥ綋鍓嶄笂甯傝偂绁ㄥ鑷寸殑骞稿瓨鑰呭亸宸?14. **data/market_features.py**: 缂哄皯瀹藉熀/鐢充竾琛屼笟鎸囨暟鏂囦欢鏃舵墦鍗版槑纭鍛婏紝閬垮厤甯傚満鐗瑰緛鍏ㄩ浂闈欓粯鍙戠敓
15. **backtest/engine.py**: 淇鍥炴祴鎴愪氦鏃ユ敹鐩婂綊鍥爈ookahead銆丄DV鍗曚綅涓嶄竴鑷淬€佸姩鎬佽皟浠揑C棣栭」涓?鏃堕櫎闆讹紝浠ュ強鎸囨暟鏂囦欢璇诲彇璺緞閿欒
16. **backtest/ensemble.py**: 淇涓€闃舵LGB鍦ㄩ獙璇佹湡鍐呮粴鍔ㄨ缁冨鑷翠簩闃舵楠岃瘉姹℃煋鐨勯棶棰?17. **data/market_features.py / core/train_utils.py**: 涓?1涓敵涓囪涓氭寚鏁板姞鍏sw_xxxxxx_available`鍙敤鎬ask锛屽競鍦虹姸鎬佺淮搴︿粠50澧炶嚦81锛屽畯瑙傚紑鍚椂`regime_dim=87`
18. **data/pipeline.py**: 淇鎶€鏈寚鏍?寰缁撴瀯鐗瑰緛鍦ㄥ仠鐗屾垨寮傚父浠锋牸涓嬩骇鐢熸瀬绔€煎鑷碻loss=nan`鐨勯棶棰橈紱鏀剁泭鐜囥€乬ap銆丮ACD/ATR姣斾緥绛夊仛winsorize锛屾渶缁圶鍋歚clip[-10,10]`锛宺isk鍓嶄笁缁翠篃鍋歝lip

### 缂撳瓨鏂囦欢鍛藉悕瑙勮寖

**闂**: hash鍛藉悕锛堝`cross_section_a11d234f38fb.pkl`锛夊彲璇绘€у樊

**鏀硅繘**: 浣跨敤鎻忚堪鎬у懡鍚?- 鏍煎紡: `cross_section_v{version}_{n_stocks}stocks_{features}_seq{seq_len}_h{horizon}.pkl`
- 绀轰緥: `cross_section_v8_allstocks_market_seq40_h5.pkl`
- 鍖呭惈鐗堟湰鍙枫€佽偂绁ㄦ暟銆佸惎鐢ㄧ殑鐗瑰緛绫诲瀷銆佸簭鍒楅暱搴︺€侀娴嬪懆鏈?
### 鍚庣画鍙€夋敼杩?/ 寰呭姙

#### 鐗瑰緛涓庢暟鎹?
- 甯傚満鐗瑰緛闄嶇淮锛?1涓涓氭寚鏁版敹鐩?+ 31涓彲鐢ㄦ€ask 鍙€冭檻PCA銆佺瓫閫夊叧閿涓氾紝鎴栧彧瀵规敹鐩婂垪闄嶇淮銆乵ask淇濈暀鍘熷褰㈡€?- **鎵╁睍鍩烘湰闈?2鍥犲瓙锛堜笅涓€闃舵瀹為獙锛?*锛氱洰鍓嶅彧鏈塦roe`銆乣revenue_yoy`銆乣pe_percentile`銆傚缓璁厛璁╁綋鍓峍10璁粌璺戝畬骞惰褰昩aseline锛屽啀瀹炵幇鍩烘湰闈?2鍥犲瓙锛屽垹闄cache/cross_section_*.pkl`鍜宍cache/fundamental_features*.parquet`鍚庨噸璁紝瀵规瘮楠岃瘉IC鍜屽洖娴嬭〃鐜般€?  - 鐩爣鍥犲瓙锛歚roe`, `roa`, `gross_margin`, `net_margin`, `revenue_yoy`, `profit_yoy`, `pe_percentile`, `pb_percentile`, `ps_percentile`, `debt_to_assets`, `current_ratio`, `ocf_to_profit`
  - 鍥犲瓙鍒嗙粍锛氱泩鍒╄兘鍔涳紙`roe`, `roa`, `gross_margin`, `net_margin`锛夈€佹垚闀挎€э紙`revenue_yoy`, `profit_yoy`锛夈€佷及鍊硷紙`pe_percentile`, `pb_percentile`, `ps_percentile`锛夈€佽储鍔¤川閲忥紙`debt_to_assets`, `current_ratio`, `ocf_to_profit`锛?  - 鏁版嵁鏉ユ簮锛歍ushare `fina_indicator`锛圧OE/ROA/姣涘埄鐜?鍑€鍒╃巼/鎴愰暱鐜?鍋垮€烘寚鏍囩瓑锛夈€乣income`锛堟敹鍏ュ拰鍒╂鼎锛夈€乣balancesheet`锛堣祫浜ц礋鍊虹巼/娴佸姩姣旂巼锛夈€乣cashflow`锛堢粡钀ョ幇閲戞祦锛夈€乣daily_basic`锛圥E/PB/PS鍜屽競鍊间及鍊兼暟鎹級
  - 瀵归綈鍘熷垯锛氬繀椤荤户缁寜`ann_date`/鍙幏寰楁棩瀵归綈锛屼笉鑳芥寜鎶ュ憡鏈焋end_date`瀵归綈锛涘彲鏇翠繚瀹堝湴浣跨敤`ann_date + 1 trading day`浣滀负鐢熸晥鏃?  - 浼板€煎鐞嗭細涓嶈鐩存帴鐢ㄥ師濮婸E/PB/PS锛屼紭鍏堝仛鑷韩鍘嗗彶鍒嗕綅鏁般€佽涓氬唴鍒嗕綅鏁版垨鎴潰鍒嗕綅鏁帮紝濡俙pe_percentile`, `pb_percentile`, `ps_percentile`
  - 缂哄け澶勭悊锛氬缓璁悓姝ュ姞鍏ュ熀鏈潰鍙敤鎬ask锛屽尯鍒嗏€滆储鎶ユ湭鍏竷/瀛楁缂哄け鈥濆拰鈥滅湡瀹炲€间负0鈥?- 缂哄け淇℃伅mask锛氶櫎鐢充竾琛屼笟鎸囨暟澶栵紝鍚庣画鍩烘湰闈€佸寳鍚戣祫閲戙€丳MI鍐峰惎鍔ㄧ瓑涔熷彲鍔犲叆`feature_available`锛岃妯″瀷鍖哄垎鈥滅湡瀹炰负0鈥濆拰鈥滃巻鍙蹭笉鍙敤/鏈叕甯冣€濄€?
#### 缃戠粶缁撴瀯浼樺寲璺嚎

**浼樺厛绾?锛堜綆椋庨櫓锛屽缓璁紭鍏堝疄楠岋級:**
1. **绉婚櫎妯″瀷鍐卹ank embedding**锛氬綋鍓峏閲屽凡缁忓寘鍚畬鏁存埅闈ank鐗瑰緛锛宍UltimateV7Model._build_rank_embed()`鍙堝熀浜巂X[...,0]`棰濆鍔爎ank embedding锛屽彲鑳介噸澶嶄笖鍙湅鍗曚竴鐗瑰緛銆傚缓璁繚鐣橷閲岀殑rank锛岀Щ闄ゆā鍨嬪唴rank embedding銆?2. **Alpha/Horizon head鍔燣ayerNorm + Dropout**锛氬皢绠€鍗昤Linear -> GELU -> Linear`鏀逛负`LayerNorm -> Linear -> GELU -> Dropout -> Linear`锛屾彁楂樿缁冪ǔ瀹氭€с€?3. **Regime铻嶅悎浠庡姞娉曟敼涓篏ate/FiLM**锛氬綋鍓嶅競鍦虹姸鎬佹槸`trans_out + regime_h`锛屾墍鏈夎偂绁ㄥ叡浜悓涓€鍔犳硶鍋忕Щ銆傚缓璁敼鎴恅h = h * (1 + gamma(regime)) + beta(regime)`鎴杇ate铻嶅悎锛岃涓嶅悓鑲＄エ琛ㄧず瀵瑰競鍦虹姸鎬佹湁鏇寸粏绮掑害鍝嶅簲銆?4. **Cross-sectional LayerNorm**锛氬湪`FeatureGrouper`杈撳嚭鎴朤ransformer鍓嶅鍔燻LayerNorm`锛岃繘涓€姝ョǔ瀹氫笉鍚屾埅闈?鎵规鐨勬縺娲诲垎甯冦€?
**浼樺厛绾?锛堢粨鏋勮皟鏁达級:**
5. **闄嶄綆Transformer澶嶆潅搴?*锛氬綋鍓峘hidden_dim=256, n_layers=4, n_heads=8`瀵瑰叏甯傚満鎴潰attention杈冮噸銆傚彲瀹為獙`hidden_dim=256, n_layers=3`锛屾垨`hidden_dim=192, n_heads=6, n_layers=3`锛岄檷浣庢樉瀛樺拰杩囨嫙鍚堥闄┿€?6. **鍏变韩澶氬懆鏈焗ead搴曞眰MLP**锛氬皢姣忎釜horizon鐙珛head鏀逛负鍏变韩浣庡眰MLP鍚庤緭鍑篳n_horizons`锛屽噺灏戝弬鏁板苟鍔犲己澶氬懆鏈熶竴鑷存€с€?7. **琛屼笟aware attention/bias**锛氬埄鐢ㄨ涓欼D缁欏悓琛屼笟鑲＄エattention澧炲姞鍙涔燽ias锛屾垨鍋氣€滆涓氬唴attention -> 琛屼笟token -> 甯傚満token鈥濈殑鍒嗗眰缁撴瀯銆傛敼鍔ㄨ緝澶э紝鏀惧悗缁€?8. **Mixture-of-Experts by regime**锛氱敤甯傚満regime浣滀负gate锛岀粍鍚堝涓猘lpha expert锛坱rend_up/range/high_vol/crisis锛夛紝澧炲己甯傚満鐘舵€佸垏鎹㈤€傚簲鎬с€?
#### 璁粌鐩爣涓巐oss浼樺寲璺嚎

**鏈€鎺ㄨ崘鐨勪笅涓€闃舵:**
1. **鏍囩涓€у寲 residual return**锛氭湭鏉ユ敹鐩婂厛瀵瑰競鍦恒€佽涓氥€乻ize銆乿ol銆乵om鍋氭埅闈㈠洖褰掞紝鏍囩鐢ㄦ畫宸敹鐩婏紝鍑忓皯妯″瀷瀛︿範椋庨櫓鍥犲瓙鏀剁泭鑰岄潪alpha銆?2. **Style exposure penalty**锛氬湪loss閲屾儵缃氭渶缁坅lpha涓巂size/vol/mom`绛夐鏍煎洜瀛愮殑鎴潰鐩稿叧锛岄伩鍏嶉娴嬬粨鏋滆繃搴︽毚闇查闄╁洜瀛愩€?3. **Rank label / residual rank label**锛氬綋鍓峘correlation_rank_loss`瀹為檯鏄疨earson鐩稿叧锛屼笉鏄弗鏍糞pearman銆傚彲灏唗arget鍏堣浆鎴潰rank锛屾垨瀵逛腑鎬у寲娈嬪樊鏀剁泭杞瑀ank鍚庤缁冿紝闄嶄綆鏋佺鏀剁泭褰卞搷銆?4. **Top-bottom spread loss**锛氱敤softmax杩戜技top/bottom缁勫悎锛岀洿鎺ヤ紭鍖栭娴媡op缁勭浉瀵筨ottom缁勭殑鏀剁泭宸紝鏇存帴杩戜氦鏄撶洰鏍囥€?5. **Horizon consistency loss**锛歵+1/t+3/t+5/t+7棰勬祴涓嶅簲瀹屽叏鍐茬獊锛屽彲鍔犲叆杞婚噺鐩搁偦horizon鏂瑰悜涓€鑷存€х害鏉燂紝浣嗕紭鍏堢骇浣庝簬鏍囩涓€у寲銆?6. **Confidence杈撳嚭**锛氭ā鍨嬮澶栬緭鍑虹疆淇″害锛岀敤`final_alpha = alpha * confidence`杩涜浠撲綅缂╂斁锛岄娴嬩笉纭畾鏃堕檷浣庢毚闇层€?
#### 璁粌閲囨牱涓庢椂闂存潈閲?
- 鏃堕棿琛板噺閲囨牱/鍔犳潈锛?010骞寸殑鏍锋湰鍜岃繎骞碅鑲＄敓鎬佸樊寮傝緝澶э紝鍙粰杩戝勾鏍锋湰鏇撮珮鏉冮噸锛屽鎸囨暟琛板噺鎴栧垎娈垫潈閲嶏紙2010-2015: 0.5, 2016-2020: 0.8, 2021-now: 1.2锛夈€?- 褰撳墠鍏堣窇瀹孷10鍩虹嚎锛屽啀閫愰」瀹為獙涓婅堪鏀瑰姩锛涙瘡娆″彧鏀逛竴绫伙紝閬垮厤鏃犳硶鍒ゆ柇鏀剁泭鏉ユ簮銆?
## Backtest绯荤粺 (2026-05-12)

### V9娣卞害妯″瀷闆嗘垚

**瀹炵幇:** `backtest/engine.py`鐜板凡鏀寔V9娣卞害妯″瀷checkpoint鍜孡ightGBM澶氬懆鏈熷熀绾跨殑瀵规瘮鍥炴祴銆?
**鍏抽敭缁勪欢:**

1. **妯″瀷鍔犺浇** (`load_v9_checkpoint`):
   - 浠巂ultimate_v7_best.pt`鍔犺浇璁粌濂界殑`UltimateV7Model`
   - 鑷姩鎺ㄦ柇妯″瀷缁村害锛歚input_dim`, `base_feat_dim`, `regime_dim`, `num_industries`
   - 鏀寔`--device auto/cpu/cuda`鍙傛暟
   - 浣跨敤`map_location="cpu"`閬垮厤涓嶅繀瑕佺殑GPU鍐呭瓨鍗犵敤

2. **棰勬祴鍣ㄦ帴鍙?*:
   - `DLPredictor`: 灏佽娣卞害妯″瀷鎺ㄧ悊锛岃緭鍑烘埅闈㈡爣鍑嗗寲鐨刟lpha淇″彿
   - `LGBPredictor`: 灏佽LightGBM澶氬懆鏈熻瀺鍚堬紝鏀寔IC琛板噺鍔犳潈鍜屽競鍦虹姸鎬佽嚜閫傚簲

3. **Alpha铻嶅悎** (`fused_alpha`):
   - 澶氬懆鏈熼娴嬫寜甯傚満鐘舵€?trend_up/panic/sideways)鍔ㄦ€佸姞鏉?   - 鍙€塈C琛板噺璋冩暣锛氱敤楠岃瘉闆咺C瀵筯orizon鏉冮噸浜屾鍔犳潈
   - 鎵€鏈夐娴嬬粡杩囨埅闈㈡爣鍑嗗寲鍜宼anh褰掍竴鍖栵紝淇濊瘉淇″彿灏哄害涓€鑷?
### 缁勫悎鏋勫缓妯″紡

**CLI鍙傛暟:** `--portfolio-mode {optimizer,simple_ls,simple_long}`

1. **optimizer** (榛樿):
   - 椋庨櫓棰勭畻浼樺寲鍣紝鍩轰簬alpha寮哄害鍒嗛厤椋庨櫓棰勭畻
   - 鑰冭檻鍥犲瓙椋庨櫓妯″瀷(F_cov, D_diag)銆佸競鍦篵eta涓€с€佹崲鎵嬫垚鏈?   - 鏀寔涓偂鏉冮噸涓婇檺鍜孉DV娴佸姩鎬х害鏉?
2. **simple_ls** (澶氱┖瀵瑰啿):
   - 鎸塧lpha鎺掑簭锛屽仛澶歵op 10%锛屽仛绌篵ottom 10%
   - 澶氱┖鍚勫崰50%鏉犳潌锛屾€绘潬鏉?.0
   - 鐢ㄤ簬楠岃瘉绾俊鍙峰己搴︼紝鎺掗櫎浼樺寲鍣ㄥ鏉傚害褰卞搷

3. **simple_long** (绾澶?:
   - 鎸塧lpha鎺掑簭锛屼粎鍋氬top 10%
   - 鎬绘潬鏉?.0
   - 鐢ㄤ簬瀵规瘮澶氱┖鍜岀函澶氱瓥鐣ヨ〃鐜?
**璁捐鍘熷垯:** 涓夌妯″紡浣跨敤鐩稿悓鐨刟lpha淇″彿銆佺浉鍚岀殑鎵ц绾︽潫銆佺浉鍚岀殑鎴愭湰妯″瀷锛岀‘淇濆姣旂殑鍏钩鎬с€?
### ADV娴佸姩鎬х害鏉?
**CLI鍙傛暟:** `--adv-mode {execution,weight_cap,both}` (榛樿: `execution`)

1. **execution** (榛樿锛屽己鐑堟帹鑽?:
   - 浠呭湪鎵ц灞傞潰搴旂敤ADV绾︽潫
   - 鐩爣鏉冮噸涓嶅彈闄愶紝浣嗗疄闄呮垚浜ゅ彈娴佸姩鎬ч檺鍒?   - `fill_ratio = min(1.0, ADV * adv_ratio / |trade_weight|)`
   - 閬垮厤鍙岄噸绾︽潫瀵艰嚧鐨勮繃搴︿繚瀹?   - **鍥炴祴琛ㄧ幇鏈€浣?*锛氬勾鍖?8.17%锛屽鏅?.99锛屽洖鎾?1.81%锛屽啿鍑绘垚鏈?.0025

2. **weight_cap** (鈿狅笍 涓嶆帹鑽愪娇鐢紝宸插純鐢?:
   - 鍦ㄤ紭鍖栧櫒灞傞潰闄愬埗涓偂鏉冮噸涓婇檺
   - `max_weight = min(鍥哄畾涓婇檺, ADV * adv_ratio / portfolio_value)`
   - **瀛樺湪璁捐缂洪櫡**锛氫紭鍖栧櫒鐢ㄥ巻鍙睞DV绾︽潫鏉冮噸锛屼絾鎵ц鏃?00%鎴愪氦锛屽綋鏃ユ祦鍔ㄦ€у樊鏃跺啿鍑绘垚鏈垎鐐?   - **鍥炴祴琛ㄧ幇鏋佸樊**锛氬勾鍖?16.32%锛屽鏅?1.50锛屽洖鎾?4.13%锛屽啿鍑绘垚鏈?.3145锛堟槸execution妯″紡鐨?25鍊嶏級
   - 淇濈暀浠ｇ爜浠呬緵鍙傝€冿紝瀹為檯浣跨敤璇烽€夋嫨execution妯″紡

3. **both**:
   - 鍚屾椂搴旂敤鏉冮噸涓婇檺鍜屾墽琛岀害鏉?   - 鏈€淇濆畧锛屼絾鍙兘杩囧害鎶戝埗alpha瀹炵幇
   - 鍙綔涓篹xecution妯″紡鐨勪繚瀹堝彉浣?
**鍐插嚮鎴愭湰妯″瀷:**
```python
turnover_ratio = |trade_exec| * portfolio_value / dollar_volume
impact_cost = impact_coeff * turnover_ratio^2 * |trade_exec|
```

**閲嶈淇:** `execute_order_with_impact()`鐜板凡姝ｇ‘澶勭悊鏃犳晥浠锋牸/鎴愪氦閲忥紝閬垮厤NaN浼犳挱鍒版垚鏈拰鏉冮噸涓€?
### 鍥炴祴璇婃柇鎸囨爣

**杈撳嚭鎸囨爣:**

1. **璋冧粨缁熻**:
   - 璋冧粨灏濊瘯娆℃暟 vs 鎴愬姛娆℃暟
   - 骞冲潎鏈夋晥鑲＄エ鏁?   - Alpha鏍囧噯宸紙淇″彿寮哄害锛?
2. **鏉犳潌涓庢崲鎵?*:
   - 鐩爣鏉犳潌 vs 鎴愪氦鍚庢潬鏉?   - 骞冲潎鎹㈡墜/璋冧粨
   - 骞冲潎濉厖鐜囷紙瀹為檯鎴愪氦/鐩爣浜ゆ槗锛?
3. **娴佸姩鎬х害鏉?*:
   - 涓嶅彲浜ゆ槗姣斾緥
   - 骞冲潎ADV鏉冮噸涓婇檺锛坵eight_cap妯″紡锛?
4. **椋庨櫓鏆撮湶**:
   - 闈為浂鎸佷粨澶╂暟
   - 骞冲潎鏉犳潌
   - 骞冲潎浼拌Beta锛堟粴鍔?0鏃ワ級

5. **鎴愭湰**:
   - 鎬诲啿鍑绘垚鏈?   - 骞冲潎鏃ユ垚鏈?
**缁╂晥鎸囨爣:**
- 鍘熷澶氱┖锛氬勾鍖栨敹鐩娿€佸鏅瘮鐜囥€佹渶澶у洖鎾?- 淇涓€э細Beta涓€у寲鍚庣殑骞村寲鏀剁泭銆佸鏅瘮鐜囥€佹渶澶у洖鎾?
**閲嶈淇 (2026-05-12):**
- 骞村寲鏀剁泭鐜颁粎璁＄畻娲昏穬鎸佷粨鏈熼棿锛屾帓闄ゅ墠鏈熸棤鎸佷粨鐨勯浂鏀剁泭澶╂暟
- 鏈€澶у洖鎾ゆ敼涓虹浉瀵瑰洖鎾わ細`(peak - cum) / peak`锛岃€岄潪缁濆鍥炴挙
- 娲昏穬鏈熼棿璇嗗埆锛歚first_active = argmax(leverage > 0)`, `last_active = len - argmax(reversed(leverage > 0))`

### 鍥炴祴绯荤粺淇璁板綍 (2026-05-12)

1. **NaN鍐插嚮鎴愭湰浼犳挱** (`execute_order_with_impact`):
   - 闂锛氭棤鏁堜环鏍?鎴愪氦閲忓鑷碞aN浼犳挱鍒癷mpact_cost鍜宖illed_weights
   - 淇锛氭坊鍔燻valid_liquidity`鎺╃爜锛屼粎瀵规湁鏁堟祦鍔ㄦ€ц绠梔ollar_volume鍜宼urnover_ratio
   - 浣跨敤`np.nan_to_num`娓呯悊鎵€鏈夎緭鍏ュ拰杈撳嚭

2. **LGB IC琛板噺绫诲瀷閿欒**:
   - 闂锛氳缁冨悗`ic_decay`鍙兘鏄疨ython list锛屽鑷撮櫎娉曡繍绠楀け璐?   - 淇锛氬湪`run_backtest_production`鍏ュ彛鏄惧紡杞崲涓篳np.asarray(ic_decay, dtype=float)`

3. **Checkpoint鍔犺浇GPU鍐呭瓨娴垂**:
   - 闂锛歚torch.load(map_location=device)`浼氬皢optimizer/scheduler鐘舵€佷篃鍔犺浇鍒癎PU
   - 淇锛氭敼涓篳map_location="cpu"`锛屼粎鍦ㄩ渶瑕佹椂灏唌odel绉诲埌GPU

4. **骞村寲鏀剁泭璁＄畻鍋忓樊**:
   - 闂锛氬寘鍚墠鏈熸棤鎸佷粨鐨勯浂鏀剁泭澶╂暟锛屽鑷村勾鍖栨敹鐩婅涓ラ噸浣庝及
   - 淇锛氳瘑鍒椿璺冩寔浠撳尯闂达紝浠呭璇ュ尯闂磋绠楀勾鍖栧拰鍥炴挙

5. **鏈€澶у洖鎾ゅ叕寮忛敊璇?*:
   - 闂锛氫娇鐢ㄧ粷瀵瑰洖鎾peak - cum`鑰岄潪鐩稿鍥炴挙
   - 淇锛氭敼涓篳(peak - cum) / (peak + 1e-12)`

6. **all_codes椤哄簭涓嶇‘瀹氭€?*:
   - 闂锛歚set()`瀵艰嚧鑲＄エ椤哄簭闅忔満锛屽奖鍝嶇煩闃电储寮曚竴鑷存€?   - 淇锛氫娇鐢╜sorted(set(...))`纭繚纭畾鎬ч『搴?
7. **weight_cap妯″紡鏋佺鍐插嚮鎴愭湰** (`execute_order_with_impact`, line 364):
   - 闂锛歸eight_cap妯″紡涓嬪啿鍑绘垚鏈垎鐐革紝杈惧埌206.1030锛堟甯稿€?.0025鐨?2,000鍊嶏級锛屽鑷村洖鎾?,307,276%
   - 鏍规湰鍘熷洜锛歸eight_cap鍦ㄤ紭鍖栧櫒灞傞潰浣跨敤鍘嗗彶20鏃ュ钩鍧囨垚浜ら噺绾︽潫鏉冮噸锛屼絾鎵ц灞傞潰浣跨敤褰撴棩瀹為檯鎴愪氦閲忚绠楀啿鍑绘垚鏈€傚綋鏃ユ垚浜ら噺杩滀綆浜庡巻鍙插潎鍊兼椂锛宍turnover_ratio = |trade_exec| * portfolio_value / dollar_vol`鐖嗙偢锛宍impact_cost 鈭?turnover_ratio虏`杩涗竴姝ユ斁澶?   - 绗竴闃舵淇锛氭坊鍔燻turnover_ratio = np.clip(turnover_ratio, 0.0, 1.0)`闃叉鏁板涓婁笉鍙兘鐨勫€?   - 淇鍚庣粨鏋滐細鍐插嚮鎴愭湰闄嶈嚦1.3145锛屽洖鎾ゆ甯稿寲鑷?2.75%锛屼絾骞村寲鏀剁泭鍙樹负璐熷€?14.46%锛屼粛姣攅xecution妯″紡宸?25鍊?   - 璁捐缂洪櫡锛歸eight_cap妯″紡鍦ㄤ紭鍖栧櫒灞傞潰绾︽潫鏉冮噸(line 526-539)锛屼絾鎵ц灞傞潰璁剧疆`exec_adv_ratio=1e9`(line 568)锛屽鑷存墍鏈夌洰鏍囦氦鏄?00%鎴愪氦锛屾棤瑙嗗綋鏃ユ祦鍔ㄦ€?   - 鐘舵€侊細**涓嶆帹鑽愪娇鐢╳eight_cap妯″紡**銆俥xecution妯″紡琛ㄧ幇鏈€浣筹紙骞村寲18.17%锛屽鏅?.99锛屽洖鎾?1.81%锛夛紝weight_cap妯″紡鍗充娇淇鍚庝粛琛ㄧ幇鏋佸樊锛堝勾鍖?16.32%锛屽鏅?1.50锛屽洖鎾?4.13%锛?
### CLI浣跨敤绀轰緥

**鍩烘湰鐢ㄦ硶:**

```bash
# 浣跨敤V9娣卞害妯″瀷锛岄粯璁ptimizer妯″紡
F:/miniconda3/envs/pytorch/python backtest/engine.py --model-type dl --checkpoint ultimate_v7_best.pt

# 浣跨敤LightGBM鍩虹嚎
F:/miniconda3/envs/pytorch/python backtest/engine.py --model-type lgb --lgb-dir models_multi_v9_tech_macro

# 瀵规瘮涓ょ妯″瀷
F:/miniconda3/envs/pytorch/python backtest/engine.py --model-type both
```

**缁勫悎妯″紡瀵规瘮:**

```bash
# 椋庨櫓棰勭畻浼樺寲鍣紙榛樿锛?F:/miniconda3/envs/pytorch/python backtest/engine.py --model-type dl --portfolio-mode optimizer

# 绠€鍗曞绌哄鍐?F:/miniconda3/envs/pytorch/python backtest/engine.py --model-type dl --portfolio-mode simple_ls

# 绾澶?F:/miniconda3/envs/pytorch/python backtest/engine.py --model-type dl --portfolio-mode simple_long
```

**ADV绾︽潫瀹為獙:**

```bash
# 浠呮墽琛屽眰绾︽潫锛堟帹鑽愶級
F:/miniconda3/envs/pytorch/python backtest/engine.py --adv-mode execution

# 浠呮潈閲嶄笂闄愮害鏉?F:/miniconda3/envs/pytorch/python backtest/engine.py --adv-mode weight_cap

# 鍙岄噸绾︽潫锛堟渶淇濆畧锛?F:/miniconda3/envs/pytorch/python backtest/engine.py --adv-mode both
```

### 寰呭姙浜嬮」涓庡凡鐭ラ棶棰?
**褰撳墠鐘舵€?(2026-05-12):**
- 鉁?V9娣卞害妯″瀷宸查泦鎴愬埌鍥炴祴绯荤粺
- 鉁?鏀寔optimizer/simple_ls/simple_long涓夌缁勫悎妯″紡瀵规瘮
- 鉁?淇骞村寲鏀剁泭鍜屾渶澶у洖鎾よ绠楀亸宸?- 鉁?淇NaN鍐插嚮鎴愭湰浼犳挱闂
- 鉁?瀹屾垚涓夌缁勫悎妯″紡瀵规瘮鍒嗘瀽
- 鉁?simple_long宸蹭紭鍖栵紙alpha鍔犳潈+甯傚満鎷╂椂锛?- 鉁?GAT鍒嗘敮宸插疄鐜帮紙琛屼笟鍥炬敞鎰忓姏锛?- 鈴?GAT鍏ㄩ噺璁粌涓?..

**鏁版嵁鏍囩璇箟 (閲嶈):**
- `sample['raw_y']`: 鏈爣鍑嗗寲鐨則arget_horizon鍓嶅悜鏀剁泭
- `sample['y']`: 鎴潰鏍囧噯鍖栧悗鐨勬爣绛撅紙璁粌鐢級
- `sample['y_seq']`: 澶氬懆鏈熸敹鐩婂簭鍒楋紝shape=(N, max_horizon)
- 鍥炴祴鏃跺簲浣跨敤`raw_y`鎴栭噸鏂拌绠楃湡瀹炴敹鐩婏紝涓嶈鐩存帴鐢ㄦ爣鍑嗗寲鍚庣殑`y`

## GAT妯″瀷瀹炵幇 (2026-05-12)

**鏋舵瀯:**
- `core/model.py`: UltimateV7Model 鏂板 GATConv 鍒嗘敮
- 1灞侴ATConv锛宧eads=2锛宑oncat=False锛宧idden_dim=256
- 鍚岃涓氳偂绁ㄦ寜 industry_id 鍏ㄨ繛鎺ワ紝姣忎釜鑲＄エ鏈€澶氳繛鎺?鍙悓琛岋紙`max_edges_per_stock=8`锛?- 杈撳嚭閫氳繃 `FusionGate` 涓?Transformer 杈撳嚭鑷€傚簲铻嶅悎: `h = gate * transformer + (1-gate) * gat`
- 鍙傛暟閲? 4.26M锛堟瘮绾疶ransformer澧炲姞~70K锛?
**GAT璁粌鑴氭湰:** `run/train_gat.py`
- 鍩轰簬 V9 璁粌鑴氭湰锛屾柊澧?`cfg.use_gat=True`
- 鐙珛 checkpoint: `ultimate_v7_gat_best.pt`锛屼笉瑕嗙洊 V9 鏉冮噸
- `save_path` 鍙傛暟: `core/train_utils.py` 鐨?`train_model()` 鏂板 `save_path` 鍙傛暟

**璁粌閰嶇疆锛圧TX 2060 6GB 涓撶敤锛?**
- `batch_size=8, accum_steps=2`锛堢瓑鏁?batch 16锛孊S=8 宄板€紐1.85GB锛岀┖浣?.3GB锛?- `keep_ratio=0.5`锛堟埅闈㈤噰鏍?0%锛屼綆浜嶸9鐨?0%锛?- `max_edges_per_stock=5`锛堥檺鍒禛AT杈规暟锛岄伩鍏峅OM锛?- `use_amp=False`锛圓MP瀵艰嚧 loss=nan锛?- `grad_clip=0.2`

**缂撳瓨澶辨晥瑙勫垯锛?* 鍚?V9锛屼慨鏀筦data/pipeline.py`/`market_features.py`/`fundamental_factors.py`鍚庨渶鍒犻櫎`cache/cross_section_*.pkl`

## GAT鍐呭瓨娉勬紡涓庢彁閫熶慨澶?(2026-05-13)

### 闂1: build_industry_edges O(n虏) GPU鏄惧瓨鐖嗙偢
**浣嶇疆:** `core/model.py:build_industry_edges`
**闂:** `[torch.randperm(n_ind) for _ in range(n_ind)]` 鍒涘缓 n脳n 鍏ㄦ帓鍒楃煩闃靛湪GPU涓婏紝澶ц涓氱粍锛垀200鑲★級浜х敓40K+涓棿tensor
**淇:**
- 杈规瀯寤虹Щ鑷?**CPU**锛堢储寮曟搷浣滀笉闇€瑕丟PU锛?- O(n虏) 鈫?**O(n脳k)** 閲囨牱锛氭瘡鍙偂绁ㄥ彧杩瀔涓殢鏈哄悓琛岋紝鏇夸唬n脳n鍏ㄦ帓鍒?- eval妯″紡鍚敤鍐呭鍝堝笇缂撳瓨锛坄(ids_bytes, mask_bytes)` 鍋歬ey锛夛紝楠岃瘉闆嗚竟绱㈠紩鍙渶绠椾竴娆?
### 闂2: _gat_forward 涓棿tensor绱Н
**浣嶇疆:** `core/model.py:_gat_forward`
**淇:** 姣忚疆寰幆鏈熬 `del batch_edges, edges, x`

### 闂3: 璁粌寰幆tensor鏈強鏃堕噴鏀?**浣嶇疆:** `core/train_utils.py` 璁粌寰幆
**淇:**
- forward鍚庣珛鍗?`del alpha_raw, alphas, horizon_preds`
- `loss.item()` 鎻愬彇鏍囬噺鍚庣珛鍗?`del loss`
- 姣?00 batch璋冪敤 `torch.cuda.empty_cache()` 鍥炴敹纰庣墖
- epoch鏈熬瀹夊叏娓呯悊锛歵ry/except 澶勭悊鍙兘宸查噴鏀剧殑鍙橀噺

### 闂4: evaluate GPU tensor鏈竻鐞?**浣嶇疆:** `core/train_utils.py:evaluate`
**淇:** 琛ュ厖 `industry_ids` 鍒癲elete鍒楄〃锛汷OM璺宠繃鏃朵篃娓呯悊宸插垎閰嶇殑杈撳叆tensor

### 闂5: Windows pipe闃诲瀵艰嚧璁粌鍋囨
**浣嶇疆:** `run/train_gat.py:Tee`
**闂:** Tee绫诲啓stdout pipe锛學indows绠￠亾缂撳啿鍖烘弧鍚?`print()` 姘镐箙闃诲锛孏PU绌鸿浆
**淇:** 绉婚櫎pipe杈撳嚭锛屾敼鐢?`LogWriter` 鍙啓log鏂囦欢

### 鏄惧瓨璇婃柇缁撹锛堝叏閲忔暟鎹? keep_ratio=0.5锛?
| batch_size | 鑲＄エ/鏍锋湰 | 璁粌fwd+bwd宄板€?| 鍓╀綑鏄惧瓨 |
|-----------|----------|----------------|---------|
| 4 | ~1675 | 934MB | 5.1GB |
| 8 | ~1675 | 1851MB | 4.3GB |
| 16 | ~1200(浼扮畻) | ~2.5-3GB(浼扮畻) | ~3GB |

### 瀹為檯璁粌姣廵poch鑰楁椂锛圔S=8, accum=2, 鍏ㄩ噺~5000鑲★級

| epoch | train | val | 绱 |
|-------|-------|-----|------|
| 绗?娆?| ~6.9min | ~2.8min锛堟棤缂撳瓨锛?| ~9.7min |
| 绗?+娆?| ~6.9min | ~1.0min锛堢紦瀛樺懡涓級 | ~7.9min |
| 25涓猠poch鎬昏 | - | - | **~3.3h** |

### 杈圭储寮曡瘎浼扮紦瀛樻晥鏋?- **eval妯″紡**锛歚build_industry_edges` 鐢?`(industry_ids_bytes, mask_bytes)` 鍋歞ict key
- 楠岃瘉闆嗕笉鍙橈紙val_loader涓峴huffle锛夛紝绗?涓猠poch cache miss鍚庡悗缁叏閮ㄥ懡涓?- 灏嗛獙璇侀樁娈典粠~2.8min闄嶈嚦~1.0min锛圕PU杈规瀯寤轰粠~1.8min闄嶈嚦~0锛?
## 鍥炴祴绯荤粺淇璁板綍 (2026-05-12)

### 淇鎬昏

鏈瀵瑰洖娴嬬郴缁熻繘琛屼簡鍏ㄩ潰鐨勯棶棰樻帓鏌ュ拰淇锛屽叡淇18涓棶棰橈紝鍒嗕负P0锛堜弗閲嶏級銆丳1锛堥噸瑕侊級銆丳2锛堜紭鍖栵級銆丳3锛堜竴鑷存€э級鍥涗釜浼樺厛绾с€?
### P0绾у埆淇锛堜弗閲嶉棶棰橈級

#### 1. 椋庨櫓妯″瀷鍥犲瓙鐭╅樀閿欒浣跨敤
**浣嶇疆锛?* `engine.py:546`  
**闂锛?* 灏嗗競鍦烘暣浣撶壒寰侊紙鎵€鏈夎偂绁ㄥ叡浜浉鍚屽€硷級璇敤涓轰釜鑲″洜瀛愭毚闇蹭紶鍏ラ闄╂ā鍨? 
**褰卞搷锛?* 椋庨櫓妯″瀷浼拌閿欒锛岀粍鍚堜紭鍖栧け鏁? 
**淇锛?*
```python
# 浠呬娇鐢ㄤ釜鑲″洜瀛?0-2: size,vol,mom)鍜岃涓歰ne-hot(87-169)
B_style = np.hstack([
    sample['risk'][valid, :3],      # 涓偂椋庢牸鍥犲瓙
    sample['risk'][valid, 87:]      # 琛屼笟one-hot
])
```

#### 2. 椋庨櫓棰勭畻鍒嗛厤涓庨闄╂ā鍨嬭劚鑺?**浣嶇疆锛?* `engine.py:292-296`  
**闂锛?* `risk_budget_allocation()`鏈娇鐢‵_cov鍜孌_diag锛岄闄╅绠楁湭鑰冭檻鍥犲瓙鐩稿叧鎬? 
**褰卞搷锛?* 鍙兘鍒嗛厤杩囧害椋庨櫓缁欓珮搴︾浉鍏崇殑鍥犲瓙  
**淇锛?* 绠€鍖栦负鑰冭檻娈嬪樊椋庨櫓鍔犳潈
```python
def risk_budget_allocation(alpha, D_diag, target_vol=0.15):
    strength = np.abs(alpha) / (np.sum(np.abs(alpha)) + 1e-8)
    inv_risk = 1.0 / (D_diag + 1e-8)
    inv_risk /= (np.sum(inv_risk) + 1e-8)
    blended = 0.5 * strength + 0.5 * inv_risk
    blended /= (np.sum(blended) + 1e-8)
    target_var = (target_vol / np.sqrt(252)) ** 2
    return blended * target_var
```

#### 3. 鎹㈡墜鎴愭湰鍗曚綅閿欒
**浣嶇疆锛?* `engine.py:371`  
**闂锛?* volume鏄墜鏁帮紙1鎵?100鑲★級锛屼絾鏈箻浠?00  
**褰卞搷锛?* 浣庝及鎴愪氦棰?0鍊嶏紝鍐插嚮鎴愭湰璁＄畻閿欒  
**淇锛?*
```python
dollar_vol = np.where(valid_liquidity, price * volume * 100, 0.0)
```

### P1绾у埆淇锛堥噸瑕侀棶棰橈級

#### 4. 浼樺寲鍣ㄦ敹鏁涘垽鎹繃浜庝弗鏍?**浣嶇疆锛?* `engine.py:330`  
**闂锛?* 缁濆闃堝€?e-6瀵筃鈮?000鐨勭粍鍚堣繃浜庝弗鏍? 
**淇锛?* 浣跨敤鐩稿鏀舵暃鍒ゆ嵁
```python
rel_change = np.linalg.norm(w_new - w) / (np.linalg.norm(w) + 1e-8)
if rel_change < 1e-4:
    converged = True
```

#### 5. IC琛板噺妫€娴嬮€昏緫閿欒
**浣嶇疆锛?* `engine.py:447-454`  
**闂锛?* 鎵€鏈塈C鈮?.5鏃朵細璁剧疆涓烘渶闀垮懆鏈燂紝閿欒繃淇″彿琛板噺  
**淇锛?* 缁撳悎缁濆闃堝€煎拰瓒嬪娍妫€娴?```python
for i in range(len(ic_norm)):
    if ic_norm[i] < 0.5:  # 缁濆闃堝€?        rebalance_freq = i + 1
        break
    if i > 0 and ic_norm[i] < ic_norm[i-1] * 0.7:  # 瓒嬪娍妫€娴?        rebalance_freq = i + 1
        break
else:
    rebalance_freq = max(3, min(5, len(ic_decay)))  # 榛樿鍊?```

#### 6. simple_ls鏈仛涓€у寲
**浣嶇疆锛?* `engine.py:273-289`  
**闂锛?* simple_ls妯″紡鐨勫绌虹粍鍚堟湭鍋氫腑鎬у寲锛屼笌optimizer妯″紡涓嶄竴鑷? 
**褰卞搷锛?* 褰卞搷鍏钩瀵规瘮  
**淇锛?*
```python
def build_simple_weights(...):
    # ... 鏋勫缓鏉冮噸 ...
    w = w - np.mean(w)  # 涓€у寲
    return w
```

### P2绾у埆淇锛堜紭鍖栭」锛?
#### 7. 鍐插嚮鎴愭湰鏈€冭檻甯傚満鐘舵€?**浣嶇疆锛?* `engine.py:388`  
**闂锛?* impact_coeff鍥哄畾涓?.1锛屾湭鏍规嵁甯傚満鐘舵€佽皟鏁? 
**淇锛?* 娣诲姞甯傚満鐘舵€佽皟鑺傜郴鏁?```python
regime_multiplier = {
    "panic": 1.5,      # 鎭愭厡甯傚満锛氭祦鍔ㄦ€ф灟绔?    "sideways": 1.0,   # 姝ｅ父甯傚満锛氬熀鍑?    "trend_up": 1.0    # 瓒嬪娍甯傚満锛氬熀鍑?}
impact_coeff_adj = impact_coeff * regime_multiplier.get(regime, 1.0)
```

#### 8. 鍒濆寤轰粨鎹㈡墜鎯╃綒澶辨晥
**浣嶇疆锛?* `engine.py:315-316`  
**闂锛?* prev_w=None鏃讹紝鎹㈡墜鎯╃綒涓嶇敓鏁堬紝瀵艰嚧棣栨璋冧粨鎹㈡墜杩囧ぇ  
**淇锛?* 娣诲姞鐙珛鐨勪粨浣嶈妯℃儵缃?```python
is_initial = prev_w is None or np.sum(np.abs(prev_w)) < 1e-8
lambda_init = 0.01 if is_initial else 0.0
if lambda_init > 0:
    grad += lambda_init * np.sign(w)  # 鎯╃綒鎬讳粨浣嶈妯?```

### P3绾у埆淇锛堜唬鐮佷竴鑷存€э級

#### 9. risk.py鐨処C琛板噺妫€娴嬩笉涓€鑷?**浣嶇疆锛?* `risk.py:98-102`  
**闂锛?* risk.py浣跨敤鏃х殑缁濆闃堝€兼柟娉曪紝涓巈ngine.py涓嶄竴鑷? 
**褰卞搷锛?* 浠呭奖鍝嶆祴璇曚唬鐮侊紝涓嶅奖鍝嶅疄闄呭洖娴? 
**淇锛?* 鍚屾engine.py鐨勯€昏緫锛堥槇鍊?瓒嬪娍妫€娴?榛樿鍊硷級

### 淇鏁堟灉瀵规瘮

#### 鍥炴祴缁撴灉瀵规瘮锛圤ptimizer妯″紡锛?
| 闃舵 | 骞村寲鏀剁泭 | 澶忔櫘姣旂巼 | 鏈€澶у洖鎾?| Calmar | Sortino | 鑳滅巼 |
|------|---------|---------|---------|--------|---------|------|
| **淇鍓?* | 13.93% | 1.46 | 9.93% | 1.40 | 2.15 | 54.71% |
| **P0+P1淇鍚?* | 32.55% | 2.61 | 11.12% | 2.93 | 3.84 | 58.84% |
| **P0+P1+P2淇鍚?* | 32.44% | 2.59 | 11.12% | 2.92 | 3.82 | 58.71% |

**鍏抽敭鍙戠幇锛?*
1. **P0+P1淇甯︽潵宸ㄥぇ鎻愬崌**锛氬勾鍖栨敹鐩婁粠13.93%鎻愬崌鍒?2.55%锛?133%锛夛紝澶忔櫘姣旂巼浠?.46鎻愬崌鍒?.61锛?79%锛?2. **P2淇鐣ュ井闄嶄綆鎬ц兘**锛氬勾鍖栨敹鐩婁笅闄?.11%锛岃繖鏄悎鐞嗙殑锛屽洜涓猴細
   - 鎭愭厡甯傚満鍐插嚮鎴愭湰澧炲姞1.5x锛堟洿鐪熷疄锛?   - 鍒濆寤轰粨娣诲姞浠撲綅瑙勬ā鎯╃綒锛堟洿淇濆畧锛?3. **椋庨櫓鎺у埗鏀瑰杽**锛氳儨鐜囦粠54.71%鎻愬崌鍒?8.71%锛孲ortino姣旂巼鎻愬崌79%

#### 鍒嗗勾搴﹁〃鐜板姣旓紙鍘熷澶氱┖锛?
| 骞翠唤 | 淇鍓嶅勾鍖?| 淇鍚庡勾鍖?| 淇鍓嶅鏅?| 淇鍚庡鏅?|
|------|-----------|-----------|-----------|-----------|
| 2010 | 4.38% | 28.16% | 0.52 | 2.32 |
| 2011 | 20.88% | 35.72% | 2.08 | 2.52 |
| 2012 | 17.40% | 35.93% | 1.70 | 3.11 |
| 2013 | 12.30% | 20.69% | 1.77 | 2.13 |

**鎵€鏈夊勾浠借〃鐜板潎鏄捐憲鏀瑰杽**

### 淇褰掑洜鍒嗘瀽

**涓昏璐＄尞鏉ヨ嚜P0绾у埆淇锛?*

1. **鎹㈡墜鎴愭湰淇锛坴olume脳100锛?*
   - 淇鍓嶄綆浼颁簡鎴愪氦棰?0鍊?   - 淇鍚庡啿鍑绘垚鏈绠楀噯纭?   - 璐＄尞锛氱害+10-15%骞村寲鏀剁泭

2. **椋庨櫓妯″瀷淇锛堜粎鐢ㄤ釜鑲″洜瀛?琛屼笟锛?*
   - 淇鍓嶅皢甯傚満鏁翠綋鐗瑰緛璇敤涓轰釜鑲″洜瀛?   - 淇鍚庨闄╀及璁″噯纭紝缁勫悎浼樺寲鏈夋晥
   - 璐＄尞锛氱害+5-10%骞村寲鏀剁泭

3. **椋庨櫓棰勭畻浼樺寲锛堣€冭檻娈嬪樊椋庨櫓锛?*
   - 淇鍓嶆湭鑰冭檻鍥犲瓙鐩稿叧鎬?   - 淇鍚庤祫閲戝垎閰嶆洿鍚堢悊
   - 璐＄尞锛氱害+3-5%骞村寲鏀剁泭

**P1绾у埆淇鎻愬崌绋冲畾鎬э細**
- 浼樺寲鍣ㄦ敹鏁涙洿绋冲畾
- IC琛板噺妫€娴嬫洿鍑嗙‘
- simple_ls瀵规瘮鏇村叕骞?
**P2绾у埆淇鎻愬崌鐪熷疄鎬э細**
- 鎭愭厡甯傚満鍐插嚮鎴愭湰鏇寸湡瀹?- 鍒濆寤轰粨鏇翠繚瀹?- 杞诲井闄嶄綆鏀剁泭锛?0.11%锛夋槸鍚堢悊浠ｄ环

### 鏈€缁堢粨璁?
鉁?**鍥炴祴绯荤粺宸蹭紭鍖栧埌鏈€浣崇姸鎬?*

#### 涓夌缁勫悎妯″紡瀵规瘮锛坋xecution ADV妯″紡锛?
| 妯″紡 | 骞村寲鏀剁泭 | 澶忔櫘姣旂巼 | 鏈€澶у洖鎾?| Calmar | Sortino | 鑳滅巼 | 鍐插嚮鎴愭湰 |
|------|---------|---------|---------|--------|---------|------|---------|
| **optimizer** | 32.55% | 2.61 | 11.12% | 2.93 | 3.84 | 58.84% | 0.0003 |
| **simple_ls (execution)** | 30.74% | 3.02 | 8.40% | 3.66 | 4.29 | 59.61% | 0.0002 |
| simple_long (鍘熷) | 8.24% | 0.99 | 12.13% | 0.68 | 1.42 | - | 0.0013 |
| simple_long (浼樺寲鍚? | 25.06% | 1.03 | 20.20% | 1.24 | 1.24 | 54.58% | 0.0012 |
| simple_ls (weight_cap) | -14.46% | -1.32 | 52.75% | -0.27 | -1.68 | - | 1.3145 |

**鍏抽敭鍙戠幇锛?*

1. **optimizer vs simple_ls (execution) 琛ㄧ幇闈炲父鎺ヨ繎**
   - simple_ls 澶忔櫘 (3.02) > optimizer (2.61)
   - simple_ls 鍥炴挙 (8.40%) < optimizer (11.12%)
   - optimizer 骞村寲鏀剁泭 (32.55%) > simple_ls (30.74%)
   - simple_ls Calmar (3.66) > optimizer (2.93)

2. **simple_ls execution 妯″紡鎺ㄨ崘鍘熷洜**
   - 鍐插嚮鎴愭湰鏈€浣?(0.0002)
   - 澶忔櫘姣旂巼鏈€楂?   - 鏈€澶у洖鎾ゆ渶浣?   - 閫傚悎浣庨闄╁亸濂芥姇璧勮€?
3. **optimizer 妯″紡鎺ㄨ崘鍘熷洜**
   - 骞村寲鏀剁泭鏈€楂?   - Calmar姣旂巼楂?   - 閫傚悎杩芥眰鏀剁泭鏈€澶у寲鐨勬姇璧勮€?
4. **weight_cap 妯″紡宸插純鐢?*
   - 鍐插嚮鎴愭湰鏄?execution 鐨?6572 鍊?   - 骞村寲鏀剁泭涓鸿礋

**绛栫暐琛ㄧ幇杈惧埌浼樼鐨勯噺鍖栫瓥鐣ユ按骞?*

## 椤圭洰鍏ㄩ潰瀹℃煡鎶ュ憡 (2026-05-12)

### 馃敶 P0 鈥?涓ラ噸锛堝繀椤讳慨澶嶏級

| # | 鏂囦欢:琛?| 闂 |
|---|---------|------|
| 1 | `backtest/engine.py:127-137` | **鍔犺浇GAT checkpoint蹇呭穿**锛歚load_v9_checkpoint` 鍒涘缓妯″瀷鏃?`use_gat=getattr(cfg,'use_gat',False)` 鎭掍负False锛屼絾GAT checkpoint鍖呭惈`gat_conv.*`鏉冮噸锛宍strict=True`鎶nexpected key(s) |
| 2 | `fundamental_factors.py:138` | **ROE浣跨敤绱YTD鍑€鍒╂鼎**锛歈1鍙湁3涓湀鍒╂鼎锛孮3鏈?涓湀锛岃法鎶ュ憡鏈熶笉鍙瘮銆傚簲鏀逛负TTM锛堟渶杩?瀛ｅ害婊氬姩姹傚拰锛?|
| 3 | `backtest/engine.py:623` | **纭紪鐮?7**锛歚sample['risk'][valid, 87:]`鍋囧畾`use_macro_features=True`銆傚叧闂畯瑙傛椂risk缁村害=167锛屽垏鐗嘸87:`浼氬垏鎺夊墠3涓涓歰ne-hot鍒?|
| 4 | `data/pipeline.py:109,113,145` | **缂撳瓨key璇**锛歚test_stocks`鏈缃椂榛樿涓篘one锛岀紦瀛榢ey鏄剧ず"allstocks"浣嗗疄闄呭彧鍔犺浇1000鍙?|
| 5 | `data/pipeline.py:128` | **缂撳瓨key缂哄皯max_horizon**锛氭敼`max_horizon`涓嶄娇缂撳瓨澶辨晥锛屽弽搴忓垪鍖栧悗鏍囩褰㈢姸涓嶅尮閰?|

### 馃煚 P1 鈥?閲嶈锛堝簲灏藉揩淇锛?
| # | 鏂囦欢:琛?| 闂 |
|---|---------|------|
| 6 | `core/train_utils.py:273` | `train_model()`榛樿`use_amp=True`锛屼笌CLAUDE.md瑕佹眰AMP OFF鐭涚浘 |
| 7 | `core/train_utils.py:492-497` | `<8GB GPU璁綽atch_size=4`锛屼絾6GB鐨凴TX 2060闇€瑕乥atch_size=2 |
| 8 | `backtest/engine.py:700-704` | **exit_day鍚庢湭娓呬粨**锛氭寔浠撴敹鐩婃湡缁撴潫鍚庯紝鏉冮噸娈嬬暀鍦╜daily_total_weights`涓洿鍒颁笅娆¤皟浠撹鐩?|
| 9 | `core/train_utils.py:218` | OOM妫€娴嬫潯浠禶"cuda" in msg`澶娉涳紝鍚炴帀鎵€鏈塁UDA閿欒 |
| 10 | `data/macro_factors.py:65-68` | 鍖楀悜璧勯噾z-score**鏈娇鐢╜.shift(1)`**锛屽寘鍚綋鍓嶅€硷紝涓嶱MI鐨刞.shift(1)`涓嶄竴鑷?|
| 11 | `data/pipeline.py:295` | risk涓璥size`鍥犲瓙鐢╜log_volume`锛堟垚浜ら噺锛夛紝鑰岄潪甯傚€尖€斺€斿洜瀛愬懡鍚嶆湁璇鎬?|

### 馃煛 P2 鈥?涓瓑锛堝彲寤跺悗锛?
| # | 鏂囦欢:琛?| 闂 |
|---|---------|------|
| 12 | `core/train_utils.py:311-312` | Resume鍙猚atch`RuntimeError`锛宍KeyError`锛堢己`model_state_dict`锛夊鑷村穿婧?|
| 13 | `core/train_utils.py:377` | 璁粌寰幆`del`閬楁紡`industry_ids`锛屾渶鍚庝竴涓猙atch鐨勮涓歵ensor鍗犵敤鏄惧瓨 |
| 14 | `core/train_utils.py:168` | `total_loss_v7()`鐨刞y`鍙傛暟浠庢湭浣跨敤锛堟鍙傛暟锛墊
| 15 | `data/fundamental_factors.py:135-136` | `effective_date = ann_cols.max(axis=1)`鍦∟aT鏃秔andas鐗堟湰琛屼负涓嶅悓 |
| 16 | `data/fundamental_factors.py:126` | `revenue_yoy`鐢╜pct_change(4)`纭€у亸绉?琛岋紝绌虹己瀛ｅ害闈欓粯閿欒瀵归綈 |
| 17 | `backtest/engine.py:536-538` | `hs300_index.csv`缂哄け鏃堕潤榛樿繑鍥為浂鏀剁泭锛屾棤浠讳綍璀﹀憡 |
| 18 | `backtest/engine.py:145-146` | `checkpoint.get('epoch')`瀵硅８state_dict闈欓粯杩斿洖None |
| 19 | `data/pipeline.py:201` | `volume_spike`鐢╜log_volume.pct_change()`锛岄潪鏍囧噯璁＄畻 |
| 20 | `data/pipeline.py:210` | `gap[0]`鐢变簬`shift(1)`濮嬬粓NaN锛屼涪寮冪涓€涓湁鏁堢獥鍙?|

### 馃煝 P3 鈥?浣庯紙鏂囨。/娓呯悊锛?
| # | 鏂囦欢:琛?| 闂 |
|---|---------|------|
| 21 | `CLAUDE.md:469` | `max_edges_per_stock=5`浣嗕唬鐮佸疄闄呯敤8 |
| 22 | `CLAUDE.md:45` | 璇碻train_v9.py`榛樿`test_mode=True`锛屼絾瀹為檯浠ｇ爜涓篳test_mode=False` |
| 23 | `core/train_utils.py:22` | 姝诲彉閲廯REGIME_DIM`鍦ㄦā鍧楃骇瀹氫箟浣嗕粠鏈娇鐢?|
| 24 | `data/market_features.py:47-63` | 搴熷純姝诲嚱鏁癭_compute_index_features`浠庢湭琚皟鐢?|
| 25 | `backtest/risk.py:112-113` | `if half_life == 0`鏄案False鐨勬浠ｇ爜 |

### 鍚庣画寤鸿

1. **闀挎湡鍥炴祴**锛氭墿灞曞洖娴嬫湡鍒?014-2024锛岄獙璇佺瓥鐣ョǔ瀹氭€?2. **瀹炵洏妯℃嫙**锛氫娇鐢ㄦ渶鏂版暟鎹繘琛屾ā鎷熶氦鏄擄紝楠岃瘉瀹炵洏鍙鎬?3. **鍙傛暟浼樺寲**锛氬 lambda_t銆乴ambda_b銆乼arget_vol 绛夊弬鏁拌繘琛屾晱鎰熸€у垎鏋?4. **simple_ls vs optimizer 閫夋嫨鎸囧崡**锛?   - 浣庨闄╁亸濂斤細閫夋嫨 simple_ls锛堟洿浣庡洖鎾ゃ€佹洿楂樺鏅級
   - 楂樻敹鐩婅拷姹傦細閫夋嫨 optimizer锛堟洿楂樺勾鍖栥€佹洿楂?Calmar锛?