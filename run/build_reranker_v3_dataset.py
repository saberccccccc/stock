"""Build state-aware candidate rows for marginal retention-portfolio fills."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run.build_reranker_v2_dataset import (
    daily_normalize,
    market_features,
    stock_label_rows,
)


SOURCE_ROOT = Path("reranker_data_20260614")
OUTPUT_ROOT = Path("reranker_v3_data_20260615")
DATA_DIR = Path("data/raw")
MAX_CANDIDATE_RANK = 80
MAX_RERANKED_FILLS = 3


def add_state_features(frame, audit):
    audit = audit.copy()
    audit["date"] = pd.to_datetime(audit["date"])
    universe_by_date = audit.set_index("date")["universe_size"].astype(int).to_dict()
    current_selected = []
    holding_ages = {}
    outputs = []
    for date, group in frame.groupby("date", sort=True):
        group = group.sort_values("candidate_position", kind="mergesort").copy()
        universe = universe_by_date[date]
        target_n = max(1, int(universe * 0.006))
        hold_n = max(target_n, int(universe * 0.10))
        rank_map = dict(zip(group["code"], group["candidate_position"]))
        kept = [
            code
            for code in current_selected
            if rank_map.get(code, universe + 1) < hold_n
        ]
        if len(kept) > target_n:
            kept = sorted(kept, key=lambda code: rank_map[code])[:target_n]
        vacancies = max(target_n - len(kept), 0)
        fill_candidates = [
            code
            for code in group["code"]
            if code not in set(kept)
        ]
        baseline_fills = fill_candidates[:vacancies]
        protected_fill_count = max(vacancies - MAX_RERANKED_FILLS, 0)
        protected_fills = set(baseline_fills[:protected_fill_count])
        eligible = {
            code
            for code in fill_candidates[protected_fill_count:]
            if rank_map[code] < MAX_CANDIDATE_RANK
        }
        current_set = set(current_selected)
        kept_set = set(kept)
        baseline_fill_set = set(baseline_fills)

        group["v3_was_held"] = group["code"].isin(current_set).astype(np.int8)
        group["v3_is_kept"] = group["code"].isin(kept_set).astype(np.int8)
        group["v3_holding_age"] = group["code"].map(holding_ages).fillna(0).astype(np.int16)
        group["v3_vacancies"] = vacancies
        group["v3_rerank_slots"] = min(vacancies, MAX_RERANKED_FILLS)
        group["v3_protected_fill"] = group["code"].isin(protected_fills).astype(np.int8)
        group["v3_eligible"] = group["code"].isin(eligible).astype(np.int8)
        group["v3_baseline_fill"] = group["code"].isin(baseline_fill_set).astype(np.int8)
        outputs.append(group)

        selected = kept + baseline_fills
        selected_set = set(selected)
        for code in list(holding_ages):
            if code not in selected_set:
                holding_ages.pop(code, None)
        for code in selected:
            holding_ages[code] = holding_ages.get(code, 0) + 1
        current_selected = selected
    return pd.concat(outputs, ignore_index=True)


def process(source):
    frame = pd.read_parquet(source)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["code"] = frame["code"].astype(str)
    max_label_date = pd.Timestamp(f"{frame['date'].dt.year.max()}-12-31")
    label_frame = frame[frame["candidate_position"] < MAX_CANDIDATE_RANK]
    wanted = {
        code: set(group["date"])
        for code, group in label_frame.groupby("code", sort=False)
    }
    records = []
    for index, (code, dates) in enumerate(wanted.items(), start=1):
        path = DATA_DIR / f"{code}.csv"
        if path.exists():
            rows = stock_label_rows(path, dates, max_label_date)
            for row in rows:
                row["code"] = code
            records.extend(rows)
        if index % 500 == 0:
            print(f"{source.parent.name}: labels {index}/{len(wanted)}", flush=True)
    labels = pd.DataFrame(records)
    frame = frame.merge(labels, on=["date", "code"], how="left", validate="one_to_one")
    market = market_features(DATA_DIR / "hs300_index.csv")
    frame = frame.merge(
        market.reset_index().rename(columns={market.index.name or "index": "date"}),
        on="date",
        how="left",
        validate="many_to_one",
    )
    frame["exec_target"] = np.nan
    labelled = frame["exec_target_raw"].notna()
    normalized = daily_normalize(frame.loc[labelled].copy())
    frame.loc[labelled, "exec_target"] = normalized["exec_target"].to_numpy()
    audit = pd.read_csv(source.parent / "daily_audit.csv")
    frame = add_state_features(frame, audit)
    frame = frame[frame["candidate_position"] < MAX_CANDIDATE_RANK].copy()

    destination = OUTPUT_ROOT / source.parent.name / "reranker_v3_dataset.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    eligible = frame["v3_eligible"].eq(1) & frame["exec_target"].notna()
    summary = {
        "source": str(source),
        "rows": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "eligible_labelled_rows": int(eligible.sum()),
        "eligible_labelled_dates": int(frame.loc[eligible, "date"].nunique()),
        "mean_vacancies": float(frame.groupby("date")["v3_vacancies"].first().mean()),
        "mean_rerank_slots": float(frame.groupby("date")["v3_rerank_slots"].first().mean()),
        "held_eligible_share": float(frame.loc[eligible, "v3_was_held"].mean()),
    }
    destination.with_name("v3_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    sources = sorted(SOURCE_ROOT.glob("oof_F[1-6]_*/reranker_dataset.parquet"))
    sources.append(SOURCE_ROOT / "m0_validation_2024/reranker_dataset.parquet")
    for source in sources:
        process(source)


if __name__ == "__main__":
    main()
