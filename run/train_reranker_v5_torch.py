"""Train a Torch listwise marginal-fill reranker.

V5 is deliberately scoped to the last few fill decisions of the coarse ranker.
It does not rewrite the whole portfolio.  The training objective is daily
listwise ranking on executable open-to-open labels, with an auxiliary
candidate-vs-baseline head used only for scoring and gate calibration.
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "PyTorch is required. Use C:\\Users\\x\\miniconda3\\envs\\torch\\python.exe"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA_ROOT = ROOT / "reranker_v3_data_20260615"
OUTPUT_ROOT = ROOT / "reranker_models_20260702" / "torch_listwise_v5"

NON_FEATURES = {
    "split",
    "date",
    "code",
    "group_size",
    "relevance",
    "market_regime",
    "industry_id",
    "v3_is_kept",
    "v3_protected_fill",
    "v3_eligible",
    "baseline_target",
    "beats_baseline",
    "year",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--train-years", default="2018,2019,2020,2021,2022")
    parser.add_argument("--validation-year", type=int, default=2023)
    parser.add_argument("--candidate-end", type=int, default=80)
    parser.add_argument("--max-reranked-fills", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-temperature", type=float, default=0.08)
    parser.add_argument("--pair-weight", type=float, default=0.30)
    parser.add_argument("--bce-weight", type=float, default=0.25)
    parser.add_argument("--max-pair-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-days", type=int, default=None)
    parser.add_argument("--max-validation-days", type=int, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_columns(frame):
    columns = []
    for column in frame.columns:
        if column in NON_FEATURES:
            continue
        if column.startswith("future_") or column.startswith("exec_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def load_frames(data_root, years):
    data_root = Path(data_root)
    frames = []
    for path in sorted(data_root.glob("oof_F*_*/reranker_v3_dataset.parquet")):
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["year"] = frame["date"].dt.year
        year = int(frame["year"].mode().iloc[0])
        if year not in years:
            continue
        frame = frame[frame["v3_eligible"].eq(1) & frame["exec_target"].notna()].copy()
        frames.append(frame)
        print(
            f"loaded {path.parent.name}: rows={len(frame):,} "
            f"dates={frame['date'].nunique()} year={year}",
            flush=True,
        )
    if not frames:
        raise ValueError(f"No reranker frames found for years={years}")
    return pd.concat(frames, ignore_index=True)


def add_relative_labels(frame):
    baseline = (
        frame[frame["v3_baseline_fill"].eq(1)]
        .groupby("date")["exec_target_raw"]
        .mean()
        .rename("baseline_target")
    )
    frame = frame.join(baseline, on="date")
    frame["beats_baseline"] = (
        frame["exec_target_raw"] > frame["baseline_target"]
    ).astype(np.float32)
    return frame


def fit_scaler(frame, features):
    values = frame[features].replace([np.inf, -np.inf], np.nan)
    median = values.median(axis=0)
    mad = (values - median).abs().median(axis=0)
    std = values.std(axis=0)
    scale = mad.where(mad > 1e-6, std).fillna(1.0).clip(lower=1e-6)
    return median.astype(np.float32), scale.astype(np.float32)


def apply_scaler(frame, features, median, scale):
    values = frame[features].replace([np.inf, -np.inf], np.nan).fillna(median)
    values = ((values - median) / scale).clip(-8.0, 8.0)
    return values.to_numpy(dtype=np.float32, copy=True)


class DailyRerankDataset(Dataset):
    def __init__(self, frame, features, median, scale, max_days=None):
        frame = frame.sort_values(["date", "candidate_position"], kind="mergesort")
        dates = list(frame["date"].drop_duplicates())
        if max_days is not None:
            dates = dates[: int(max_days)]
            frame = frame[frame["date"].isin(dates)].copy()
        x_all = apply_scaler(frame, features, median, scale)
        self.groups = []
        offset = 0
        for date, group in frame.groupby("date", sort=False):
            n = len(group)
            x = x_all[offset : offset + n]
            offset += n
            slots = int(group["v3_rerank_slots"].iloc[0])
            baseline_mask = group["v3_baseline_fill"].to_numpy(dtype=np.float32)
            if slots <= 0 or baseline_mask.sum() != slots:
                continue
            target = group["exec_target"].to_numpy(dtype=np.float32)
            raw_target = group["exec_target_raw"].to_numpy(dtype=np.float32)
            beats = group["beats_baseline"].to_numpy(dtype=np.float32, copy=True)
            self.groups.append(
                {
                    "date": str(pd.Timestamp(date).date()),
                    "x": x,
                    "target": target,
                    "raw_target": raw_target,
                    "beats": beats,
                    "baseline": baseline_mask,
                    "slots": slots,
                }
            )

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        return self.groups[index]


def collate_daily(batch):
    max_n = max(item["x"].shape[0] for item in batch)
    feature_dim = batch[0]["x"].shape[1]
    size = len(batch)
    x = torch.zeros(size, max_n, feature_dim, dtype=torch.float32)
    target = torch.zeros(size, max_n, dtype=torch.float32)
    raw_target = torch.zeros(size, max_n, dtype=torch.float32)
    beats = torch.zeros(size, max_n, dtype=torch.float32)
    baseline = torch.zeros(size, max_n, dtype=torch.float32)
    mask = torch.zeros(size, max_n, dtype=torch.bool)
    slots = torch.zeros(size, dtype=torch.long)
    dates = []
    for i, item in enumerate(batch):
        n = item["x"].shape[0]
        x[i, :n] = torch.from_numpy(item["x"])
        target[i, :n] = torch.from_numpy(item["target"])
        raw_target[i, :n] = torch.from_numpy(item["raw_target"])
        beats[i, :n] = torch.from_numpy(item["beats"])
        baseline[i, :n] = torch.from_numpy(item["baseline"])
        mask[i, :n] = True
        slots[i] = int(item["slots"])
        dates.append(item["date"])
    return {
        "x": x,
        "target": target,
        "raw_target": raw_target,
        "beats": beats,
        "baseline": baseline,
        "mask": mask,
        "slots": slots,
        "dates": dates,
    }


class ListwiseReranker(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.15):
        super().__init__()
        self.item_net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Linear(hidden_dim * 2, 1)
        self.beat_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, mask):
        item = self.item_net(x)
        valid = mask.unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp_min(1.0)
        mean = (item * valid).sum(dim=1) / denom
        centered = (item - mean.unsqueeze(1)) * valid
        std = torch.sqrt((centered.square().sum(dim=1) / denom).clamp_min(1e-6))
        context = self.context_net(torch.cat([mean, std], dim=-1))
        context = context.unsqueeze(1).expand_as(item)
        fused = torch.cat([item, context], dim=-1)
        score = self.score_head(fused).squeeze(-1)
        beat_logit = self.beat_head(fused).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        beat_logit = beat_logit.masked_fill(~mask, 0.0)
        return score, beat_logit


def pairwise_loss(score, target, mask, max_pairs):
    losses = []
    batch = score.shape[0]
    for i in range(batch):
        valid = mask[i]
        s = score[i, valid]
        y = target[i, valid]
        n = len(y)
        if n < 2:
            continue
        diff_y = y[:, None] - y[None, :]
        pair_mask = diff_y > 0.05
        rows, cols = torch.where(pair_mask)
        if len(rows) == 0:
            continue
        if len(rows) > max_pairs:
            choice = torch.randperm(len(rows), device=rows.device)[:max_pairs]
            rows = rows[choice]
            cols = cols[choice]
        margin = s[rows] - s[cols]
        weight = diff_y[rows, cols].clamp(max=5.0)
        losses.append((F.softplus(-margin) * weight).mean())
    if not losses:
        return score.new_tensor(0.0)
    return torch.stack(losses).mean()


def compute_loss(batch, score, beat_logit, label_temperature, pair_weight, bce_weight, max_pairs):
    mask = batch["mask"]
    target = batch["target"]
    label_logits = (target / float(label_temperature)).masked_fill(~mask, -1e9)
    label_prob = torch.softmax(label_logits, dim=1)
    pred_log_prob = torch.log_softmax(score, dim=1)
    listwise = -(label_prob * pred_log_prob).sum(dim=1).mean()
    pair = pairwise_loss(score, target, mask, max_pairs)
    bce = F.binary_cross_entropy_with_logits(
        beat_logit[mask],
        batch["beats"][mask],
    )
    return listwise + pair_weight * pair + bce_weight * bce, {
        "listwise": float(listwise.detach().cpu()),
        "pair": float(pair.detach().cpu()),
        "bce": float(bce.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        score, beat_logit = model(x, mask)
        pred = score.detach().cpu().numpy()
        pred_win = torch.sigmoid(beat_logit).detach().cpu().numpy()
        raw = batch["raw_target"].numpy()
        baseline = batch["baseline"].numpy()
        mask_np = batch["mask"].numpy()
        for i, date in enumerate(batch["dates"]):
            valid = mask_np[i]
            slots = int(batch["slots"][i])
            raw_i = raw[i, valid]
            score_i = pred[i, valid]
            win_i = pred_win[i, valid]
            baseline_i = baseline[i, valid].astype(bool)
            if slots <= 0 or baseline_i.sum() != slots:
                continue
            selected_idx = np.argsort(-score_i, kind="mergesort")[:slots]
            baseline_idx = np.where(baseline_i)[0]
            delta = raw_i[selected_idx].mean() - raw_i[baseline_idx].mean()
            confidence = win_i[selected_idx].mean() - win_i[baseline_idx].mean()
            rows.append(
                {
                    "date": date,
                    "delta": float(delta),
                    "confidence": float(confidence),
                    "selected_raw": float(raw_i[selected_idx].mean()),
                    "baseline_raw": float(raw_i[baseline_idx].mean()),
                    "changed": int(len(set(selected_idx) - set(baseline_idx))),
                }
            )
    report = pd.DataFrame(rows)
    if report.empty:
        return {
            "dates": 0,
            "mean_delta": 0.0,
            "positive_days": 0.0,
            "mean_confidence": 0.0,
            "mean_changed": 0.0,
        }, report
    return {
        "dates": int(len(report)),
        "mean_delta": float(report["delta"].mean()),
        "positive_days": float((report["delta"] > 0).mean()),
        "mean_confidence": float(report["confidence"].mean()),
        "mean_changed": float(report["changed"].mean()),
    }, report


def to_device(batch, device):
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device) if torch.is_tensor(value) else value
    return output


def train(args):
    set_seed(args.seed)
    train_years = {int(x) for x in args.train_years.split(",") if x.strip()}
    validation_year = int(args.validation_year)
    years = sorted(train_years | {validation_year})
    data = add_relative_labels(load_frames(args.data_root, set(years)))
    train_frame = data[data["year"].isin(train_years)].copy()
    validation_frame = data[data["year"].eq(validation_year)].copy()
    features = feature_columns(train_frame)
    median, scale = fit_scaler(train_frame, features)

    train_ds = DailyRerankDataset(
        train_frame,
        features,
        median,
        scale,
        max_days=args.max_train_days if args.smoke else args.max_train_days,
    )
    validation_ds = DailyRerankDataset(
        validation_frame,
        features,
        median,
        scale,
        max_days=args.max_validation_days if args.smoke else args.max_validation_days,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_daily,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_daily,
        num_workers=0,
    )
    requested = args.device
    device = torch.device(
        "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = ListwiseReranker(
        feature_dim=len(features),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        parts = []
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            score, beat_logit = model(batch["x"], batch["mask"])
            loss, detail = compute_loss(
                batch,
                score,
                beat_logit,
                args.label_temperature,
                args.pair_weight,
                args.bce_weight,
                args.max_pair_samples,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            parts.append(detail)
        scheduler.step()
        metrics, daily = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "lr": float(scheduler.get_last_lr()[0]),
            **metrics,
            "listwise": float(np.mean([x["listwise"] for x in parts])),
            "pair": float(np.mean([x["pair"] for x in parts])),
            "bce": float(np.mean([x["bce"] for x in parts])),
        }
        history.append(row)
        key = (
            row["mean_delta"],
            row["positive_days"],
            -abs(row["mean_changed"] - args.max_reranked_fills),
            -epoch,
        )
        if best is None or key > best[0]:
            best = (key, epoch, row)
            torch.save(model.state_dict(), output / "model_best.pt")
            daily.to_csv(output / "validation_daily_best.csv", index=False)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
                f"val_delta={row['mean_delta']:.6f} "
                f"pos={row['positive_days']:.3f} changed={row['mean_changed']:.2f}",
                flush=True,
            )
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output / "training_history.csv", index=False)
    best_epoch = int(best[1])
    summary = {
        "model": "torch_listwise_v5",
        "data_root": str(Path(args.data_root)),
        "train_years": sorted(train_years),
        "validation_year": validation_year,
        "features": len(features),
        "train_days": len(train_ds),
        "validation_days": len(validation_ds),
        "best_epoch": best_epoch,
        "best_validation": best[2],
        "candidate_end": int(args.candidate_end),
        "max_reranked_fills": int(args.max_reranked_fills),
        "device": str(device),
        "label": "exec_target_raw/open-to-open executable marginal fill target",
        "selection_rule": "choose by validation mean_delta, positive_days; 2024/2025 open-ledger still required before promotion",
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output / "preprocess.pkl").open("wb") as handle:
        pickle.dump(
            {
                "feature_columns": features,
                "median": median,
                "scale": scale,
                "args": vars(args),
            },
            handle,
        )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
