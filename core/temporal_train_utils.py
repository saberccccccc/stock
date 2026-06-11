"""Dataset and collate helpers for temporal-tower experiments."""

import numpy as np
import torch
from torch.utils.data import Dataset

SCALE = 1000
SENTINEL = np.int16(-32768)


def _open_memmap(path, dtype, shape):
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _split_indices(meta, split):
    if split == "train":
        return list(meta["train_indices"])
    if split == "val":
        return list(meta["val_indices"])
    if split == "test":
        return list(meta.get("test_indices", []))
    if split == "trainval":
        return list(meta["train_indices"]) + list(meta["val_indices"])
    if split == "all":
        return list(meta.get("valid_indices", []))
    raise ValueError(f"unknown temporal split: {split}")


class TemporalMemmapDataset(Dataset):
    """Lazy temporal cross-section dataset.

    Each item is one trading date. It returns the existing cross-section feature
    tensor X plus a true per-stock sequence X_seq ending at the same date.
    """

    def __init__(self, meta, split="train", min_stocks=None, validate_windows=True):
        self.meta = meta
        self.split = split
        self.all_codes = np.asarray(meta["all_codes"])
        self.all_dates = list(meta["all_dates"])
        self.industry_array = meta["industry_array"]
        self.max_horizon = int(meta["max_horizon"])
        self.lookback = int(meta["seq_lookback"])
        self.min_stocks = int(min_stocks if min_stocks is not None else meta.get("min_stocks", 30))

        n_stocks = len(self.all_codes)
        n_dates = len(self.all_dates)
        self.X_mm = _open_memmap(meta["x_norm_path"], np.int16, (n_stocks, n_dates, int(meta["x_dim"])))
        self.R_mm = _open_memmap(meta["risk_full_path"], np.int16, (n_stocks, n_dates, int(meta["risk_full_dim"])))
        self.Y_mm = _open_memmap(meta["y_norm_path"], np.int16, (n_stocks, n_dates))
        self.YS_mm = _open_memmap(meta["y_seq_norm_path"], np.int16, (n_stocks, n_dates, self.max_horizon))
        self.SEQ_mm = _open_memmap(meta["seq_norm_path"], np.int16, (n_stocks, n_dates, int(meta["seq_dim"])))

        time_indices = _split_indices(meta, split)
        if validate_windows:
            self.time_indices = self._filter_valid_times(time_indices)
        else:
            self.time_indices = list(time_indices)

    def _filter_valid_times(self, time_indices):
        valid_times = []
        for t in time_indices:
            if t - self.lookback + 1 < 0:
                continue
            valid_today = (self.X_mm[:, t, 0] != SENTINEL) & (self.Y_mm[:, t] != SENTINEL)
            seq_ok = (self.SEQ_mm[:, t - self.lookback + 1:t + 1, 0] != SENTINEL).all(axis=1)
            if int((valid_today & seq_ok).sum()) >= self.min_stocks:
                valid_times.append(t)
        return valid_times

    def __len__(self):
        return len(self.time_indices)

    def __getitem__(self, idx):
        t = self.time_indices[idx]
        start = t - self.lookback + 1

        valid_today = (self.X_mm[:, t, 0] != SENTINEL) & (self.Y_mm[:, t] != SENTINEL)
        valid_seq = (self.SEQ_mm[:, start:t + 1, 0] != SENTINEL).all(axis=1)
        valid_idx = np.where(valid_today & valid_seq)[0]

        x = self.X_mm[valid_idx, t, :].astype(np.float32) / SCALE
        x_seq = self.SEQ_mm[valid_idx, start:t + 1, :].astype(np.float32) / SCALE
        y = self.Y_mm[valid_idx, t].astype(np.float32) / SCALE
        y_seq = self.YS_mm[valid_idx, t, :].astype(np.float32) / SCALE
        risk = self.R_mm[valid_idx, t, :].astype(np.float32) / SCALE
        industry_ids = self.industry_array[valid_idx, t].astype(np.int64)

        return {
            "X": torch.from_numpy(x).float(),
            "X_seq": torch.from_numpy(x_seq).float(),
            "y": torch.from_numpy(y).float(),
            "y_seq": torch.from_numpy(y_seq).float(),
            "risk": torch.from_numpy(risk).float(),
            "industry_ids": torch.from_numpy(industry_ids).long(),
            "time_index": torch.tensor(t, dtype=torch.long),
        }


def _deterministic_subsample(n, max_stocks, seed):
    if max_stocks is None or n <= max_stocks:
        return None
    gen = torch.Generator()
    gen.manual_seed(int(seed) % (2**31 - 1))
    return torch.randperm(n, generator=gen)[:max_stocks]


def collate_temporal_eval(batch, max_stocks=None):
    batch = [item for item in batch if item["X"].shape[0] > 0]
    if not batch:
        raise ValueError("empty temporal batch")

    if max_stocks is not None:
        capped = []
        for item in batch:
            n = item["X"].shape[0]
            idx = _deterministic_subsample(n, max_stocks, int(item["time_index"]))
            if idx is None:
                capped.append(item)
            else:
                capped.append({
                    "X": item["X"][idx],
                    "X_seq": item["X_seq"][idx],
                    "y": item["y"][idx],
                    "y_seq": item["y_seq"][idx],
                    "risk": item["risk"][idx],
                    "industry_ids": item["industry_ids"][idx],
                    "time_index": item["time_index"],
                })
        batch = capped

    batch_size = len(batch)
    max_n = max(item["X"].shape[0] for item in batch)
    x_dim = batch[0]["X"].shape[1]
    lookback = batch[0]["X_seq"].shape[1]
    seq_dim = batch[0]["X_seq"].shape[2]
    horizon = batch[0]["y_seq"].shape[1]
    risk_dim = batch[0]["risk"].shape[1]

    X = torch.zeros(batch_size, max_n, x_dim)
    X_seq = torch.zeros(batch_size, max_n, lookback, seq_dim)
    y = torch.zeros(batch_size, max_n)
    y_seq = torch.zeros(batch_size, max_n, horizon)
    risk = torch.zeros(batch_size, max_n, risk_dim)
    industry_ids = torch.full((batch_size, max_n), -1, dtype=torch.long)
    mask = torch.zeros(batch_size, max_n, dtype=torch.bool)
    time_index = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["X"].shape[0]
        X[i, :n] = item["X"]
        X_seq[i, :n] = item["X_seq"]
        y[i, :n] = item["y"]
        y_seq[i, :n] = item["y_seq"]
        risk[i, :n] = item["risk"]
        industry_ids[i, :n] = item["industry_ids"]
        mask[i, :n] = True
        time_index[i] = item["time_index"]

    return {
        "X": X,
        "X_seq": X_seq,
        "y": y,
        "y_seq": y_seq,
        "risk": risk,
        "industry_ids": industry_ids,
        "mask": mask,
        "time_index": time_index,
    }


def _sample_top_bottom_random(item, keep_n, top_frac=0.20, bottom_frac=0.20, random_frac=0.60, target_horizon=4):
    n = item["X"].shape[0]
    if keep_n >= n:
        return torch.arange(n)

    y_seq = item["y_seq"]
    h_idx = min(max(int(target_horizon), 0), y_seq.shape[1] - 1)
    target = y_seq[:, h_idx]
    finite = torch.isfinite(target)
    if finite.sum().item() < max(10, keep_n // 4):
        return torch.randperm(n)[:keep_n]

    valid_idx = finite.nonzero(as_tuple=True)[0]
    valid_target = target[valid_idx]
    order = torch.argsort(valid_target)

    n_top = min(int(round(keep_n * float(top_frac))), valid_idx.numel() // 2)
    n_bottom = min(int(round(keep_n * float(bottom_frac))), valid_idx.numel() // 2)
    n_top = max(0, n_top)
    n_bottom = max(0, n_bottom)

    parts = []
    used = torch.zeros(n, dtype=torch.bool)
    if n_top > 0:
        top_pool = valid_idx[order[-max(n_top * 3, n_top):]]
        top_take = top_pool[torch.randperm(top_pool.numel())[:n_top]]
        parts.append(top_take)
        used[top_take] = True
    if n_bottom > 0:
        bottom_pool = valid_idx[order[:max(n_bottom * 3, n_bottom)]]
        bottom_take = bottom_pool[torch.randperm(bottom_pool.numel())[:n_bottom]]
        parts.append(bottom_take)
        used[bottom_take] = True

    remain_n = keep_n - sum(p.numel() for p in parts)
    if remain_n > 0:
        candidates = (~used).nonzero(as_tuple=True)[0]
        if candidates.numel() > 0:
            take = candidates[torch.randperm(candidates.numel())[:min(remain_n, candidates.numel())]]
            parts.append(take)

    idx = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
    if idx.numel() < keep_n:
        used = torch.zeros(n, dtype=torch.bool)
        if idx.numel() > 0:
            used[idx] = True
        candidates = (~used).nonzero(as_tuple=True)[0]
        if candidates.numel() > 0:
            extra = candidates[torch.randperm(candidates.numel())[:keep_n - idx.numel()]]
            idx = torch.cat([idx, extra])
    return idx[torch.randperm(idx.numel())[:keep_n]]


def collate_temporal_train(
    batch,
    keep_ratio=0.7,
    min_keep=20,
    max_stocks=None,
    sample_mode="random",
    top_frac=0.20,
    bottom_frac=0.20,
    random_frac=0.60,
    target_horizon=4,
):
    sampled = []
    for item in batch:
        n = item["X"].shape[0]
        if n == 0:
            continue
        keep_n = max(min_keep, int(n * keep_ratio))
        if max_stocks is not None:
            keep_n = min(keep_n, int(max_stocks))
        keep_n = min(keep_n, n)
        if sample_mode == "target_top_bottom_random":
            idx = _sample_top_bottom_random(
                item,
                keep_n,
                top_frac=top_frac,
                bottom_frac=bottom_frac,
                random_frac=random_frac,
                target_horizon=target_horizon,
            )
        else:
            idx = torch.randperm(n)[:keep_n]
        sampled.append({
            "X": item["X"][idx],
            "X_seq": item["X_seq"][idx],
            "y": item["y"][idx],
            "y_seq": item["y_seq"][idx],
            "risk": item["risk"][idx],
            "industry_ids": item["industry_ids"][idx],
            "time_index": item["time_index"],
        })
    return collate_temporal_eval(sampled)
