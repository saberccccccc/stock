import numpy as np


class V9GATIntersectionPredictor:
    name = "v9_gat_intersection"

    def __init__(self, v9_predictor, gat_predictor, top_frac=0.10):
        self.v9_predictor = v9_predictor
        self.gat_predictor = gat_predictor
        self.top_frac = top_frac
        self.reset_stats()

    def reset_stats(self):
        self.top_counts = []
        self.bottom_counts = []
        self.n_counts = []

    def predict_alpha(self, sample, valid, regime):
        alpha_v9 = self.v9_predictor.predict_alpha(sample, valid, regime)
        alpha_gat = self.gat_predictor.predict_alpha(sample, valid, regime)
        n = len(alpha_v9)
        if n == 0:
            return np.array([], dtype=np.float32)

        k = max(1, int(n * self.top_frac))
        v9_order = np.argsort(alpha_v9)
        gat_order = np.argsort(alpha_gat)
        top_idx = sorted(set(v9_order[-k:]) & set(gat_order[-k:]))
        bottom_idx = sorted(set(v9_order[:k]) & set(gat_order[:k]))

        rank_v9 = np.argsort(np.argsort(alpha_v9)).astype(np.float64) / max(n - 1, 1)
        rank_gat = np.argsort(np.argsort(alpha_gat)).astype(np.float64) / max(n - 1, 1)
        score = np.zeros(n, dtype=np.float32)

        if top_idx:
            score[top_idx] = ((rank_v9[top_idx] + rank_gat[top_idx]) / 2.0).astype(np.float32)
        if bottom_idx:
            score[bottom_idx] = (-((1.0 - rank_v9[bottom_idx]) + (1.0 - rank_gat[bottom_idx])) / 2.0).astype(np.float32)

        self.top_counts.append(len(top_idx))
        self.bottom_counts.append(len(bottom_idx))
        self.n_counts.append(n)
        return score

    def stats(self):
        if not self.n_counts:
            return {}
        return {
            "avg_universe": float(np.mean(self.n_counts)),
            "avg_top_intersection": float(np.mean(self.top_counts)),
            "avg_bottom_intersection": float(np.mean(self.bottom_counts)),
            "avg_top_intersection_pct": float(np.mean(np.array(self.top_counts) / np.array(self.n_counts))),
            "avg_bottom_intersection_pct": float(np.mean(np.array(self.bottom_counts) / np.array(self.n_counts))),
        }


class V9GATEnsemblePredictor:
    def __init__(self, v9_predictor, gat_predictor, strategy, top_pct=0.10, cache=None):
        self.v9_predictor = v9_predictor
        self.gat_predictor = gat_predictor
        self.strategy = strategy
        self.top_pct = top_pct
        self.cache = {} if cache is None else cache
        self.name = f"v9_gat_{strategy}"
        self.reset_stats()

    def reset_stats(self):
        self.n_counts = []
        self.long_counts = []
        self.short_counts = []

    def _base_alpha(self, sample, valid, regime):
        key = sample["date"]
        if key not in self.cache:
            self.cache[key] = (
                self.v9_predictor.predict_alpha(sample, valid, regime),
                self.gat_predictor.predict_alpha(sample, valid, regime),
            )
        return self.cache[key]

    def predict_alpha(self, sample, valid, regime):
        alpha_v9, alpha_gat = self._base_alpha(sample, valid, regime)
        n = len(alpha_v9)
        if n == 0:
            return np.array([], dtype=np.float32)

        if self.strategy == "avg_score":
            score = ((alpha_v9 + alpha_gat) / 2.0).astype(np.float32)
            k = max(1, int(n * self.top_pct))
            self.n_counts.append(n)
            self.long_counts.append(k)
            self.short_counts.append(k)
            return score

        k = max(1, int(n * self.top_pct))
        v9_order = np.argsort(alpha_v9)
        gat_order = np.argsort(alpha_gat)
        v9_top = set(v9_order[-k:])
        gat_top = set(gat_order[-k:])
        v9_bottom = set(v9_order[:k])
        gat_bottom = set(gat_order[:k])

        if self.strategy in ("union", "union_max_score"):
            long_idx = sorted(v9_top | gat_top)
            short_idx = sorted(v9_bottom | gat_bottom)
        elif self.strategy == "top_union_bottom_intersection":
            long_idx = sorted(v9_top | gat_top)
            short_idx = sorted(v9_bottom & gat_bottom)
        else:
            raise ValueError(f"unknown strategy: {self.strategy}")

        rank_v9 = np.argsort(np.argsort(alpha_v9)).astype(np.float64) / max(n - 1, 1)
        rank_gat = np.argsort(np.argsort(alpha_gat)).astype(np.float64) / max(n - 1, 1)
        score = np.zeros(n, dtype=np.float32)
        if long_idx:
            if self.strategy == "union_max_score":
                score[long_idx] = np.maximum(rank_v9[long_idx], rank_gat[long_idx]).astype(np.float32)
            else:
                score[long_idx] = ((rank_v9[long_idx] + rank_gat[long_idx]) / 2.0).astype(np.float32)
        if short_idx:
            if self.strategy == "union_max_score":
                score[short_idx] = -np.maximum(1.0 - rank_v9[short_idx], 1.0 - rank_gat[short_idx]).astype(np.float32)
            else:
                score[short_idx] = (-((1.0 - rank_v9[short_idx]) + (1.0 - rank_gat[short_idx])) / 2.0).astype(np.float32)

        self.n_counts.append(n)
        self.long_counts.append(len(long_idx))
        self.short_counts.append(len(short_idx))
        return score

    def stats(self):
        if not self.n_counts:
            return {}
        n = np.array(self.n_counts, dtype=np.float64)
        long = np.array(self.long_counts, dtype=np.float64)
        short = np.array(self.short_counts, dtype=np.float64)
        return {
            "avg_universe": float(n.mean()),
            "avg_long_count": float(long.mean()),
            "avg_short_count": float(short.mean()),
            "avg_long_pct": float((long / n).mean()),
            "avg_short_pct": float((short / n).mean()),
        }


class PersistentPredictor:
    """Multi-date persistent signal aggregator.

    Wraps any predictor and smooths alpha using a rolling window of past
    predictions, reducing single-day noise.

    Modes:
      - "average": mean alpha over the window (persistent high scorers)
      - "momentum": current alpha + (current - oldest) * momentum_boost (rising stars)
      - "composite": 0.5*avg + 0.3*consistency + 0.2*trend (combined)
    """

    def __init__(self, base_predictor, window=5, mode="composite", momentum_boost=0.5):
        self.base = base_predictor
        self.window = window
        self.mode = mode
        self.momentum_boost = momentum_boost
        self.name = f"{getattr(base_predictor, 'name', 'base')}_persistent_{mode}_w{window}"
        self._history = []  # list of {code: alpha}
        self._call_count = 0
        self._blend_weights = []

    def reset_stats(self):
        self._history = []
        self._call_count = 0
        self._blend_weights = []
        if hasattr(self.base, "reset_stats"):
            self.base.reset_stats()

    def predict_alpha(self, sample, valid, regime):
        codes = [sample["codes"][i] for i in range(len(sample["codes"])) if valid[i]]
        current = self.base.predict_alpha(sample, valid, regime)

        code_to_alpha = dict(zip(codes, current))
        self._history.append(code_to_alpha)
        if len(self._history) > self.window:
            self._history.pop(0)
        self._call_count += 1

        # For the first few calls, fall back to current alpha until window fills
        if self.window <= 1 or self._call_count < 2:
            return current

        n_codes = len(codes)
        result = np.zeros(n_codes, dtype=np.float32)

        for i, code in enumerate(codes):
            past = []
            for h in self._history:
                if code in h:
                    past.append(h[code])
            if len(past) < 2:
                result[i] = current[i]
                continue

            past_arr = np.array(past, dtype=np.float64)
            n_past = len(past_arr)
            avg = np.mean(past_arr)
            std = np.std(past_arr)
            current_val = past_arr[-1]

            if self.mode == "average":
                result[i] = float(avg)
            elif self.mode == "momentum":
                oldest = past_arr[0]
                mom = (current_val - oldest) * self.momentum_boost
                result[i] = float(current_val + mom)
            elif self.mode == "composite":
                # Percentiles within this stock's own history (0-1)
                ranks = np.argsort(np.argsort(past_arr)).astype(np.float64) / max(n_past - 1, 1)
                avg_pct = np.mean(ranks)
                pct_std = np.std(ranks)
                consistency = 1.0 / (1.0 + pct_std)
                # Trend: slope of rank over time
                x = np.arange(n_past, dtype=np.float64)
                if n_past >= 3 and pct_std > 1e-8:
                    slope = np.polyfit(x, ranks, 1)[0]
                else:
                    slope = 0.0
                composite = avg_pct * 0.5 + consistency * 0.3 + max(0.0, slope) * 10.0 * 0.2
                # Map composite (0~1) back to alpha scale via current alpha sign
                alpha_scale = np.std(past_arr) if std > 1e-8 else 1.0
                result[i] = float((composite - 0.5) * 2.0 * alpha_scale)
            else:
                result[i] = current[i]

        if len(codes) > 1:
            if np.std(result) > 1e-8:
                result = (result - np.mean(result)) / np.std(result)
            result = np.tanh(np.asarray(result, dtype=np.float32))
        return result.astype(np.float32)

    def stats(self):
        base_stats = self.base.stats() if hasattr(self.base, "stats") else {}
        return {
            **base_stats,
            "persistent_calls": self._call_count,
            "persistent_window_n": self.window,
        }
