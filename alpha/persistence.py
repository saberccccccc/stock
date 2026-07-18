"""Pure alpha-score transforms that do not require model inference."""

from __future__ import annotations

from collections import deque

import numpy as np


def rank_alpha_rows(score_rows):
    """Sort score rows into the project's JSONL alpha-ranking representation."""
    ranked_rows = []
    for row in score_rows:
        codes = list(row["codes"])
        scores = np.asarray(row["alpha"], dtype=np.float32)
        if len(codes) != len(scores):
            raise ValueError("alpha row has mismatched codes and alpha lengths")
        order = np.argsort(scores)[::-1]
        ranked_rows.append(
            {
                "date": row["date"],
                "codes": np.asarray(codes, dtype=object)[order].tolist(),
                "alpha": scores[order].astype(float).tolist(),
                "n_stocks": int(row.get("n_stocks", len(codes))),
            }
        )
    return ranked_rows


def rolling_average_alpha_scores(score_rows, window):
    """Apply ``PersistentPredictor(mode='average')`` without re-running a model.

    Rows must retain each day's original code order. The first row, and stocks
    with fewer than two observations in the rolling window, retain their
    current score. Later rows use the rolling per-code mean, then the same
    cross-sectional z-score/tanh normalization as ``PersistentPredictor``.
    """
    if int(window) < 1:
        raise ValueError("window must be >= 1")

    history = deque(maxlen=int(window))
    transformed = []
    for row in score_rows:
        codes = list(row["codes"])
        current = np.asarray(row["alpha"], dtype=np.float32)
        if len(codes) != len(current):
            raise ValueError("alpha row has mismatched codes and alpha lengths")

        history.append(dict(zip(codes, current)))
        result = current.copy()
        if int(window) > 1 and len(history) >= 2:
            result = np.zeros(len(codes), dtype=np.float32)
            for index, code in enumerate(codes):
                values = [past[code] for past in history if code in past]
                if len(values) < 2:
                    result[index] = current[index]
                else:
                    result[index] = float(np.mean(np.asarray(values, dtype=np.float64)))
            if len(codes) > 1:
                if np.std(result) > 1e-8:
                    result = (result - np.mean(result)) / np.std(result)
                result = np.tanh(np.asarray(result, dtype=np.float32))

        transformed.append(
            {
                "date": row["date"],
                "codes": codes,
                "alpha": result.astype(float).tolist(),
                "n_stocks": int(row.get("n_stocks", len(codes))),
            }
        )
    return transformed
