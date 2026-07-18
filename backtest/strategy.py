"""Pure score-to-target-holdings policies for the realistic open ledger.

Policies never inspect prices, cash, lots, ADV, or price limits.  They propose
ranked target names; ``open_ledger`` remains the only execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyProposal:
    selected: tuple[str, ...]
    retained: tuple[str, ...]
    target_n: int
    rationale: str


def retention_target(ranked_codes, current_codes, *, target_frac, hold_frac):
    """Build the existing rank-retention target without execution assumptions."""
    ranked = tuple(dict.fromkeys(str(code) for code in ranked_codes))
    n = len(ranked)
    if n == 0:
        return StrategyProposal((), (), 0, "empty_ranked_universe")
    target_n = max(1, int(n * float(target_frac)))
    hold_n = max(target_n, int(n * float(hold_frac)))
    rank_map = {code: rank for rank, code in enumerate(ranked)}
    retained = []
    retained_set = set()
    for code in current_codes:
        code = str(code)
        if code not in retained_set and rank_map.get(code, n + 1) < hold_n:
            retained.append(code)
            retained_set.add(code)
        if len(retained) >= target_n:
            break
    selected = list(retained)
    selected_set = set(selected)
    for code in ranked:
        if len(selected) >= target_n:
            break
        if code not in selected_set:
            selected.append(code)
            selected_set.add(code)
    return StrategyProposal(tuple(selected), tuple(retained), target_n, "rank_retention")


def topk_dropout_target(ranked_codes, current_codes, *, top_k, n_drop):
    """Propose TopK holdings while replacing at most ``n_drop`` current names."""
    ranked = tuple(dict.fromkeys(str(code) for code in ranked_codes))
    top_k = max(0, min(int(top_k), len(ranked)))
    n_drop = max(0, int(n_drop))
    if top_k == 0:
        return StrategyProposal((), (), 0, "topk_dropout_empty")
    rank_map = {code: rank for rank, code in enumerate(ranked)}
    current_ranked = sorted(
        (str(code) for code in current_codes if str(code) in rank_map),
        key=lambda code: rank_map[code],
    )
    if not current_ranked:
        return StrategyProposal(tuple(ranked[:top_k]), (), top_k, "topk_dropout_initial_build")
    retained = current_ranked[: max(top_k - n_drop, 0)]
    selected = list(retained)
    selected_set = set(selected)
    new_count = 0
    for code in ranked:
        if len(selected) >= top_k:
            break
        if code in selected_set:
            continue
        is_new = code not in set(current_ranked)
        if is_new and new_count >= n_drop:
            continue
        selected.append(code)
        selected_set.add(code)
        new_count += int(is_new)
    for code in current_ranked:
        if len(selected) >= top_k:
            break
        if code not in selected_set:
            selected.append(code)
            selected_set.add(code)
    return StrategyProposal(tuple(selected), tuple(retained), top_k, "topk_dropout")


def select_target_policy(
    ranked_codes,
    current_codes,
    *,
    policy,
    target_frac,
    hold_frac,
    top_k=0,
    n_drop=0,
):
    """Dispatch a named policy while keeping its target-size rule explicit."""
    if policy == "retention":
        return retention_target(
            ranked_codes,
            current_codes,
            target_frac=target_frac,
            hold_frac=hold_frac,
        )
    if policy == "topk_dropout":
        resolved_top_k = int(top_k) if int(top_k) > 0 else max(1, int(len(ranked_codes) * float(target_frac)))
        return topk_dropout_target(
            ranked_codes,
            current_codes,
            top_k=resolved_top_k,
            n_drop=n_drop,
        )
    raise ValueError(f"unknown selection policy: {policy}")
