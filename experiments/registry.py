"""Candidate registry for reproducible experiment comparisons."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ResultSource:
    path: str
    split: str | None = None
    stress: str = "normal"
    source_type: str = "open_ledger_summary"
    target_frac: float | None = None
    hold_frac: float | None = None
    portfolio_value: float | None = None


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    status: str
    execution_family: str
    description: str
    result_sources: tuple[ResultSource, ...] = field(default_factory=tuple)
    notes: str = ""


EXECUTION_OFFICIAL_OPEN_LEDGER = "official_open_price_share_ledger"


def _source(
    path,
    split=None,
    stress="normal",
    source_type="open_ledger_summary",
    target_frac=None,
    hold_frac=None,
    portfolio_value=None,
):
    return ResultSource(
        path=path,
        split=split,
        stress=stress,
        source_type=source_type,
        target_frac=target_frac,
        hold_frac=hold_frac,
        portfolio_value=portfolio_value,
    )


DEFAULT_CANDIDATES = (
    CandidateSpec(
        name="official",
        status="official",
        execution_family=EXECUTION_OFFICIAL_OPEN_LEDGER,
        description="V9 avgw3 + maxret095 + open-price share-ledger.",
        result_sources=(
            _source(
                "v9_avgw3_open_ledger_20260617/low_target_history_grid/val/open_ledger_summary.csv",
                "val",
                target_frac=0.006,
                hold_frac=0.10,
            ),
            _source("v9_avgw3_open_ledger_20260617/main_candidate_single_diag/test/open_ledger_summary.csv", "test"),
            _source("v9_avgw3_open_ledger_20260617/sweep_main_candidate_lag1/sweep_summary.csv", None, "lag1", "sweep_summary"),
            _source("v9_avgw3_open_ledger_20260617/sweep_main_candidate_cost2x/sweep_summary.csv", None, "cost2x", "sweep_summary"),
            _source("forward_results/frozen_v9_avgw3/primary_ledger_20260519_20260616_maxret095_mn5/open_ledger_summary.csv", "forward"),
            _source("forward_results/frozen_v9_avgw3/primary_ledger_20260519_20260616_maxret095_mn5_lag1/open_ledger_summary.csv", "forward", "lag1"),
            _source("forward_results/frozen_v9_avgw3/primary_ledger_20260519_20260616_maxret095_mn5_cost2x/open_ledger_summary.csv", "forward", "cost2x"),
        ),
    ),
    CandidateSpec(
        name="breadth_m085",
        status="shadow",
        execution_family=EXECUTION_OFFICIAL_OPEN_LEDGER,
        description="Official ranking plus weak-breadth market multiplier cap at 0.85.",
        result_sources=(
            _source("v9_avgw3_open_ledger_20260617/breadth_triggered_market/val_ma3_035_m085/open_ledger_summary.csv", "val"),
            _source("v9_avgw3_open_ledger_20260617/breadth_triggered_market/test_ma3_035_m085/open_ledger_summary.csv", "test"),
            _source("v9_avgw3_open_ledger_20260617/breadth_triggered_market/val_ma3_035_m085_lag1/open_ledger_summary.csv", "val", "lag1"),
            _source("v9_avgw3_open_ledger_20260617/breadth_triggered_market/test_ma3_035_m085_lag1/open_ledger_summary.csv", "test", "lag1"),
            _source("v9_avgw3_open_ledger_20260617/breadth_triggered_market/val_ma3_035_m085_cost2x/open_ledger_summary.csv", "val", "cost2x"),
            _source("v9_avgw3_open_ledger_20260617/breadth_triggered_market/test_ma3_035_m085_cost2x/open_ledger_summary.csv", "test", "cost2x"),
            _source("forward_results/frozen_v9_avgw3/breadth_triggered_market_20260519_20260616_ma3_035_m085/open_ledger_summary.csv", "forward"),
            _source("forward_results/frozen_v9_avgw3/breadth_triggered_market_20260519_20260616_ma3_035_m085_lag1/open_ledger_summary.csv", "forward", "lag1"),
            _source("forward_results/frozen_v9_avgw3/breadth_triggered_market_20260519_20260616_ma3_035_m085_cost2x/open_ledger_summary.csv", "forward", "cost2x"),
        ),
    ),
    CandidateSpec(
        name="edge_r030_100",
        status="research",
        execution_family=EXECUTION_OFFICIAL_OPEN_LEDGER,
        description="Rerank only base ranks [30, 100).",
        result_sources=(
            _source("v9_avgw3_open_ledger_20260617/sweep_edge_r030_100_w095/sweep_summary.csv", None, "normal", "sweep_summary"),
            _source("v9_avgw3_open_ledger_20260617/sweep_edge_r030_100_w095_lag1/sweep_summary.csv", None, "lag1", "sweep_summary"),
            _source("v9_avgw3_open_ledger_20260617/sweep_edge_r030_100_w095_cost2x/sweep_summary.csv", None, "cost2x", "sweep_summary"),
            _source("forward_results/frozen_v9_avgw3/edge_r030_100_w095_20260519_20260616/open_ledger_summary.csv", "forward"),
            _source("forward_results/frozen_v9_avgw3/edge_r030_100_w095_20260519_20260616_lag1/open_ledger_summary.csv", "forward", "lag1"),
            _source("forward_results/frozen_v9_avgw3/edge_r030_100_w095_20260519_20260616_cost2x/open_ledger_summary.csv", "forward", "cost2x"),
        ),
    ),
    CandidateSpec(
        name="negfilter_drop3",
        status="shadow",
        execution_family=EXECUTION_OFFICIAL_OPEN_LEDGER,
        description="Remove 3 worst full-reranker names from base ranks [30, 100).",
        result_sources=(
            _source("v9_avgw3_open_ledger_20260617/negfilter_drop3_single_diag/val/open_ledger_summary.csv", "val"),
            _source("v9_avgw3_open_ledger_20260617/negfilter_drop3_single_diag/test/open_ledger_summary.csv", "test"),
            _source("v9_avgw3_open_ledger_20260617/sweep_negfilter_r030_100_drop3_lag1/sweep_summary.csv", None, "lag1", "sweep_summary"),
            _source("v9_avgw3_open_ledger_20260617/sweep_negfilter_r030_100_drop3_cost2x/sweep_summary.csv", None, "cost2x", "sweep_summary"),
            _source("forward_results/frozen_v9_avgw3/negfilter_r030_100_drop3_20260519_20260616/open_ledger_summary.csv", "forward"),
            _source("forward_results/frozen_v9_avgw3/negfilter_r030_100_drop3_20260519_20260616_lag1/open_ledger_summary.csv", "forward", "lag1"),
            _source("forward_results/frozen_v9_avgw3/negfilter_r030_100_drop3_20260519_20260616_cost2x/open_ledger_summary.csv", "forward", "cost2x"),
        ),
    ),
    CandidateSpec(
        name="risk_target_r004",
        status="research",
        execution_family=EXECUTION_OFFICIAL_OPEN_LEDGER,
        description="Weak-state target shrink to 0.004.",
        result_sources=(
            _source("v9_avgw3_open_ledger_20260617/risk_target_switch/val_r004/open_ledger_summary.csv", "val"),
            _source("v9_avgw3_open_ledger_20260617/risk_target_switch/test_r004/open_ledger_summary.csv", "test"),
            _source("forward_results/frozen_v9_avgw3/risk_target_switch_20260519_20260616_r004/open_ledger_summary.csv", "forward"),
        ),
    ),
)


def default_registry():
    return {candidate.name: candidate for candidate in DEFAULT_CANDIDATES}


def resolve_source_path(source, root="."):
    return Path(root) / source.path
