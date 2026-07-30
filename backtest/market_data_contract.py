"""Shared execution-market backend contract for ledger entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


OHLC_BACKENDS = ("legacy", "csv", "monthly")
DEFAULT_OHLC_BACKEND = "legacy"
DEFAULT_MARKET_DAILY_STORE_ROOT = "data/market_daily_candidate_v2"
DEFAULT_MONTHLY_CACHE_ROOT = "cache/ohlcv_monthly_v3_candidate"


@dataclass(frozen=True)
class ExecutionMarketDataContract:
    backend: str = DEFAULT_OHLC_BACKEND
    market_daily_store_root: str | Path = DEFAULT_MARKET_DAILY_STORE_ROOT
    monthly_cache_root: str | Path = DEFAULT_MONTHLY_CACHE_ROOT

    def __post_init__(self) -> None:
        backend = str(self.backend).strip().lower()
        if backend not in OHLC_BACKENDS:
            raise ValueError(f"unknown OHLC backend: {backend}")
        object.__setattr__(self, "backend", backend)

    def cli_args(self) -> list[str]:
        return [
            "--ohlc-backend",
            self.backend,
            "--market-daily-store-root",
            str(self.market_daily_store_root),
            "--ohlc-monthly-cache-dir",
            str(self.monthly_cache_root),
        ]

    def manifest(self, *, project_root: str | Path | None = None) -> dict[str, Any]:
        def display(value: str | Path) -> str:
            path = Path(value)
            if project_root is not None and not path.is_absolute():
                path = Path(project_root) / path
            return str(path.resolve())

        return {
            "schema": "execution_market_data_contract_v1",
            "backend": self.backend,
            "data_role": (
                "legacy_compatibility"
                if self.backend == "legacy"
                else "csv_parity_oracle"
                if self.backend == "csv"
                else "monthly_candidate"
            ),
            "market_daily_store_root": (
                display(self.market_daily_store_root) if self.backend == "monthly" else ""
            ),
            "monthly_cache_root": (
                display(self.monthly_cache_root) if self.backend == "monthly" else ""
            ),
        }


def add_execution_market_data_args(parser, *, default_backend: str = DEFAULT_OHLC_BACKEND):
    parser.add_argument(
        "--ohlc-backend",
        choices=OHLC_BACKENDS,
        default=default_backend,
        help="Execution-data backend. legacy remains the formal default until MD9.",
    )
    parser.add_argument(
        "--market-daily-store-root",
        default=DEFAULT_MARKET_DAILY_STORE_ROOT,
    )
    parser.add_argument(
        "--ohlc-monthly-cache-dir",
        default=DEFAULT_MONTHLY_CACHE_ROOT,
    )
    return parser


def contract_from_args(args) -> ExecutionMarketDataContract:
    return ExecutionMarketDataContract(
        backend=getattr(args, "ohlc_backend", DEFAULT_OHLC_BACKEND),
        market_daily_store_root=getattr(
            args, "market_daily_store_root", DEFAULT_MARKET_DAILY_STORE_ROOT
        ),
        monthly_cache_root=getattr(
            args, "ohlc_monthly_cache_dir", DEFAULT_MONTHLY_CACHE_ROOT
        ),
    )
