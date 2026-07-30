"""Shared execution-market backend contract for ledger entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


OHLC_BACKENDS = ("legacy", "csv", "monthly")
DEFAULT_OHLC_BACKEND = "legacy"
DEFAULT_MARKET_DAILY_STORE_ROOT = "data/market_daily_candidate_v2"
DEFAULT_MONTHLY_CACHE_ROOT = "cache/ohlcv_monthly_v3_candidate"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_POLICY_PATH = PROJECT_ROOT / "configs" / "execution_market_backend_policy.json"


def configured_default_backend() -> str:
    if not BACKEND_POLICY_PATH.is_file():
        return DEFAULT_OHLC_BACKEND
    from data.execution_market_backend_policy import load_policy

    return str(load_policy(BACKEND_POLICY_PATH)["active_backend"])


def configured_candidate_paths() -> tuple[str, str]:
    if not BACKEND_POLICY_PATH.is_file():
        return DEFAULT_MARKET_DAILY_STORE_ROOT, DEFAULT_MONTHLY_CACHE_ROOT
    from data.execution_market_backend_policy import load_policy

    policy = load_policy(BACKEND_POLICY_PATH)
    return (
        str(policy["candidate_store_root"]),
        str(policy["candidate_monthly_cache_root"]),
    )


@dataclass(frozen=True)
class ExecutionMarketDataContract:
    backend: str | None = None
    market_daily_store_root: str | Path | None = None
    monthly_cache_root: str | Path | None = None
    shadow_backend: str | None = None
    shadow_report: str | Path | None = None

    def __post_init__(self) -> None:
        backend = str(self.backend or configured_default_backend()).strip().lower()
        if backend not in OHLC_BACKENDS:
            raise ValueError(f"unknown OHLC backend: {backend}")
        object.__setattr__(self, "backend", backend)
        configured_store, configured_cache = configured_candidate_paths()
        object.__setattr__(
            self,
            "market_daily_store_root",
            self.market_daily_store_root or configured_store,
        )
        object.__setattr__(
            self,
            "monthly_cache_root",
            self.monthly_cache_root or configured_cache,
        )
        shadow = str(self.shadow_backend or "").strip().lower() or None
        if shadow is not None:
            if {backend, shadow} != {"csv", "monthly"}:
                raise ValueError("OHLC dual-read requires one csv and one monthly backend")
            if self.shadow_report is None or not str(self.shadow_report).strip():
                raise ValueError("OHLC dual-read requires a shadow report path")
        object.__setattr__(self, "shadow_backend", shadow)

    def cli_args(self) -> list[str]:
        result = [
            "--ohlc-backend",
            self.backend,
            "--market-daily-store-root",
            str(self.market_daily_store_root),
            "--ohlc-monthly-cache-dir",
            str(self.monthly_cache_root),
        ]
        if self.shadow_backend is not None:
            result.extend(
                [
                    "--ohlc-shadow-backend",
                    self.shadow_backend,
                    "--ohlc-shadow-report",
                    str(self.shadow_report),
                ]
            )
        return result

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
            "shadow_backend": self.shadow_backend or "",
            "shadow_report": (
                display(self.shadow_report) if self.shadow_backend is not None else ""
            ),
        }


def add_execution_market_data_args(parser, *, default_backend: str | None = None):
    configured_store, configured_cache = configured_candidate_paths()
    parser.add_argument(
        "--ohlc-backend",
        choices=OHLC_BACKENDS,
        default=default_backend or configured_default_backend(),
        help="Execution-data backend. legacy remains the formal default until MD9.",
    )
    parser.add_argument(
        "--market-daily-store-root",
        default=configured_store,
    )
    parser.add_argument(
        "--ohlc-monthly-cache-dir",
        default=configured_cache,
    )
    parser.add_argument("--ohlc-shadow-backend", choices=("csv", "monthly"), default=None)
    parser.add_argument("--ohlc-shadow-report", default=None)
    return parser


def contract_from_args(args) -> ExecutionMarketDataContract:
    return ExecutionMarketDataContract(
        backend=getattr(args, "ohlc_backend", None),
        market_daily_store_root=getattr(
            args, "market_daily_store_root", None
        ),
        monthly_cache_root=getattr(
            args, "ohlc_monthly_cache_dir", None
        ),
        shadow_backend=getattr(args, "ohlc_shadow_backend", None),
        shadow_report=getattr(args, "ohlc_shadow_report", None),
    )
