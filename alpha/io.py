"""JSONL helpers for daily alpha ranking files and capital-aware manifests."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class ResolvedAlphaSource:
    request_path: str
    alpha_jsonl: str
    mode: str
    manifest_path: str = ""
    rule_name: str = ""
    rule_index: int = -1
    min_portfolio_value: Optional[float] = None
    max_portfolio_value: Optional[float] = None


def normalize_date(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def iter_alpha_rows(path, normalize_dates=True):
    """Yield non-empty JSONL rows from an alpha ranking file."""
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if normalize_dates and "date" in row:
                row["date"] = normalize_date(row["date"])
            yield row


def load_alpha_rows(path, timestamp_dates=False):
    rows = list(iter_alpha_rows(path, normalize_dates=not timestamp_dates))
    validate_alpha_rows(rows)
    if timestamp_dates:
        for row in rows:
            row["date"] = pd.Timestamp(row["date"])
        rows.sort(key=lambda row: row["date"])
    return rows


def load_alpha_dates(path):
    rows = list(iter_alpha_rows(path))
    validate_alpha_rows(rows)
    dates = [pd.Timestamp(row["date"]).normalize() for row in rows]
    if not dates:
        raise ValueError("alpha file is empty")
    return sorted(dates)


def validate_alpha_rows(rows):
    seen_dates = set()
    for row_number, row in enumerate(rows, start=1):
        if "date" not in row:
            raise ValueError(f"alpha row {row_number} is missing date")
        date = normalize_date(row["date"])
        if date in seen_dates:
            raise ValueError(f"duplicate alpha date: {date}")
        seen_dates.add(date)

        if "codes" not in row:
            continue
        codes = row["codes"]
        if not isinstance(codes, list):
            raise ValueError(f"alpha row {date} codes must be a list")
        if len(codes) != len(set(codes)):
            raise ValueError(f"alpha row {date} contains duplicate stock codes")
        if "alpha" in row:
            resolve_row_scores(row)


def resolve_row_scores(row):
    """Return scores in code order for list- or code-mapped alpha artifacts."""

    codes = list(row.get("codes", []))
    raw = row.get("scores", row.get("alpha"))
    if raw is None:
        raise ValueError("alpha row is missing alpha/scores")
    if isinstance(raw, Mapping):
        missing = [code for code in codes if code not in raw]
        extra = set(raw) - set(codes)
        if missing or extra:
            raise ValueError(
                f"alpha mapping does not match codes: missing={missing[:3]} extra={sorted(extra)[:3]}"
            )
        return [raw[code] for code in codes]
    values = list(raw)
    if len(values) != len(codes):
        date = row.get("date", "unknown")
        raise ValueError(f"alpha row {date} has mismatched codes and alpha lengths")
    return values


def write_alpha_rows(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def assert_same_date(left, right):
    left_date = normalize_date(left["date"])
    right_date = normalize_date(right["date"])
    if left_date != right_date:
        raise ValueError(f"date mismatch: {left_date} vs {right_date}")
    return left_date


def iter_aligned_alpha_rows(left_path, right_path):
    """Yield two Alpha files in lockstep and reject missing or shifted rows."""

    missing = object()
    pairs = zip_longest(
        iter_alpha_rows(left_path),
        iter_alpha_rows(right_path),
        fillvalue=missing,
    )
    for left, right in pairs:
        if left is missing or right is missing:
            raise ValueError("alpha files have different row counts")
        assert_same_date(left, right)
        yield left, right


def _normalize_path(path):
    return str(Path(path).expanduser().resolve())


def _optional_float(value, field_name):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc


def _resolve_manifest_alpha_path(value, base_dir):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_dir) / path
    return str(path.resolve())


@lru_cache(maxsize=None)
def load_alpha_manifest(path):
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("alpha manifest must be a JSON object")

    default_alpha_jsonl = payload.get("default_alpha_jsonl")
    if default_alpha_jsonl in (None, ""):
        default_alpha_jsonl = None
    else:
        default_alpha_jsonl = _resolve_manifest_alpha_path(
            default_alpha_jsonl,
            manifest_path.parent,
        )

    raw_rules = payload.get("rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError("alpha manifest rules must be a list")
    if not raw_rules and default_alpha_jsonl is None:
        raise ValueError("alpha manifest must contain rules or default_alpha_jsonl")

    rules = []
    for rule_index, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"alpha manifest rule {rule_index} must be a JSON object")
        alpha_jsonl = raw_rule.get("alpha_jsonl")
        if not alpha_jsonl:
            raise ValueError(f"alpha manifest rule {rule_index} is missing alpha_jsonl")

        exact_value = _optional_float(
            raw_rule.get("portfolio_value"),
            f"alpha manifest rule {rule_index} portfolio_value",
        )
        min_value = _optional_float(
            raw_rule.get("min_portfolio_value"),
            f"alpha manifest rule {rule_index} min_portfolio_value",
        )
        max_value = _optional_float(
            raw_rule.get("max_portfolio_value"),
            f"alpha manifest rule {rule_index} max_portfolio_value",
        )
        if exact_value is not None:
            if min_value is not None or max_value is not None:
                raise ValueError(
                    f"alpha manifest rule {rule_index} cannot mix portfolio_value with min/max bounds"
                )
            min_value = exact_value
            max_value = exact_value
        if min_value is None and max_value is None:
            raise ValueError(
                f"alpha manifest rule {rule_index} must define portfolio_value or min/max bounds"
            )
        if min_value is not None and max_value is not None and max_value < min_value:
            raise ValueError(
                f"alpha manifest rule {rule_index} has max_portfolio_value < min_portfolio_value"
            )

        rules.append({
            "rule_index": rule_index,
            "rule_name": str(raw_rule.get("name", "") or "").strip(),
            "alpha_jsonl": _resolve_manifest_alpha_path(
                alpha_jsonl,
                manifest_path.parent,
            ),
            "min_portfolio_value": min_value,
            "max_portfolio_value": max_value,
        })

    return {
        "manifest_path": str(manifest_path),
        "default_alpha_jsonl": default_alpha_jsonl,
        "rules": rules,
    }


def resolve_alpha_source(*, alpha_jsonl=None, alpha_manifest=None, portfolio_value=None):
    has_jsonl = alpha_jsonl not in (None, "")
    has_manifest = alpha_manifest not in (None, "")
    if has_jsonl == has_manifest:
        raise ValueError("exactly one of alpha_jsonl or alpha_manifest must be provided")

    if has_jsonl:
        return ResolvedAlphaSource(
            request_path=str(alpha_jsonl),
            alpha_jsonl=_normalize_path(alpha_jsonl),
            mode="direct",
        )

    if portfolio_value is None:
        raise ValueError("portfolio_value is required when resolving alpha_manifest")

    manifest = load_alpha_manifest(alpha_manifest)
    value = float(portfolio_value)
    for rule in manifest["rules"]:
        min_value = rule["min_portfolio_value"]
        max_value = rule["max_portfolio_value"]
        if min_value is not None and value < min_value:
            continue
        if max_value is not None and value > max_value:
            continue
        return ResolvedAlphaSource(
            request_path=str(alpha_manifest),
            alpha_jsonl=rule["alpha_jsonl"],
            mode="manifest",
            manifest_path=manifest["manifest_path"],
            rule_name=rule["rule_name"],
            rule_index=int(rule["rule_index"]),
            min_portfolio_value=min_value,
            max_portfolio_value=max_value,
        )

    if manifest["default_alpha_jsonl"] is not None:
        return ResolvedAlphaSource(
            request_path=str(alpha_manifest),
            alpha_jsonl=manifest["default_alpha_jsonl"],
            mode="manifest",
            manifest_path=manifest["manifest_path"],
            rule_name="default",
        )

    raise ValueError(
        f"alpha manifest {manifest['manifest_path']} has no rule for portfolio_value={value}"
    )
