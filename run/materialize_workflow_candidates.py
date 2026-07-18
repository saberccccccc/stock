"""Create an experiment-local candidate registry from a completed rolling run."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
root_path = str(ROOT)
if root_path in sys.path:
    sys.path.remove(root_path)
sys.path.insert(0, root_path)

from experiments.recording import MANIFEST_NAME, validate_manifest_for_formal_use


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-experiment-dir", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--comparison-candidate-id", action="append", default=[])
    parser.add_argument("--source-candidates-csv", default="registry/candidates.csv")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def materialize_candidates(
    *,
    rolling_experiment_dir: str | Path,
    candidate_id: str,
    comparison_candidate_ids: list[str],
    source_candidates_csv: str | Path,
    output: str | Path,
):
    experiment_dir = Path(rolling_experiment_dir).resolve()
    validate_manifest_for_formal_use(experiment_dir / MANIFEST_NAME)
    rolling_manifest = experiment_dir / "rolling_manifest.json"
    if not rolling_manifest.is_file():
        raise FileNotFoundError(rolling_manifest)

    source_path = Path(source_candidates_csv)
    source_frame = pd.read_csv(source_path)
    source = source_frame.set_index("candidate_id", drop=False)
    missing = sorted(set(comparison_candidate_ids) - set(source.index))
    if missing:
        raise KeyError(f"comparison candidates are missing: {missing}")
    rows = [source.loc[item].to_dict() for item in comparison_candidate_ids]
    rows.append(
        {
            "candidate_id": candidate_id,
            "display_name": candidate_id,
            "family": "workflow_rolling",
            "status": "research",
            "selection_eligible": True,
            "forward_observation_only": True,
            "signal_path": str(rolling_manifest),
            "backtest_path": "",
            "execution_mode": "realistic",
            "base_alpha": "",
            "created_at": date.today().isoformat(),
            "notes": "Experiment-local rolling candidate; not a global registry promotion.",
        }
    )
    frame = pd.DataFrame(rows, columns=list(source_frame.columns))
    if frame["candidate_id"].duplicated().any():
        raise ValueError("workflow candidate registry contains duplicate candidate IDs")
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    return target


def main(argv=None):
    args = parse_args(argv)
    target = materialize_candidates(
        rolling_experiment_dir=args.rolling_experiment_dir,
        candidate_id=args.candidate_id,
        comparison_candidate_ids=args.comparison_candidate_id,
        source_candidates_csv=ROOT / args.source_candidates_csv,
        output=args.output,
    )
    print(json.dumps({"candidates_csv": str(target)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
