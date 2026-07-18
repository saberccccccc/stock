"""Generate deployable no-lookahead Ledger Path V3 alpha signals.

Pipeline:
1. Build state-aware policy features from a base alpha JSONL.
2. Build feature-only pairwise ledger inference rows using previous diagnostics.
3. Apply the trained pairwise V3 model with slot-swap rewriting.

This is the deployable path.  It must not use future labels or next-execution
diagnostics.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


DEFAULT_MODEL = (
    "reports/state_aware_policy_training_20260704/"
    "multi_downside_e19_sa_p05_pairwise_ledger_path_v3/model.pkl"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-alpha-jsonl", required=True)
    parser.add_argument("--baseline-diagnostics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--global-features", default=None)
    parser.add_argument("--industry-csv", default="data/stock_industry.csv")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-data-date", default=None)
    parser.add_argument("--candidate-end", type=int, default=120)
    parser.add_argument("--target-frac", type=float, default=0.006)
    parser.add_argument("--hold-frac", type=float, default=0.10)
    parser.add_argument("--max-reranked-fills", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0001)
    parser.add_argument("--industry-hhi-penalty", type=float, default=0.0)
    parser.add_argument("--top-industry-share-penalty", type=float, default=0.0)
    parser.add_argument("--candidate-industry-share-penalty", type=float, default=0.0)
    parser.add_argument(
        "--concentration-penalty-condition",
        choices=(
            "always",
            "fragile_beta",
            "fragile_vol",
            "fragile_beta_or_vol",
            "pair_risk_positive",
            "fragile_or_pair_risk",
        ),
        default="always",
    )
    parser.add_argument("--penalty-beta-threshold", type=float, default=1.2)
    parser.add_argument("--penalty-specific-vol-threshold", type=float, default=0.08)
    parser.add_argument("--penalty-pair-risk-threshold", type=float, default=0.0)
    parser.add_argument("--specific-vol-worsen-penalty", type=float, default=0.0)
    parser.add_argument("--ret20-worsen-penalty", type=float, default=0.0)
    parser.add_argument("--pair-risk-worsen-penalty", type=float, default=0.0)
    parser.add_argument("--beta-worsen-penalty", type=float, default=0.0)
    parser.add_argument("--gate-max-pair-risk-delta", type=float, default=None)
    parser.add_argument("--gate-max-pair-downside-delta", type=float, default=None)
    parser.add_argument("--gate-max-diff-specific-vol-60d", type=float, default=None)
    parser.add_argument("--gate-min-diff-ret20", type=float, default=None)
    parser.add_argument(
        "--gate-condition",
        choices=(
            "always",
            "active_drawdown_negative",
            "global_pressure_high",
            "active_or_global",
            "active_and_global",
        ),
        default="always",
    )
    parser.add_argument(
        "--risk-guard-condition",
        choices=("always", "when_decrowding", "when_pair_risk_positive", "when_decrowding_or_pair_risk"),
        default="always",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_commands(args):
    output = Path(args.output_dir)
    policy_dir = output / "policy_features"
    pairwise_dir = output / "pairwise_inference"
    policy_dir.mkdir(parents=True, exist_ok=True)
    pairwise_dir.mkdir(parents=True, exist_ok=True)
    alpha_out = output / "alpha_policy.jsonl"
    policy_cmd = [
        PYTHON,
        "run/build_state_aware_policy_dataset.py",
        "--alpha-jsonl",
        str(args.base_alpha_jsonl),
        "--data-dir",
        str(args.data_dir),
        "--industry-csv",
        str(args.industry_csv),
        "--output-dir",
        str(policy_dir),
        "--split-name",
        str(args.split_name),
        "--candidate-end",
        str(args.candidate_end),
        "--target-frac",
        str(args.target_frac),
        "--hold-frac",
        str(args.hold_frac),
        "--max-reranked-fills",
        str(args.max_reranked_fills),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.global_features:
        policy_cmd.extend(["--global-features", str(args.global_features)])
    if args.start_date:
        policy_cmd.extend(["--start-date", str(args.start_date)])
    if args.end_date:
        policy_cmd.extend(["--end-date", str(args.end_date)])
    if args.max_data_date:
        # build_state_aware_policy_dataset uses max_label_date only to cap labels.
        # In deployable mode this just limits cache loading and should not be a
        # forward-selection knob.
        policy_cmd.extend(["--max-label-date", str(args.max_data_date)])

    pairwise_cmd = [
        PYTHON,
        "run/build_pairwise_ledger_inference_dataset.py",
        "--policy-dataset",
        str(policy_dir / "policy_dataset.parquet"),
        "--diagnostics-csv",
        str(args.baseline_diagnostics),
        "--output-dir",
        str(pairwise_dir),
        "--split-name",
        str(args.split_name),
        "--diag-timing",
        "previous",
    ]
    apply_cmd = [
        PYTHON,
        "run/apply_pairwise_replacement_policy_lgbm.py",
        "--input-alpha-jsonl",
        str(args.base_alpha_jsonl),
        "--pairwise-dataset",
        str(pairwise_dir / "pairwise_ledger_inference_dataset.parquet"),
        "--model",
        str(args.model),
        "--output-alpha-jsonl",
        str(alpha_out),
        "--threshold",
        str(args.threshold),
        "--rewrite-mode",
        "slot_swap",
        "--target-frac",
        str(args.target_frac),
        "--hold-frac",
        str(args.hold_frac),
    ]
    if float(args.industry_hhi_penalty) != 0.0:
        apply_cmd.extend(["--industry-hhi-penalty", str(args.industry_hhi_penalty)])
    if float(args.top_industry_share_penalty) != 0.0:
        apply_cmd.extend(["--top-industry-share-penalty", str(args.top_industry_share_penalty)])
    if float(args.candidate_industry_share_penalty) != 0.0:
        apply_cmd.extend(["--candidate-industry-share-penalty", str(args.candidate_industry_share_penalty)])
    if args.concentration_penalty_condition != "always":
        apply_cmd.extend(["--concentration-penalty-condition", str(args.concentration_penalty_condition)])
        apply_cmd.extend(["--penalty-beta-threshold", str(args.penalty_beta_threshold)])
        apply_cmd.extend(["--penalty-specific-vol-threshold", str(args.penalty_specific_vol_threshold)])
        apply_cmd.extend(["--penalty-pair-risk-threshold", str(args.penalty_pair_risk_threshold)])
    if float(args.specific_vol_worsen_penalty) != 0.0:
        apply_cmd.extend(["--specific-vol-worsen-penalty", str(args.specific_vol_worsen_penalty)])
    if float(args.ret20_worsen_penalty) != 0.0:
        apply_cmd.extend(["--ret20-worsen-penalty", str(args.ret20_worsen_penalty)])
    if float(args.pair_risk_worsen_penalty) != 0.0:
        apply_cmd.extend(["--pair-risk-worsen-penalty", str(args.pair_risk_worsen_penalty)])
    if float(args.beta_worsen_penalty) != 0.0:
        apply_cmd.extend(["--beta-worsen-penalty", str(args.beta_worsen_penalty)])
    if args.risk_guard_condition != "always":
        apply_cmd.extend(["--risk-guard-condition", str(args.risk_guard_condition)])
    if args.gate_max_pair_risk_delta is not None:
        apply_cmd.extend(["--gate-max-pair-risk-delta", str(args.gate_max_pair_risk_delta)])
    if args.gate_max_pair_downside_delta is not None:
        apply_cmd.extend(["--gate-max-pair-downside-delta", str(args.gate_max_pair_downside_delta)])
    if args.gate_max_diff_specific_vol_60d is not None:
        apply_cmd.extend(["--gate-max-diff-specific-vol-60d", str(args.gate_max_diff_specific_vol_60d)])
    if args.gate_min_diff_ret20 is not None:
        apply_cmd.extend(["--gate-min-diff-ret20", str(args.gate_min_diff_ret20)])
    if args.gate_condition != "always":
        apply_cmd.extend(["--gate-condition", str(args.gate_condition)])
    return [policy_cmd, pairwise_cmd, apply_cmd]


def run_commands(commands, dry_run=False):
    if dry_run:
        return [{"cmd": cmd, "returncode": None} for cmd in commands]
    results = []
    for cmd in commands:
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        results.append({"cmd": cmd, "returncode": 0})
    return results


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    commands = build_commands(args)
    results = run_commands(commands, dry_run=args.dry_run)
    manifest = {
        "base_alpha_jsonl": str(args.base_alpha_jsonl),
        "baseline_diagnostics": str(args.baseline_diagnostics),
        "output_alpha_jsonl": str(output / "alpha_policy.jsonl"),
        "split_name": args.split_name,
        "model": str(args.model),
        "threshold": args.threshold,
        "industry_hhi_penalty": args.industry_hhi_penalty,
        "top_industry_share_penalty": args.top_industry_share_penalty,
        "candidate_industry_share_penalty": args.candidate_industry_share_penalty,
        "concentration_penalty_condition": args.concentration_penalty_condition,
        "penalty_beta_threshold": args.penalty_beta_threshold,
        "penalty_specific_vol_threshold": args.penalty_specific_vol_threshold,
        "penalty_pair_risk_threshold": args.penalty_pair_risk_threshold,
        "specific_vol_worsen_penalty": args.specific_vol_worsen_penalty,
        "ret20_worsen_penalty": args.ret20_worsen_penalty,
        "pair_risk_worsen_penalty": args.pair_risk_worsen_penalty,
        "beta_worsen_penalty": args.beta_worsen_penalty,
        "gate_max_pair_risk_delta": args.gate_max_pair_risk_delta,
        "gate_max_pair_downside_delta": args.gate_max_pair_downside_delta,
        "gate_max_diff_specific_vol_60d": args.gate_max_diff_specific_vol_60d,
        "gate_min_diff_ret20": args.gate_min_diff_ret20,
        "gate_condition": args.gate_condition,
        "risk_guard_condition": args.risk_guard_condition,
        "diag_timing": "previous",
        "uses_future_labels": False,
        "uses_future_execution_diagnostics": False,
        "commands": results,
    }
    (output / "ledger_path_v3_signal_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
