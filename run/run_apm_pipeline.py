"""Generic APM pipeline: backtest -> scorecard -> attribution -> audit."""
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
PYTHON = sys.executable

from core.research_protocol import get_split_spec

STRESSES = ["normal", "lag1", "cost2x", "capacity_3pct"]
SPLITS = [
    ("validation_2024", "val", "val_2024"),
    ("test_2025", "test", "test_2025"),
    ("forward_2026", "forward", "forward_2026"),
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--alpha-val")
    p.add_argument("--alpha-test")
    p.add_argument("--alpha-forward")
    p.add_argument("--output-base", default="reports")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--forward-data-dir", default="data/forward_raw")
    p.add_argument("--skip-backtest", action="store_true")
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--skip-attribution", action="store_true")
    return p.parse_args()

def run_backtest(model, alpha_path, split_label, split_key, split_name, data_dir, output_base):
    if not alpha_path or not Path(alpha_path).exists():
        print(f"[SKIP] {split_label}: no alpha file")
        return []
    summary_files = []
    for stress in STRESSES:
        out_dir = Path(output_base) / f"apm_{model}_{split_label}_{stress}"
        spec = get_split_spec(split_name)
        cmd = [PYTHON, "run/backtest_retention_open_ledger.py",
               "--alpha-jsonl", alpha_path, "--output-dir", str(out_dir),
               "--data-dir", data_dir,
               "--preset", "official_open_price_share_ledger",
               "--stress", stress,
               "--portfolio-values", "500000,1000000",
               "--target-fracs", "0.004", "--hold-fracs", "0.08"]
        if spec.is_forward:
            cmd.append("--allow-forward")
        cmd.extend(["--max-data-date", spec.command_dates()[2]])
        result = subprocess.run(cmd, capture_output=True, text=True)
        summary = out_dir / "open_ledger_summary.csv"
        if result.returncode == 0 and Path(summary).exists():
            summary_files.append(f"{model}:{split_key}:{stress}:{summary}")
            print(f"  [OK] {split_key}/{stress}")
        else:
            print(f"  [FAIL] {split_key}/{stress}")
    return summary_files

def run_scorecard(model, summary_specs, output_base):
    if not summary_specs: return
    out_dir = Path(output_base) / f"apm_scorecard_{model}"
    cmd = [PYTHON, "run/summarize_apm_scorecard.py", "--output-dir", out_dir]
    for spec in summary_specs: cmd.extend(["--input", spec])
    subprocess.run(cmd, check=True)
    print(f"  Scorecard -> {out_dir}")

def run_attribution(model, split_label, split_key, data_dir, output_base):
    base_dir = Path(output_base) / f"apm_{model}_{split_label}_normal"
    diag = base_dir / "diagnostics_pv0100w_target004_hold080.csv"
    rets = base_dir / "returns_pv0100w_target004_hold080.csv"
    if not Path(diag).exists() or not Path(rets).exists(): return None
    out_dir = Path(output_base) / f"apm_attribution_{model}_{split_key}"
    cmd = [PYTHON, "run/summarize_apm_attribution.py",
           "--returns-csv", rets, "--diagnostics-csv", diag,
           "--industry-csv", "data/stock_industry.csv",
           "--data-dir", data_dir, "--output-dir", out_dir]
    subprocess.run(cmd, check=True)
    print(f"  Attribution -> {out_dir}")
    return out_dir

def run_audit(model, summary_specs, attribution_dirs, output_base):
    if not summary_specs: return
    out_dir = Path(output_base) / f"apm_completeness_audit_{model}"
    cmd = [PYTHON, "run/audit_apm_completeness.py",
           "--candidate", model, "--output-dir", out_dir]
    for spec in summary_specs: cmd.extend(["--summary", spec])
    for d in attribution_dirs:
        if d: cmd.extend(["--attribution-dir", d])
    subprocess.run(cmd, check=True)
    print(f"  Audit -> {out_dir}")

def main():
    args = parse_args()
    print(f"\\n=== APM Pipeline: {args.model_name} ===\\n")
    all_summaries = []
    if not args.skip_backtest:
        for split_label, split_key, split_name in SPLITS:
            alpha = getattr(args, f"alpha_{split_key}", None)
            if split_key == "val": alpha = args.alpha_val
            elif split_key == "test": alpha = args.alpha_test
            elif split_key == "forward": alpha = args.alpha_forward
            spec = get_split_spec(split_name)
            data_dir = args.forward_data_dir if spec.is_forward else args.data_dir
            s = run_backtest(
                args.model_name,
                alpha,
                split_label,
                split_key,
                split_name,
                data_dir,
                args.output_base,
            )
            all_summaries.extend(s)
    if all_summaries: run_scorecard(args.model_name, all_summaries, args.output_base)
    att_dirs = []
    if not args.skip_attribution:
        for split_label, split_key, split_name in SPLITS:
            spec = get_split_spec(split_name)
            data_dir = args.forward_data_dir if spec.is_forward else args.data_dir
            d = run_attribution(args.model_name, split_label, split_key, data_dir, args.output_base)
            att_dirs.append(d)
    if not args.skip_audit and all_summaries:
        run_audit(args.model_name, all_summaries, att_dirs, args.output_base)
    print(f"\\n=== Pipeline complete: {args.model_name} ===")

if __name__ == "__main__": main()
