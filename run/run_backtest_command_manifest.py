"""Run a JSON command manifest with resume support."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)
os.chdir(ROOT)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Maximum commands to run; 0 means all.")
    parser.add_argument("--proposal", action="append", default=None)
    parser.add_argument("--stress", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_commands(path):
    commands = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(commands, list):
        raise ValueError("manifest must be a JSON list")
    return commands


def summary_path(item):
    return Path(item["output_dir"]) / "open_ledger_summary.csv"


def filter_commands(commands, proposals=None, stresses=None):
    proposal_set = set(proposals or [])
    stress_set = set(stresses or [])
    out = []
    for item in commands:
        if proposal_set and item.get("proposal") not in proposal_set:
            continue
        if stress_set and item.get("stress") not in stress_set:
            continue
        out.append(item)
    return out


def run_manifest(args):
    commands = filter_commands(load_commands(args.manifest), args.proposal, args.stress)
    status = []
    run_count = 0
    for idx, item in enumerate(commands, start=1):
        summary = summary_path(item)
        if summary.exists():
            status.append({**item, "status": "skipped_existing", "summary": str(summary)})
            print(f"[{idx}/{len(commands)}] skip existing {item.get('proposal')}/{item.get('stress')}", flush=True)
            continue
        if args.limit and run_count >= args.limit:
            status.append({**item, "status": "pending_limit", "summary": str(summary)})
            continue
        cmd = [str(part) for part in item["command"]]
        print(f"[{idx}/{len(commands)}] run {item.get('proposal')}/{item.get('stress')}", flush=True)
        if args.dry_run:
            status.append({**item, "status": "dry_run", "summary": str(summary)})
            run_count += 1
            continue
        result = subprocess.run(cmd, text=True)
        run_count += 1
        if result.returncode == 0 and summary.exists():
            status.append({**item, "status": "ok", "summary": str(summary)})
        else:
            status.append(
                {
                    **item,
                    "status": "failed",
                    "returncode": int(result.returncode),
                    "summary": str(summary),
                }
            )
            break
    return status


def main(argv=None):
    args = parse_args(argv)
    status = run_manifest(args)
    manifest_path = Path(args.manifest)
    out_path = manifest_path.with_name(manifest_path.stem + "_run_status.json")
    out_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {}
    for item in status:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    print(json.dumps({"status_path": str(out_path), "counts": counts}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
