from pathlib import Path

from run.generate_ledger_path_v3_signal import build_commands, parse_args


def test_generate_ledger_path_v3_signal_uses_previous_diagnostics(tmp_path):
    args = parse_args(
        [
            "--base-alpha-jsonl",
            "alpha.jsonl",
            "--baseline-diagnostics",
            "diag.csv",
            "--output-dir",
            str(tmp_path),
            "--split-name",
            "test_2025",
        ]
    )

    commands = build_commands(args)
    flattened = [part for cmd in commands for part in cmd]

    assert "run/build_state_aware_policy_dataset.py" in commands[0]
    assert "run/build_pairwise_ledger_inference_dataset.py" in commands[1]
    assert "run/apply_pairwise_replacement_policy_lgbm.py" in commands[2]
    assert "--diag-timing" in commands[1]
    assert "previous" in commands[1]
    assert "next" not in commands[1]
    assert "--rewrite-mode" in commands[2]
    assert "slot_swap" in commands[2]
    assert str(Path(tmp_path, "policy_features", "policy_dataset.parquet")) in flattened
    assert "pairwise_ledger_path_dataset.parquet" not in flattened


def test_generate_ledger_path_v3_signal_keeps_selection_defaults(tmp_path):
    args = parse_args(
        [
            "--base-alpha-jsonl",
            "alpha.jsonl",
            "--baseline-diagnostics",
            "diag.csv",
            "--output-dir",
            str(tmp_path),
            "--split-name",
            "forward_2026",
            "--threshold",
            "0.0001",
        ]
    )

    commands = build_commands(args)
    apply_cmd = commands[2]

    assert apply_cmd[apply_cmd.index("--threshold") + 1] == "0.0001"
    assert apply_cmd[apply_cmd.index("--target-frac") + 1] == "0.006"
    assert apply_cmd[apply_cmd.index("--hold-frac") + 1] == "0.1"
    assert "--industry-hhi-penalty" not in apply_cmd
    assert "--top-industry-share-penalty" not in apply_cmd
    assert "--candidate-industry-share-penalty" not in apply_cmd
    assert "--concentration-penalty-condition" not in apply_cmd
    assert "--specific-vol-worsen-penalty" not in apply_cmd
    assert "--ret20-worsen-penalty" not in apply_cmd
    assert "--risk-guard-condition" not in apply_cmd


def test_generate_ledger_path_v3_signal_passes_concentration_penalty(tmp_path):
    args = parse_args(
        [
            "--base-alpha-jsonl",
            "alpha.jsonl",
            "--baseline-diagnostics",
            "diag.csv",
            "--output-dir",
            str(tmp_path),
            "--split-name",
            "test_2025",
            "--industry-hhi-penalty",
            "0.02",
            "--top-industry-share-penalty",
            "0.01",
            "--candidate-industry-share-penalty",
            "0.003",
            "--concentration-penalty-condition",
            "fragile_or_pair_risk",
            "--penalty-beta-threshold",
            "1.1",
            "--penalty-specific-vol-threshold",
            "0.07",
            "--specific-vol-worsen-penalty",
            "0.001",
            "--ret20-worsen-penalty",
            "0.002",
            "--risk-guard-condition",
            "when_decrowding",
        ]
    )

    commands = build_commands(args)
    apply_cmd = commands[2]

    assert apply_cmd[apply_cmd.index("--industry-hhi-penalty") + 1] == "0.02"
    assert apply_cmd[apply_cmd.index("--top-industry-share-penalty") + 1] == "0.01"
    assert apply_cmd[apply_cmd.index("--candidate-industry-share-penalty") + 1] == "0.003"
    assert apply_cmd[apply_cmd.index("--concentration-penalty-condition") + 1] == "fragile_or_pair_risk"
    assert apply_cmd[apply_cmd.index("--penalty-beta-threshold") + 1] == "1.1"
    assert apply_cmd[apply_cmd.index("--penalty-specific-vol-threshold") + 1] == "0.07"
    assert apply_cmd[apply_cmd.index("--specific-vol-worsen-penalty") + 1] == "0.001"
    assert apply_cmd[apply_cmd.index("--ret20-worsen-penalty") + 1] == "0.002"
    assert apply_cmd[apply_cmd.index("--risk-guard-condition") + 1] == "when_decrowding"
