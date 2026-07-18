import json

from run.run_backtest_command_manifest import filter_commands, main


def test_filter_commands_by_proposal_and_stress():
    commands = [
        {"proposal": "baseline", "stress": "normal"},
        {"proposal": "risk_mild", "stress": "normal"},
        {"proposal": "risk_mild", "stress": "lag1"},
    ]

    out = filter_commands(commands, proposals=["risk_mild"], stresses=["lag1"])

    assert out == [{"proposal": "risk_mild", "stress": "lag1"}]


def test_manifest_runner_skips_existing_summary(tmp_path):
    out_dir = tmp_path / "open_ledger" / "baseline" / "normal"
    out_dir.mkdir(parents=True)
    (out_dir / "open_ledger_summary.csv").write_text("x\n", encoding="utf-8")
    manifest = tmp_path / "backtest_commands.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "proposal": "baseline",
                    "stress": "normal",
                    "output_dir": str(out_dir),
                    "command": ["python", "-c", "raise SystemExit(99)"],
                }
            ]
        ),
        encoding="utf-8",
    )

    main(["--manifest", str(manifest)])

    status = json.loads((tmp_path / "backtest_commands_run_status.json").read_text(encoding="utf-8"))
    assert status[0]["status"] == "skipped_existing"
