from pathlib import Path


REPRO_SCRIPTS = (
    "resume_downside_topfocus_remaining_20260616.ps1",
    "run_forward_observation_candidates_20260617.ps1",
    "run_lag1_loss_ablation_after_sweep_20260616.ps1",
    "run_unified_good_ops_validation_20260616.ps1",
    "validate_downside_topfocus_candidates_20260616.ps1",
    "validate_m0_epoch_lag1_sweep_20260616.ps1",
)


def test_all_tracked_powershell_scripts_avoid_user_specific_paths():
    scripts = tuple(Path(".").glob("*.ps1")) + tuple(Path("scripts").glob("*.ps1"))

    assert scripts
    for script in scripts:
        text = script.read_text(encoding="utf-8-sig")
        assert "C:\\Users\\" not in text, script


def test_repro_scripts_are_portable_and_present():
    for name in REPRO_SCRIPTS:
        text = Path(name).read_text(encoding="utf-8-sig")
        assert "C:\\Users\\" not in text
        assert "$PSScriptRoot" in text
        assert "miniconda3\\envs\\torch\\python.exe" in text


def test_lag1_runner_has_no_stale_fixed_pid():
    text = Path("run_lag1_loss_ablation_after_sweep_20260616.ps1").read_text(
        encoding="utf-8-sig"
    )

    assert "19484" not in text
    assert "$env:WAIT_PID" in text
