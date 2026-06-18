import json
import subprocess
import sys

from core.training_presets import (
    load_training_suite,
    load_training_suites,
    params_to_train_argv,
    render_train_command,
)


def test_load_training_suite_expands_common_params(tmp_path):
    config = tmp_path / "demo.json"
    config.write_text(
        json.dumps(
            {
                "common": {
                    "model": "v9",
                    "epochs": 6,
                    "save_every_epoch": True,
                    "top_focus_loss_weight": 0.0,
                },
                "experiments": [
                    {
                        "id": "A0",
                        "output_dir": "checkpoints_A0",
                        "top_focus_loss_weight": 0.005,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite = load_training_suite(config)

    assert suite.name == "demo"
    assert len(suite.experiments) == 1
    experiment = suite.experiments[0]
    assert experiment.suite == "demo"
    assert experiment.experiment_id == "A0"
    assert experiment.output_dir == "checkpoints_A0"
    assert experiment.params["model"] == "v9"
    assert experiment.params["top_focus_loss_weight"] == 0.005


def test_params_to_train_argv_uses_train_cli_flags():
    argv = params_to_train_argv(
        {
            "model": "v9",
            "epochs": 6,
            "save_every_epoch": True,
            "reset_optimizer": False,
            "output_dir": "checkpoints_A0",
            "unused": None,
        }
    )

    assert argv == [
        "--model",
        "v9",
        "--epochs",
        "6",
        "--save-every-epoch",
        "--output-dir",
        "checkpoints_A0",
    ]


def test_render_train_command_quotes_values():
    command = render_train_command(
        {
            "model": "v9",
            "output_dir": "checkpoints with spaces",
            "save_every_epoch": True,
        },
        python_exe="python",
        train_script="run/train.py",
    )

    assert command == 'python run/train.py --model v9 --output-dir "checkpoints with spaces" --save-every-epoch'


def test_existing_training_configs_are_loadable():
    suites = load_training_suites("configs")
    names = {suite.name for suite in suites}

    assert names >= {
        "loss_ablation_20260613",
        "m0_topfocus_validation_20260614",
        "lag1_loss_ablation_20260616",
    }
    assert all(suite.experiments for suite in suites)
    assert all("--model" in exp.train_argv() for suite in suites for exp in suite.experiments)


def test_render_training_commands_cli_filters_experiment():
    result = subprocess.run(
        [
            sys.executable,
            "run/render_training_commands.py",
            "--config",
            "configs/m0_topfocus_validation_20260614.json",
            "--experiment-id",
            "M1",
            "--python-exe",
            "python",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "# m0_topfocus_validation_20260614:M1" in result.stdout
    assert "--output-dir checkpoints_loss_ablation_M1_nomulti_topfocus_w005" in result.stdout
    assert "--top-focus-loss-weight 0.005" in result.stdout
