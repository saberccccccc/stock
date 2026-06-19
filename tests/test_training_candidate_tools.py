import importlib
import sys

import pytest

from core.training_presets import load_training_suite
from run.run_loss_ablation import build_command


TOOL_MODULES = (
    "run.analyze_alpha_execution_quality",
    "run.confirm_locked_candidate",
    "run.generate_v9_inference_alpha",
    "run.run_loss_ablation",
    "run.screen_stall_execution",
    "run.summarize_loss_ablation",
    "run.validate_candidate_models",
)


@pytest.mark.parametrize("module_name", TOOL_MODULES)
def test_training_candidate_tool_imports(module_name):
    importlib.import_module(module_name)


def test_loss_ablation_runner_uses_validated_training_argv():
    suite = load_training_suite("configs/loss_ablation_20260613.json")
    experiment = suite.experiments[0]

    assert build_command(experiment) == [
        sys.executable,
        "run/train.py",
        *experiment.train_argv(),
    ]
