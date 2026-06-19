import importlib


RERANKER_MODULES = (
    "run.build_reranker_dataset",
    "run.build_reranker_v2_dataset",
    "run.build_reranker_v3_dataset",
    "run.build_reranker_v3_forward",
    "run.train_reranker",
    "run.train_reranker_v2",
    "run.train_reranker_v3",
    "run.train_reranker_v4",
    "run.train_reranker_v41",
    "run.validate_reranker_2024",
    "run.validate_conservative_reranker_2024",
    "run.validate_reranker_v2_2024",
    "run.validate_reranker_v3_2024",
    "run.validate_reranker_v4_2024",
    "run.validate_reranker_v41",
    "run.confirm_reranker_v3_history",
    "run.confirm_reranker_v4_history",
    "run.evaluate_reranker_v3_forward",
    "run.evaluate_reranker_v4_forward",
    "run.diagnose_reranker_replacements",
)


def test_reranker_research_modules_are_importable():
    for module_name in RERANKER_MODULES:
        importlib.import_module(module_name)
