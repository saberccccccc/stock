from run.audit_provider_contracts import build_data_view, parse_args


def test_provider_audit_defaults_to_combined_selection_view():
    args = parse_args(["--v14-meta", "meta.pkl", "--output", "audit.json"])

    assert args.split == "selection_2024_2025"
    assert args.data_dir is None


def test_provider_audit_builds_forward_view_from_canonical_protocol(tmp_path):
    view, role = build_data_view("forward_2026", tmp_path)

    assert role == "forward"
    assert str(view.evaluation.start.date()) == "2026-01-01"
    assert str(view.evaluation.end.date()) == "2026-06-30"
    assert str(view.max_data_date.date()) == "2026-06-30"
