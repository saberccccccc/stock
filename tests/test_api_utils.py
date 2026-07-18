import pytest

from data.api_utils import SafeAPICaller


def test_safe_api_caller_fails_fast_for_non_retryable_access_error():
    calls = []

    def denied():
        calls.append(1)
        raise RuntimeError("permission denied: endpoint requires more points")

    caller = SafeAPICaller(
        min_interval=0,
        max_retries=3,
        retry_base_delay=0,
        jitter=None,
        data_source="test",
        non_retryable_markers=("permission", "points"),
    )

    with pytest.raises(RuntimeError, match="permission denied"):
        caller(denied)

    assert calls == [1]


def test_safe_api_caller_still_retries_transient_errors():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("temporary network failure")
        return "ok"

    caller = SafeAPICaller(
        min_interval=0,
        max_retries=3,
        retry_base_delay=0,
        jitter=None,
        data_source="test",
        non_retryable_markers=("permission",),
    )

    assert caller(flaky) == "ok"
    assert len(calls) == 2
