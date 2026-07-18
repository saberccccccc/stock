import pandas as pd
import pytest

from data.global_market_features import add_composite_scores


def test_us_sector_breadth_excludes_missing_sector_etfs():
    frame = pd.DataFrame(
        {
            "global_xlk_ret_1d": [0.01],
            "global_xlk_missing": [0.0],
            "global_xlf_ret_1d": [-0.02],
            "global_xlf_missing": [0.0],
            "global_xlre_ret_1d": [0.0],
            "global_xlre_missing": [1.0],
            "global_xlc_ret_1d": [0.0],
            "global_xlc_missing": [1.0],
        },
        index=pd.to_datetime(["2025-01-02"]),
    )

    result = add_composite_scores(frame.copy())

    assert result.loc[frame.index[0], "global_us_sector_available_count"] == 2
    assert result.loc[frame.index[0], "global_us_sector_breadth"] == pytest.approx(0.5)
    assert result.loc[frame.index[0], "global_us_sector_avg_ret_1d"] == pytest.approx(-0.005)
