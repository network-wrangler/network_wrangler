"""Tests for scoped link values."""

# Rest of the code...
import pandas as pd
import pytest
from network_wrangler.roadway.links.scopes import (
    _filter_to_conflicting_scopes,
    _filter_to_conflicting_timespan_scopes,
    _filter_to_matching_scope,
    _filter_to_matching_timespan_scopes,
    _filter_to_overlapping_scopes,
    _filter_to_overlapping_timespan_scopes,
    prop_for_scope,
    props_for_scopes,
)


def test_filter_to_overlapping_timespan_scopes():
    scoped_values = [
        {"value": 1, "timespan": ["6:00", "9:00"]},
        {"value": 1, "timespan": ["10:00", "12:00"]},
        {"value": 1, "timespan": ["13:00", "15:00"]},
    ]
    timespan = ["8:00", "11:00"]
    expected_result = [
        {"value": 1, "timespan": ["6:00", "9:00"], "category": "any"},
        {"value": 1, "timespan": ["10:00", "12:00"], "category": "any"},
    ]
    result = _filter_to_overlapping_timespan_scopes(scoped_values, timespan)
    assert [vars(i) for i in result] == expected_result


def test_filter_to_matching_timespan_scopes():
    scoped_values = [
        {"value": 1, "timespan": ["6:00", "9:00"]},
        {"value": 1, "timespan": ["10:00", "12:00"]},
        {"value": 1, "timespan": ["13:00", "15:00"]},
    ]
    timespan = ["8:00", "9:00"]
    expected_result = [
        {"value": 1, "timespan": ["6:00", "9:00"], "category": "any"},
    ]
    result = _filter_to_matching_timespan_scopes(scoped_values, timespan)
    assert [vars(i) for i in result] == expected_result


def test_filter_to_conflicting_timespan_scopes():
    scoped_values = [
        {"value": 1, "timespan": ["6:00", "9:00"]},
        {"value": 1, "timespan": ["10:00", "12:00"]},
        {"value": 1, "timespan": ["13:00", "15:00"]},
    ]
    timespan = ["8:00", "11:00"]
    expected_result = [
        {"value": 1, "timespan": ["6:00", "9:00"], "category": "any"},
        {"value": 1, "timespan": ["10:00", "12:00"], "category": "any"},
    ]
    result = _filter_to_conflicting_timespan_scopes(scoped_values, timespan)
    assert [vars(i) for i in result] == expected_result


def test_filter_to_conflicting_scopes():
    scoped_values = [
        {"value": 1, "category": "A", "timespan": ["6:00", "9:00"]},
        {"value": 1, "category": "B", "timespan": ["11:00", "12:00"]},
        {"value": 1, "category": "C", "timespan": ["13:00", "15:00"]},
    ]
    category = ["A", "B"]
    timespan = ["8:00", "11:00"]
    expected_result = [
        {"value": 1, "category": "A", "timespan": ["6:00", "9:00"]},
    ]
    result = _filter_to_conflicting_scopes(scoped_values, timespan, category)
    assert [vars(i) for i in result] == expected_result


def test_filter_to_matching_scope():
    scoped_values = [
        {"value": 1, "category": "A", "timespan": ["6:00", "9:00"]},
        {"value": 1, "category": "B", "timespan": ["10:00", "12:00"]},
        {"value": 1, "category": "C", "timespan": ["13:00", "15:00"]},
    ]
    category = ["A", "B"]
    timespan = ["8:00", "9:00"]
    expected_result = [
        {"value": 1, "category": "A", "timespan": ["6:00", "9:00"]},
    ]
    result, _ = _filter_to_matching_scope(scoped_values, category, timespan)
    assert [vars(i) for i in result] == expected_result


def test_filter_to_overlapping_scopes():
    scoped_prop_list = [
        {"value": 1, "category": "A", "timespan": ["6:00", "9:00"]},
        {"value": 1, "category": "B", "timespan": ["10:00", "12:00"]},
        {"value": 1, "category": "C", "timespan": ["13:00", "15:00"]},
        {"value": 1, "category": "A", "timespan": ["00:00", "24:00"]},
        {"value": 1, "category": "any", "timespan": ["8:00", "9:00"]},
        {"value": 1, "category": "any", "timespan": ["00:00", "24:00"]},
    ]
    category = ["A", "B"]
    timespan = ["8:00", "11:00"]
    expected_result = [
        {"value": 1, "category": "A", "timespan": ["6:00", "9:00"]},
        {"value": 1, "category": "B", "timespan": ["10:00", "12:00"]},
        {"value": 1, "category": "A", "timespan": ["00:00", "24:00"]},
        {"value": 1, "category": "any", "timespan": ["8:00", "9:00"]},
        {"value": 1, "category": "any", "timespan": ["00:00", "24:00"]},
    ]
    result = _filter_to_overlapping_scopes(scoped_prop_list, category, timespan)
    assert [vars(i) for i in result] == expected_result


# ---------------------------------------------------------------------------
# props_for_scopes
# ---------------------------------------------------------------------------


@pytest.fixture
def links_with_scoped_lanes():
    """Minimal links DataFrame with a scoped lanes column."""
    from network_wrangler.models.roadway.types import ScopedLinkValueItem

    return pd.DataFrame(
        {
            "model_link_id": [1, 2, 3],
            "lanes": [2, 3, 4],
            "sc_lanes": [
                [ScopedLinkValueItem(value=3, timespan=["6:00", "9:00"])],
                None,
                [
                    ScopedLinkValueItem(value=5, timespan=["6:00", "9:00"]),
                    ScopedLinkValueItem(value=6, timespan=["16:00", "19:00"]),
                ],
            ],
        }
    )


def test_props_for_scopes_matches_prop_for_scope(small_net):
    """Batch result must equal calling prop_for_scope individually for each scope."""
    links_df = small_net.links_df
    scopes = [
        {"label": "lanes_AM", "timespan": ["6:00", "9:00"]},
        {"label": "lanes_PM", "timespan": ["16:00", "19:00"]},
    ]
    batch = props_for_scopes(links_df, "lanes", scopes)

    for scope in scopes:
        single = prop_for_scope(links_df, "lanes", timespan=scope["timespan"])["lanes"]
        pd.testing.assert_series_equal(
            batch[scope["label"]].reset_index(drop=True),
            single.reset_index(drop=True),
            check_names=False,
        )


def test_props_for_scopes_no_scoped_column():
    df = pd.DataFrame({"model_link_id": [1, 2], "lanes": [2, 3]})
    result = props_for_scopes(df, "lanes", [{"label": "lanes_AM", "timespan": ["6:00", "9:00"]}])
    pd.testing.assert_series_equal(result["lanes_AM"], df["lanes"])


def test_props_for_scopes_missing_prop_raises():
    df = pd.DataFrame({"model_link_id": [1], "lanes": [2]})
    with pytest.raises(ValueError, match="not in dataframe"):
        props_for_scopes(df, "nonexistent", [{"label": "x", "timespan": ["6:00", "9:00"]}])
