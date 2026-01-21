"""Tests for scoped link values."""

# Rest of the code...
from network_wrangler.roadway.links.scopes import (
    _filter_to_conflicting_scopes,
    _filter_to_conflicting_timespan_scopes,
    _filter_to_matching_scope,
    _filter_to_matching_timespan_scopes,
    _filter_to_overlapping_scopes,
    _filter_to_overlapping_timespan_scopes,
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
