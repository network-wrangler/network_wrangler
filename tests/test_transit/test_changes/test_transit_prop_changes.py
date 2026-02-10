"""Tests for transit property change.

Run just these tests using `pytest tests/test_transit/test_transit_prop_changes.py`
"""

import copy

import pandas as pd
import pytest
from pandera.errors import SchemaError

from network_wrangler import WranglerLogger


def test_invalid_field_value_set(request, small_transit_net):
    """Checks that changing data to an invalid field value will fail."""
    from network_wrangler.utils.models import TableValidationError

    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = copy.deepcopy(small_transit_net)
    feed = net.feed
    # For Enum/Categorical will fail when mutated.
    with pytest.raises(TypeError):
        feed.stops.loc[0, "wheelchair_boarding"] = 9999

    # Should fail to be coerced - add optional column with invalid string value
    _new_stop_times = copy.deepcopy(feed.stop_times)
    _new_stop_times["shape_dist_traveled"] = "abc"  # Invalid string for float column
    with pytest.raises(TableValidationError):
        feed.stop_times = _new_stop_times

    # Should fail to be coerced - create with object dtype to allow string value
    _new_stop_times = copy.deepcopy(feed.stop_times)
    _new_stop_times["arrival_time"] = _new_stop_times["arrival_time"].astype(object)
    _new_stop_times.loc[4, "arrival_time"] = "123"
    with pytest.raises(TableValidationError):
        feed.stop_times = _new_stop_times

    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_force_invalid_field_value(request, small_transit_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = copy.deepcopy(small_transit_net)
    feed = net.feed
    feed.stop_times.loc[3, "shape_dist_traveled"] = "abc"

    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_valid_field_value_set(request, small_transit_net):
    """Checks that setting a valid field value will pass."""
    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = copy.deepcopy(small_transit_net)
    feed = net.feed

    # Should be coerced - convert column to object dtype to allow string value assignment
    _new_stop_times = copy.deepcopy(feed.stop_times)
    _new_stop_times["stop_sequence"] = _new_stop_times["stop_sequence"].astype(object)
    _new_stop_times.loc[0, "stop_sequence"] = "1"
    feed.stop_times = _new_stop_times
    assert feed.stop_times.loc[0, "stop_sequence"] == 1

    _new_stop_times = copy.deepcopy(feed.stop_times)
    _new_stop_times["arrival_time"] = _new_stop_times["arrival_time"].astype(object)
    _new_stop_times.loc[1, "arrival_time"] = "12:00:00"
    feed.stop_times = _new_stop_times
    assert feed.stop_times.loc[1, "arrival_time"] == pd.Timestamp("12:00:00")

    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_coerce_times(request, small_transit_net):
    """Checks that setting a valid field value will pass."""
    WranglerLogger.info(f"--Starting: {request.node.name}")
    feed = small_transit_net.feed
    # Should be ok b/c GTFS can last "several days"
    _new_stop_times = copy.deepcopy(feed.stop_times)
    _new_stop_times.loc[1, "departure_time"] = "12:00:00"
    feed.stop_times = _new_stop_times
    assert isinstance(feed.stop_times.loc[1, "departure_time"], pd.Timestamp)
    assert feed.stop_times.loc[1, "departure_time"] == pd.Timestamp("12:00:00")
    WranglerLogger.debug(f"feed.stop_times: \n{feed.stop_times}")
    WranglerLogger.info(f"--Finished: {request.node.name}")


@pytest.mark.skip(reason="This test is not working as expected")
def test_coerce_over24hr_times(request, small_transit_net):
    """Checks that setting a valid field value will pass."""
    WranglerLogger.info(f"--Starting: {request.node.name}")
    feed = small_transit_net.feed
    # Should be ok b/c GTFS can last "several days"
    _new_stop_times = copy.deepcopy(feed.stop_times)
    _new_stop_times.loc[2, "arrival_time"] = "42:00:00"
    _new_stop_times.loc[1, "departure_time"] = "12:00:00"
    feed.stop_times = _new_stop_times
    assert isinstance(feed.stop_times.loc[2, "arrival_time"], pd.Timestamp)
    assert feed.stop_times.loc[2, "arrival_time"] == pd.Timestamp("18:00:00")
    WranglerLogger.debug(f"feed.stop_times: \n{feed.stop_times}")
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_transit_property_change(request, small_transit_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    net = copy.deepcopy(small_transit_net)
    trip_ids = ["blue-1"]
    timespan = ["04:00", "06:00:00"]
    new_headway = 1600
    project_card = {
        "project": "Bus Frequency Blue1",
        "transit_property_change": {
            "service": {"trip_properties": {"trip_id": trip_ids}, "timespans": [timespan]},
            "property_changes": {"headway_secs": {"set": new_headway}},
        },
    }
    og_df = copy.deepcopy(net.feed.frequencies)
    WranglerLogger.debug(f"Initial Frequencies: \n{net.feed.frequencies}")
    net.apply(project_card)

    # Filter the DataFrame correctly
    WranglerLogger.debug(f"Result Frequencies: \n{net.feed.frequencies}")
    target_df = net.feed.frequencies.loc[
        (net.feed.frequencies.trip_id.isin(trip_ids))
        & (net.feed.frequencies.start_time.dt.strftime("%H:%M") == timespan[0])
    ]

    if not (target_df["headway_secs"] == new_headway).all():
        WranglerLogger.error("Headway not changed as expected:")
        WranglerLogger.debug(f"Targeted Frequencies: \n{target_df}")
        raise AssertionError()

    unchanged_result_df = net.feed.frequencies.loc[
        ~net.feed.frequencies.index.isin(target_df.index)
    ]
    unchanged_og_df_df = og_df.loc[~og_df.index.isin(target_df.index)]

    try:
        pd.testing.assert_frame_equal(unchanged_result_df, unchanged_og_df_df)
    except AssertionError as e:
        WranglerLogger.error("Frequencies changed that shouldn't have:")
        WranglerLogger.debug(f"(Supposed to be) Unchanged Frequencies: \n{unchanged_result_df}")
        WranglerLogger.debug(f"Original Frequencies: \n{unchanged_og_df_df}")
        raise e

    if not (target_df["projects"] == f"{project_card['project']},").all():
        WranglerLogger.error(f"Projects not updated as expected with {project_card['project']}:")
        WranglerLogger.debug(f"Targeted Frequencies: \n{target_df}")
        raise AssertionError()
