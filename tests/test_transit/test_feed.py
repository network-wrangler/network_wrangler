"""Tests for public api of feed.py.

Run just these tests using `pytest tests/test_transit/test_feed.py`
"""

import pandas as pd
import pytest

from network_wrangler import WranglerLogger
from network_wrangler.transit.feed.shapes import (
    shape_id_for_trip_id,
    shapes_for_trip_id,
)
from network_wrangler.transit.feed.stop_times import (
    stop_times_for_pickup_dropoff_trip_id,
    stop_times_for_trip_id,
)
from network_wrangler.transit.feed.stops import stop_id_pattern_for_trip

TEST_TABLES_W_PROP = [
    ("route_short_name", ["routes"]),
    ("agency_id", ["routes"]),
    ("wheelchair_boarding", ["stops"]),
]


@pytest.mark.parametrize(("prop", "expected_tables"), TEST_TABLES_W_PROP)
def test_table_names_with_field(request, small_transit_net, prop, expected_tables):
    WranglerLogger.info(f"--Starting: {request.node.name}")

    tables = small_transit_net.feed.table_names_with_field(prop)
    assert tables == expected_tables

    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_trip_stop_times(request, small_transit_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    from network_wrangler.transit.feed.stop_times import stop_times_for_trip_id

    trip_id = "blue-2"
    stop_times = stop_times_for_trip_id(small_transit_net.feed.stop_times, trip_id)

    result = stop_times.stop_id.to_list()
    expected = [1, 3, 4]

    assert result == expected

    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_trip_shape_id(request, small_transit_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    from network_wrangler.transit.feed.shapes import shape_id_for_trip_id

    trip_id = "blue-1"
    result = shape_id_for_trip_id(small_transit_net.feed.trips, trip_id)
    WranglerLogger.debug(f"test_trip_shape_id result: {result}")
    expected = "shape1"

    assert result == expected

    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_trip_shape(request, small_transit_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    from network_wrangler.transit.feed.shapes import shapes_for_trip_id

    trip_id = "blue-2"
    shape = shapes_for_trip_id(
        small_transit_net.feed.shapes, small_transit_net.feed.trips, trip_id
    )

    # shape_id is "9020001"
    result = shape.shape_model_node_id.to_list()
    expected = [1, 2, 3, 4]

    assert result == expected

    WranglerLogger.info(f"--Finished: {request.node.name}")


TEST_TRIP_PATTERNS = [
    {
        "trip_id": "14944019-JUN19-MVS-BUS-Weekday-01",
        "pickup_type": "either",
        "answer": [
            45983,
            150855,
            46666,
            68609,
            62146,
            70841,
            69793,
            7688,
            100784,
            91685,
            71086,
            133183,
            44298,
            68417,
            72311,
            46083,
            75783,
            71964,
            71456,
            44190,
            44190,
            74898,
            51814,
            75787,
            75122,
            75788,
            123002,
        ],
    },
    {
        "trip_id": "14944019-JUN19-MVS-BUS-Weekday-01",
        "pickup_type": "both",
        "answer": [
            150855,
            46666,
            68609,
            62146,
            70841,
            69793,
            7688,
            100784,
            91685,
            71086,
            133183,
            44298,
            68417,
            72311,
            46083,
            75783,
            71964,
            71456,
            44190,
            44190,
            74898,
            51814,
            75787,
            75122,
            75788,
            123002,
        ],
    },
    {
        "trip_id": "14944019-JUN19-MVS-BUS-Weekday-01",
        "pickup_type": "pickup_only",
        "answer": [45983],
    },
    {
        "trip_id": "14944019-JUN19-MVS-BUS-Weekday-01",
        "pickup_type": "dropoff_only",
        "answer": [],
    },
]


@pytest.mark.parametrize("tpat_test", TEST_TRIP_PATTERNS)
def test_trip_stop_pattern(request, tpat_test, stpaul_transit_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    from network_wrangler.transit.feed.stops import stop_id_pattern_for_trip

    result = stop_id_pattern_for_trip(
        stpaul_transit_net.feed.stop_times,
        tpat_test["trip_id"],
        tpat_test["pickup_type"],
    )

    assert result == tpat_test["answer"]

    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_stop_times_for_trip_id(request, small_transit_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    trip_id = "blue-2"
    stoptimes = stop_times_for_trip_id(small_transit_net.feed.stop_times, trip_id)
    result = stoptimes[["trip_id", "stop_id"]].reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "trip_id": ["blue-2", "blue-2", "blue-2"],
            "stop_id": [1, 3, 4],
        }
    )
    pd.testing.assert_frame_equal(result, expected)
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_shape_id_for_trip_id(request, small_transit_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    trip_id = "blue-1"
    result = shape_id_for_trip_id(small_transit_net.feed.trips, trip_id)
    expected = "shape1"
    assert result == expected
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_shapes_for_trip_id(request, small_transit_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    trip_id = "blue-2"
    shapes = shapes_for_trip_id(
        small_transit_net.feed.shapes, small_transit_net.feed.trips, trip_id
    )
    _cols = ["shape_id", "shape_pt_sequence", "shape_model_node_id"]
    result = shapes[_cols].reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "shape_id": ["shape2", "shape2", "shape2", "shape2"],
            "shape_pt_sequence": [1, 2, 3, 4],
            "shape_model_node_id": [1, 2, 3, 4],
        }
    )
    pd.testing.assert_frame_equal(result, expected)
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_stop_times_for_pickup_dropoff_trip_id(request, small_transit_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    trip_id = "blue-2"
    pickup_dropoff = "either"
    stop_times = stop_times_for_pickup_dropoff_trip_id(
        small_transit_net.feed.stop_times, trip_id, pickup_dropoff
    )
    _cols = ["trip_id", "stop_id"]
    result = stop_times[_cols].reset_index(drop=True)

    expected = pd.DataFrame(
        {
            "trip_id": ["blue-2", "blue-2", "blue-2"],
            "stop_id": [1, 3, 4],
        }
    )
    pd.testing.assert_frame_equal(result, expected)
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_stop_id_pattern_for_trip(request, small_transit_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    trip_id = "blue-2"
    pickup_dropoff = "either"
    result = stop_id_pattern_for_trip(
        small_transit_net.feed.stop_times, trip_id, pickup_dropoff=pickup_dropoff
    )
    expected = [1, 3, 4]
    assert result == expected
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_feed_equality(request, small_transit_net):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    feed1 = small_transit_net.feed
    feed2 = feed1.deepcopy()

    # should be equal even though they are different instances
    # Note: Hash comparison might fail due to version differences in pandas/numpy
    # between local and CI environments, so we focus on logical equality
    assert feed1 == feed2

    # should still equal, because should only look at hash of "tables"
    feed2.__dict__["arbitrary_property"] = "blah"
    assert feed1 == feed2

    # if I change one of them, it shouldn't be the same anymore
    import copy

    new_stops = copy.deepcopy(feed2.stops)
    new_stops.loc[new_stops.stop_id == 1, "stop_name"] = 999
    feed2.stops = new_stops
    assert feed1 != feed2
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_filter_shapes_to_links(request):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    from network_wrangler.transit.feed.shapes import shapes_for_road_links

    links_df = pd.DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]})
    shapes_df = pd.DataFrame(
        {
            "shape_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            "shape_pt_sequence": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8],
            "shape_model_node_id": [1, 2, 3, 4, 5, 1, 2, 3, 1, 5, 4, 1, 2],
            "should_retain": [
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
        }
    )

    result = shapes_for_road_links(shapes_df, links_df)
    WranglerLogger.debug(f"result: \n{result}")

    expected = shapes_df.loc[shapes_df.should_retain].reset_index(drop=True)
    WranglerLogger.debug(f"expected: \n{expected}")
    pd.testing.assert_frame_equal(result, expected)
    WranglerLogger.info(f"--Finished: {request.node.name}")


def test_filter_stop_times_to_links(request):
    WranglerLogger.info(f"--Starting: {request.node.name}")
    from network_wrangler.transit.feed.stop_times import stop_times_for_shapes

    stop_times = pd.DataFrame(
        {
            "trip_id": ["t1", "t1", "t1", "t1", "t2", "t2", "t2"],
            "stop_sequence": [1, 2, 3, 4, 1, 2, 3],
            "stop_id": [1, 2, 3, 5, 1, 3, 7],
        }
    )

    shapes = pd.DataFrame(
        {
            "shape_id": ["s1", "s1", "s1", "s1", "s2", "s2", "s2"],
            "shape_pt_sequence": [1, 2, 3, 4, 1, 2, 3],
            "shape_model_node_id": [1, 2, 3, 4, 1, 2, 3],
        }
    )

    trips = pd.DataFrame({"trip_id": ["t1", "t2"], "shape_id": ["s1", "s2"]})

    # Expected DataFrame
    expected = pd.DataFrame(
        {
            "trip_id": ["t1", "t1", "t1", "t2", "t2"],
            "stop_sequence": [1, 2, 3, 1, 2],
            "stop_id": [1, 2, 3, 1, 3],
        }
    )

    # Function under test
    result = stop_times_for_shapes(stop_times, shapes, trips)
    # WranglerLogger.debug(f"original:\n{stop_times}")
    # WranglerLogger.debug(f"result:\n{result}")
    # WranglerLogger.debug(f"expected:\n{expected}")

    # Validate the results
    pd.testing.assert_frame_equal(result, expected, check_like=True)
