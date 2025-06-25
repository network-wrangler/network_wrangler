"""Filters, queries of a gtfs shapes table and node patterns."""

from __future__ import annotations

import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandera.typing import DataFrame

from ...logger import WranglerLogger
from ...models.gtfs.tables import (
    WranglerShapesTable,
    WranglerStopTimesTable,
    WranglerTripsTable,
)
from ...utils.data import concat_with_attr
from ..feed.transit_links import shapes_to_shape_links
from ..feed.transit_segments import (
    filter_shapes_to_segments,
    shape_links_to_longest_shape_segments,
)
from .feed import PickupDropoffAvailability


def shape_ids_for_trip_ids(trips: DataFrame[WranglerTripsTable], trip_ids: list[str]) -> list[str]:
    """Returns a list of shape_ids for a given list of trip_ids."""
    return trips[trips["trip_id"].isin(trip_ids)].shape_id.unique().tolist()


def shapes_for_shape_id(
    shapes: DataFrame[WranglerShapesTable], shape_id: str
) -> DataFrame[WranglerShapesTable]:
    """Returns shape records for a given shape_id."""
    shapes = shapes.loc[shapes.shape_id == shape_id]
    return shapes.sort_values(by=["shape_pt_sequence"])


def shapes_for_trip_id(
    shapes: DataFrame[WranglerShapesTable], trips: DataFrame[WranglerTripsTable], trip_id: str
) -> DataFrame[WranglerShapesTable]:
    """Returns shape records for a single given trip_id."""
    shape_id = shape_id_for_trip_id(trips, trip_id)
    return shapes.loc[shapes.shape_id == shape_id]


def shapes_for_trip_ids(
    shapes: DataFrame[WranglerShapesTable],
    trips: DataFrame[WranglerTripsTable],
    trip_ids: list[str],
) -> DataFrame[WranglerShapesTable]:
    """Returns shape records for list of trip_ids."""
    shape_ids = shape_ids_for_trip_ids(trips, trip_ids)
    return shapes.loc[shapes.shape_id.isin(shape_ids)]


def shapes_with_stop_id_for_trip_id(
    shapes: DataFrame[WranglerShapesTable],
    trips: DataFrame[WranglerTripsTable],
    stop_times: DataFrame[WranglerStopTimesTable],
    trip_id: str,
    pickup_dropoff: PickupDropoffAvailability = "either",
) -> DataFrame[WranglerShapesTable]:
    """Returns shapes.txt for a given trip_id with the stop_id added based on pickup_type.

    Args:
        shapes: WranglerShapesTable
        trips: WranglerTripsTable
        stop_times: WranglerStopTimesTable
        trip_id: trip id to select
        pickup_dropoff: str indicating logic for selecting stops based on piackup and dropoff
            availability at stop. Defaults to "either".
            "either": either pickup_type or dropoff_type > 0
            "both": both pickup_type or dropoff_type > 0
            "pickup_only": only pickup > 0
            "dropoff_only": only dropoff > 0
    """
    from .stop_times import stop_times_for_pickup_dropoff_trip_id  # noqa: PLC0415

    shapes = shapes_for_trip_id(shapes, trips, trip_id)
    trip_stop_times = stop_times_for_pickup_dropoff_trip_id(
        stop_times, trip_id, pickup_dropoff=pickup_dropoff
    )

    stop_times_cols = [
        "stop_id",
        "trip_id",
        "pickup_type",
        "drop_off_type",
    ]

    shape_with_trip_stops = shapes.merge(
        trip_stop_times[stop_times_cols],
        how="left",
        right_on="stop_id",
        left_on="shape_model_node_id",
    )
    shape_with_trip_stops = shape_with_trip_stops.sort_values(by=["shape_pt_sequence"])
    return shape_with_trip_stops


def node_pattern_for_shape_id(shapes: DataFrame[WranglerShapesTable], shape_id: str) -> list[int]:
    """Returns node pattern of a shape."""
    shape_df = shapes.loc[shapes["shape_id"] == shape_id]
    shape_df = shape_df.sort_values(by=["shape_pt_sequence"])
    return shape_df["shape_model_node_id"].to_list()


def shapes_with_stops_for_shape_id(
    shapes: DataFrame[WranglerShapesTable],
    trips: DataFrame[WranglerTripsTable],
    stop_times: DataFrame[WranglerStopTimesTable],
    shape_id: str,
) -> DataFrame[WranglerShapesTable]:
    """Returns a DataFrame containing shapes with associated stops for a given shape_id.

    Parameters:
        shapes (DataFrame[WranglerShapesTable]): DataFrame containing shape data.
        trips (DataFrame[WranglerTripsTable]): DataFrame containing trip data.
        stop_times (DataFrame[WranglerStopTimesTable]): DataFrame containing stop times data.
        shape_id (str): The shape_id for which to retrieve shapes with stops.

    Returns:
        DataFrame[WranglerShapesTable]: DataFrame containing shapes with associated stops.
    """
    from .trips import trip_ids_for_shape_id  # noqa: PLC0415

    trip_ids = trip_ids_for_shape_id(trips, shape_id)
    all_shape_stop_times = concat_with_attr(
        [shapes_with_stop_id_for_trip_id(shapes, trips, stop_times, t) for t in trip_ids]
    )
    shapes_with_stops = all_shape_stop_times[all_shape_stop_times["stop_id"].notna()]
    shapes_with_stops = shapes_with_stops.sort_values(by=["shape_pt_sequence"])
    return shapes_with_stops


def shapes_for_trips(
    shapes: DataFrame[WranglerShapesTable], trips: DataFrame[WranglerTripsTable]
) -> DataFrame[WranglerShapesTable]:
    """Filter shapes dataframe to records associated with trips table."""
    _sel_shapes = trips.shape_id.unique().tolist()
    filtered_shapes = shapes[shapes.shape_id.isin(_sel_shapes)]
    WranglerLogger.debug(
        f"Filtered shapes to {len(filtered_shapes)}/{len(shapes)} \
                         records that referenced one of {len(trips)} trips."
    )
    return filtered_shapes


def shapes_for_road_links(
    shapes: DataFrame[WranglerShapesTable], links_df: pd.DataFrame
) -> DataFrame[WranglerShapesTable]:
    """Filter shapes dataframe to records associated with links dataframe.

    EX:

    > shapes = pd.DataFrame({
        "shape_id": ["1", "1", "1", "1", "2", "2", "2", "2", "2"],
        "shape_pt_sequence": [1, 2, 3, 4, 1, 2, 3, 4, 5],
        "shape_model_node_id": [1, 2, 3, 4, 2, 3, 1, 5, 4]
    })

    > links_df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [2, 3, 4]
    })

    > shapes

    shape_id   shape_pt_sequence   shape_model_node_id *should retain*
    1          1                  1                        TRUE
    1          2                  2                        TRUE
    1          3                  3                        TRUE
    1          4                  4                        TRUE
    1          5                  5                       FALSE
    2          1                  1                        TRUE
    2          2                  2                        TRUE
    2          3                  3                        TRUE
    2          4                  1                       FALSE
    2          5                  5                       FALSE
    2          6                  4                       FALSE
    2          7                  1                       FALSE - not largest segment
    2          8                  2                       FALSE - not largest segment

    > links_df

    A   B
    1   2
    2   3
    3   4
    """
    """
    > shape_links

    shape_id  shape_pt_sequence_A  shape_model_node_id_A shape_pt_sequence_B shape_model_node_id_B
    1          1                        1                       2                        2
    1          2                        2                       3                        3
    1          3                        3                       4                        4
    1          4                        4                       5                        5
    2          1                        1                       2                        2
    2          2                        2                       3                        3
    2          3                        3                       4                        1
    2          4                        1                       5                        5
    2          5                        5                       6                        4
    2          6                        4                       7                        1
    2          7                        1                       8                        2
    """
    shape_links = shapes_to_shape_links(shapes)

    """
    > shape_links_w_links

    shape_id  shape_pt_sequence_A shape_pt_sequence_B  A  B
    1          1                         2             1  2
    1          2                         3             2  3
    1          3                         4             3  4
    2          1                         2             1  2
    2          2                         3             2  3
    2          7                         8             1  2
    """

    shape_links_w_links = shape_links.merge(
        links_df[["A", "B"]],
        how="inner",
        on=["A", "B"],
    )

    """
    Find largest segment of each shape_id that is in the links

    > longest_shape_segments
    shape_id, segment_id, segment_start_shape_pt_seq, segment_end_shape_pt_seq
    1          1                        1                       4
    2          1                        1                       3
    """
    longest_shape_segments = shape_links_to_longest_shape_segments(shape_links_w_links)

    """
    > shapes

    shape_id   shape_pt_sequence   shape_model_node_id
    1          1                  1
    1          2                  2
    1          3                  3
    1          4                  4
    2          1                  1
    2          2                  2
    2          3                  3
    """
    filtered_shapes = filter_shapes_to_segments(shapes, longest_shape_segments)
    filtered_shapes = filtered_shapes.reset_index(drop=True)
    return filtered_shapes


def shape_id_for_trip_id(trips: WranglerTripsTable, trip_id: str) -> str:
    """Returns a shape_id for a given trip_id."""
    return trips.loc[trips.trip_id == trip_id, "shape_id"].values[0]


def find_nearest_stops(
    shapes: WranglerShapesTable,
    trips: WranglerTripsTable,
    stop_times: WranglerStopTimesTable,
    trip_id: str,
    node_id: int,
    pickup_dropoff: PickupDropoffAvailability = "either",
) -> tuple[int, int]:
    """Returns node_ids (before and after) of nearest node_ids that are stops for a given trip_id.

    Args:
        shapes: WranglerShapesTable
        trips: WranglerTripsTable
        stop_times: WranglerStopTimesTable
        trip_id: trip id to find nearest stops for
        node_id: node_id to find nearest stops for
        pickup_dropoff: str indicating logic for selecting stops based on piackup and dropoff
            availability at stop. Defaults to "either".
            "either": either pickup_type or dropoff_type > 0
            "both": both pickup_type or dropoff_type > 0
            "pickup_only": only pickup > 0
            "dropoff_only": only dropoff > 0

    Returns:
        tuple: node_ids for stop before and stop after
    """
    shapes = shapes_with_stop_id_for_trip_id(
        shapes, trips, stop_times, trip_id, pickup_dropoff=pickup_dropoff
    )
    WranglerLogger.debug(f"Looking for stops near node_id: {node_id}")
    if node_id not in shapes["shape_model_node_id"].values:
        msg = f"Node ID {node_id} not in shapes for trip {trip_id}"
        raise ValueError(msg)
    # Find index of node_id in shapes
    node_idx = shapes[shapes["shape_model_node_id"] == node_id].index[0]

    # Find stops before and after new stop in shapes sequence
    nodes_before = shapes.loc[: node_idx - 1]
    stops_before = nodes_before.loc[nodes_before["stop_id"].notna()]
    stop_node_before = 0 if stops_before.empty else stops_before.iloc[-1]["shape_model_node_id"]

    nodes_after = shapes.loc[node_idx + 1 :]
    stops_after = nodes_after.loc[nodes_after["stop_id"].notna()]
    stop_node_after = 0 if stops_after.empty else stops_after.iloc[0]["shape_model_node_id"]

    return stop_node_before, stop_node_after
