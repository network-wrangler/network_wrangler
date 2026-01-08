"""Utilities for getting GTFS into wrangler."""

import pprint
from pathlib import Path
from typing import Any, Literal, Optional, Union
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely.geometry

from ..errors import NodeNotFoundError, TransitValidationError
from ..logger import WranglerLogger
from ..models.gtfs.converters import (
    convert_stop_times_to_wrangler_stop_times,
    convert_stops_to_wrangler_stops,
)
from ..models.gtfs.gtfs import (
    FERRY_ROUTE_TYPES,
    GtfsModel,
    MIXED_TRAFFIC_ROUTE_TYPES,
    RAIL_ROUTE_TYPES,
    STATION_ROUTE_TYPES,
)
from ..models.gtfs.types import RouteType
from ..params import FEET_PER_MILE, LAT_LON_CRS, METERS_PER_KILOMETER
from ..roadway.network import RoadwayNetwork
from ..transit.feed.feed import Feed
# Import filter functions from their new location for backwards compatibility
from ..transit.filter import (
    drop_transit_agency,
    filter_transit_by_boundary,
    truncate_route_at_stop,
)
from .time import time_to_seconds

__all__ = [
    "create_feed_frequencies",
    "assess_stop_name_roadway_compatibility",
    "match_bus_stops_to_roadway_nodes",
    "add_unmatched_bus_stops_to_network",
    "create_connector_links_for_poor_match_stops",
    "create_links_for_failed_bus_paths",
    "create_bus_routes",
    "add_additional_data_to_stops",
    "add_additional_data_to_shapes",
    "add_stations_and_links_to_roadway_network",
    "create_feed_from_gtfs_model",
    "get_original_shape_points_between_stops",
    "calculate_path_deviation_from_shape",
    "find_shape_aware_shortest_path",
    # Re-exported from transit.filter for backwards compatibility
    "filter_transit_by_boundary",
    "drop_transit_agency",
    "truncate_route_at_stop",
]

# Constants
K_NEAREST_CANDIDATES = 20
"""Number of nearest candidate nodes to consider in match_bus_stops_to_roadway_nodes() when using name scoring."""
NAME_MATCH_WEIGHT = 0.9
"""Weight for name match score in combined scoring in match_bus_stops_to_roadway_nodes(). 0.9 means 90% name match, 10% distance."""
MIN_SUBSTRING_MATCH_LENGTH = 3
"""Minimum string length required for substring matching in assess_stop_name_roadway_compatibility(). Prevents spurious matches with single letters."""
SHAPE_DISTANCE_TOLERANCE = 1.10
"""Maximum ratio of path distance to shortest distance in shape-aware routing. Used in create_bus_routes() and find_shape_aware_shortest_path(). 1.10 means paths up to 110% of shortest distance are considered."""
MAX_SHAPE_CANDIDATE_PATHS = 20
"""Maximum number of candidate paths to evaluate in find_shape_aware_shortest_path() when doing shape-aware routing."""
NEAREST_K_SHAPES_TO_STOPS = 20
"""Number of nearest shape points to check for each stop"""
FIRST_LAST_SHAPE_STOP_IDX = 10
"""For loops, the first stop much match one of the first FIRST_LAST_SHAPE_STOP_IDX shapes,
and the last stop must match one of the last FIRST_LAST_SHAPE_STOP_IDX shapes"""

MAX_DISTANCE_STOP = {
    "feet": 0.10 * FEET_PER_MILE,
    "meters": 0.15 * METERS_PER_KILOMETER,
}
"""Maximum distance for a stop to match to a node."""


def create_feed_frequencies(  # noqa: PLR0915
    feed_tables: dict[str, pd.DataFrame],
    timeperiods: dict[str, tuple[str, str]],
    frequency_method: str,
    default_frequency_for_onetime_route: int = 180,
    trace_shape_ids: Optional[list[str]] = None,
):
    """Create frequencies table and convert GTFS-style tables to Wrangler-style Feed tables.

    This function transforms detailed GTFS trip schedules into a frequency-based representation
    used by the Wrangler Feed object. While GTFS specifies each individual transit trip with
    exact times, Wrangler uses representative trips with headways/frequencies for time periods.

    Process Steps:
    1. Adds route_id, direction_id, shape_id to stop_times from trips table
    2. Adds departure_minutes column (minutes after midnight) to stop_times
    3. Groups trips by stop pattern to verify consistency per shape_id
    4. Assigns each trip to a time period based on departure time
    5. Calculates headways using the specified frequency_method
    6. Creates new trip_ids based on shape_id (one trip per unique shape)
    7. Updates stop_times and trips tables to use new consolidated trip definitions

    See [GTFS frequencies.txt](https://gtfs.org/documentation/schedule/reference/#frequenciestxt)

    Args:
        feed_tables (dict[str, pd.DataFrame]): dictionary of feed tables to modify in place:
            Required input columns:
            - 'stop_times': trip_id, stop_id, stop_sequence, departure_time
            - 'trips': trip_id, route_id, direction_id, shape_id, service_id

            Columns added/modified:
            - 'stop_times': Adds route_id, direction_id, shape_id, departure_minutes;
                           Renames trip_id to orig_trip_id, creates new trip_id from shape_id
            - 'trips': Consolidated to one row per shape_id with new trip_id
            - 'frequencies': Created with trip_id, start_time, end_time, headway_secs

        timeperiods (dict[str, tuple[str, str]]): Time period labels to time spans.
            Example: {'EA': ('03:00','06:00'), 'AM': ('06:00','10:00')}
        frequency_method (str): Method for calculating headways:
            - 'uniform_headway': timeperiod_duration / number_of_trips
            - 'mean_headway': mean of time gaps between consecutive trips
            - 'median_headway': median of time gaps between consecutive trips
        default_frequency_for_onetime_route (int, optional): Default headway in minutes
            for routes with only one trip in a period. Defaults to 180 (3 hours).
        trace_shape_ids (Optional[list[str]]): Shape IDs to log detailed debug output for.

    Returns:
        None - Modifies feed_tables in place

    Notes:
        - Assumes shape_id uniquely identifies a route/direction/stop pattern combination
        - Handles time periods crossing midnight (e.g., '19:00' to '03:00')
        - For single-trip routes, uses default_frequency_for_onetime_route
        - Original trip_ids are preserved as 'orig_trip_id' for reference
    """
    WranglerLogger.info("Creating frequencies table from feed stop_times data")

    VALID_FREQUENCY_METHOD = ["uniform_headway", "mean_headway", "median_headway"]
    if frequency_method not in VALID_FREQUENCY_METHOD:
        msg = f"frequency_method must be on of {VALID_FREQUENCY_METHOD}; received {frequency_method}"
        raise ValueError(msg)

    # convert timeperiods to { timeperiod_label: [start_time, end_time]} where times are in minutes from midnight
    timeperiod_minutes_after_midnight = {}
    for label in timeperiods:
        timeperiod_minutes_after_midnight[label] = [
            time_to_seconds(f"{timeperiods[label][0]}:00") / 60.0,
            time_to_seconds(f"{timeperiods[label][1]}:00") / 60.0,
        ]
    # sort by the start_time of the timeperiod
    WranglerLogger.debug(
        f"timeperiod_minutes_after_midnight:\n{timeperiod_minutes_after_midnight}"
    )

    # Add route_id, direction_id, and shape_id to stop_times
    feed_tables["stop_times"] = pd.merge(
        left=feed_tables["stop_times"],
        right=feed_tables["trips"][
            ["trip_id", "shape_id", "direction_id", "route_id"]
        ].drop_duplicates(),
        how="left",
        validate="many_to_one",
    )
    # Add departure_minutes, which is departure_time in minutes after midnight (easier to read than seconds)
    feed_tables["stop_times"]["departure_minutes"] = feed_tables["stop_times"][
        "departure_time"
    ].apply(time_to_seconds)
    feed_tables["stop_times"]["departure_minutes"] = (
        feed_tables["stop_times"]["departure_minutes"] / 60.0
    )  # convert to minutes
    feed_tables["stop_times"]["departure_minutes"] = feed_tables["stop_times"][
        "departure_minutes"
    ].round(decimals=0)
    WranglerLogger.debug(f"feed_tables['stop_times']:\n{feed_tables['stop_times']}")

    # In GTFS, each trip_id is a vehicle trip with a single set of times for the stops.
    # In NetworkWrangler, each trip is a set of vehicle trips with a frequency per time period.
    # Assuming that the GTFS data includes shape_ids, which follow the shape of a route/direction,
    # we'll transform shape_ids to be the new trip_id, and determine frequencies for them
    WranglerLogger.info(
        f"There are {feed_tables['stop_times']['shape_id'].nunique():,} unique shape_ids."
    )
    WranglerLogger.info(f"These will serve as our new trip_ids.")

    # Check that stop patterns are the same for all trips for a given route_id, direction_id, shape_id
    # Group by trip_id and get ordered stop sequences
    stop_patterns_df = (
        (feed_tables["stop_times"].sort_values(["trip_id", "stop_sequence"]).groupby("trip_id"))
        .aggregate(
            trip_depart_time=pd.NamedAgg(column="departure_minutes", aggfunc="first"),
            stop_pattern=pd.NamedAgg(column="stop_id", aggfunc=list),
        )
        .reset_index(drop=False)
    )
    WranglerLogger.debug(f"stop_patterns_df:\n{stop_patterns_df}")
    # columns are trip_id, trip_depart_time, stop_pattern

    # Merge with trips to get route_id, direction_id, shape_id
    stop_patterns_df = pd.merge(
        stop_patterns_df,
        feed_tables["trips"][
            ["trip_id", "route_id", "direction_id", "shape_id"]
        ].drop_duplicates(),
        on="trip_id",
        how="left",
        validate="one_to_one",
    )
    # Assign each trip to a timeperiod using the label
    # And calculate timeperiod_duration_minutes
    timeperiod_duration_minutes = {}
    stop_patterns_df["timeperiod"] = "not_set"
    for timeperiod in timeperiod_minutes_after_midnight:
        # for the time period, in minutes after midnight
        start_time = timeperiod_minutes_after_midnight[timeperiod][0]
        end_time = timeperiod_minutes_after_midnight[timeperiod][1]

        assert start_time >= 0
        assert start_time < 24 * 60
        assert end_time >= 0
        assert end_time < 24 * 60
        assert start_time != end_time
        # doesn't roll over midnight
        if start_time < end_time:
            stop_patterns_df.loc[
                (stop_patterns_df["trip_depart_time"] >= start_time)
                & (stop_patterns_df["trip_depart_time"] < end_time),
                "timeperiod",
            ] = timeperiod
            timeperiod_duration_minutes[timeperiod] = end_time - start_time
        # rolls over
        else:
            stop_patterns_df.loc[
                (stop_patterns_df["trip_depart_time"] >= start_time)
                | (stop_patterns_df["trip_depart_time"] < end_time),
                "timeperiod",
            ] = timeperiod
            timeperiod_duration_minutes[timeperiod] = ((24 * 60) - start_time) + (end_time)

    WranglerLogger.debug(f"timeperiod_duration_minutes: {timeperiod_duration_minutes}")

    # Make sure all trips are assigned
    WranglerLogger.debug(
        f"trip_id by timeperiod:\n{stop_patterns_df['timeperiod'].value_counts()}"
    )
    assert len(stop_patterns_df.loc[stop_patterns_df.timeperiod == "not_set"]) == 0

    WranglerLogger.debug(f"stop_patterns_df:\n{stop_patterns_df}")
    #                   trip_id  trip_depart_time                                       stop_pattern route_id direction_id           shape_id timeperiod
    # 0        3D:1090:20230930            261.00  [811425, 810752, 810747, 819910, 810641, 81063...  3D:300X            1     3D:35:20230930         EA
    # 1        3D:1091:20230930            321.00  [811425, 810752, 810747, 819910, 810641, 81063...  3D:300X            1     3D:35:20230930         EA
    # 2        3D:1092:20230930            381.00  [811425, 810752, 810747, 819910, 810641, 81063...  3D:300X            1     3D:35:20230930         AM
    # 3        3D:1093:20230930            438.00  [817028, 812735, 812719, 817899, 814801, 81454...   3D:391            0     3D:51:20230930         AM
    # 4        3D:1094:20230930            542.00  [817754, 813644, 811043, 819910, 810948, 81093...  3D:300X            0     3D:36:20230930         AM

    # First, group without timeperiods to make sure there is one shape_id per route_id and direction_id
    shape_per_route_dir_df = (
        stop_patterns_df.groupby(by=["route_id", "direction_id", "shape_id"], observed=False)
        .aggregate(
            num_trip_ids=pd.NamedAgg(column="trip_id", aggfunc="nunique"),
            trip_ids=pd.NamedAgg(column="trip_id", aggfunc=list),
            first_trip_id=pd.NamedAgg(column="trip_id", aggfunc="first"),
            trip_depart_time=pd.NamedAgg(column="trip_depart_time", aggfunc=list),
            stop_pattern=pd.NamedAgg(column="stop_pattern", aggfunc="first"),
        )
        .reset_index(drop=False)
    )
    # drop lines with no relationship between shape_id and the stop_pattern
    shape_per_route_dir_df = shape_per_route_dir_df.loc[
        shape_per_route_dir_df.stop_pattern.notna()
    ].reset_index(drop=True)
    WranglerLogger.debug(f"shape_per_route_dir_df:\n{shape_per_route_dir_df}")

    # Check if there are shape_ids that appears for more than one (e.g. one shape_id for 1+ unique route_id, direction_id)
    # This is what showed that 'PE:t263-sl17-p182-r1A:20230930' appears to have the wrong direction_id
    shape_id_multiple_rows = shape_per_route_dir_df.loc[
        shape_per_route_dir_df.duplicated(subset="shape_id", keep=False)
    ]
    if len(shape_id_multiple_rows) > 0:
        WranglerLogger.error(f"shape_id_multiple_rows:\n{shape_id_multiple_rows}")
        # Hopefully there are none!
        assert len(shape_id_multiple_rows) == 0

    # Use shape_per_route_dir_df to create a mapping from the new trip_id (shape_id + '_trip') and
    # the representative (first) orig_trip_id
    new_old_trip_id_df = shape_per_route_dir_df[["shape_id", "first_trip_id"]].copy()
    new_old_trip_id_df["trip_id"] = new_old_trip_id_df["shape_id"] + "_trip"
    new_old_trip_id_df.rename(columns={"first_trip_id": "orig_trip_id"}, inplace=True)
    new_old_trip_id_df.drop(columns=["shape_id"], inplace=True)
    WranglerLogger.debug(f"new_old_trip_id_df:\n{new_old_trip_id_df}")

    # Now, groupby route_id, direction_id, shape_id, timeperiod and count unique trip_ids, stop_patterns
    shape_patterns_df = (
        stop_patterns_df.groupby(
            by=["route_id", "direction_id", "shape_id", "timeperiod"], observed=False
        )
        .aggregate(
            num_trip_ids=pd.NamedAgg(column="trip_id", aggfunc="nunique"),
            trip_ids=pd.NamedAgg(column="trip_id", aggfunc=list),
            trip_depart_time=pd.NamedAgg(
                column="trip_depart_time", aggfunc=lambda x: sorted(x)
            ),
            stop_pattern=pd.NamedAgg(column="stop_pattern", aggfunc="first"),
        )
        .reset_index(drop=False)
    )
    # WranglerLogger.debug(f"shape_patterns_df:\n{shape_patterns_df}")

    # drop lines with no relationship between shape_id and the stop_pattern
    shape_patterns_df = shape_patterns_df.loc[shape_patterns_df.stop_pattern.notna()].reset_index(
        drop=True
    )
    WranglerLogger.debug(f"shape_patterns_df:\n{shape_patterns_df}")
    #      route_id direction_id           shape_id timeperiod  num_trip_ids                                           trip_ids        trip_depart_time                                       stop_pattern
    # 0     3D:200X            0     3D:83:20230930         AM             3  [3D:1118:20230930, 3D:1130:20230930, 3D:1166:2...   [445.0, 505.0, 588.0]  [819191, 818955, 819209, 813979, 814568, 81399...
    # 1     3D:200X            0     3D:83:20230930         PM             3  [3D:1100:20230930, 3D:1354:20230930, 3D:1430:2...  [912.0, 972.0, 1032.0]  [819191, 818955, 819209, 813979, 814568, 81399...
    # 2     3D:200X            1     3D:82:20230930         AM             3  [3D:1117:20230930, 3D:1129:20230930, 3D:1165:2...   [380.0, 440.0, 520.0]  [818889, 816287, 816481, 819221, 819213, 81456...
    # 3     3D:200X            1     3D:82:20230930         MD             1                                 [3D:1099:20230930]                 [856.0]  [818889, 816287, 816481, 819221, 819213, 81456...
    # 4     3D:200X            1     3D:82:20230930         PM             2               [3D:1353:20230930, 3D:1429:20230930]          [916.0, 976.0]  [818889, 816287, 816481, 819221, 819213, 81456...

    # Determine frequencies for the shape_id, timeperiod
    # if num_trip_ids == 1, use default_frequency_for_onetime_route
    # if num_trip_ids > 1, we can use:
    #    timeperiod duration / num_trip_ids OR
    #    mean time between trip_depart_times OR
    #    median time between trip_depart_times
    # Let's set all of those

    shape_patterns_df["timeperiod_duration_minutes"] = shape_patterns_df["timeperiod"].map(
        timeperiod_duration_minutes
    )
    shape_patterns_df["uniform_headway"] = (
        shape_patterns_df["timeperiod_duration_minutes"] / shape_patterns_df["num_trip_ids"]
    )

    # Mean of differences (as integers)
    shape_patterns_df["mean_headway"] = shape_patterns_df["trip_depart_time"].apply(
        lambda x: int(np.mean(np.diff(x))) if len(x) > 1 else None
    )

    # Median of differences (as integers)
    shape_patterns_df["median_headway"] = shape_patterns_df["trip_depart_time"].apply(
        lambda x: int(np.median(np.diff(x))) if len(x) > 1 else None
    )

    # set it based on frequency_method argument
    shape_patterns_df["headway_mins"] = shape_patterns_df[frequency_method]
    # Make it default_frequency_for_onetime_route if needed
    shape_patterns_df.loc[shape_patterns_df["num_trip_ids"] <= 1, "headway_mins"] = (
        default_frequency_for_onetime_route
    )
    # if it's zero, use uniform_headway
    WranglerLogger.debug(f"Updating zero headway_mins to uniform_headway:\n{shape_patterns_df.loc[ shape_patterns_df['headway_mins']==0]}")
    shape_patterns_df.loc[ shape_patterns_df['headway_mins']==0, 'headway_mins'] = shape_patterns_df['uniform_headway']

    WranglerLogger.debug(
        f"After calculating different versions of headways, shape_patterns_df:\n{shape_patterns_df}"
    )
    #      route_id direction_id           shape_id timeperiod  num_trip_ids                                           trip_ids         trip_depart_time                                       stop_pattern  timeperiod_duration_minutes  uniform_headway  mean_headway  median_headway  headway_mins
    # 0     3D:200X            0     3D:83:20230930         AM             3  [3D:1118:20230930, 3D:1130:20230930, 3D:1166:2...    [445.0, 505.0, 588.0]  [819191, 818955, 819209, 813979, 814568, 81399...                       240.00            80.00         71.00           71.00         71.00
    # 1     3D:200X            0     3D:83:20230930         PM             3  [3D:1100:20230930, 3D:1354:20230930, 3D:1430:2...   [912.0, 972.0, 1032.0]  [819191, 818955, 819209, 813979, 814568, 81399...                       240.00            80.00         60.00           60.00         60.00
    # 2     3D:200X            1     3D:82:20230930         AM             3  [3D:1117:20230930, 3D:1129:20230930, 3D:1165:2...    [380.0, 440.0, 520.0]  [818889, 816287, 816481, 819221, 819213, 81456...                       240.00            80.00         70.00           70.00         70.00
    # 3     3D:200X            1     3D:82:20230930         MD             1                                 [3D:1099:20230930]                  [856.0]  [818889, 816287, 816481, 819221, 819213, 81456...                       300.00           300.00           NaN             NaN      10800.00
    # 4     3D:200X            1     3D:82:20230930         PM             2               [3D:1353:20230930, 3D:1429:20230930]           [916.0, 976.0]  [818889, 816287, 816481, 819221, 819213, 81456...                       240.00           120.00         60.00           60.00         60.00
    if trace_shape_ids:
        WranglerLogger.debug(
            f"trace_shape_ids shape_patterns_df:\n{shape_patterns_df.loc[shape_patterns_df.shape_id.isin(trace_shape_ids)]}"
        )

    # trip_id is now the same as shape_id (but let's add suffix '_trip')
    # Create feed_tables['frequencies'] compatible with WranglerFrequenciesTable
    timeperiod_start_times = {period: times[0] for period, times in timeperiods.items()}
    timeperiod_end_times = {period: times[1] for period, times in timeperiods.items()}
    WranglerLogger.debug(f"timeperiod_start_times:\n{timeperiod_start_times}")

    shape_patterns_df["trip_id"] = shape_patterns_df["shape_id"] + "_trip"
    shape_patterns_df["start_time"] = shape_patterns_df["timeperiod"].map(timeperiod_start_times)
    shape_patterns_df["end_time"] = shape_patterns_df["timeperiod"].map(timeperiod_end_times)
    shape_patterns_df["headway_secs"] = shape_patterns_df["headway_mins"] * 60
    WranglerLogger.debug(f"shape_patterns_df:\n{shape_patterns_df}")
    #      route_id direction_id           shape_id timeperiod  num_trip_ids                                           trip_ids          trip_depart_time                                       stop_pattern  timeperiod_duration_minutes  uniform_headway  mean_headway  median_headway  headway_mins                 trip_id start_time end_time  headway_secs
    # 0     3D:200X            0     3D:83:20230930         AM             3  [3D:1118:20230930, 3D:1130:20230930, 3D:1166:2...     [445.0, 505.0, 588.0]  [819191, 818955, 819209, 813979, 814568, 81399...                       240.00            80.00         71.00           71.00         71.00     3D:83:20230930_trip      06:00    10:00       4260.00
    # 1     3D:200X            0     3D:83:20230930         PM             3  [3D:1100:20230930, 3D:1354:20230930, 3D:1430:2...    [912.0, 972.0, 1032.0]  [819191, 818955, 819209, 813979, 814568, 81399...                       240.00            80.00         60.00           60.00         60.00     3D:83:20230930_trip      15:00    19:00       3600.00
    # 2     3D:200X            1     3D:82:20230930         AM             3  [3D:1117:20230930, 3D:1129:20230930, 3D:1165:2...     [380.0, 440.0, 520.0]  [818889, 816287, 816481, 819221, 819213, 81456...                       240.00            80.00         70.00           70.00         70.00     3D:82:20230930_trip      06:00    10:00       4200.00
    # 3     3D:200X            1     3D:82:20230930         MD             1                                 [3D:1099:20230930]                   [856.0]  [818889, 816287, 816481, 819221, 819213, 81456...                       300.00           300.00           NaN             NaN      10800.00     3D:82:20230930_trip      10:00    15:00     648000.00
    # 4     3D:200X            1     3D:82:20230930         PM             2               [3D:1353:20230930, 3D:1429:20230930]            [916.0, 976.0]  [818889, 816287, 816481, 819221, 819213, 81456...                       240.00           120.00         60.00           60.00         60.00     3D:82:20230930_trip      15:00    19:00       3600.00
    if trace_shape_ids:
        WranglerLogger.debug(
            f"trace_shape_ids shape_patterns_df:\n{shape_patterns_df.loc[shape_patterns_df.shape_id.isin(trace_shape_ids)]}"
        )

    feed_tables["frequencies"] = shape_patterns_df[
        ["trip_id", "start_time", "end_time", "headway_secs"]
    ]
    WranglerLogger.debug(f"feed_tables['frequencies']:\n{feed_tables['frequencies']}")

    # Update feed_tables['trips'] to be a WranglerTripsTable
    # Only one trip_id per shape_id, so drop the remaining trip_ids
    feed_tables["trips"].rename(columns={"trip_id": "orig_trip_id"}, inplace=True)
    feed_tables["trips"]["trip_id"] = feed_tables["trips"]["shape_id"] + "_trip"
    # make trip_id unique
    TRIP_TABLE_COLUMNS = [
        "shape_id",
        "direction_id",
        "service_id",
        "route_id",
        "trip_short_name",
        "trip_headsign",
        "block_id",
        "wheelchair_accessible",
        "bikes_allowed",
    ]
    aggregate_dict = {}
    for colname in TRIP_TABLE_COLUMNS:
        if colname not in feed_tables["trips"].columns:
            continue
        aggregate_dict[colname] = pd.NamedAgg(column=colname, aggfunc="first")
        aggregate_dict[f"{colname}_n"] = pd.NamedAgg(column=colname, aggfunc="nunique")

    feed_tables["trips"] = feed_tables["trips"].groupby("trip_id").agg(**aggregate_dict)
    # move trip_id back to regular column from index
    feed_tables["trips"].reset_index(drop=False, inplace=True)
    WranglerLogger.debug(
        f"After aggregating to new shape_id version of trip_id, "
        f"feed_tables['trips']:\n{feed_tables['trips']}"
    )
    # Log summary
    WranglerLogger.debug(
        f"After aggregating to new shape_id version of trip_id, "
        f"feed_tables['trips'].describe():\n{feed_tables['trips'].describe()}"
    )
    feed_tables["trips"] = feed_tables["trips"][["trip_id", *TRIP_TABLE_COLUMNS]]
    WranglerLogger.debug(f"feed_tables['trips']:\n{feed_tables['trips']}")

    # Update feed_tables['stop_times'] to be a WranglerStopTimesTable
    feed_tables["stop_times"].rename(columns={"trip_id": "orig_trip_id"}, inplace=True)
    feed_tables["stop_times"]["trip_id"] = feed_tables["stop_times"]["shape_id"] + "_trip"
    WranglerLogger.debug(
        f"Before trimming, feed_tables['stop_times']:\n{feed_tables['stop_times']}"
    )
    # There are stop_times for every original trip_id but we have fewer now, so filter out the extra
    feed_tables["stop_times"] = pd.merge(
        left=feed_tables["stop_times"],
        right=new_old_trip_id_df,  # trip_id, orig_trip_id
        on=["trip_id", "orig_trip_id"],
        how="inner",
    )
    # drop columns no long relevant
    feed_tables["stop_times"].drop(
        columns=[
            "orig_trip_id",
            "arrival_time",
            "departure_time",  # not relevant for frequency-based trips
            "departure_minutes",
        ],
        inplace=True,
    )
    WranglerLogger.debug(f"feed_tables['stop_times']:\n{feed_tables['stop_times']}")


def assess_stop_name_roadway_compatibility(
    stop_name: str,
    node_link_names: list[str],
    threshold: float = 0.5
) -> tuple[bool, float, list[str]]:
    """Assess if a transit stop name is compatible with a roadway node's link names.

    Checks if street names in the stop name match any of the node's connected link names.
    Handles common patterns like "Street1 & Street2" or "Street1 at Street2".

    Exact name matches receive a special high score of 10.0 to enable users to force
    specific stop-to-node matches by ensuring the stop name exactly matches a link name.

    Args:
        stop_name: Name of the transit stop (e.g., "Van Ness Ave & Market St")
        node_link_names: List of link names connected to the roadway node
        threshold: Minimum fraction of stop streets that must match node links (default 0.5)

    Returns:
        Tuple of (is_compatible, match_score, matched_streets) where:
            - is_compatible: True if stop name is compatible with node
            - match_score: 10.0 for exact match (to force selection), otherwise fraction
                           of stop streets found in node links (0.0 to 1.0)
            - matched_streets: List of street names from stop that matched node links
    """
    import re

    if not stop_name or (node_link_names is None or len(node_link_names) == 0):
        return False, 0.0, []

    # Check for exact match first - allows users to force specific matches by ensuring
    # stop names and link names are identical (case-insensitive)
    stop_name_normalized = stop_name.lower().strip()
    for node_link in node_link_names:
        if node_link.lower().strip() == stop_name_normalized:
            # Exact match gets special high score to strongly prefer/force this match
            return True, 10.0, [stop_name]

    # Common separators in stop names
    separators = [' & ', ' and ', ' at ', ' @ ', ' / ', ' near ']

    # Split stop name by separators to get individual street names
    stop_streets = [stop_name]
    for sep in separators:
        if sep in stop_name.lower():
            # Split and clean up each part
            parts = re.split(re.escape(sep), stop_name, flags=re.IGNORECASE)
            stop_streets = [part.strip() for part in parts if part.strip()]
            break

    # Normalize for comparison (lowercase, remove extra spaces)
    normalized_node_links = [link.lower().strip() for link in node_link_names]

    matched_streets = []
    for street in stop_streets:
        street_normalized = street.lower().strip()

        # Check for exact match or substring match
        for node_link in normalized_node_links:
            # Check if the street name is contained in the node link name or vice versa
            # Only do substring matching if both strings meet minimum length to avoid
            # spurious matches with single letters (e.g., "E" matching "Deer Creek")
            if len(street_normalized) >= MIN_SUBSTRING_MATCH_LENGTH and len(node_link) >= MIN_SUBSTRING_MATCH_LENGTH:
                if street_normalized in node_link or node_link in street_normalized:
                    matched_streets.append(street)
                    break

            # Also check for partial matches (e.g., "Market St" matches "Market Street")
            # Remove common suffixes for comparison
            suffixes = [' street', ' st', ' avenue', ' ave', ' road', ' rd', ' boulevard', ' blvd',
                       ' drive', ' dr', ' lane', ' ln', ' way', ' court', ' ct', ' place', ' pl',
                       ' parkway', ' pkwy', ' highway', ' hwy']

            street_base = street_normalized
            for suffix in suffixes:
                if street_base.endswith(suffix):
                    street_base = street_base[:-len(suffix)].strip()
                    break

            node_base = node_link
            for suffix in suffixes:
                if node_base.endswith(suffix):
                    node_base = node_base[:-len(suffix)].strip()
                    break

            # Apply same minimum length requirement for suffix-removed matching
            if (street_base and node_base and
                len(street_base) >= MIN_SUBSTRING_MATCH_LENGTH and
                len(node_base) >= MIN_SUBSTRING_MATCH_LENGTH and
                (street_base in node_base or node_base in street_base)):
                matched_streets.append(street)
                break

    # Calculate match score
    if len(stop_streets) > 0:
        match_score = len(matched_streets) / len(stop_streets)
    else:
        match_score = 0.0

    is_compatible = match_score >= threshold

    return is_compatible, match_score, matched_streets



def match_bus_stops_to_roadway_nodes(  # noqa: PLR0912, PLR0915
    feed_tables: dict[str, pd.DataFrame],
    roadway_net: RoadwayNetwork,
    local_crs: str,
    crs_units: str,
    max_distance: float,
    trace_shape_ids: Optional[list[str]] = None,
    use_name_matching: bool = True,
    name_match_weight: float = NAME_MATCH_WEIGHT,
):
    """Match bus stops to bus-accessible nodes in the roadway network.

    Matches bus and trolleybus stops to the nearest bus-accessible nodes in the roadway
    network using spatial proximity and optionally street name compatibility.
    Updates stop and shape locations to snap to road nodes.

    Process Steps:
    1. Identifies bus stops (route_types BUS or TROLLEYBUS) in feed_tables['stops']
    2. Builds bus network graph from roadway to find accessible nodes
    3. Projects geometries to local CRS for accurate distance calculations
    4. Uses BallTree spatial index to find candidate nodes within max_distance
    5. If name matching is enabled and link_names exist, scores candidates by both
       distance and name compatibility, selecting best match within max_distance
    6. Marks stops with combined_score > 0.9 as poor_match=True (only when name matching enabled)
       - Excludes stops that serve station route types (rail, ferry, etc.) - these are handled separately
    7. For poor_match stops, their model_node_id is the nearest bus-accessible node
       (to use for creating connector links later)
    8. Updates stop locations to matched road node locations (except poor_match stops)
    9. Updates shape point locations for matched bus stops

    Modifies feed_tables in place:

    feed_tables['stops'] - Adds/modifies columns:
        - is_bus_stop (bool): True if stop serves BUS or TROLLEYBUS routes
        - model_node_id (int): Matched roadway node ID (None if no close match)
        - match_distance_{crs_units} (float): Distance to matched node
        - close_match (bool): True if match found within max_distance
        - poor_match (bool): True if combined_score > 0.9 AND stop doesn't serve station routes (only when name matching enabled)
        - stop_lon, stop_lat, geometry: Updated to road node location if close_match and not poor_match

    feed_tables['shapes'] - Adds/modifies columns:
        - shape_model_node_id (int): Matched roadway node ID for bus stops
        - match_distance_{crs_units} (float): Distance to matched node
        - shape_pt_lon, shape_pt_lat, geometry: Updated to road node location if valid match

    feed_tables['stop_times'] - If GeoDataFrame, updates:
        - geometry: Updated to matched road node location for bus stops

    Args:
        feed_tables: dictionary of GTFS feed tables. Expects:
            - 'stops': Must have route_types column (list of RouteType enums)
            - 'shapes': Shape points to update
            - 'stop_times': Optional, updated if present as GeoDataFrame
        roadway_net: RoadwayNetwork with nodes to match against.
            Will be converted to GeoDataFrame if needed.
        local_crs: Coordinate reference system for projections (e.g., "EPSG:2227")
        crs_units: Distance units for local_crs ('feet' or 'meters')
        max_distance: Maximum matching distance in crs_units
        trace_shape_ids: Optional list of shape_ids for debug logging
        use_name_matching: If True and nodes have 'link_names', will consider name
            compatibility when selecting best match within max_distance. Default is True.
        name_match_weight: Weight for name match score in combined scoring (0.0 to 1.0).
            Final score = (1 - name_match_weight) * normalized_distance + name_match_weight * name_score
            Defaults to NAME_MATCH_WEIGHT constant.

    Raises:
        TransitValidationError: If no bus-accessible nodes found near any bus stops

    Notes:
        - Only matches stops that serve BUS or TROLLEYBUS routes
        - Uses bus modal graph to ensure matched nodes are bus-accessible
        - Preserves original locations for non-bus stops
    """
    if crs_units not in ["feet", "meters"]:
        msg = f"crs_units must be on of 'feet' or 'meters'; received {crs_units}"
        raise ValueError(msg)

    # Make roadway network nodes a GeoDataFrame if it's not already
    if not isinstance(roadway_net.nodes_df, gpd.GeoDataFrame):
        # Convert to GeoDataFrame if needed
        roadway_net.nodes_df = gpd.GeoDataFrame(
            roadway_net.nodes_df,
            geometry=gpd.points_from_xy(roadway_net.nodes_df.X, roadway_net.nodes_df.Y),
            crs=LAT_LON_CRS,
        )

    # Collect bus stops to match
    feed_tables["stops"]["is_bus_stop"] = False
    feed_tables["stops"].loc[
        feed_tables["stops"]["route_types"].apply(
            lambda x: RouteType.BUS in x if isinstance(x, list) else False
        )
        | feed_tables["stops"]["route_types"].apply(
            lambda x: RouteType.TROLLEYBUS in x if isinstance(x, list) else False
        ),
        "is_bus_stop",
    ] = True
    bus_stops_gdf = (
        feed_tables["stops"].loc[feed_tables["stops"]["is_bus_stop"] == True].copy(deep=True)
    )

    # Reset index to ensure continuous indices from 0 to n-1
    bus_stops_gdf.reset_index(drop=True, inplace=True)

    WranglerLogger.info(f"Matching {len(bus_stops_gdf):,} bus stops to roadway nodes")
    WranglerLogger.debug(f"bus_stops_gdf:\n{bus_stops_gdf}")
    WranglerLogger.debug(f"feed_tables['shapes']:\n{feed_tables['shapes']}")
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace shapes for {trace_shape_id}:\n"
                f"{feed_tables['shapes'].loc[feed_tables['shapes']['shape_id'] == trace_shape_id]}"
            )

    # Build BallTree for bus-access nodes that are in the bus graph
    # This ensures we only match to connected nodes
    G_bus = roadway_net.get_modal_graph("bus")
    bus_graph_nodes = set(G_bus.nodes())
    roadway_net.nodes_df["bus_access"] = False
    roadway_net.nodes_df.loc[
        roadway_net.nodes_df["model_node_id"].isin(bus_graph_nodes), "bus_access"
    ] = True
    WranglerLogger.debug(
        f"bus-accessible nodes from graph: {roadway_net.nodes_df['bus_access'].sum():,}"
    )

    # Extract bus-accessible nodes
    bus_accessible_nodes_gdf = roadway_net.nodes_df[
        roadway_net.nodes_df["bus_access"] == True
    ].copy()
    WranglerLogger.debug(
        f"Found {len(bus_accessible_nodes_gdf):,} bus-accessible nodes (for mixed-traffic transit) "
        f"out of {len(roadway_net.nodes_df):,} total"
    )

    # Project bus nodes and bus_stops_gdf to specified CRS and bus_stops
    bus_accessible_nodes_gdf.to_crs(local_crs, inplace=True)
    bus_stops_gdf.to_crs(local_crs, inplace=True)

    # Save original projected geometries AFTER projection (for restoring unmatched stops later)
    bus_stops_gdf["geometry_original"] = bus_stops_gdf["geometry"].copy()

    # Initialize results
    bus_stops_gdf["model_node_id"] = None
    bus_stops_gdf[f"match_distance_{crs_units}"] = np.inf  # in crs_units

    # Build spatial index for bus nodes
    try:
        from sklearn.neighbors import BallTree  # noqa: PLC0415
    except ImportError as e:
        msg = "sklearn is required for transit stop matching. Install with: pip install scikit-learn"
        raise ImportError(msg) from e
    
    bus_node_coords = np.array([(geom.x, geom.y) for geom in bus_accessible_nodes_gdf.geometry])
    bus_nodes_tree = BallTree(bus_node_coords)
    WranglerLogger.debug(f"Created BallTree for {len(bus_node_coords):,} bus nodes")

    # Get coordinates of stops to match
    bus_stop_coords = np.array([(geom.x, geom.y) for geom in bus_stops_gdf.geometry])

    # Query nearest neighbors - use more candidates if name matching is enabled
    k = 1  # Default to nearest neighbor only
    if use_name_matching and 'link_names' in bus_accessible_nodes_gdf.columns:
        k = min(K_NEAREST_CANDIDATES, len(bus_accessible_nodes_gdf))
        WranglerLogger.info(
            f"Using name-aware matching within {max_distance} {crs_units} "
            f"with name weight {name_match_weight}"
        )

    match_distances, match_indices = bus_nodes_tree.query(bus_stop_coords, k=k)

    # Process results based on whether we're doing name matching
    if k > 1:  # Name matching with multiple candidates
        # Initialize arrays to store best matches
        best_indices = np.zeros(len(bus_stops_gdf), dtype=int)
        best_distances = np.zeros(len(bus_stops_gdf))
        name_match_scores = np.zeros(len(bus_stops_gdf))
        normalized_dists = np.zeros(len(bus_stops_gdf))
        combined_scores = np.zeros(len(bus_stops_gdf))

        # Find best match for each stop considering both distance and name
        for stop_idx in range(len(bus_stops_gdf)):
            stop_name = bus_stops_gdf.iloc[stop_idx]["stop_name"]
            distances = match_distances[stop_idx]
            indices = match_indices[stop_idx]

            best_score = float('inf')
            best_idx = 0
            best_name_score = 0.0
            best_normalized_dist = 0.0
            best_combined_score = float('inf')

            # Evaluate candidates within max_distance
            candidates_found = False
            for i, (dist, node_idx) in enumerate(zip(distances, indices)):
                # only look at candidates within max_distance
                if dist > max_distance: continue

                candidates_found = True
                node_link_names = bus_accessible_nodes_gdf.iloc[node_idx].get("link_names", [])

                # Calculate name match score
                _, name_score, _ = assess_stop_name_roadway_compatibility(
                    stop_name, node_link_names if (node_link_names is not None and len(node_link_names) > 0) else []
                )

                # Combined score (lower is better)
                normalized_dist = dist / max_distance
                combined_score = (1 - name_match_weight) * normalized_dist + name_match_weight * (1 - name_score)

                if combined_score < best_score:
                    best_score = combined_score
                    best_idx = i
                    best_name_score = name_score
                    best_normalized_dist = normalized_dist
                    best_combined_score = combined_score

            # If no candidates within max_distance, use closest regardless
            if not candidates_found:
                # best_idx is already 0 (closest)
                # Calculate normalized_dist as >1 to indicate beyond max_distance
                best_normalized_dist = distances[0] / max_distance
                best_combined_score = (1 - name_match_weight) * best_normalized_dist + name_match_weight * (1 - best_name_score)

            # Store best match
            best_indices[stop_idx] = indices[best_idx]
            best_distances[stop_idx] = distances[best_idx]
            name_match_scores[stop_idx] = best_name_score
            normalized_dists[stop_idx] = best_normalized_dist
            combined_scores[stop_idx] = best_combined_score

        # Create matches dataframe
        matches_df = pd.DataFrame({
            "stop_idx": bus_stops_gdf.index,
            "match_node_idx": best_indices,
            "match_distance": best_distances,
            "name_match_score": name_match_scores,
            "normalized_dist": normalized_dists,
            "combined_score": combined_scores
        })
    else:
        # Simple nearest neighbor matching (k=1)
        matches_df = pd.DataFrame({
            "stop_idx": bus_stops_gdf.index,
            "match_node_idx": match_indices.flatten(),
            "match_distance": match_distances.flatten(),
        })

    # Check for close matches (within max_distance)
    matches_df["close_match"] = False
    matches_df.loc[matches_df["match_distance"] <= max_distance, "close_match"] = True
    WranglerLogger.debug(f"matches_df:\n{matches_df}")

    WranglerLogger.info(
        f"Found {matches_df.close_match.sum():,} close matches out of {len(bus_stops_gdf):,} total bus stops"
    )

    if matches_df.close_match.sum() == 0:
        exception = TransitValidationError("Found no bus-accessible nodes near bus stops.")
        raise exception

    # Get matched node information
    matched_nodes_gdf = bus_accessible_nodes_gdf.iloc[matches_df["match_node_idx"]]
    WranglerLogger.debug(f"matched_nodes_gdf:\n{matched_nodes_gdf}")

    # Update bus stops with matched node information (vectorized)
    # Update all matches with their corresponding node information
    # Note: geometry_original was already saved right after projection at line 682
    bus_stops_gdf.loc[matches_df["stop_idx"], f"match_distance_{crs_units}"] = matches_df[
        "match_distance"
    ].values
    bus_stops_gdf.loc[matches_df["stop_idx"], "close_match"] = matches_df["close_match"].values
    bus_stops_gdf.loc[matches_df["stop_idx"], "model_node_id"] = matched_nodes_gdf[
        "model_node_id"
    ].values
    bus_stops_gdf.loc[matches_df["stop_idx"], "stop_lon"] = matched_nodes_gdf["X"].values
    bus_stops_gdf.loc[matches_df["stop_idx"], "stop_lat"] = matched_nodes_gdf["Y"].values
    bus_stops_gdf.loc[matches_df["stop_idx"], "geometry"] = matched_nodes_gdf["geometry"].values

    # Add node link_names and name match scores if available
    if "link_names" in matched_nodes_gdf.columns:
        bus_stops_gdf.loc[matches_df["stop_idx"], "node_link_names"] = matched_nodes_gdf["link_names"].values
    if "name_match_score" in matches_df.columns:
        bus_stops_gdf.loc[matches_df["stop_idx"], "name_match_score"] = matches_df["name_match_score"].values
        bus_stops_gdf.loc[matches_df["stop_idx"], "normalized_dist"] = matches_df["normalized_dist"].values
        bus_stops_gdf.loc[matches_df["stop_idx"], "combined_score"] = matches_df["combined_score"].values
        # Report poor name matches
        poor_name_matches = matches_df[(matches_df["close_match"] == True) & (matches_df["name_match_score"] < 0.5)]
        if len(poor_name_matches) > 0:
            WranglerLogger.info(f"Found {len(poor_name_matches)} bus stops with low name compatibility (score < 0.5)")

    # Mark stops with poor combined_score (> 0.9) as poor_match
    # poor_match only applies when name matching is enabled (combined_score exists)
    # These will be handled by add_unmatched_bus_stops_to_network()
    debug_cols = ['stop_id','stop_name','model_node_id',f'match_distance_{crs_units}','close_match',
                  'name_match_score','node_link_names','geometry']

    # combined_score only exists when use_name_matching=True and link_names column exists
    # poor_match is defined as having combined_score > 0.9
    # BUT exclude stops that serve station route types (rail, ferry, etc.) - they're handled separately
    if "combined_score" in bus_stops_gdf.columns:
        debug_cols = debug_cols + ['combined_score', 'poor_match']

        # Check if stop serves any station route types (rail, ferry, etc.)
        bus_stops_gdf["serves_station_routes"] = bus_stops_gdf["route_types"].apply(
            lambda x: any(rt in STATION_ROUTE_TYPES for rt in x) if isinstance(x, list) else False
        )

        # poor_match = poor score AND not a station stop
        bus_stops_gdf["poor_match"] = (
            (bus_stops_gdf["close_match"] == True) &
            (bus_stops_gdf["combined_score"] > 0.9) &
            (bus_stops_gdf["serves_station_routes"] == False)
        )
        poor_score_stops = bus_stops_gdf[bus_stops_gdf["poor_match"] == True]

        # Log excluded stops (poor score but serve station routes)
        excluded_station_stops = bus_stops_gdf[
            (bus_stops_gdf["close_match"] == True) &
            (bus_stops_gdf["combined_score"] > 0.9) &
            (bus_stops_gdf["serves_station_routes"] == True)
        ]
        if len(excluded_station_stops) > 0:
            WranglerLogger.info(
                f"Found {len(excluded_station_stops)} stops with poor combined_score (> 0.9) "
                f"that serve station route types (rail/ferry/etc). These will NOT be marked as "
                f"poor_match and will be handled as stations in step 7."
            )

        if len(poor_score_stops) > 0:
            WranglerLogger.info(
                f"Found {len(poor_score_stops)} bus-only stops with poor_match=True (combined_score > 0.9). "
                f"These will be treated as unmatched stops and added to the network with "
                f"connector links to nearest bus-accessible nodes."
            )

            # Restore original geometry for poor_match stops (they shouldn't snap to matched nodes)
            # Use the saved geometry from after projection but before matching
            bus_stops_gdf.loc[bus_stops_gdf["poor_match"], "geometry"] = bus_stops_gdf.loc[bus_stops_gdf["poor_match"], "geometry_original"]

            debug_cols_with_poor = debug_cols + ['poor_match']
            WranglerLogger.debug(
                f"poor_match stops:\n"
                f"{bus_stops_gdf.loc[bus_stops_gdf['poor_match'], debug_cols_with_poor]}"
            )

        # Clean up temporary column
        bus_stops_gdf.drop(columns=["serves_station_routes"], inplace=True)
    else:
        # No name matching, so no poor_match stops (all matches are based on distance only)
        bus_stops_gdf["poor_match"] = False

    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace bus_stops_gdf for {trace_shape_id}:\n"
                f"{bus_stops_gdf.loc[bus_stops_gdf['shape_ids'].apply(lambda x, tid=trace_shape_id: tid in x), debug_cols]}"
            )

    # Clean up temporary column
    if "geometry_original" in bus_stops_gdf.columns:
        bus_stops_gdf.drop(columns=["geometry_original"], inplace=True)

    # verify model_node_id, f'match_distance_{crs_units}' and 'close_match' are not in feed_tables['stops']
    assert "model_node_id" not in feed_tables["stops"].columns
    assert f"match_distance_{crs_units}" not in feed_tables["stops"].columns
    assert "close_match" not in feed_tables["stops"].columns

    # Update feed_tables['stops'] by merging the updates
    merge_cols = [
        "stop_id",
        "model_node_id",
        f"match_distance_{crs_units}",
        "stop_lon",
        "stop_lat",
        "geometry",
        "close_match",
        "poor_match",
    ]
    feed_tables["stops"] = feed_tables["stops"].merge(
        bus_stops_gdf[merge_cols],
        on="stop_id",
        how="left",
        suffixes=("", "_bus"),
        validate="one_to_one",
    )
    # Only update stop location for close match AND not unmatched
    # (unmatched stops keep original location until they're added to network in step 6a)
    update_mask = (feed_tables["stops"]["close_match"] == True) & (feed_tables["stops"]["poor_match"] == False)
    feed_tables["stops"].loc[update_mask, "stop_lon"] = (
        feed_tables["stops"]["stop_lon_bus"]
    )
    feed_tables["stops"].loc[update_mask, "stop_lat"] = (
        feed_tables["stops"]["stop_lat_bus"]
    )
    feed_tables["stops"].loc[update_mask, "geometry"] = (
        feed_tables["stops"]["geometry_bus"]
    )
    # Drop bus-specific columns
    feed_tables["stops"].drop(
        columns=["stop_lon_bus", "stop_lat_bus", "geometry_bus"], inplace=True
    )

    WranglerLogger.debug(
        f"After merging with bus_stops_gdf, feed_tables['stops']:\n{feed_tables['stops']}"
    )
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace feed_tables['stops'] for {trace_shape_id}:\n"
                f"{feed_tables['stops'].loc[feed_tables['stops']['shape_ids'].apply(lambda x, tid=trace_shape_id: isinstance(x, list) and tid in x)]}"
            )

    # summary of match distances
    WranglerLogger.debug(
        f"\n{feed_tables['stops'][['close_match', f'match_distance_{crs_units}']].describe()}"
    )

    # Update shapes table similarly
    WranglerLogger.debug(f"feed_tables['shapes']:\n{feed_tables['shapes']}")
    # columns: shape_id, shape_pt_lat, shape_pt_lon, shape_pt_sequence, shape_dist_traveled, geometry
    #   trip_id, direction_id, route_id, agency_id, route_short_name, route_type, agency_name,
    #   match_distance_feet, stop_id, stop_name, stop_sequence
    # Note: this is adding: 'model_node_id'
    feed_tables["shapes"] = pd.merge(
        left=feed_tables["shapes"],
        right=bus_stops_gdf[
            [
                "stop_id",
                "model_node_id",
                f"match_distance_{crs_units}",
                "stop_lon",
                "stop_lat",
                "geometry",
                "close_match",
                "poor_match",
            ]
        ],
        on="stop_id",
        how="left",
        suffixes=("", "_bus"),
        validate="many_to_one",
    ).rename(columns={"model_node_id": "shape_model_node_id"})
    # Only update stop location for close match AND not unmatched
    # (unmatched stops keep original location in shapes until they're added to network in step 6a)
    shape_update_mask = (feed_tables["shapes"]["close_match"] == True) & (feed_tables["shapes"]["poor_match"] == False)
    feed_tables["shapes"].loc[shape_update_mask, "shape_pt_lon"] = (
        feed_tables["shapes"]["stop_lon"]
    )
    feed_tables["shapes"].loc[shape_update_mask, "shape_pt_lat"] = (
        feed_tables["shapes"]["stop_lat"]
    )
    feed_tables["shapes"].loc[shape_update_mask, "geometry"] = (
        feed_tables["shapes"]["geometry_bus"]
    )
    feed_tables["shapes"].loc[
        shape_update_mask, f"match_distance_{crs_units}"
    ] = feed_tables["shapes"][f"match_distance_{crs_units}_bus"]

    if trace_shape_ids:
        debug_cols = ["shape_pt_sequence","stop_sequence","stop_id","stop_name","shape_model_node_id","match_distance_feet_bus","poor_match"]
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace feed_tables['shapes'] for {trace_shape_id}"
                f"at the end of match_bus_stops_to_roadway_nodes():\n"
                f"{feed_tables['shapes'].loc[feed_tables['shapes']['shape_id'] == trace_shape_id][debug_cols]}"
            )

    # Drop bus-specific columns
    feed_tables["shapes"].drop(
        columns=[
            "stop_lon",
            "stop_lat",
            "geometry_bus",
            f"match_distance_{crs_units}_bus",
            "close_match",
        ],
        inplace=True,
    )

    # update geometry of feed_tables['stop_times'] if needed
    if isinstance(feed_tables["stop_times"], gpd.GeoDataFrame):
        feed_tables["stop_times"] = pd.merge(
            left=feed_tables["stop_times"],
            right=bus_stops_gdf[["stop_id", "geometry"]],
            how="left",
            on="stop_id",
            validate="many_to_one",
            suffixes=("", "_bus"),
            indicator=True,
        )
        WranglerLogger.debug(
            f"Updating feed['stop_times'] stop locations for bus stops:\n{feed_tables['stop_times']._merge.value_counts()}"
        )
        feed_tables["stop_times"].loc[
            feed_tables["stop_times"]["geometry_bus"].notna(), "geometry"
        ] = feed_tables["stop_times"]["geometry_bus"]
        feed_tables["stop_times"].drop(columns=["geometry_bus", "_merge"], inplace=True)


def add_unmatched_bus_stops_to_network(
    feed_tables: dict[str, pd.DataFrame],
    roadway_net: RoadwayNetwork,
    local_crs: str,
    max_distance: float,
    trace_shape_ids: Optional[list[str]] = None,
) -> gpd.GeoDataFrame:
    """Add unmatched bus stops as new nodes in the roadway network.

    Creates new roadway nodes for bus stops that couldn't be matched to the bus-accessible
    network. Clusters nearby unmatched stops (e.g., at transit stations) and creates one
    node per cluster at the centroid location.

    Process Steps:
    1. Identifies unmatched bus stops (poor_match=True)
    2. Clusters stops using max_distance threshold with DBSCAN
    3. Calculates centroid for each cluster
    4. Creates new roadway nodes at cluster centroids
    5. Updates stops table with new node IDs and locations
    6. Adds nodes to roadway network

    Args:
        feed_tables: dictionary of GTFS feed tables with 'stops' containing:
            - poor_match (bool): True for unmatched stops
            - model_node_id (int): Nearest bus-accessible node (for creating connector links)
            - Other stop attributes
        roadway_net: RoadwayNetwork to add nodes to
        local_crs: Coordinate reference system for projections (e.g., "EPSG:2227")
        max_distance: Distance threshold in crs_units for clustering
        trace_shape_ids: Optional list of shape_ids for debug logging

    Returns:
        GeoDataFrame of added nodes with columns:
            - model_node_id (int): New node ID
            - X, Y (float): Node coordinates in lat/lon
            - geometry (Point): Node geometry
            - cluster_id (int): Cluster assignment
            - stop_ids (list): List of stop_ids in this cluster
            - stop_names (list): List of stop names in this cluster
            - nearest_bus_node (int): Nearest bus-accessible node for connectivity
            - is_transit_stop_node (bool): True (marks these as special transit nodes)

    Notes:
        - Only processes bus stops (not rail/ferry stations)
        - Clusters stops within max_distance of each other
        - One node created per cluster at centroid
        - Original GTFS stop locations preserved before updating
        - Modifies feed_tables['stops'] in place
        - Modifies roadway_net.nodes_df in place
    """
    WranglerLogger.info("Adding unmatched bus stops to roadway network")

    # Get unmatched bus stops (poor_match=True means they have a model_node_id for nearest bus node)
    stops_df = feed_tables["stops"]
    unmatched_mask = (
        (stops_df["is_bus_stop"] == True) &
        (stops_df["poor_match"] == True) &
        (stops_df["model_node_id"].notna())
    )
    unmatched_stops_gdf = stops_df[unmatched_mask].copy()

    if len(unmatched_stops_gdf) == 0:
        WranglerLogger.info("No unmatched bus stops (poor_match=True) found; skipping")
        return gpd.GeoDataFrame()

    WranglerLogger.info(
        f"Processing {len(unmatched_stops_gdf)} unmatched bus stops "
        f"(clustering with max_distance={max_distance})"
    )

    # Ensure it's a GeoDataFrame
    if not isinstance(unmatched_stops_gdf, gpd.GeoDataFrame):
        unmatched_stops_gdf = gpd.GeoDataFrame(unmatched_stops_gdf, geometry="geometry")

    # Project to local CRS for clustering
    unmatched_stops_proj = unmatched_stops_gdf.to_crs(local_crs)

    # Extract coordinates for clustering
    coords = np.array([(geom.x, geom.y) for geom in unmatched_stops_proj.geometry])

    # Cluster using DBSCAN with max_distance threshold
    try:
        from sklearn.cluster import DBSCAN  # noqa: PLC0415
    except ImportError as e:
        msg = "sklearn is required for clustering. Install with: pip install scikit-learn"
        raise ImportError(msg) from e

    # DBSCAN eps parameter is the maximum distance between two samples in a cluster
    # min_samples=1 means a single point can be a cluster
    clustering = DBSCAN(eps=max_distance, min_samples=1, metric='euclidean')
    unmatched_stops_proj["cluster_id"] = clustering.fit_predict(coords)

    WranglerLogger.info(
        f"Clustered {len(unmatched_stops_gdf)} unmatched stops into "
        f"{unmatched_stops_proj['cluster_id'].nunique()} clusters"
    )

    # Calculate centroid for each cluster
    cluster_centroids = []
    for cluster_id in unmatched_stops_proj["cluster_id"].unique():
        cluster_stops = unmatched_stops_proj[unmatched_stops_proj["cluster_id"] == cluster_id]

        # Calculate centroid in projected CRS
        centroid_x = cluster_stops.geometry.x.mean()
        centroid_y = cluster_stops.geometry.y.mean()
        centroid_geom_proj = shapely.geometry.Point(centroid_x, centroid_y)

        # Convert back to lat/lon
        centroid_gdf = gpd.GeoDataFrame(
            {"geometry": [centroid_geom_proj]},
            crs=local_crs
        ).to_crs(LAT_LON_CRS)
        centroid_geom = centroid_gdf.geometry.iloc[0]

        # Get representative nearest bus node (use model_node_id from first stop in cluster)
        nearest_bus_node = cluster_stops.iloc[0]["model_node_id"]

        cluster_centroids.append({
            "cluster_id": cluster_id,
            "geometry": centroid_geom,
            "X": centroid_geom.x,
            "Y": centroid_geom.y,
            "stop_id_GTFS": str(list(set(cluster_stops["stop_id"].tolist()))), # make unique list
            "stop_name": str(list(set(cluster_stops["stop_name"].tolist()))), # make unique
            "nearest_bus_node": nearest_bus_node,
            "is_transit_stop_node": True,
            "num_stops_in_cluster": len(cluster_stops),
        })

    # Create GeoDataFrame of new nodes
    new_nodes_gdf = gpd.GeoDataFrame(cluster_centroids, crs=LAT_LON_CRS)
    WranglerLogger.debug(f"new_nodes_gdf:\n{new_nodes_gdf}")

    # Assign new model_node_ids
    # TODO: county numbering
    max_node_id = roadway_net.nodes_df["model_node_id"].max()
    new_nodes_gdf["model_node_id"] = range(max_node_id + 1, max_node_id + 1 + len(new_nodes_gdf))

    WranglerLogger.info(
        f"Creating {len(new_nodes_gdf)} new roadway nodes "
        f"(IDs {max_node_id + 1} to {max_node_id + len(new_nodes_gdf)})"
    )
    WranglerLogger.debug(f"Before adding nodes, roadway network has {len(roadway_net.nodes_df)} nodes")
    roadway_net.add_nodes(new_nodes_gdf)
    WranglerLogger.debug(f"After adding nodes, roadway network has {len(roadway_net.nodes_df)} nodes")

    # Update feed_tables['stops'] with new node IDs and locations
    # Create mapping from stop_id to new model_node_id
    stop_to_cluster = unmatched_stops_proj[["stop_id", "cluster_id"]].copy()
    cluster_to_node = new_nodes_gdf[["cluster_id", "model_node_id", "X", "Y", "geometry"]].copy()
    stop_to_node = stop_to_cluster.merge(cluster_to_node, on="cluster_id")
    WranglerLogger.debug(f"stop_to_node:\n{stop_to_node}")
    # Update stops table
    for _, row in stop_to_node.iterrows():
        stop_id = row["stop_id"]
        new_node_id = row["model_node_id"]
        new_x = row["X"]
        new_y = row["Y"]
        new_geom = row["geometry"]

        # Update in feed_tables['stops']
        mask = feed_tables["stops"]["stop_id"] == stop_id
        feed_tables["stops"].loc[mask, "model_node_id"] = new_node_id
        feed_tables["stops"].loc[mask, "stop_lon"] = new_x
        feed_tables["stops"].loc[mask, "stop_lat"] = new_y
        feed_tables["stops"].loc[mask, "geometry"] = new_geom

    WranglerLogger.info(
        f"Updated {len(stop_to_node)} stops to point to new cluster nodes"
    )

    # Log for trace_shape_ids
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            trace_stops = feed_tables["stops"][
                feed_tables["stops"]["shape_ids"].apply(
                    lambda x, tid=trace_shape_id: isinstance(x, list) and tid in x
                )
            ]
            WranglerLogger.debug(f"trace_stops for {trace_shape_id}:\n{trace_stops}")

    return new_nodes_gdf


def create_connector_links_for_poor_match_stops(
        roadway_net: RoadwayNetwork,
        unmatched_stops_gdf: gpd.GeoDataFrame,
        local_crs: str,
        crs_units: str,
        trace_shape_ids: Optional[list[str]] = None,
    ):
    """Create connector links between poor match bus stop nodes and nearest bus-accessible nodes.

    Creates bidirectional bus-only connector links in the roadway network to enable routing
    through bus stops that couldn't be matched directly to existing roadway nodes. These are
    typically stops that are too far from the road network (poor_match stops added as new nodes).

    Args:
        roadway_net: RoadwayNetwork to add connector links to
        unmatched_stops_gdf: GeoDataFrame of unmatched stops with columns:
            - model_node_id (int): New transit stop node ID
            - nearest_bus_node (int): Nearest bus-accessible node for connectivity
            - stop_ids, stop_names: Stop information
            - geometry (Point): Stop location
        local_crs: Coordinate reference system for distance calculations
        crs_units: Distance units ('feet' or 'meters')
        trace_shape_ids: Optional list of shape_ids for debug logging

    Notes:
        - Creates bidirectional links (forward and reverse) for each stop
        - Links are marked with ref="unmatched_bus_stop" for identification
        - All links are bus_only=True, no other mode access
        - Modifies roadway_net.links_df and roadway_net.shapes in place
    """
    if unmatched_stops_gdf is None or len(unmatched_stops_gdf) == 0:
        WranglerLogger.info("No unmatched stops to create connector links for")
        return

    WranglerLogger.info(
        f"Creating connector links for {len(unmatched_stops_gdf)} unmatched stop clusters"
    )
    WranglerLogger.debug(f"unmatched_stops_gdf:\n{unmatched_stops_gdf}")

    # Create bidirectional links between each new stop node and its nearest bus node
    connector_links = []
    for _, stop_node in unmatched_stops_gdf.iterrows():
        stop_node_id = stop_node["model_node_id"]
        nearest_node_id = stop_node["nearest_bus_node"]

        # Get node geometries
        stop_geom = stop_node["geometry"]
        nearest_node_row = roadway_net.nodes_df[
            roadway_net.nodes_df["model_node_id"] == nearest_node_id
        ]
        if len(nearest_node_row) == 0:
            WranglerLogger.warning(
                f"Could not find nearest_bus_node {nearest_node_id} for stop node {stop_node_id}"
            )
            continue
        nearest_geom = nearest_node_row.iloc[0]["geometry"]

        # Create LineString geometry for the link
        link_geom = shapely.geometry.LineString([
            (stop_geom.x, stop_geom.y),
            (nearest_geom.x, nearest_geom.y)
        ])

        # Forward link: stop -> nearest
        connector_links.append({
            "A": stop_node_id,
            "B": nearest_node_id,
            "geometry": link_geom,
            "name": f"unmatched_stop_{stop_node_id}_connector",
            "ref": "unmatched_bus_stop",
        })

        # Reverse link: nearest -> stop
        connector_links.append({
            "A": nearest_node_id,
            "B": stop_node_id,
            "geometry": link_geom,
            "name": f"unmatched_stop_{stop_node_id}_connector_reverse",
            "ref": "unmatched_bus_stop",
        })

    if len(connector_links) == 0:
        WranglerLogger.info("No valid connector links created")
        return

    connector_links_gdf = gpd.GeoDataFrame(connector_links, crs=LAT_LON_CRS)
    connector_links_gdf["shape_id"] = connector_links_gdf["A"].astype(str) + " to " + connector_links_gdf["B"].astype(str)
    connector_links_gdf["name"] = "bus stop connector"

    # Set transit access attributes
    connector_links_gdf["bus_only"] = True
    connector_links_gdf["rail_only"] = False
    connector_links_gdf["ferry_only"] = False
    connector_links_gdf["drive_access"] = True

    # No non-transit access
    connector_links_gdf["truck_access"] = False
    connector_links_gdf["bike_access"] = False
    connector_links_gdf["walk_access"] = False

    # Other attributes
    connector_links_gdf["roadway"] = "transit"
    connector_links_gdf["lanes"] = 1
    connector_links_gdf["managed"] = 0

    # Calculate distance
    connector_links_gdf.to_crs(local_crs, inplace=True)
    connector_links_gdf["length"] = connector_links_gdf.length
    if crs_units == "feet":
        connector_links_gdf["distance"] = connector_links_gdf["length"] / FEET_PER_MILE
    else:
        connector_links_gdf["distance"] = connector_links_gdf["length"] / METERS_PER_KILOMETER
    connector_links_gdf.to_crs(LAT_LON_CRS, inplace=True)

    # Assign model_link_ids
    max_model_link_id = roadway_net.links_df.model_link_id.max()
    connector_links_gdf["model_link_id"] = range(
        max_model_link_id + 1, max_model_link_id + 1 + len(connector_links_gdf)
    )

    WranglerLogger.info(
        f"Adding {len(connector_links_gdf)} connector links for unmatched stops "
        f"(IDs {max_model_link_id + 1} to {max_model_link_id + len(connector_links_gdf)})"
    )
    WranglerLogger.debug(
        f"connector_links_gdf:\n{connector_links_gdf}"
    )

    # Add to roadway network
    roadway_net.add_links(connector_links_gdf)
    roadway_net.add_shapes(connector_links_gdf)


def create_links_for_failed_bus_paths(
        roadway_net: RoadwayNetwork,
        no_bus_path_gdf: gpd.GeoDataFrame,
        local_crs: str,
        crs_units: str,
        trace_shape_ids: Optional[list[str]] = None,
    ):
    """Create direct transit-only links for bus stop pairs that couldn't be routed.

    When pathfinding through the bus network fails for consecutive bus stop pairs,
    this function creates direct transit-only links connecting them. These links
    enable the transit route to continue even when the underlying road network
    doesn't provide a valid bus path.

    Args:
        roadway_net: RoadwayNetwork to add links to
        no_bus_path_gdf: GeoDataFrame of failed bus path segments with columns:
            - A, B (int): Stop node IDs that couldn't be connected via pathfinding
            - trip_id, stop_sequence: Trip and sequence information
            - stop_id, stop_name: Stop identifiers
            - next_stop_id, next_stop_name: Next stop identifiers
            - geometry (LineString): Direct connection geometry between stops
            - route_type, route_id, direction_id, shape_id: Route metadata
        local_crs: Coordinate reference system for distance calculations
        crs_units: Distance units ('feet' or 'meters')
        trace_shape_ids: Optional list of shape_ids for debug logging

    Notes:
        - Links are marked with ref="bad_bus_path" for identification
        - Links allow all transit modes (bus_only, rail_only, ferry_only = True)
        - If a link already exists for an A->B pair, transit access is added to existing link
        - Modifies roadway_net.links_df and roadway_net.shapes in place
    """
    if no_bus_path_gdf is None or len(no_bus_path_gdf) == 0:
        WranglerLogger.info("No failed bus paths to create links for")
        return

    WranglerLogger.info(f"Creating links for {len(no_bus_path_gdf)} failed bus path segments")

    # no_bus_path_gdf columns:
    # A, B, shape_id, stop_sequence, route_type, route_id, direction_id, trip_id,
    # stop_id, stop_name, next_stop_id, next_stop_name, num_points, geometry
    add_links_gdf = no_bus_path_gdf.copy()
    # drop some unneeded columns
    add_links_gdf.drop(columns=["route_type","route_id","direction_id","shape_id","num_points"])
    # roll up to unique A,B, using the first
    add_links_gdf = gpd.GeoDataFrame(data=add_links_gdf.groupby(by=["A","B"]).agg(
        trip_ids       = pd.NamedAgg(column="trip_id", aggfunc=list),
        stop_seqs      = pd.NamedAgg(column="stop_sequence", aggfunc=list),
        stop_id        = pd.NamedAgg(column="stop_id", aggfunc="first"),
        stop_name      = pd.NamedAgg(column="stop_name", aggfunc="first"),
        next_stop_id   = pd.NamedAgg(column="next_stop_id", aggfunc="first"),
        next_stop_name = pd.NamedAgg(column="next_stop_name", aggfunc="first"),
        geometry       = pd.NamedAgg(column="geometry", aggfunc="first")
    ).reset_index(drop=False), geometry="geometry", crs=no_bus_path_gdf.crs)
    add_links_gdf["name"] = add_links_gdf["trip_ids"].astype(str)
    add_links_gdf["shape_id"] = (
        add_links_gdf["stop_id"] + " to " + add_links_gdf["next_stop_id"]
    )
    # make ok for buses (but not ferry or rail, since that's confusing)
    add_links_gdf["rail_only"] = False
    add_links_gdf["ferry_only"] = False
    add_links_gdf["bus_only"] = True
    add_links_gdf["drive_access"] = True
    # not ok for others
    add_links_gdf["truck_access"] = False
    add_links_gdf["bike_access"] = False
    add_links_gdf["walk_access"] = False
    # fill in some defaults
    add_links_gdf["roadway"] = "transit"
    add_links_gdf["lanes"] = 1
    add_links_gdf["managed"] = 0
    # this is how you find me
    add_links_gdf["ref"] = "bad_bus_path"

    add_links_gdf.to_crs(local_crs, inplace=True)
    add_links_gdf["length"] = add_links_gdf.length
    if crs_units == "feet":
        add_links_gdf["distance"] = add_links_gdf["length"] / FEET_PER_MILE
    else:
        add_links_gdf["distance"] = (
            add_links_gdf["length"] / METERS_PER_KILOMETER
        )
    add_links_gdf.to_crs(LAT_LON_CRS, inplace=True)
    add_links_gdf.reset_index(drop=True, inplace=True)

    # check if any exist already
    add_links_gdf["temp_model_link_id"] = add_links_gdf.index
    exists_already_df = pd.merge(
        left=roadway_net.links_df,
        right=add_links_gdf[["A","B","temp_model_link_id"]],
        on=["A","B"],
        how="inner"
    )
    if len(exists_already_df) > 0:
        WranglerLogger.warning(
            f"Can't add the following links because they exist already; adding transit modes:\n"
            f"{exists_already_df}")
        # set transit usability for those links
        roadway_net.links_df.loc[ roadway_net.links_df["model_link_id"].isin(exists_already_df["model_link_id"]), "rail_only"] = True
        roadway_net.links_df.loc[ roadway_net.links_df["model_link_id"].isin(exists_already_df["model_link_id"]), "bus_only" ] = True
        roadway_net.links_df.loc[ roadway_net.links_df["model_link_id"].isin(exists_already_df["model_link_id"]), "ferry_only"] = True
        # remove the duplicate from add_links_gdf
        add_links_gdf = add_links_gdf.loc[ ~add_links_gdf["temp_model_link_id"].isin(exists_already_df["temp_model_link_id"])]
        add_links_gdf.reset_index(drop=True, inplace=True)

    # we're done with this
    add_links_gdf.drop(columns=["temp_model_link_id"], inplace=True)

    if len(add_links_gdf) == 0:
        WranglerLogger.info("All failed bus path links already exist")
        return

    # assign model_link_id
    max_model_link_id = roadway_net.links_df.model_link_id.max()
    add_links_gdf["model_link_id"] = add_links_gdf.index + max_model_link_id + 1

    # log for trace_shape_ids
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            trace_trip_id = f'{trace_shape_id}_trip'
            shape_mask = add_links_gdf['trip_ids'].apply(lambda x: trace_trip_id in x if isinstance(x, list) else False)
            if shape_mask.any():
                WranglerLogger.debug(
                    f"adding links for trace {trace_shape_id} in create_links_for_failed_bus_paths:\n"
                    f"{add_links_gdf.loc[shape_mask]}"
                )

    # Add links
    WranglerLogger.debug(f"add_links_gdf:\n{add_links_gdf}")
    roadway_net.add_links(add_links_gdf)
    WranglerLogger.info(f"Adding {len(add_links_gdf):,} links for failed bus paths")

    # Add shapes
    roadway_net.add_shapes(add_links_gdf)


def create_bus_routes(  # noqa: PLR0912, PLR0915
    bus_stop_links_gdf: gpd.GeoDataFrame,
    feed_tables: dict[str, pd.DataFrame],
    roadway_net: RoadwayNetwork,
    local_crs: str,
    crs_units: str,
    trace_shape_ids: Optional[list[str]] = None,
    errors: Literal["raise", "ignore"] = "raise",
):
    """Find shortest paths through the bus network between consecutive bus stops.

    Replaces original bus route shapes with new shapes that follow the actual bus network
    by finding shortest paths between consecutive stops through bus-accessible roads.

    Process Steps:
    1. Sorts bus stop links by shape_id and stop_sequence
    2. Gets bus modal graph from roadway network
    3. For each consecutive stop pair in each shape:
       - Finds shortest path through bus network using NetworkX
       - Creates shape points for all nodes in the path
       - Preserves stop information at stop nodes
    4. Replaces bus shapes in feed_tables['shapes'] with new routed shapes

    Modifies feed_tables['shapes'] in place:
    - Removes existing bus/trolleybus shapes
    - Adds new shapes with points following road network paths
    - Each shape point has shape_model_node_id from roadway network
    - Stop points retain stop_id, stop_name, stop_sequence

    Args:
        bus_stop_links_gdf: GeoDataFrame of bus stop pairs, required columns:
            - shape_id (str): Shape identifier
            - stop_sequence (int): Stop order in route
            - stop_id (str): Current stop ID
            - stop_name (str): Current stop name
            - next_stop_id (str): Next stop ID
            - next_stop_name (str): Next stop name
            - A (int): Current stop's model_node_id
            - B (int): Next stop's model_node_id
            - route_id, route_type, trip_id, direction_id: Route metadata
        feed_tables: dictionary with required tables:
            - 'stops': Must have is_bus_stop column
            - 'shapes': Will be modified to replace bus shapes
        roadway_net: RoadwayNetwork with bus modal graph
        local_crs: Coordinate reference system for projections
        crs_units: Distance units ('feet' or 'meters')
        trace_shape_ids: Optional shape IDs for debug logging
        errors: 'raise' or 'ignore'

    Raises:
        TransitValidationError: If no path exists between any consecutive stops.
            Exception includes no_bus_path_gdf with failed stop sequences.

    Notes:
        - Uses NetworkX shortest_path for routing
        - Intermediate nodes between stops are added as shape points
        - Original shape geometry is replaced with routed paths
    """
    if crs_units not in ["feet", "meters"]:
        msg = f"crs_units must be on of 'feet' or 'meters'; received {crs_units}"
        raise ValueError(msg)

    WranglerLogger.info(f"Creating bus routes between bus stops")
    WranglerLogger.debug(
        f"bus stops:\n{feed_tables['stops'].loc[feed_tables['stops']['is_bus_stop'] == True]}"
    )
    WranglerLogger.debug(
        f"bus shapes:\n{feed_tables['shapes'].loc[feed_tables['shapes']['route_type'].isin([RouteType.BUS, RouteType.TROLLEYBUS])]}"
    )
    WranglerLogger.debug(f"bus_stop_links_gdf:\n{bus_stop_links_gdf}")
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace bus_stop_links_gdf for {trace_shape_id}:\n"
                f"{bus_stop_links_gdf.loc[bus_stop_links_gdf.shape_id==trace_shape_id]}"
            )

    # Create a shortest path through the bus network between each consecutive bus stop for a given shape_id,
    # traversing through intermediate roadway nodes
    # Create new shape links between those nodes
    # Check how far away the new shape links are from the given stop-to-stop shape links
    bus_stop_links_gdf.sort_values(by=["shape_id", "stop_sequence"], inplace=True)
    G_bus_multi = roadway_net.get_modal_graph("bus")

    # Convert MultiDiGraph to DiGraph for pathfinding
    # DiGraph is required for nx.shortest_simple_paths() used in shape-aware routing
    # Keep directionality but collapse multiple edges to shortest
    G_bus = nx.DiGraph()
    for u, v, data in G_bus_multi.edges(data=True):
        if G_bus.has_edge(u, v):
            # Keep edge with minimum distance
            if data.get('distance', float('inf')) < G_bus[u][v]['distance']:
                G_bus[u][v]['distance'] = data.get('distance', 0)
        else:
            G_bus.add_edge(u, v, distance=data.get('distance', 0))

    WranglerLogger.debug(f"Converted MultiDiGraph ({G_bus_multi.number_of_edges()} edges) to DiGraph ({G_bus.number_of_edges()} edges)")

    # collect node sequences for these shapes
    bus_node_sequence = []
    # also collect failed stop sequences
    no_path_sequence = []

    current_shape_id = None
    current_shape_pt_sequence = None
    for _idx, row in bus_stop_links_gdf.iterrows():
        # restart for each shape_id
        if current_shape_id != row["shape_id"]:
            current_shape_pt_sequence = 1
            current_shape_id = row["shape_id"]

        if trace_shape_ids and current_shape_id in trace_shape_ids:
            WranglerLogger.debug(f"trace path {current_shape_id}: {_idx} Looking for path from {row['A']} to {row['B']}")
            WranglerLogger.debug(f"\n{row}")

        # Check if either node is not in the bus graph (e.g., poor_match=True)
        # If so, skip pathfinding and add simple A->B connection to bus_node_sequence
        if not G_bus.has_node(row["A"]) or not G_bus.has_node(row["B"]):
            WranglerLogger.warning(
                f"Node not in bus graph for {row['shape_id']} from {row['A']} to {row['B']} "
                f"(likely poor_match). Adding direct A->B connection."
            )
            no_path_sequence.append(
                {
                    "shape_id": row["shape_id"],
                    "stop_id": row["stop_id"],
                    "next_stop_id": row["next_stop_id"],
                    "stop_sequence": row["stop_sequence"],
                }
            )

            # Add simple A->B path to bus_node_sequence (in correct order)
            path = [row["A"], row["B"]]
            if current_shape_pt_sequence != 1:
                path = path[1:]  # Skip first node to avoid duplication

            for path_node_id in path:
                bus_node_dict = {
                    "shape_id": row["shape_id"],
                    "route_id": row["route_id"],
                    "route_type": row["route_type"],
                    "trip_id": row["trip_id"],
                    "direction_id": row["direction_id"],
                    "shape_pt_sequence": current_shape_pt_sequence,
                    "shape_model_node_id": path_node_id,
                }
                # Set stop info for the nodes
                if path_node_id == row["A"]:
                    bus_node_dict["stop_id"] = row["stop_id"]
                    bus_node_dict["stop_name"] = row["stop_name"]
                    bus_node_dict["stop_sequence"] = row["stop_sequence"]
                elif path_node_id == row["B"]:
                    bus_node_dict["stop_id"] = row["next_stop_id"]
                    bus_node_dict["stop_name"] = row["next_stop_name"]
                    bus_node_dict["stop_sequence"] = row["stop_sequence"] + 1

                bus_node_sequence.append(bus_node_dict)
                current_shape_pt_sequence += 1
                last_stop_sequence = row["stop_sequence"]

            continue

        try:
            # Find shortest path with optional shape-aware selection
            use_shape_aware_routing = True  # Could be made a parameter

            if use_shape_aware_routing:
                # Get original shape points between these stops for comparison
                original_shape_points = get_original_shape_points_between_stops(
                    feed_tables, row["shape_id"], row["stop_sequence"],
                    row["stop_sequence"] + 1, trace_shape_ids and current_shape_id in trace_shape_ids
                )

                path = find_shape_aware_shortest_path(
                    G_bus, row["A"], row["B"], original_shape_points,
                    roadway_net, SHAPE_DISTANCE_TOLERANCE, trace_shape_ids and current_shape_id in trace_shape_ids
                )
            else:
                # Standard shortest path
                path = nx.shortest_path(G_bus, row["A"], row["B"], weight="distance")

            if trace_shape_ids and current_shape_id in trace_shape_ids:
                WranglerLogger.debug(f"trace path {current_shape_id}: Found path for {row['A']} to {row['B']}: len={len(path)} {path}")

            # Create shape point rows for that path
            # Only include first point if it's the first path for the shape,
            # otherwise it'll be added twice -- as the last point of the previous path
            # and the first point of the current one
            if current_shape_pt_sequence != 1:
                path = path[1:]
            for path_node_id in path:
                bus_node_dict = {
                    "shape_id": row["shape_id"],
                    "route_id": row["route_id"],
                    "route_type": row["route_type"],
                    "trip_id": row["trip_id"],
                    "direction_id": row["direction_id"],
                    "shape_pt_sequence": current_shape_pt_sequence,
                    "shape_model_node_id": path_node_id,
                }
                # set these for the stops but leave blank for intermediate nodes
                if path_node_id == row["A"]:
                    bus_node_dict["stop_id"] = row["stop_id"]
                    bus_node_dict["stop_name"] = row["stop_name"]
                    bus_node_dict["stop_sequence"] = row["stop_sequence"]
                elif path_node_id == row["B"]:
                    bus_node_dict["stop_id"] = row["next_stop_id"]
                    bus_node_dict["stop_name"] = row["next_stop_name"]
                    bus_node_dict["stop_sequence"] = last_stop_sequence + 1

                bus_node_sequence.append(bus_node_dict)
                current_shape_pt_sequence += 1
                last_stop_sequence = row["stop_sequence"]

        except nx.NetworkXNoPath as e:
            WranglerLogger.warning(
                f"No path exists for {row['shape_id']} from {row['A']} ({row['stop_name']}) "
                f"to {row['B']} ({row['next_stop_name']})"
            )
            WranglerLogger.warning(e)
            # No path exists - add to no_path_sequence and add simple A->B to bus_node_sequence
            no_path_sequence.append(
                {
                    "shape_id": row["shape_id"],
                    "stop_id": row["stop_id"],
                    "next_stop_id": row["next_stop_id"],
                    "stop_sequence": row["stop_sequence"],
                }
            )

            # Add simple A->B path to bus_node_sequence (in correct order)
            path = [row["A"], row["B"]]
            if current_shape_pt_sequence != 1:
                path = path[1:]  # Skip first node to avoid duplication

            for path_node_id in path:
                bus_node_dict = {
                    "shape_id": row["shape_id"],
                    "route_id": row["route_id"],
                    "route_type": row["route_type"],
                    "trip_id": row["trip_id"],
                    "direction_id": row["direction_id"],
                    "shape_pt_sequence": current_shape_pt_sequence,
                    "shape_model_node_id": path_node_id,
                }
                # Set stop info for the nodes
                if path_node_id == row["A"]:
                    bus_node_dict["stop_id"] = row["stop_id"]
                    bus_node_dict["stop_name"] = row["stop_name"]
                    bus_node_dict["stop_sequence"] = row["stop_sequence"]
                elif path_node_id == row["B"]:
                    bus_node_dict["stop_id"] = row["next_stop_id"]
                    bus_node_dict["stop_name"] = row["next_stop_name"]
                    bus_node_dict["stop_sequence"] = row["stop_sequence"] + 1

                bus_node_sequence.append(bus_node_dict)
                current_shape_pt_sequence += 1
                last_stop_sequence = row["stop_sequence"]

        except nx.NodeNotFound as e:
            WranglerLogger.warning(
                f"Node not found for {row['shape_id']} from {row['A']} to {row['B']}: {e}. "
                f"Adding simple A->B connection."
            )
            # Node not in graph - add to no_path_sequence and add simple A->B to bus_node_sequence
            no_path_sequence.append(
                {
                    "shape_id": row["shape_id"],
                    "stop_id": row["stop_id"],
                    "next_stop_id": row["next_stop_id"],
                    "stop_sequence": row["stop_sequence"],
                }
            )

            # Add simple A->B path to bus_node_sequence (in correct order)
            path = [row["A"], row["B"]]
            if current_shape_pt_sequence != 1:
                path = path[1:]  # Skip first node to avoid duplication

            for path_node_id in path:
                bus_node_dict = {
                    "shape_id": row["shape_id"],
                    "route_id": row["route_id"],
                    "route_type": row["route_type"],
                    "trip_id": row["trip_id"],
                    "direction_id": row["direction_id"],
                    "shape_pt_sequence": current_shape_pt_sequence,
                    "shape_model_node_id": path_node_id,
                }
                # Set stop info for the nodes
                if path_node_id == row["A"]:
                    bus_node_dict["stop_id"] = row["stop_id"]
                    bus_node_dict["stop_name"] = row["stop_name"]
                    bus_node_dict["stop_sequence"] = row["stop_sequence"]
                elif path_node_id == row["B"]:
                    bus_node_dict["stop_id"] = row["next_stop_id"]
                    bus_node_dict["stop_name"] = row["next_stop_name"]
                    bus_node_dict["stop_sequence"] = row["stop_sequence"] + 1

                bus_node_sequence.append(bus_node_dict)
                current_shape_pt_sequence += 1
                last_stop_sequence = row["stop_sequence"]

    bus_node_sequence_df = pd.DataFrame(bus_node_sequence)
    WranglerLogger.debug(f"bus_node_sequence_df:\n{bus_node_sequence_df}")
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace bus_node_sequence_df for {trace_shape_id}:\n"
                f"{bus_node_sequence_df.loc[bus_node_sequence_df.shape_id==trace_shape_id]}"
            )

    if len(no_path_sequence) == 0:
        WranglerLogger.info(f"All bus route shapes mapped to roadway nodes")
    else:
        no_path_sequence_df = pd.DataFrame(no_path_sequence)
        WranglerLogger.debug(f"no_path_sequence_df:\n{no_path_sequence_df}")

        # join with bus_stop_links_gdf
        no_bus_path_gdf = gpd.GeoDataFrame(
            pd.merge(
                left=no_path_sequence_df,
                right=bus_stop_links_gdf,
                how="left",
                on=["stop_id", "next_stop_id", "stop_sequence", "shape_id"],
                validate="one_to_one",
            ),
            crs=bus_stop_links_gdf.crs,
        )
        WranglerLogger.debug(f"no_bus_path_gdf:\n{no_bus_path_gdf}")
        if trace_shape_ids:
            debug_cols = ["A","B","stop_sequence","stop_id","next_stop_id","stop_name","next_stop_name"]
            for trace_shape_id in trace_shape_ids:
                WranglerLogger.debug(
                    f"trace no_bus_path_gdf for {trace_shape_id}:\n"
                    f"{no_bus_path_gdf.loc[no_bus_path_gdf['shape_id'] == trace_shape_id, debug_cols]}"
                )

        # raise an error if requested
        if errors == "raise":
            e = TransitValidationError(
                "Some bus stop sequences failed to find paths. See e.no_bus_path_gdf"
            )
            e.no_bus_path_gdf = no_bus_path_gdf
            raise e
        
        # if we're ignoring, then we need to create roadway network links for these - and mark them
        create_links_for_failed_bus_paths(
            roadway_net=roadway_net,
            no_bus_path_gdf=no_bus_path_gdf,
            local_crs=local_crs,
            crs_units=crs_units,
            trace_shape_ids=trace_shape_ids
        )

    # create bus shapes
    # current shapes columns:
    #  shape_id, shape_pt_lat, shape_pt_lon, shape_pt_sequence, shape_dist_traveled, geometry,
    #  trip_id, direction_id, route_id, agency_id, route_short_name, route_type, agency_name, match_distance_feet,
    #  stop_id, stop_name, stop_sequence, shape_model_node_id

    # we have:
    #  shape_id, route_id, route_type, trip_id, direction_id, shape_pt_sequence, shape_model_node_id, stop_id, stop_name

    # Reorder to be similar
    # bus_node_sequence_df.sort_values(by=["trip_id",])
    bus_node_sequence_df = bus_node_sequence_df[
        [
            "shape_id",
            "shape_pt_sequence",
            "trip_id",
            "direction_id",
            "route_id",
            "route_type",
            "stop_id",
            "stop_name",
            "stop_sequence",
            "shape_model_node_id",
        ]
    ]
    # get agency_id, route_short_name, agency_name from existing feed_tables['shapes']
    bus_node_sequence_df = pd.merge(
        left=bus_node_sequence_df,
        right=feed_tables["shapes"][
            ["shape_id", "agency_id", "route_short_name", "agency_name"]
        ].drop_duplicates(),
        on="shape_id",
        how="left",
        validate="many_to_one",
    )

    # get lon, lat and geometry from roadway_net.nodes
    bus_node_sequence_gdf = gpd.GeoDataFrame(
        pd.merge(
            left=bus_node_sequence_df,
            right=roadway_net.nodes_df[["model_node_id", "X", "Y", "geometry"]].rename(
                columns={"model_node_id": "shape_model_node_id"}
            ),
            how="left",
            on="shape_model_node_id",
            validate="many_to_one",
        ).rename(columns={"X": "shape_pt_lon", "Y": "shape_pt_lat"}),
        crs=roadway_net.nodes_df.crs,
    )
    WranglerLogger.debug(f"Final bus_node_sequence_gdf:\n{bus_node_sequence_gdf}")
    if trace_shape_ids:
        debug_cols = ["shape_pt_sequence","stop_sequence","stop_id","stop_name","shape_model_node_id"]
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace {trace_shape_id} bus_node_sequence_gdf:\n"
                f"{bus_node_sequence_gdf.loc[ bus_node_sequence_gdf['shape_id']==trace_shape_id, debug_cols]}"
            )

    feed_tables["shapes"].to_crs(LAT_LON_CRS, inplace=True)
    # replace bus links in feed_tables['shapes'] with bus_node_sequence_gdf
    feed_tables["shapes"] = pd.concat(
        [
            feed_tables["shapes"].loc[
                ~feed_tables["shapes"]["route_type"].isin([RouteType.BUS, RouteType.TROLLEYBUS])
            ],
            bus_node_sequence_gdf,
        ]
    )

    if trace_shape_ids:
        debug_cols = ["shape_pt_sequence","stop_sequence","stop_id","stop_name","shape_model_node_id"]
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace feed_tables['shapes'] for {trace_shape_id} at the end of create_bus_routes():\n"
                f"{feed_tables['shapes'].loc[feed_tables['shapes']['shape_id'] == trace_shape_id, debug_cols]}"
            )


def add_additional_data_to_stops(
    feed_tables: dict[str, pd.DataFrame],
):
    """Updates feed_tables['stops'] with additional metadata about routes and agencies.

    Aggregates information from stop_times, trips, routes, and agencies tables to add
    comprehensive metadata about which routes and agencies serve each stop.

    Process Steps:

    1. Joins stop_times with trips to get route and shape information
    2. Joins with routes and agencies to get route types and agency names
    3. Groups by stop_id to aggregate all serving routes/agencies
    4. Identifies mixed-traffic stops based on route types
    5. Handles parent stations by checking child stop characteristics

    Modifies feed_tables['stops'] in place, adding columns:

    Route/Agency Information:
    - agency_ids (list of str): All agencies serving this stop
    - agency_names (list of str): Names of agencies serving this stop
    - route_ids (list of str): All routes serving this stop
    - route_names (list of str): Short names of routes serving this stop
    - route_types (list of int): Types of routes serving this stop
    - shape_ids (list of str): All shapes associated with this stop

    Stop Type Flags:
    - is_parent (bool): True if other stops reference this as parent_station

    Args:
        feed_tables: dictionary with required tables:
            - 'stop_times': Links stops to trips
            - 'trips': Links trips to routes and shapes
            - 'routes': Route information including type
            - 'agencies': Agency names
            - 'stops': Table to be updated

    Notes:
        - Parent stations may not appear in trips but are retained if referenced
        - Empty lists are used for stops with no associated routes
        - Handles missing parent_station column gracefully
    """
    # Add information about agencies, routes, directions and shapes to stops
    # Join stop_times with trips and routes
    stop_trips = pd.merge(
        feed_tables["stop_times"][["stop_id", "trip_id"]].drop_duplicates(),
        feed_tables["trips"][["trip_id", "direction_id", "route_id", "shape_id"]],
        on="trip_id",
        how="left",
    )
    WranglerLogger.debug(f"After joining stop_times with trips: {len(stop_trips):,} records")

    # Create stop to route, agency mapping with direction information
    stop_agencies = pd.merge(
        stop_trips,
        feed_tables["routes"][["route_id", "agency_id", "route_short_name", "route_type"]],
        on="route_id",
        how="left",
    )[
        [
            "stop_id",
            "agency_id",
            "route_id",
            "direction_id",
            "shape_id",
            "route_short_name",
            "route_type",
        ]
    ].drop_duplicates()
    WranglerLogger.debug(f"stop_agencies.head():\n{stop_agencies.head()}")

    # pick up agency information
    stop_agencies = pd.merge(
        stop_agencies,
        feed_tables["agencies"][["agency_id", "agency_name"]],
        on="agency_id",
        how="left",
    )
    WranglerLogger.debug(f"stop_agencies.head():\n{stop_agencies.head()}")

    # Group by stop to get all agencies and routes serving each stop
    # Now including route_dir_ids as list of (route_id, direction_id) tuples
    stop_agency_info = (
        stop_agencies.groupby("stop_id")
        .agg(
            {
                "agency_id": lambda x: list(x.dropna().unique()),
                "agency_name": lambda x: list(x.dropna().unique()) if x.notna().any() else [],
                "route_id": lambda x: list(x.dropna().unique()),
                "route_short_name": lambda x: list(x.dropna().unique()),
                "route_type": lambda x: list(x.dropna().unique()),
                "shape_id": lambda x: list(x.dropna().unique()),
            }
        )
        .reset_index()
    )

    stop_agency_info.columns = [
        "stop_id",
        "agency_ids",
        "agency_names",
        "route_ids",
        "route_names",
        "route_types",
        "shape_ids",
    ]

    # columns: stop_id (str), agency_ids (list of str), agency_names (list of str),
    #   route_ids (list of str), route_names (list of str), route_types (list of int),
    #   shape_ids (list of str)
    WranglerLogger.debug(f"stop_agency_info.head():\n{stop_agency_info.head()}")

    # Merge this information back to stops
    feed_tables["stops"] = pd.merge(
        feed_tables["stops"], stop_agency_info, on="stop_id", how="left"
    )

    # Handle parent stations that may not be included in trips
    if "parent_station" in feed_tables["stops"].columns:
        # Find which stops are referenced as parent stations
        child_stops = feed_tables["stops"][
            feed_tables["stops"]["parent_station"].notna()
            & (feed_tables["stops"]["parent_station"] != "")
        ]

        if len(child_stops) > 0:
            # Get unique parent station IDs
            parent_station_ids = child_stops["parent_station"].unique()

            # Mark parent stations
            feed_tables["stops"]["is_parent"] = False
            feed_tables["stops"].loc[
                feed_tables["stops"]["stop_id"].isin(parent_station_ids), "is_parent"
            ] = True

            # Log parent stations
            WranglerLogger.debug(
                f"Found {len(parent_station_ids)} parent stations:\n"
                + f"{feed_tables['stops'].loc[feed_tables['stops']['is_parent'] == True, ['stop_id', 'stop_name']]}"
            )
        else:
            feed_tables["stops"]["is_parent"] = False
    else:
        feed_tables["stops"]["is_parent"] = False

    WranglerLogger.debug(
        f"add_additional_data_to_stops() completed. feed_tables['stops']:\n{feed_tables['stops']}"
    )


def _match_stop_to_shape_points(
    stop_geom: shapely.geometry.Point,
    shape_df: pd.DataFrame,
    prev_matched_shape_idx: int,
    is_circular_route: bool,
    is_first_stop: bool,
    is_last_stop: bool,
    max_distance: float,
    feed_tables: dict[str, pd.DataFrame],
    crs_units: str,
) -> tuple[Optional[int], float]:
    """Find the nearest shape point to a stop within forward search constraints.

    Searches for the nearest shape point to a stop, only looking forward from the
    previous matched position to handle routes that double back on themselves.

    Args:
        stop_geom: Stop location geometry
        shape_df: DataFrame with shape points for this shape_id (includes 'global_idx' column)
        prev_matched_shape_idx: Index in shape_df of previously matched stop (-1 if none)
        is_circular_route: True if first and last stops have same stop_id
        is_first_stop: True if this is the first stop in the sequence
        is_last_stop: True if this is the last stop in the sequence
        max_distance: Maximum distance threshold for matching
        feed_tables: Dict containing 'shapes' DataFrame for checking existing matches
        crs_units: Units for distance ('feet' or 'meters')

    Returns:
        Tuple of (best_shape_idx, best_distance) where:
            - best_shape_idx: Index in shape_df of best match (None if no match found)
            - best_distance: Distance to best match (inf if no match found)

    Called by:
        _align_shape_with_stops()
    """
    # Determine search range based on whether this is a circular route
    search_start = prev_matched_shape_idx + 1
    next_matched_shape_idx = len(shape_df)

    # For circular routes, constrain first and last stops
    if is_circular_route:
        if is_first_stop:
            # First stop must match within first FIRST_LAST_SHAPE_STOP_IDX points
            search_start = 0
            next_matched_shape_idx = min(FIRST_LAST_SHAPE_STOP_IDX, len(shape_df))
        elif is_last_stop:
            # Last stop must match within last FIRST_LAST_SHAPE_STOP_IDX points
            search_start = max(prev_matched_shape_idx + 1, len(shape_df) - FIRST_LAST_SHAPE_STOP_IDX)
            next_matched_shape_idx = len(shape_df)

    # Search for nearest shape point within the forward search range
    # IMPORTANT: Find the LOCAL MINIMUM distance - the closest point in a local neighborhood.
    # This handles two cases:
    # 1. Routes that double back: stops at first local minimum, not a later occurrence
    # 2. Avoids picking too early: finds closest match in neighborhood, not just first within threshold
    best_shape_idx = None
    best_distance = np.inf
    found_candidate = False
    prev_distance = np.inf

    for shape_idx in range(search_start, next_matched_shape_idx):
        if shape_idx >= len(shape_df):
            break

        shape_point = shape_df.iloc[shape_idx]
        distance = stop_geom.distance(shape_point["geometry"])

        if distance < max_distance:
            # This shape point is within threshold - check if it's available
            global_idx = shape_point["global_idx"]
            existing_dist = feed_tables["shapes"].loc[global_idx, f"match_distance_{crs_units}"]

            if distance < existing_dist:
                found_candidate = True

                # Track the closest point we've seen
                if distance < best_distance:
                    best_shape_idx = shape_idx
                    best_distance = distance

                # If distance is increasing, we've passed the local minimum - stop searching
                if found_candidate and distance > prev_distance:
                    break

                prev_distance = distance

    return (best_shape_idx, best_distance if best_shape_idx is not None else np.inf)


def _insert_stop_into_shape(
    stop_row: pd.Series,
    shape_id: str,
    shape_df: pd.DataFrame,
    insert_after_idx: int,
    feed_tables: dict[str, pd.DataFrame],
    local_crs: str,
    crs_units: str,
) -> tuple[int, pd.DataFrame]:
    """Insert an unmatched stop into the shape with proper positioning.

    Calculates the proper shape_pt_sequence for the stop by projecting it onto
    the remaining forward portion of the route, creates a new shape point row,
    inserts it into feed_tables["shapes"], and returns the updated shape_df.

    Args:
        stop_row: Series with stop information (stop_id, stop_sequence, stop_name, geometry)
        shape_id: Shape ID this stop belongs to
        shape_df: DataFrame with shape points for this shape_id (includes 'global_idx' column)
        insert_after_idx: Index in shape_df to insert after (-1 to insert at beginning)
        feed_tables: Dict containing 'shapes' and 'stops' DataFrames
        local_crs: Coordinate reference system for geometries
        crs_units: Units for distance ('feet' or 'meters')

    Returns:
        Tuple of (updated_prev_matched_shape_idx, updated_shape_df) where:
            - updated_prev_matched_shape_idx: Index in new shape_df of inserted stop
            - updated_shape_df: Refreshed shape_df including the inserted stop

    Called by:
        _align_shape_with_stops()
    """
    stop_id = stop_row["stop_id"]
    stop_sequence = stop_row["stop_sequence"]
    stop_geom = stop_row["geometry"]

    # Get stop info from stops table
    stop_info = feed_tables["stops"][feed_tables["stops"]["stop_id"] == stop_id].iloc[0]

    # Calculate shape_pt_sequence for the inserted stop
    # Build LineString from remaining shape points only (after insert_after_idx)
    start_search_idx = max(0, insert_after_idx + 1)
    remaining_coords = [
        (shape_df.iloc[i]["geometry"].x, shape_df.iloc[i]["geometry"].y)
        for i in range(start_search_idx, len(shape_df))
    ]

    if len(remaining_coords) >= 2 and start_search_idx < len(shape_df):
        # Project stop onto remaining portion of route
        remaining_line = shapely.geometry.LineString(remaining_coords)
        stop_distance_on_remaining = remaining_line.project(stop_geom)

        # Find which segment in the remaining portion
        cumulative_dist = 0.0
        new_seq = None
        for i in range(start_search_idx, len(shape_df) - 1):
            pt1 = shape_df.iloc[i]["geometry"]
            pt2 = shape_df.iloc[i + 1]["geometry"]
            segment_len = pt1.distance(pt2)

            if cumulative_dist <= stop_distance_on_remaining <= cumulative_dist + segment_len:
                # Interpolate shape_pt_sequence
                curr_seq = shape_df.iloc[i]["shape_pt_sequence"]
                next_seq = shape_df.iloc[i + 1]["shape_pt_sequence"]
                if segment_len > 0:
                    ratio = (stop_distance_on_remaining - cumulative_dist) / segment_len
                    new_seq = curr_seq + ratio * (next_seq - curr_seq)
                else:
                    new_seq = (curr_seq + next_seq) / 2.0
                break
            cumulative_dist += segment_len

        # Fallback: place between adjacent shape points
        if new_seq is None:
            if insert_after_idx >= 0 and insert_after_idx < len(shape_df) - 1:
                curr_seq = shape_df.iloc[insert_after_idx]["shape_pt_sequence"]
                next_seq = shape_df.iloc[insert_after_idx + 1]["shape_pt_sequence"]
                new_seq = (curr_seq + next_seq) / 2.0
            elif insert_after_idx >= 0:
                new_seq = shape_df.iloc[insert_after_idx]["shape_pt_sequence"] + 1.0
            else:
                new_seq = 0.5
    else:
        # Not enough remaining points, use simple interpolation
        if insert_after_idx >= 0 and insert_after_idx < len(shape_df) - 1:
            curr_seq = shape_df.iloc[insert_after_idx]["shape_pt_sequence"]
            next_seq = shape_df.iloc[insert_after_idx + 1]["shape_pt_sequence"]
            new_seq = (curr_seq + next_seq) / 2.0
        elif insert_after_idx >= 0 and insert_after_idx < len(shape_df):
            new_seq = shape_df.iloc[insert_after_idx]["shape_pt_sequence"] + 1.0
        else:
            new_seq = 0.5

    # Ensure new_seq doesn't conflict with existing shape points
    # Check if any remaining shape points have this exact sequence
    remaining_sequences = shape_df.iloc[start_search_idx:]["shape_pt_sequence"].values
    if new_seq in remaining_sequences:
        # Find the index of the conflicting sequence
        conflict_idx = start_search_idx + list(remaining_sequences).index(new_seq)
        # Place this stop slightly before the conflicting point
        if insert_after_idx >= 0 and insert_after_idx < len(shape_df):
            prev_seq = shape_df.iloc[insert_after_idx]["shape_pt_sequence"]
            new_seq = (prev_seq + new_seq) / 2.0
        else:
            new_seq = new_seq - 0.5

    # Create new row for the inserted stop
    # Use template from insert_after_idx position for route metadata
    if insert_after_idx >= 0 and insert_after_idx < len(shape_df):
        template_global_idx = shape_df.iloc[insert_after_idx]["global_idx"]
        template_row = feed_tables["shapes"].loc[template_global_idx].to_dict()
    else:
        template_global_idx = shape_df.iloc[0]["global_idx"]
        template_row = feed_tables["shapes"].loc[template_global_idx].to_dict()

    new_row = template_row.copy()
    new_row.update(
        {
            "shape_id": shape_id,
            "shape_pt_sequence": new_seq,
            "shape_pt_lat": stop_info["stop_lat"],
            "shape_pt_lon": stop_info["stop_lon"],
            "geometry": stop_info["geometry"],
            "stop_id": stop_id,
            "stop_name": stop_row["stop_name"],
            "stop_sequence": stop_sequence,
            f"match_distance_{crs_units}": 0.0,
            "shape_dist_traveled": None,
        }
    )
    # Copy poor_match if available (will be present for bus stops after match_bus_stops_to_roadway_nodes)
    if "poor_match" in feed_tables["stops"].columns:
        new_row["poor_match"] = stop_info.get("poor_match", False)

    # Insert into feed_tables["shapes"]
    # Create new row as GeoDataFrame
    new_row_gdf = gpd.GeoDataFrame([new_row], geometry="geometry", crs=local_crs)

    # Suppress FutureWarning about concatenation with empty/NA entries
    # This is a pandas deprecation warning that we can't easily avoid
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*DataFrame concatenation.*"
        )
        feed_tables["shapes"] = pd.concat(
            [feed_tables["shapes"], new_row_gdf], ignore_index=True
        )

    # Refresh shape_df to reflect the insertion
    shape_mask = feed_tables["shapes"]["shape_id"] == shape_id
    shape_df = feed_tables["shapes"].loc[shape_mask].copy().reset_index(drop=False)
    shape_df.rename(columns={"index": "global_idx"}, inplace=True)
    # Sort by shape_pt_sequence to maintain order
    shape_df.sort_values(by="shape_pt_sequence", inplace=True, ignore_index=True)

    # Update prev_matched_shape_idx to the position of the newly inserted stop
    # Find it in the refreshed shape_df by both stop_id and stop_sequence
    new_stop_mask = (shape_df["stop_id"] == stop_id) & (
        shape_df["stop_sequence"] == stop_sequence
    )
    new_stop_positions = shape_df[new_stop_mask].index.tolist()
    if len(new_stop_positions) > 0:
        updated_prev_matched_shape_idx = new_stop_positions[-1]  # Use the last occurrence
    else:
        # Fallback: find by shape_pt_sequence
        closest_idx = shape_df.iloc[
            (shape_df["shape_pt_sequence"] - new_seq).abs().argsort()[:1]
        ].index[0]
        updated_prev_matched_shape_idx = closest_idx

    return (updated_prev_matched_shape_idx, shape_df)


def _align_shape_with_stops(
    shape_id: str,
    feed_tables: dict[str, pd.DataFrame],
    local_crs: str,
    crs_units: str,
    max_distance: float,
    trace_shape_ids: Optional[list[str]],
    stoptime_debug_cols: list[str],
    shape_debug_cols: list[str],
) -> tuple[int, int, list[dict]]:
    """Align stops with shape points for a single shape_id.

    For each stop in the shape's trip, either matches it to an existing shape point
    or inserts a new shape point at the stop location. Processes stops in sequence
    order, always searching forward to handle routes that double back.

    Args:
        shape_id: Shape ID to process
        feed_tables: Dict with 'shapes', 'stops', 'stop_times' DataFrames
        local_crs: Coordinate reference system for geometries
        crs_units: Units for distance ('feet' or 'meters')
        max_distance: Maximum distance threshold for matching stops
        trace_shape_ids: Optional list of shape IDs for debug logging
        stoptime_debug_cols: Column names for debug output of stop_times
        shape_debug_cols: Column names for debug output of shapes

    Returns:
        Tuple of (matched_count, inserted_count, debug_features) where:
            - matched_count: Number of stops matched to existing shape points
            - inserted_count: Number of stops inserted as new shape points
            - debug_features: List of dicts for debug GeoJSON output

    Called by:
        add_additional_data_to_shapes()
    """
    # Get stops for this shape, ordered by stop_sequence
    stoptimes_for_shape_df = (
        feed_tables["stop_times"]
        .loc[feed_tables["stop_times"]["shape_id"] == shape_id]
        .copy()
        .reset_index(drop=True)
    )

    # Get shape points for this shape, ordered by shape_pt_sequence
    shape_mask = feed_tables["shapes"]["shape_id"] == shape_id
    shape_df = feed_tables["shapes"].loc[shape_mask].copy().reset_index(drop=False)
    shape_df.rename(columns={"index": "global_idx"}, inplace=True)

    if len(stoptimes_for_shape_df) == 0:
        return (0, 0, [])

    if trace_shape_ids and shape_id in trace_shape_ids:
        WranglerLogger.debug(
            f"trace {shape_id}: Processing {len(stoptimes_for_shape_df)} stops, "
            f"{len(shape_df)} shape points"
        )
        WranglerLogger.debug(
            f"trace {shape_id}: stoptimes_for_shape_df:\n"
            f"{stoptimes_for_shape_df[stoptime_debug_cols]}"
        )

    # Detect circular/loop routes (first and last stops have same stop_id)
    is_circular_route = False
    if len(stoptimes_for_shape_df) >= 2:
        first_stop_id = stoptimes_for_shape_df.iloc[0]["stop_id"]
        last_stop_id = stoptimes_for_shape_df.iloc[-1]["stop_id"]
        is_circular_route = first_stop_id == last_stop_id

        if is_circular_route and trace_shape_ids and shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace {shape_id}: Detected circular route (first and last stop both {first_stop_id})"
            )

    # Track which shape points have been matched
    prev_matched_shape_idx = -1  # Start before first shape point
    matched_count = 0
    inserted_count = 0

    # Iterate through stops in stop_sequence order
    # Insert stops immediately when they don't match (rather than batching)
    # so that each insertion updates prev_matched_shape_idx for the next stop
    for stop_idx, stop_row in stoptimes_for_shape_df.iterrows():
        stop_id = stop_row["stop_id"]
        stop_sequence = stop_row["stop_sequence"]
        stop_geom = stop_row["geometry"]

        # Determine if this is first or last stop
        is_first_stop = stop_idx == 0
        is_last_stop = stop_idx == len(stoptimes_for_shape_df) - 1

        if trace_shape_ids and shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace {shape_id}: Matching {stop_sequence=} {stop_id=}, "
                f"prev_matched_shape_idx={prev_matched_shape_idx}"
            )

        # Try to match stop to an existing shape point
        best_shape_idx, best_distance = _match_stop_to_shape_points(
            stop_geom=stop_geom,
            shape_df=shape_df,
            prev_matched_shape_idx=prev_matched_shape_idx,
            is_circular_route=is_circular_route,
            is_first_stop=is_first_stop,
            is_last_stop=is_last_stop,
            max_distance=max_distance,
            feed_tables=feed_tables,
            crs_units=crs_units,
        )

        if best_shape_idx is not None:
            # Found a match - update the shape point with stop information
            shape_point = shape_df.iloc[best_shape_idx]
            global_idx = shape_point["global_idx"]

            if trace_shape_ids and shape_id in trace_shape_ids:
                WranglerLogger.debug(
                    f"trace {shape_id}: Matched {stop_sequence=} {stop_id=} to "
                    f"shape_pt_sequence={shape_point['shape_pt_sequence']}, distance={best_distance:.2f}"
                )

            # Get stop info from stops table
            stop_info = feed_tables["stops"][feed_tables["stops"]["stop_id"] == stop_id].iloc[0]

            # Update shape point
            feed_tables["shapes"].loc[global_idx, f"match_distance_{crs_units}"] = best_distance
            feed_tables["shapes"].loc[global_idx, "stop_id"] = stop_id
            feed_tables["shapes"].loc[global_idx, "stop_name"] = stop_row["stop_name"]
            feed_tables["shapes"].loc[global_idx, "stop_sequence"] = stop_sequence
            feed_tables["shapes"].loc[global_idx, "shape_pt_lon"] = stop_info["stop_lon"]
            feed_tables["shapes"].loc[global_idx, "shape_pt_lat"] = stop_info["stop_lat"]
            feed_tables["shapes"].loc[global_idx, "geometry"] = stop_info["geometry"]
            # Copy poor_match if available (will be present for bus stops after match_bus_stops_to_roadway_nodes)
            if "poor_match" in feed_tables["stops"].columns:
                feed_tables["shapes"].loc[global_idx, "poor_match"] = stop_info.get("poor_match", False)

            prev_matched_shape_idx = best_shape_idx
            matched_count += 1

        else:
            # No nearby shape point found - insert stop immediately
            insert_after_idx = prev_matched_shape_idx

            if trace_shape_ids and shape_id in trace_shape_ids:
                WranglerLogger.debug(
                    f"trace {shape_id}: No match for {stop_sequence=} {stop_id=}, "
                    f"inserting after shape_pt_idx {insert_after_idx}"
                )

            # Insert the stop
            prev_matched_shape_idx, shape_df = _insert_stop_into_shape(
                stop_row=stop_row,
                shape_id=shape_id,
                shape_df=shape_df,
                insert_after_idx=insert_after_idx,
                feed_tables=feed_tables,
                local_crs=local_crs,
                crs_units=crs_units,
            )

            inserted_count += 1

    # Sort feed_tables["shapes"] by shape_id and shape_pt_sequence to ensure correct order
    # Use stop_sequence as a secondary key to handle cases where multiple stops
    # have the same shape_pt_sequence (e.g., when an inserted stop ends up at same position as matched stop)
    # This is needed because we inserted stops inline
    feed_tables["shapes"].sort_values(
        by=["shape_id", "shape_pt_sequence", "stop_sequence"],
        inplace=True,
        ignore_index=True,
        na_position='first'  # Put shape points without stops first
    )

    # Renumber shape_pt_sequence if there are non-integer values or duplicates
    shape_mask = feed_tables["shapes"]["shape_id"] == shape_id
    shape_sequences = feed_tables["shapes"].loc[shape_mask, "shape_pt_sequence"]

    # Check if renumbering is needed (non-integers or duplicates)
    has_non_integers = not all(shape_sequences == shape_sequences.astype(int))
    has_duplicates = shape_sequences.duplicated().any()

    if has_non_integers or has_duplicates:
        shape_indices = feed_tables["shapes"][shape_mask].index
        feed_tables["shapes"].loc[shape_indices, "shape_pt_sequence"] = range(len(shape_indices))
        WranglerLogger.info(
            f"Renumbered shape_pt_sequence for {shape_id} to sequential integers"
        )

    # Verify all stops are matched
    shape_mask = feed_tables["shapes"]["shape_id"] == shape_id
    shape_stops_df = feed_tables["shapes"].loc[
        shape_mask & pd.notnull(feed_tables["shapes"]["stop_id"])
    ].copy()

    # Check if stop_sequence is monotonically increasing
    if not shape_stops_df["stop_sequence"].is_monotonic_increasing:
        WranglerLogger.warning(f"stop_sequence isn't monotonic_increasing for shape {shape_id}")
        WranglerLogger.debug(f"shape_stops_df:\n{shape_stops_df[shape_debug_cols]}")

    # Debug logging for traced shapes
    if trace_shape_ids and shape_id in trace_shape_ids:
        WranglerLogger.debug(
            f"trace {shape_id}: Final shape_stops_df:\n" f"{shape_stops_df[shape_debug_cols]}"
        )

    # Collect debug features for all shapes
    debug_features = []
    try:
        # Get all shape points for this shape
        debug_shape_df = feed_tables["shapes"].loc[shape_mask].copy()
        debug_shape_df = debug_shape_df.sort_values("shape_pt_sequence")

        # Create LineString from shape points
        shape_coords = [(pt.x, pt.y) for pt in debug_shape_df.geometry]
        shape_line = shapely.geometry.LineString(shape_coords)

        # Add the shape line
        debug_features.append(
            {
                "geometry": shape_line,
                "shape_id": shape_id,
                "feature_type": "shape_line",
                "stop_type": None,
                "num_points": len(shape_coords),
                "shape_pt_sequence": None,
                "shape_dist_traveled": None,
                "stop_id": None,
                "stop_name": None,
                "stop_sequence": None,
                "match_distance": None,
            }
        )

        # Add all shape points with their attributes
        for idx, row in debug_shape_df.iterrows():
            is_stop = pd.notnull(row.get("stop_id"))

            # Determine if stop was inserted (match_distance = 0) or matched (match_distance > 0)
            stop_type = None
            if is_stop:
                match_dist = row.get(f"match_distance_{crs_units}")
                if match_dist == 0.0:
                    stop_type = "inserted"
                elif match_dist > 0.0:
                    stop_type = "matched"

            debug_features.append(
                {
                    "geometry": row["geometry"],
                    "shape_id": shape_id,
                    "feature_type": "stop" if is_stop else "shape_point",
                    "stop_type": stop_type,  # "inserted", "matched", or None
                    "num_points": None,
                    "shape_pt_sequence": row["shape_pt_sequence"],
                    "shape_dist_traveled": row.get("shape_dist_traveled"),
                    "stop_id": row.get("stop_id") if is_stop else None,
                    "stop_name": row.get("stop_name") if is_stop else None,
                    "stop_sequence": row.get("stop_sequence") if is_stop else None,
                    "match_distance": row.get(f"match_distance_{crs_units}") if is_stop else None,
                    "poor_match": row.get("poor_match") if is_stop else None,
                }
            )

    except Exception as e:
        WranglerLogger.warning(f"Failed to collect debug features for {shape_id}: {e}")

    WranglerLogger.debug(f"Completed adding stop information to shape_id {shape_id}")

    return (matched_count, inserted_count, debug_features)


def _write_debug_shapes(debug_features: list[dict], local_crs: str) -> None:
    """Write consolidated debug GeoJSON for all shapes with their stops.

    Args:
        debug_features: List of feature dicts with geometry and attributes
        local_crs: Coordinate reference system for the GeoDataFrame

    Called by:
        add_additional_data_to_shapes()
    """
    if len(debug_features) == 0:
        return

    try:
        debug_gdf = gpd.GeoDataFrame(debug_features, crs=local_crs)
        output_file = "debug_shapes.geojson"
        debug_gdf.to_file(output_file, driver="GeoJSON")
        num_shapes = len(debug_gdf[debug_gdf["feature_type"] == "shape_line"])
        WranglerLogger.info(
            f"Wrote debug output for {num_shapes} shapes "
            f"({len(debug_features)} features) to {output_file}"
        )
    except Exception as e:
        WranglerLogger.warning(f"Failed to write consolidated debug output: {e}")


def add_additional_data_to_shapes(  # noqa: PLR0915
    feed_tables: dict[str, pd.DataFrame],
    local_crs: str,
    crs_units: str,
    trace_shape_ids: Optional[list[str]] = None,
):
    """Updates feed_tables['shapes'] with route/trip metadata and snaps shape points to stops.

    Enriches shape points with information from trips, routes, and agencies tables,
    then matches shape points to nearby stops and updates their locations. Processes
    stops in sequence order, always searching forward to handle routes that double back.

    Process Steps:

    1. Converts shapes to GeoDataFrame if needed (using shape_pt_lon/lat)
    2. Joins with trips, routes, and agencies to add metadata
    3. Projects to local CRS for distance calculations
    4. For each shape, calls _align_shape_with_stops() to:
       - Match stops to existing shape points (via _match_stop_to_shape_points())
       - Insert unmatched stops as new shape points (via _insert_stop_into_shape())
       - Verify stop_sequence is monotonically increasing
    5. Writes debug GeoJSON output (via _write_debug_shapes())

    Assumes create_feed_frequencies() has already run, so each shape corresponds
    to one consolidated trip_id.

    Modifies feed_tables in place:

    feed_tables['shapes'] - Adds/modifies columns:
        Route/Trip Metadata:
        - trip_id (str): Associated trip ID
        - direction_id (int): Direction of travel (0 or 1)
        - route_id (str): Route identifier
        - agency_id (str): Agency identifier
        - agency_name (str): Agency name
        - route_short_name (str): Route short name
        - route_type (int): GTFS route type

        Stop Matching (for matched points only):
        - stop_id (str): Matched stop ID
        - stop_name (str): Matched stop name
        - stop_sequence (int): Order of stop in trip
        - match_distance_{crs_units} (float): Distance from original to stop location
        - shape_pt_lon, shape_pt_lat: Updated to stop coordinates
        - geometry: Updated to stop location

    feed_tables['stop_times'] - Converted to GeoDataFrame with:
        - geometry: Stop location added from stops table

    Args:
        feed_tables: dictionary with required tables:
            - 'shapes': Shape points to update
            - 'trips': Trip information
            - 'routes': Route information
            - 'agencies': Agency information
            - 'stops': Stop locations
            - 'stop_times': Stop sequences
        local_crs: Coordinate reference system for projections
        crs_units: Distance units ('feet' or 'meters')
        trace_shape_ids: Optional shape IDs for debug logging

    Helper Functions:
        _align_shape_with_stops(): Process all stops for one shape
        _match_stop_to_shape_points(): Find nearest shape point for a stop
        _insert_stop_into_shape(): Insert unmatched stop as new shape point
        _write_debug_shapes(): Write debug GeoJSON output

    Notes:
        - Searches forward from previous matched position to handle routes that double back
        - Inserts stops immediately when unmatched (not batched) for accurate positioning
        - Handles circular routes (first/last stop same) with constrained search ranges
        - Writes debug_shapes.geojson with all shapes, stops, and shape points
    """
    # Step 1: Convert shapes to GeoDataFrame if needed and add geometry from lat/lon coordinates
    # Create GeoDataFrame from shape points if not already one
    if not isinstance(feed_tables["shapes"], gpd.GeoDataFrame):
        shape_geometry = [
            shapely.geometry.Point(lon, lat)
            for lon, lat in zip(
                feed_tables["shapes"]["shape_pt_lon"], feed_tables["shapes"]["shape_pt_lat"]
            )
        ]
        feed_tables["shapes"] = gpd.GeoDataFrame(
            feed_tables["shapes"], geometry=shape_geometry, crs=LAT_LON_CRS
        )
        WranglerLogger.debug(f"Converted feed_tables['shapes'] to GeoDataFrame")
    else:
        WranglerLogger.debug(f"feed_tables['shapes'].crs={feed_tables['shapes'].crs}")

    # Step 2: Add agency, route, and trip information to shapes by joining with trips and routes tables
    # Get unique shape_ids to trips mapping
    trip_shapes_df = feed_tables["trips"][
        ["shape_id", "trip_id", "direction_id", "route_id"]
    ].drop_duplicates()
    WranglerLogger.debug(f"trip_shapes_df\n{trip_shapes_df}")
    # assumes trip_id and shape_id are equivalent due to add_additional_data_to_shapes
    assert feed_tables["trips"]["trip_id"].nunique() == feed_tables["trips"]["shape_id"].nunique()

    # Get route information: Join routes with agencies to get agency names
    routes_with_agency_df = pd.merge(
        feed_tables["routes"][["route_id", "agency_id", "route_short_name", "route_type"]],
        feed_tables["agencies"][["agency_id", "agency_name"]],
        on="agency_id",
        how="left",
    )
    # Add agency information to trips_shapes_df
    trip_shapes_df = pd.merge(
        left=trip_shapes_df, right=routes_with_agency_df, how="left", on="route_id"
    )
    # Add this data to shapes table
    feed_tables["shapes"] = pd.merge(
        feed_tables["shapes"], trip_shapes_df, on="shape_id", how="left", validate="many_to_one"
    )

    WranglerLogger.debug(f"Added agency and route information to shapes table")
    WranglerLogger.debug(f"feed_tables['shapes'].head():\n{feed_tables['shapes'].head()}")
    # shapes columns: shape_id, shape_pt_lat, shape_pt_lon, shape_pt_sequence, shape_dist_traveled, geometry
    #                 trip_id, direction_id, route_id, agency_id, route_short_name, route_type, agency_name

    # Match stops to shape points using segment iteration
    WranglerLogger.info(f"Matching stops to shape points using segment iteration")

    # Project both GeoDataFrames to specified CRS for distance calculations
    feed_tables["shapes"].to_crs(local_crs, inplace=True)
    feed_tables["stops"].to_crs(local_crs, inplace=True)

    # Initialize shape columns for stop information
    feed_tables["shapes"][f"match_distance_{crs_units}"] = np.inf
    feed_tables["shapes"]["stop_id"] = None
    feed_tables["shapes"]["stop_name"] = ""
    feed_tables["shapes"]["stop_sequence"] = None
    # these are the most useful columns for debugging
    stoptime_debug_cols = ['stop_sequence','stop_id','stop_name']
    shape_debug_cols = ['shape_id','shape_pt_sequence','shape_dist_traveled','stop_sequence','stop_id','stop_name',f'match_distance_{crs_units}']

    # Add stop geometry to stop_times and convert it a GeoDataFrame
    WranglerLogger.debug(f"Before merge, {len(feed_tables['stop_times'])=:,}")
    feed_tables["stop_times"] = gpd.GeoDataFrame(
        pd.merge(
            left=feed_tables["stop_times"],
            right=feed_tables["stops"][["stop_id", "stop_name", "geometry"]],
            how="left",
            on="stop_id",
            validate="many_to_one",
        ),
        geometry="geometry",
        crs=feed_tables["stops"].crs,
    )

    WranglerLogger.debug(
        f"feed_tables['trips'] type={type(feed_tables['trips'])}:\n{feed_tables['trips']}"
    )
    WranglerLogger.debug(
        f"feed_tables['stops'] type={type(feed_tables['stops'])}:\n{feed_tables['stops']}"
    )
    WranglerLogger.debug(
        f"feed_tables['stop_times'] type={type(feed_tables['stop_times'])}:\n{feed_tables['stop_times']}"
    )

    # Sort tables for processing
    feed_tables["stop_times"].sort_values(by=["trip_id", "stop_sequence"], inplace=True, ignore_index=True)
    feed_tables["shapes"].sort_values(by=["shape_id","shape_pt_sequence"], inplace=True, ignore_index=True)

    # Process each shape_id
    unique_shape_ids = sorted(feed_tables["shapes"]["shape_id"].unique())
    WranglerLogger.info(f"Finding stops for {len(unique_shape_ids):,} unique shape_ids")

    matched_count = 0
    inserted_count = 0
    max_distance = MAX_DISTANCE_STOP[crs_units]

    # Collect debug features for all shapes
    debug_features = []

    for shape_id in unique_shape_ids:
        # Process this shape: match and/or insert all stops
        shape_matched, shape_inserted, shape_debug_features = _align_shape_with_stops(
            shape_id=shape_id,
            feed_tables=feed_tables,
            local_crs=local_crs,
            crs_units=crs_units,
            max_distance=max_distance,
            trace_shape_ids=trace_shape_ids,
            stoptime_debug_cols=stoptime_debug_cols,
            shape_debug_cols=shape_debug_cols,
        )

        matched_count += shape_matched
        inserted_count += shape_inserted
        debug_features.extend(shape_debug_features)

    WranglerLogger.info(
        f"Finished adding stop information to shapes: "
        f"{matched_count} stops matched, {inserted_count} stops inserted"
    )

    # Write consolidated debug GeoJSON
    _write_debug_shapes(debug_features, local_crs)


def add_stations_and_links_to_roadway_network(  # noqa: PLR0912, PLR0915
    feed_tables: dict[str, pd.DataFrame],
    roadway_net: RoadwayNetwork,
    local_crs: str,
    crs_units: str,
    trace_shape_ids: Optional[list[str]] = None,
) -> tuple[dict[str, int], gpd.GeoDataFrame]:
    """Add transit station nodes and dedicated transit links to the roadway network.

    Creates new roadway nodes for transit stations and adds dedicated transit links
    between stations for fixed-guideway transit (rail, subway, ferry, etc.). Bus stops
    use existing roadway nodes from match_bus_stops_to_roadway_nodes().

    Process Steps:
    1. Creates stop link pairs from consecutive stops in stop_times
    2. Aggregates intermediate shape points between stops into multi-point lines
    3. Filters links to STATION_ROUTE_TYPES for network addition
    4. Creates new roadway nodes for stations not already in network
    5. Creates dedicated transit links with appropriate access restrictions
    6. Updates feed_tables['stops'] with model_node_id for all stops
    7. Updates feed_tables['shapes'] with shape_model_node_id for all stations
    8. Returns bus stop links separately (not added to network)

    Modifies in place:

    roadway_net - Adds:
        - New nodes for transit stations with model_node_id
        - New links between stations with:
            - rail_only=True for rail types
            - ferry_only=True for ferry types
            - drive/bike/walk/truck_access=False
            - Geometry following shape points if available

    feed_tables['stops'] - Adds/updates:
        - model_node_id (int): Roadway node ID for the stop
        - Updates existing bus stop model_node_ids
        - Adds new station model_node_ids

    feed_tables['shapes'] - Adds/updates:
        - shape_model_node_id (int): Roadway node ID for the shape point

    Args:
        feed_tables: dictionary with required tables:
            - 'stops': Stop information with geometry
            - 'stop_times': Stop sequences for trips
            - 'shapes': Shape points between stops
            - 'routes': Route types
        roadway_net: RoadwayNetwork to modify with new nodes/links
        local_crs: Coordinate reference system for projections
        crs_units: Distance units ('feet' or 'meters')
        trace_shape_ids: Optional shape IDs for debug logging

    Returns:
        tuple[dict[str,int], gpd.GeoDataFrame]:
            - dictionary mapping new station stop_ids to model_node_ids
            - GeoDataFrame of bus stop links (not added to network) with columns:
                shape_id, stop_sequence, stop_id, stop_name, next_stop_id,
                next_stop_name, A, B, geometry

    Notes:
        - Stations are new nodes; bus stops use existing road nodes
        - Self-loops (stop appearing twice consecutively) are filtered out
        - Links follow actual shape geometry when available
        - Parent stations without trips are handled correctly
    """
    WranglerLogger.info(f"Adding transit stations and station-based links to the roadway network")
    WranglerLogger.debug(
        f"feed_tables['shapes'] type={type(feed_tables['shapes'])}:\n{feed_tables['shapes']}"
    )
    WranglerLogger.debug(
        f"feed_tables['stops'] type={type(feed_tables['stops'])}:\n{feed_tables['stops']}"
    )
    WranglerLogger.debug(
        f"feed_tables['stop_times'] type={type(feed_tables['stop_times'])}:\n{feed_tables['stop_times']}"
    )

    # Add route_type to stop_times
    if "route_type" not in feed_tables["stop_times"].columns:
        feed_tables["stop_times"] = pd.merge(
            feed_tables["stop_times"],
            feed_tables["routes"][["route_id", "route_type"]],
            how="left",
            on="route_id",
            validate="many_to_one",
        )
    # keep trace_stop_id_set
    trace_stop_id_set = None
    if trace_shape_ids:
        trace_stop_id_set = set(feed_tables["stop_times"].loc[ 
            feed_tables["stop_times"]["shape_id"].isin(trace_shape_ids), "stop_id"
        ].to_list())
        WranglerLogger.debug(f"trace_stop_id_set:{trace_stop_id_set}")

    # Prepare new link list first
    # For all consecutive stop_ids in feed_table['stop_times'], create consecutive node pairs for each shape
    stop_links_df = feed_tables["stop_times"][
        [
            "route_type",
            "route_id",
            "direction_id",
            "shape_id",
            "trip_id",
            "stop_sequence",
            "stop_id",
            "stop_name",
            "geometry",
        ]
    ].copy()
    stop_links_df.rename(columns={"geometry": "stop_geometry"}, inplace=True)
    stop_links_df.sort_values(by=["trip_id", "stop_sequence"], inplace=True)

    stop_links_df["next_stop_id"] = stop_links_df.groupby("trip_id")["stop_id"].shift(-1)
    stop_links_df["next_stop_name"] = stop_links_df.groupby("trip_id")["stop_name"].shift(-1)
    stop_links_df["next_stop_geometry"] = stop_links_df.groupby("trip_id")["stop_geometry"].shift(
        -1
    )

    # Filter to only rows that have a next node (excludes last point of each shape)
    # and filter out self-loops where the stop occurs twice in a row
    has_next = stop_links_df["next_stop_id"].notna()
    is_self_loop = stop_links_df["next_stop_id"] == stop_links_df["stop_id"]

    num_self_loops = (has_next & is_self_loop).sum()
    if num_self_loops > 0:
        WranglerLogger.debug(
            f"Filtering out {num_self_loops:,} self-loop segments where stop_id == next_stop_id"
        )

    stop_links_df = stop_links_df[has_next & ~is_self_loop]
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace stop_links_df for {trace_shape_id}:\n"
                f"{stop_links_df.loc[stop_links_df.shape_id == trace_shape_id]}"
            )

    WranglerLogger.debug(f"stop_links_df.dtypes:\n{stop_links_df.dtypes}")
    # route_type            category
    # route_id                object
    # direction_id          category
    # shape_id                object
    # trip_id                 object
    # stop_sequence            int64
    # stop_id                 object
    # stop_name               object
    # stop_geometry         geometry
    # next_stop_id            object
    # next_stop_name          object
    # next_stop_geometry    geometry
    # dtype: object

    # feed_tables['shapes'] is a GeoDataFrame of points, 3 the columns 'shape_id', 'stop_id' and 'stop_sequence'
    # Match these sequences with the stop_id/next_stop_id in stop_links_gdf based on shape_id and add intermediate points to the shape
    shape_links_df = feed_tables["shapes"][
        ["shape_id", "geometry", "stop_sequence", "stop_id"]
    ].copy()
    # set the stop_sequence to 1 for the first row of each shape_id if it's not set
    shape_links_df.loc[
        (~shape_links_df["shape_id"].duplicated()) & (shape_links_df["stop_sequence"].isna()),
        "stop_sequence",
    ] = -1
    shape_links_df.loc[
        (~shape_links_df["shape_id"].duplicated()) & (shape_links_df["stop_id"].isna()), "stop_id"
    ] = -1
    # fill forward
    # Suppress downcasting warning for ffill
    with pd.option_context("future.no_silent_downcasting", True):
        shape_links_df["shape_stop_sequence"] = shape_links_df["stop_sequence"].ffill()
        shape_links_df["shape_stop_id"] = shape_links_df["stop_id"].ffill()
    # drop the first one - that's already covered by the stop point
    shape_links_df.loc[shape_links_df["stop_id"].notna(), "shape_stop_sequence"] = np.nan
    shape_links_df.loc[shape_links_df["stop_id"].notna(), "shape_stop_id"] = np.nan

    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace shape_links_df for {trace_shape_id} after ffill():\n"
                f"{shape_links_df.loc[shape_links_df.shape_id == trace_shape_id]}"
            )
    # Example: this shape's first stop is stop_sequence=2, the shape points at the beginning are pre-trip points
    #                 shape_id                         geometry stop_sequence stop_id  shape_stop_sequence shape_stop_id
    # 909222  SF:9717:20230930  POINT (6013196.693 2109556.951)            -1      -1                  NaN           NaN
    # 909223  SF:9717:20230930  POINT (6013227.638 2109614.595)          None    None                -1.00            -1
    # 909224  SF:9717:20230930  POINT (6013265.791 2109685.568)          None    None                -1.00            -1
    # 909225  SF:9717:20230930  POINT (6013316.874 2109752.637)          None    None                -1.00            -1
    # 909226  SF:9717:20230930  POINT (6013604.053 2110029.431)          None    None                -1.00            -1
    # 909227  SF:9717:20230930  POINT (6013637.289 2110057.529)          None    None                -1.00            -1
    # 909228  SF:9717:20230930  POINT (6014283.122 2110658.483)             2   15240                  NaN           NaN
    # 909229  SF:9717:20230930  POINT (6014293.455 2110683.403)          None    None                 2.00         15240
    # 909230  SF:9717:20230930  POINT (6014940.906 2111322.944)          None    None                 2.00         15240
    # 909231  SF:9717:20230930  POINT (6015538.148 2111853.892)             3   15237                  NaN           NaN
    # 909232  SF:9717:20230930  POINT (6015603.806 2111941.794)          None    None                 3.00         15237
    # 909233  SF:9717:20230930  POINT (6015814.592 2112130.926)          None    None                 3.00         15237
    # 909234  SF:9717:20230930  POINT (6015897.484 2112242.153)          None    None                 3.00         15237
    # 909235  SF:9717:20230930   POINT (6015932.62 2112307.364)          None    None                 3.00         15237
    # 909236  SF:9717:20230930  POINT (6015955.806 2112367.716)          None    None                 3.00         15237
    # 909237  SF:9717:20230930  POINT (6015971.693 2112438.778)          None    None                 3.00         15237
    # 909238  SF:9717:20230930  POINT (6015985.888 2112526.263)          None    None                 3.00         15237
    # 909239  SF:9717:20230930   POINT (6015989.04 2112639.464)          None    None                 3.00         15237
    # 909240  SF:9717:20230930  POINT (6016045.522 2113160.945)          None    None                 3.00         15237
    # 909241  SF:9717:20230930  POINT (6016108.599 2113665.541)             4   17145                  NaN           NaN

    # aggregate and convert to list
    shape_links_agg_df = (
        shape_links_df.groupby(by=["shape_id", "shape_stop_sequence"])
        .aggregate(
            point_list=pd.NamedAgg(column="geometry", aggfunc=list),
            num_points=pd.NamedAgg(column="geometry", aggfunc="nunique"),
        )
        .reset_index(drop=False)
    )
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace shape_links_agg_df for {trace_shape_id}:\n"
                f"{shape_links_agg_df.loc[shape_links_agg_df.shape_id == trace_shape_id]}"
            )
    # columns are shape_id, stop_sequence, point_list, num_points
    #                shape_id  shape_stop_sequence                                         point_list  num_points
    # 36235  SF:9717:20230930                -1.00  [POINT (6013227.6381617775 2109614.5949144145)...           5
    # 36236  SF:9717:20230930                 2.00  [POINT (6014293.454981883 2110683.4032201055),...           2
    # 36237  SF:9717:20230930                 3.00  [POINT (6015603.805947471 2111941.793745517), ...           9
    # 36238  SF:9717:20230930                 4.00  [POINT (6016092.661659231 2113720.8560314486),...          13

    # add intermediate stops to stop_links
    stop_links_df = pd.merge(
        left=stop_links_df,
        right=shape_links_agg_df.rename(columns={"shape_stop_sequence": "stop_sequence"}),
        on=["shape_id", "stop_sequence"],
        how="left",
        indicator=True,
    )
    WranglerLogger.debug(
        f"stop_links_df._merge.value_counts():\n{stop_links_df._merge.value_counts()}"
    )
    # for links without intermediate shape points
    stop_links_df.loc[stop_links_df["_merge"] == "left_only", "num_points"] = 0
    if trace_shape_ids:
        WranglerLogger.debug(
            f"trace stop_links_df:\n{stop_links_df.loc[stop_links_df.shape_id.isin(trace_shape_ids)]}"
        )

    # turn them into multi-point lines
    stop_links_df["geometry"] = stop_links_df.apply(
        lambda row:
        # if intermediate points
        shapely.geometry.LineString(
            [row["stop_geometry"]] + row["point_list"] + [row["next_stop_geometry"]]
        )
        if row["num_points"] > 0
        # no intermediate points
        else shapely.geometry.LineString([row["stop_geometry"], row["next_stop_geometry"]]),
        axis=1,
    )
    WranglerLogger.debug(f"stop_links_df including multi-point lines:\n{stop_links_df}")

    # create GeoDataFrame; this is in the local crs
    stop_links_df.drop(
        columns=["stop_geometry", "next_stop_geometry", "point_list", "_merge"], inplace=True
    )
    stop_links_gdf = gpd.GeoDataFrame(
        stop_links_df, geometry="geometry", crs=feed_tables["stops"].crs
    )
    WranglerLogger.debug(f"stop_links_gdf.dtypes\n{stop_links_gdf.dtypes}")
    # route_type        category
    # route_id            object
    # direction_id      category
    # shape_id            object
    # trip_id             object
    # stop_sequence        int64
    # stop_id             object
    # stop_name           object
    # next_stop_id        object
    # next_stop_name      object
    # num_points           int64
    # geometry          geometry
    # dtype: object

    # Filter to STATION_ROUTE_TYPES for adding to the roadway network
    station_stop_links_gdf = stop_links_gdf.loc[
        stop_links_gdf.route_type.isin(STATION_ROUTE_TYPES)
    ].copy()
    station_stop_id_set = set(station_stop_links_gdf["stop_id"]) | set(
        station_stop_links_gdf["next_stop_id"]
    )
    # Also add parent stations
    parent_station_id_set = set(
        feed_tables["stops"].loc[feed_tables["stops"]["is_parent"] == True, "stop_id"].tolist()
    )
    WranglerLogger.debug(
        f"parent_station_id_set len={len(parent_station_id_set)}:\n{parent_station_id_set}"
    )

    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace station_stop_links_gdf for {trace_shape_id}:\n"
                f"{station_stop_links_gdf.loc[station_stop_links_gdf.shape_id == trace_shape_id]}"
            )

    # Prepare nodes to add - station stops and parent stops
    station_stop_ids_gdf = (
        feed_tables["stops"]
        .loc[
            (feed_tables["stops"]["stop_id"].isin(station_stop_id_set))
            | (feed_tables["stops"]["stop_id"].isin(parent_station_id_set)),
            [
                "stop_id",
                "stop_name",
                "is_parent",
                "stop_lat",
                "stop_lon",
                "model_node_id",
                "geometry",
            ],
        ]
        .reset_index(drop=True)
        .copy()
    )
    station_stop_ids_gdf.rename(columns={
        "stop_lon": "X",
        "stop_lat": "Y",
        "stop_id":"stop_id_GTFS"
    }, inplace=True)
    station_stop_ids_gdf.to_crs(LAT_LON_CRS, inplace=True)
    WranglerLogger.debug(f"station_stop_ids_gdf:\n{station_stop_ids_gdf}")
    if trace_stop_id_set:
        WranglerLogger.debug(
            f"trace station_stop_ids_gdf for trace_stop_id_set:\n"
            f"{station_stop_ids_gdf.loc[ station_stop_ids_gdf['stop_id_GTFS'].isin(trace_stop_id_set)]}"
        )

    # Don't create new stations where one already exists! (stops that serve both bus and light rail)
    # => Filter to station ONLY
    new_station_stop_ids_gdf = station_stop_ids_gdf.loc[
        station_stop_ids_gdf["model_node_id"].isna()
    ].reset_index(drop=False)
    new_station_stop_ids_gdf.drop(columns={"model_node_id"}, inplace=True)

    # Assign model_node_id and add new stations to roadway network as roadway nodes
    max_node_num = roadway_net.nodes_df.model_node_id.max()
    new_station_stop_ids_gdf["model_node_id"] = new_station_stop_ids_gdf.index + max_node_num + 1
    new_station_stop_ids_gdf.drop(columns=['index'], inplace=True)
    WranglerLogger.info(f"Adding {len(new_station_stop_ids_gdf):,} nodes to roadway network")
    WranglerLogger.debug(f"new_station_stop_ids_gdf:\n{new_station_stop_ids_gdf}")
    WranglerLogger.debug(f"Before adding nodes, {len(roadway_net.nodes_df)=:,}")
    roadway_net.add_nodes(new_station_stop_ids_gdf)
    WranglerLogger.debug(f"After adding nodes, {len(roadway_net.nodes_df)=:,}")


    # get stop_id -> model_node_id for new nodes and stations that mapped to roadway nodes
    # (e.g. for LRT that have road node stations)
    new_stop_id_to_model_node_id_dict = (
         new_station_stop_ids_gdf[["stop_id_GTFS", "model_node_id"]]
        .set_index("stop_id_GTFS")
        .to_dict()["model_node_id"]
    )
    stop_id_to_model_node_id_dict = (
        station_stop_ids_gdf[["stop_id_GTFS", "model_node_id"]]
        .set_index("stop_id_GTFS")
        .to_dict()["model_node_id"]
    )
    stop_id_to_model_node_id_dict.update(new_stop_id_to_model_node_id_dict)
    WranglerLogger.debug(f"stop_id_to_model_node_id_dict:\n{stop_id_to_model_node_id_dict}")

    # Prepare links to add
    station_stop_links_gdf["A"] = station_stop_links_gdf["stop_id"].map(
        stop_id_to_model_node_id_dict
    )
    station_stop_links_gdf["B"] = station_stop_links_gdf["next_stop_id"].map(
        stop_id_to_model_node_id_dict
    )
    # Set rail/ferry only values
    station_stop_links_gdf = station_stop_links_gdf[
        [
            "route_type",
            "A",
            "stop_id",
            "stop_name",
            "B",
            "next_stop_id",
            "next_stop_name",
            "shape_id",
            "geometry",
        ]
    ]
    station_stop_links_gdf["rail_only"] = False
    station_stop_links_gdf.loc[
        station_stop_links_gdf.route_type.isin(RAIL_ROUTE_TYPES), "rail_only"
    ] = True
    station_stop_links_gdf["ferry_only"] = False
    station_stop_links_gdf.loc[
        station_stop_links_gdf.route_type.isin(FERRY_ROUTE_TYPES), "ferry_only"
    ] = True
    # Aggregate by A,B, choosing first values, and convert back to GeoDataFrame
    station_road_links_gdf = gpd.GeoDataFrame(
        station_stop_links_gdf.groupby(by=["A", "B"])
        .aggregate(
            stop_id=pd.NamedAgg(column="stop_id", aggfunc="first"),
            stop_name=pd.NamedAgg(column="stop_name", aggfunc="first"),
            next_stop_id=pd.NamedAgg(column="next_stop_id", aggfunc="first"),
            next_stop_name=pd.NamedAgg(column="next_stop_name", aggfunc="first"),
            geometry=pd.NamedAgg(column="geometry", aggfunc="first"),
            rail_only=pd.NamedAgg(column="rail_only", aggfunc=any),
            ferry_only=pd.NamedAgg(column="ferry_only", aggfunc=any),
            shape_ids=pd.NamedAgg(column="shape_id", aggfunc=list),
        )
        .reset_index(drop=False),
        crs=station_stop_links_gdf.crs,
    )
    station_road_links_gdf["A"] = station_road_links_gdf["A"].astype(int)
    station_road_links_gdf["B"] = station_road_links_gdf["B"].astype(int)

    # Drop links that are already in roadway network - this may happen for LRT links on roadways
    # But first, make sure rail_only or ferry_only is set to True in the roadway links version

    # save this to re-apply
    links_df_name = roadway_net.links_df.attrs["name"]
    roadway_net.links_df = roadway_net.links_df.merge(
        right=station_road_links_gdf[["A","B","rail_only","ferry_only"]],
        how="left",
        on=["A","B"],
        validate="one_to_one",
        suffixes=["","_update"],
        indicator=True
    )
    # re-apply
    roadway_net.links_df.attrs["name"] = links_df_name
    WranglerLogger.debug(
        f"Making sure existing roadway links corresponding to station pairs have transit access\n"
        f"{roadway_net.links_df.loc[roadway_net.links_df._merge == 'both']}"
    )
    # if any of these are footway or cycleway, warn
    if 'roadway' in roadway_net.links_df.columns:
        ACTIVE_OSM_HIGHWAY = ['footway','cycleway','path','pedestrian']
        active_only = roadway_net.links_df.loc[
            (roadway_net.links_df["_merge"] == "both") &
            roadway_net.links_df["roadway"].isin(ACTIVE_OSM_HIGHWAY)
        ]
        if len(active_only) > 0:
            WranglerLogger.warning(f"Adding rail or ferry access to {len(active_only)} active links -- See debug log")
            WranglerLogger.debug(f"Updating the following:\n{active_only}")

    roadway_net.links_df.loc[ roadway_net.links_df["_merge"] == "both", "rail_only"] = \
        roadway_net.links_df["rail_only"] | roadway_net.links_df["rail_only_update"]
    roadway_net.links_df.loc[ roadway_net.links_df["_merge"] == "both", "ferry_only"] = \
        roadway_net.links_df["ferry_only"] | roadway_net.links_df["ferry_only_update"]
    WranglerLogger.debug(
        f"After updating:\n"
        f"{roadway_net.links_df.loc[roadway_net.links_df._merge == 'both']}"
    )
    roadway_net.links_df.drop(columns=["_merge","rail_only_update","ferry_only_update"], inplace=True)

    # Now drop those that are already in the roadway network
    station_road_links_gdf = station_road_links_gdf.merge(
        right=roadway_net.links_df[["A","B"]],
        how="left",
        validate="one_to_one",
        indicator=True
    )
    WranglerLogger.debug(f"Dropping the following station_road_links_gdf rows that are already in the roadway network:\n"
                         f"{station_road_links_gdf.loc[ station_road_links_gdf['_merge'] == 'both']}")

    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace station_road_links_gdf including check for roadway_net.links_df for {trace_shape_id}:\n"
                f"{station_road_links_gdf.loc[station_road_links_gdf['shape_ids'].apply(lambda x, tid=trace_shape_id: isinstance(x, list) and tid in x)]}"
            )    
    station_road_links_gdf = station_road_links_gdf.loc[ station_road_links_gdf["_merge"] == "left_only"]
    station_road_links_gdf.drop(columns={"_merge"}, inplace=True)

    # Assign model_link_id, access for drive,walk,bike,truck,bus
    max_model_link_id = roadway_net.links_df.model_link_id.max()
    station_road_links_gdf["model_link_id"] = station_road_links_gdf.index + max_model_link_id + 1
    station_road_links_gdf["name"] = (
        station_road_links_gdf["stop_name"] + " to " + station_road_links_gdf["next_stop_name"]
    )
    station_road_links_gdf["shape_id"] = (
        station_road_links_gdf["stop_id"] + " to " + station_road_links_gdf["next_stop_id"]
    )
    station_road_links_gdf["drive_access"] = False
    station_road_links_gdf["bike_access"] = False
    station_road_links_gdf["walk_access"] = False
    station_road_links_gdf["truck_access"] = False
    station_road_links_gdf["bus_only"] = False
    station_road_links_gdf["lanes"] = 0
    if "roadway" in roadway_net.links_df.columns:
        station_road_links_gdf["roadway"] = "transit"

    # Set distance
    station_road_links_gdf.to_crs(local_crs, inplace=True)
    station_road_links_gdf["length"] = station_road_links_gdf.length
    if crs_units == "feet":
        station_road_links_gdf["distance"] = station_road_links_gdf["length"] / FEET_PER_MILE
    else:
        station_road_links_gdf["distance"] = (
            station_road_links_gdf["length"] / METERS_PER_KILOMETER
        )
    station_road_links_gdf.to_crs(LAT_LON_CRS, inplace=True)

    # Add to roadway network
    WranglerLogger.info(f"Adding {len(station_road_links_gdf):,} links to roadway network")
    WranglerLogger.debug(f"station_road_links_gdf:\n{station_road_links_gdf}")

    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace station_road_links_gdf for {trace_shape_id}:\n"
                f"{station_road_links_gdf.loc[station_road_links_gdf['shape_ids'].apply(lambda x, tid=trace_shape_id: isinstance(x, list) and tid in x)]}"
            )

    WranglerLogger.debug(f"Before adding links, {len(roadway_net.links_df)=:,}")
    roadway_net.add_links(station_road_links_gdf)
    WranglerLogger.debug(f"After adding links, {len(roadway_net.links_df)=:,}")

    WranglerLogger.info(f"Adding {len(station_road_links_gdf):,} shapes to roadway network")
    roadway_net.add_shapes(station_road_links_gdf)

    # Update feed_table['stops']: set model_node_id for stations in feed_table['stops']
    WranglerLogger.debug(
        f"Before updating, feed_tables['stops'] with model_node_id set: "
        f"{feed_tables['stops']['model_node_id'].notna().sum():,}"
    )
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace feed_tables['stops'] for {trace_shape_id}:\n"
                f"{feed_tables['stops'].loc[feed_tables['stops']['shape_ids'].apply(lambda x, tid=trace_shape_id: isinstance(x, list) and tid in x)]}"
            )

    feed_tables["stops"]["station_node_id"] = feed_tables["stops"]["stop_id"].map(
        stop_id_to_model_node_id_dict
    )

    # Verify no stops table stops have a model_node_id already set (from match_bus_stops_to_roadway_nodes() *and* have a station_model_node_id
    have_both_df = feed_tables["stops"].loc[
        feed_tables["stops"]["model_node_id"].notna()
        & feed_tables["stops"]["station_node_id"].notna()
        & (feed_tables["stops"]["model_node_id"] != feed_tables["stops"]["station_node_id"])
    ]
    assert len(have_both_df) == 0

    feed_tables["stops"].loc[feed_tables["stops"]["station_node_id"].notna(), "model_node_id"] = (
        feed_tables["stops"]["station_node_id"]
    )
    WranglerLogger.debug(
        f"After updating, feed_tables['stops'] with model_node_id set:\n"
        f"{feed_tables['stops']['model_node_id'].notna().sum():,}"
    )
    # Log feed_tables['stops'] for trace_shape_ids
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace feed_tables['stops'] for {trace_shape_id}:\n"
                f"{feed_tables['stops'].loc[feed_tables['stops']['shape_ids'].apply(lambda x, tid=trace_shape_id: isinstance(x, list) and tid in x)]}"
            )

    feed_tables["stops"].drop(columns=["station_node_id"], inplace=True)
    WranglerLogger.debug(f"feed_tables['stops']:\n{feed_tables['stops']}")
    # TODO: I think these are all parent nodes...
    WranglerLogger.debug(
        f"feed_tables['stops'] without model_node_id:\n{feed_tables['stops'].loc[feed_tables['stops']['model_node_id'].isna()]}"
    )

    # Update feed_table['shapes']: set shape_model_node_id for stations in feed_table['shapes'] and delete the other nodes.
    # Those are now in the shape of the roadway network link
    WranglerLogger.debug(
        f"About to update feed['shapes'] in add_stations_and_links_to_roadway_network:\n{feed_tables['shapes']}"
    )
    feed_tables["shapes"]["station_node_id"] = feed_tables["shapes"]["stop_id"].map(
        stop_id_to_model_node_id_dict
    )

    # Verify no shapes table stops have a shape_model_node_id already set *and* have a station_model_node_id
    have_both_df = feed_tables["shapes"].loc[
        feed_tables["shapes"]["shape_model_node_id"].notna()
        & feed_tables["shapes"]["station_node_id"].notna()
        & (feed_tables["shapes"]["shape_model_node_id"] != feed_tables["shapes"]["station_node_id"])
    ]
    if len(have_both_df) > 0:
        WranglerLogger.fatal(f"have_both_df:\n{have_both_df}")
    assert len(have_both_df) == 0
    feed_tables["shapes"].loc[
        feed_tables["shapes"]["station_node_id"].notna(), "shape_model_node_id"
    ] = feed_tables["shapes"]["station_node_id"]
    feed_tables["shapes"].drop(columns=["station_node_id"], inplace=True)
    # Delete other nodes
    feed_tables["shapes"] = feed_tables["shapes"].loc[
        # leave non station route types (bus) alone -- these will be handled elsewhere
        ~feed_tables["shapes"]["route_type"].isin(STATION_ROUTE_TYPES)
        |
        # for station route types, only keep those with model_node_id set
        feed_tables["shapes"]["shape_model_node_id"].notna()
    ]

    # Log feed_tables['shapes'] for trace_shape_ids
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace feed_tables['shapes'] for {trace_shape_id} at the end of add_stations_and_links_to_roadway_network():\n"
                f"{feed_tables['shapes'].loc[feed_tables['shapes']['shape_id'] == trace_shape_id]}"
            )

    # set A for stop_id's model_node_id
    stop_links_gdf = pd.merge(
        left=stop_links_gdf,
        right=feed_tables["stops"][["stop_id", "model_node_id"]],
        how="left",
        validate="many_to_one",
    ).rename(columns={"model_node_id": "A"})
    # set B for next_stop_id's model_node_id
    stop_links_gdf = pd.merge(
        left=stop_links_gdf,
        right=feed_tables["stops"][["stop_id", "model_node_id"]].rename(
            columns={"stop_id": "next_stop_id"}
        ),
        how="left",
        validate="many_to_one",
    ).rename(columns={"model_node_id": "B"})
    WranglerLogger.debug(f"stop_links_gdf after setting A and B:\n{stop_links_gdf}")

    # Filter non-station stop links to return
    bus_stop_links_gdf = stop_links_gdf.loc[~stop_links_gdf.route_type.isin(STATION_ROUTE_TYPES)]

    return stop_id_to_model_node_id_dict, bus_stop_links_gdf


def create_feed_from_gtfs_model(  # noqa: PLR0915
    gtfs_model: GtfsModel,
    roadway_net: RoadwayNetwork,
    local_crs: str,
    crs_units: str,
    timeperiods: dict[str, tuple[str, str]],
    frequency_method: str,
    default_frequency_for_onetime_route: int = 180,
    add_stations_and_links: bool = True,
    max_stop_distance: Optional[float] = None,
    trace_shape_ids: Optional[list[str]] = None,
    errors: Literal["raise", "ignore"] = "raise",
) -> Feed:
    """Convert GTFS model to Wrangler Feed with stops mapped to roadway network.

    Comprehensive conversion that transforms GTFS schedule data into a frequency-based
    Feed representation compatible with travel modeling. Maps transit stops to roadway
    nodes and optionally adds station infrastructure to the network.

    Process Steps:
    1. Prepare roadway network:
       - Convert roadway_net.nodes_df to GeoDataFrame if needed
       - Create Point geometries from X, Y coordinates
       - Set CRS to LAT_LON_CRS (EPSG:4326)
       - Modifies roadway_net.nodes_df in place

    2. Copy GTFS tables to feed_tables dictionary:
       - Copy routes, trips, agencies, stops, stop_times, shapes from gtfs_model
       - Convert stops to GeoDataFrame with Point geometries from stop_lon/stop_lat
       - Creates feed_tables dict for all subsequent operations

    3. Enrich stops with route/agency metadata:
       - Calls: [`add_additional_data_to_stops()`][network_wrangler.utils.transit.add_additional_data_to_stops]
       - Joins route and agency information to each stop via stop_times and trips
       - Adds columns: agency_ids, agency_names, route_ids, route_names, route_types,
         shape_ids, is_parent, is_bus_stop
       - Modifies feed_tables['stops'] in place

    4. Create frequency-based schedules from timetables:
       - Calls: [`create_feed_frequencies()`][network_wrangler.utils.transit.create_feed_frequencies]
       - Converts GTFS trip-based schedules to frequency-based representation
       - Groups trips by stop pattern (shape_id) and time period
       - Calculates headways using specified method (uniform/mean/median)
       - Creates one representative trip per shape_id
       - Creates feed_tables['frequencies'] table
       - Modifies: feed_tables['stop_times'] (adds departure_minutes),
         feed_tables['trips'] (one row per shape_id)

    5. Match stops to shape points and enrich shapes:
       - Calls: [`add_additional_data_to_shapes()`][network_wrangler.utils.transit.add_additional_data_to_shapes]
       - For each shape_id, processes stops in sequence order
       - For each stop:
           - Match: Find nearest existing shape point within threshold (forward-only search)
           - Insert: If no match, create new shape point at stop location
       - Uses local minimum matching to handle routes that double back
       - Renumbers shape_pt_sequence if duplicates or non-integers detected
       - Writes debug_shapes.geojson with stop matching information
       - Calls helpers: `_match_stop_to_shape_points()`, `_insert_stop_into_shape()`,
         `_align_shape_with_stops()`, `_write_debug_shapes()`
       - Modifies feed_tables['shapes']: adds stop_id, stop_name, stop_sequence,
         match_distance_{crs_units}, poor_match

    6. Match bus stops to roadway nodes:
       - Calls: [`match_bus_stops_to_roadway_nodes()`][network_wrangler.utils.transit.match_bus_stops_to_roadway_nodes]
       - Gets bus modal graph from roadway network (bus-accessible nodes only)
       - For each bus stop:
           - Finds K nearest bus-accessible nodes using BallTree spatial index
           - If use_name_matching=True: scores by distance + name compatibility
             (combined_score = 0.1 * normalized_dist + 0.9 * (1 - name_score))
           - Selects best match within max_distance threshold
       - If name matching enabled: marks stops with combined_score > 0.9 as poor_match:
           - Sets poor_match = True (does not update location yet)
           - Keeps model_node_id as nearest bus-accessible node (for connector links)
           - These stops will be added to network in step 6a
       - If name matching disabled: poor_match = False for all stops
       - Updates stop locations to matched node positions (except poor_match stops)
       - Modifies feed_tables['stops']: adds model_node_id, match_distance_{crs_units},
         close_match, poor_match (always added, but only True when name matching enabled),
         node_link_names, name_match_score, normalized_dist, combined_score
       - Modifies feed_tables['shapes']: updates bus stop shape points to node locations,
         adds poor_match flag
       - Modifies roadway_net.nodes_df: adds bus_access column

    6a. Add unmatched bus stops to network:
        - Calls: [`add_unmatched_bus_stops_to_network()`][network_wrangler.utils.transit.add_unmatched_bus_stops_to_network]
        - Identifies unmatched bus stops (poor_match=True)
        - Clusters nearby stops using DBSCAN (max_distance threshold)
        - Creates new roadway nodes at cluster centroids
        - One node per cluster for grouped transit stations
        - Updates feed_tables['stops'] with new node IDs
        - Modifies roadway_net.nodes_df: adds new transit stop nodes with is_transit_stop_node flag

    7. Add rail/ferry stations and links to roadway network:
       - Calls: [`add_stations_and_links_to_roadway_network()`][network_wrangler.utils.transit.add_stations_and_links_to_roadway_network]
       - For STATION_ROUTE_TYPES (rail, light rail, subway, etc.):
           - Creates new nodes for each station
           - Adds dedicated transit links between consecutive stations
           - Links follow original GTFS shape geometry
       - For BUS/TROLLEYBUS routes:
           - Creates bus_stop_links_gdf with consecutive stop pairs (A->B)
           - Includes route metadata (route_id, trip_id, direction_id, etc.)
           - No new nodes/links added yet (handled in next step)
       - Returns station_id_to_model_node_id_dict and bus_stop_links_gdf
       - Modifies roadway_net.nodes_df: adds station nodes
       - Modifies roadway_net.links_df: adds station-to-station links
       - Modifies feed_tables['stops']: updates station stops with new model_node_ids
       - Modifies feed_tables['shapes']: removes intermediate shape points for station routes,
         keeps only station stop points

    7a. Create connector links for unmatched bus stops:
        - Calls: [`create_connector_links_for_poor_match_stops()`][network_wrangler.utils.transit.create_connector_links_for_poor_match_stops]
        - Creates bidirectional bus-only links between:
            - New transit stop nodes (from step 6a) to their nearest bus-accessible node
        - Links marked with ref="unmatched_bus_stop"
        - Enables routing through these previously unmatched stops
        - Modifies roadway_net.links_df: adds connector links
        - Modifies roadway_net.shapes: adds link geometries

    8. Route bus services through road network:
       - Calls: [`create_bus_routes()`][network_wrangler.utils.transit.create_bus_routes]
       - Gets bus modal graph (DiGraph for pathfinding)
       - For each consecutive bus stop pair (A->B):
           - Check: If either node not in bus graph (e.g., poor_match):
               - Add direct A->B connection to bus_node_sequence
               - Add to no_path_sequence for special handling
               - Skip pathfinding
           - Find path: Use NetworkX shortest path through bus network
               - Optional: Shape-aware routing (prefers paths close to original shape)
           - Create shape points: Add all intermediate nodes in path as shape points
       - Handles exceptions (NetworkXNoPath, NodeNotFound): adds to no_path_sequence
       - If errors='raise' and no_path_sequence not empty: raises TransitValidationError
       - If errors='ignore' and no_path_sequence not empty: calls create_links_for_failed_bus_paths()
         to create special bus-only links (marked ref="bad_bus_path") for failed routing segments
       - Calls helpers: `get_original_shape_points_between_stops()`,
         `find_shape_aware_shortest_path()`
       - Modifies feed_tables['shapes']: replaces bus route shapes with routed paths
         through road network, adds shape_model_node_id from roadway

    9. Consolidate duplicate stops mapped to same node:
       - Renames stop_id -> stop_id_GTFS (original GTFS IDs)
       - Renames model_node_id -> stop_id (now uses network node IDs as stop IDs)
       - Multiple GTFS stops may map to same network node
       - Groups by stop_id (model_node_id) and aggregates:
           - Converts singular fields to lists (stop_id_GTFS, stop_name, etc.)
           - Takes first geometry/location
           - Merges route/agency lists (flattens and deduplicates)
       - Creates stop_id_to_model_node_id_dict mapping GTFS stop_id -> model_node_id
       - Modifies feed_tables['stops']: consolidated rows, one per unique network node

    10. Update stop references and create Feed object:
        - Updates feed_tables['stop_times']: maps stop_id_GTFS -> stop_id (model_node_id)
        - Converts stop_times to Wrangler format
        - Creates Feed object with all processed tables:
            - routes, trips, agencies: from GTFS
            - stops: consolidated by model_node_id with metadata
            - stop_times: with updated stop_id references
            - shapes: routed through road network with shape_model_node_id
            - frequencies: frequency-based schedules by time period
        - Returns Feed object ready for network modeling

    Args:
        gtfs_model: Source GTFS data model
        roadway_net: Target roadway network for stop mapping
        local_crs: Coordinate system for distance calculations
        crs_units: Distance units ('feet' or 'meters')
        timeperiods: Time period definitions for frequencies
            Example: {'EA': ('03:00','06:00'), 'AM': ('06:00','10:00')}
        frequency_method: How to calculate headways
            ('uniform_headway', 'mean_headway', or 'median_headway')
        default_frequency_for_onetime_route: Default headway in minutes
            for routes with one trip per period (default: 180)
        add_stations_and_links: If True, add stations to roadway network
            (recommended, False not implemented)
        max_stop_distance: Maximum distance in crs_units for matching bus stops
            to roadway nodes. If None, uses default MAX_DISTANCE_STOP values
        trace_shape_ids: Shape IDs for detailed debug logging
        errors: How to handle routing errors ('raise' or 'ignore')

    Returns:
        Feed: Wrangler Feed object with:
            - Stops mapped to roadway nodes
            - Frequency-based trip representation
            - Routes following road network paths

    Raises:
        TransitValidationError: If bus stops can't be matched to roadway
        NodeNotFoundError: If required nodes aren't found

    Notes:
        - Bus routes are re-routed through actual road network
        - Station routes keep original alignment with new nodes/links
    """
    WranglerLogger.debug(f"create_feed_from_gtfsmodel()")
    if crs_units not in ["feet", "meters"]:
        msg = f"crs_units must be on of 'feet' or 'meters'; received {crs_units}"
        raise ValueError(msg)

    # Convert roadway_net.nodes_df GeoDataFrame if needed (modifying in place)
    if not isinstance(roadway_net.nodes_df, gpd.GeoDataFrame):
        if "geometry" not in roadway_net.nodes_df.columns:
            node_geometry = [
                shapely.geometry.Point(x, y)
                for x, y in zip(roadway_net.nodes_df["X"], roadway_net.nodes_df["Y"])
            ]
            roadway_net.nodes_df = gpd.GeoDataFrame(
                roadway_net.nodes_df, geometry=node_geometry, crs=LAT_LON_CRS
            )
        else:
            roadway_net.nodes_df = gpd.GeoDataFrame(roadway_net.nodes_df, crs=LAT_LON_CRS)
    elif roadway_net.nodes_df.crs is None:
        roadway_net.nodes_df = roadway_net.nodes_df.set_crs(LAT_LON_CRS)

    # Start with the tables from the GTFS model
    feed_tables = {}

    # Copy over standard tables that don't need modification
    # GtfsModel guarantees routes and trips exist
    # create a copies of this which we'll manipulate
    feed_tables["routes"] = gtfs_model.routes.copy()
    feed_tables["trips"] = gtfs_model.trips.copy()
    feed_tables["agencies"] = gtfs_model.agency.copy()
    feed_tables["stops"] = gtfs_model.stops.copy()
    feed_tables["stop_times"] = gtfs_model.stop_times.copy()
    feed_tables["shapes"] = gtfs_model.shapes.copy()

    # create mapping from gtfs_model stop to RoadwayNetwork nodes
    # GtfsModel guarantees stops exists
    if not isinstance(feed_tables["stops"], gpd.GeoDataFrame):
        stop_geometry = [
            shapely.geometry.Point(lon, lat)
            for lon, lat in zip(gtfs_model.stops["stop_lon"], gtfs_model.stops["stop_lat"])
        ]
        feed_tables["stops"] = gpd.GeoDataFrame(
            feed_tables["stops"], geometry=stop_geometry, crs=LAT_LON_CRS
        )

    # Add helpful extra data to stops table
    add_additional_data_to_stops(feed_tables)

    # create frequencies table from GTFS stop_times (if no frequencies table is specified)
    if hasattr(gtfs_model, "frequencies") and gtfs_model.frequencies is not None:
        feed_tables["frequencies"] = gtfs_model.frequencies
        # TODO: What if the frequencies are specified for the wrong time periods?
    else:
        # GtfsModel specifies every individual trip but Feed expects the trip to be
        # representative with frequencies. This makes that conversion
        create_feed_frequencies(
            feed_tables,
            timeperiods,
            frequency_method,
            default_frequency_for_onetime_route,
            trace_shape_ids,
        )

    if not add_stations_and_links:
        msg = "create_feed_from_gtfs_feed() doesn't implement add_stations_and_links==False."
        raise NotImplementedError(msg)

    # Add helpful extra data to shapes table
    add_additional_data_to_shapes(feed_tables, local_crs, crs_units, trace_shape_ids)

    # Use provided max_stop_distance or default
    if max_stop_distance is None:
        max_stop_distance = MAX_DISTANCE_STOP[crs_units]

    match_bus_stops_to_roadway_nodes(
        feed_tables,
        roadway_net,
        local_crs,
        crs_units,
        max_stop_distance,
        trace_shape_ids,
        use_name_matching=True  # Use name matching when available
    )

    # Add unmatched bus stops as new nodes in the roadway network
    unmatched_stops_nodes_gdf = add_unmatched_bus_stops_to_network(
        feed_tables,
        roadway_net,
        local_crs,
        max_stop_distance,
        trace_shape_ids
    )

    # for fixed route transit, add the links and stops to the roadway network
    station_id_to_model_node_id_dict, bus_stop_links_gdf = (
        add_stations_and_links_to_roadway_network(
            feed_tables, roadway_net, local_crs, crs_units, trace_shape_ids
        )
    )

    # Create connector links for unmatched bus stops
    if len(unmatched_stops_nodes_gdf) > 0:
        create_connector_links_for_poor_match_stops(
            roadway_net=roadway_net,
            unmatched_stops_gdf=unmatched_stops_nodes_gdf,
            local_crs=local_crs,
            crs_units=crs_units,
            trace_shape_ids=trace_shape_ids
        )

    WranglerLogger.debug(f"bus_stop_links_gdf:\n{bus_stop_links_gdf}")

    # finally, we need to find shortest paths through the bus network
    # between bus stops and update stops and shapes accordingly
    try:
        create_bus_routes(
            bus_stop_links_gdf, feed_tables, roadway_net, local_crs, crs_units, trace_shape_ids, errors
        )
    except Exception as e:
        raise e

    # Getting ready to create Feed object
    # stop_id is now really the model_node_id -- set it
    feed_tables["stops"].rename(
        columns={"stop_id": "stop_id_GTFS", "model_node_id": "stop_id"}, inplace=True
    )
    # But some of the stops are mapped the the same model_node_id (now, stop_id) -- merge them.
    duplicate_stop_ids_df = feed_tables["stops"].loc[
        feed_tables["stops"].duplicated(subset=["stop_id"], keep=False)
    ]
    WranglerLogger.debug(f"duplicate_stop_ids_df:\n{duplicate_stop_ids_df}")
    WranglerLogger.debug(f"feed_tables['stops'].dtypes:\n{feed_tables['stops'].dtypes}")
    # stop_id_GTFS                     object
    # stop_name                        object
    # stop_lat                        float64
    # stop_lon                        float64
    # zone_id                          object
    # location_type                  category
    # parent_station                   object
    # level_id                         object
    # geometry                       geometry
    # agency_ids                       object
    # agency_names                     object
    # route_ids                        object
    # route_names                      object
    # route_types                      object
    # shape_ids                        object
    # is_parent                          bool
    # is_bus_stop                        bool
    # stop_id                          object
    # match_distance_feet             float64
    # valid_match                      object

    # create full stop_id_to_model_node_id_dict mapping
    stop_id_to_model_node_id_dict = (
        feed_tables["stops"][["stop_id_GTFS", "stop_id"]]
        .set_index("stop_id_GTFS")
        .to_dict()["stop_id"]
    )
    WranglerLogger.debug(f"stop_id_to_model_node_id_dict: {stop_id_to_model_node_id_dict}")

    # Convert NaN to empty lists before aggregation
    feed_tables["stops"]["agency_ids"] = feed_tables["stops"]["agency_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    feed_tables["stops"]["agency_names"] = feed_tables["stops"]["agency_names"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    feed_tables["stops"]["route_ids"] = feed_tables["stops"]["route_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    feed_tables["stops"]["route_names"] = feed_tables["stops"]["route_names"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    feed_tables["stops"]["route_types"] = feed_tables["stops"]["route_types"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    feed_tables["stops"]["shape_ids"] = feed_tables["stops"]["shape_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    feed_tables["stops"] = (
        feed_tables["stops"]
        .groupby(by=["stop_id"])
        .aggregate(
            stop_id_GTFS=pd.NamedAgg(column="stop_id_GTFS", aggfunc=list),
            stop_name=pd.NamedAgg(column="stop_name", aggfunc=list),
            stop_lat=pd.NamedAgg(column="stop_lat", aggfunc="first"),
            stop_lon=pd.NamedAgg(column="stop_lon", aggfunc="first"),
            zone_id=pd.NamedAgg(column="zone_id", aggfunc=list),
            location_type=pd.NamedAgg(column="location_type", aggfunc="first"),
            parent_station=pd.NamedAgg(column="parent_station", aggfunc="first"),
            level_id=pd.NamedAgg(column="level_id", aggfunc=list),
            geometry=pd.NamedAgg(column="geometry", aggfunc="first"),
            agency_ids=pd.NamedAgg(
                column="agency_ids",
                aggfunc=lambda x: list({item for sublist in x for item in sublist}),
            ),
            agency_names=pd.NamedAgg(
                column="agency_names",
                aggfunc=lambda x: list({item for sublist in x for item in sublist}),
            ),
            route_ids=pd.NamedAgg(
                column="route_ids",
                aggfunc=lambda x: list({item for sublist in x for item in sublist}),
            ),
            route_names=pd.NamedAgg(
                column="route_names",
                aggfunc=lambda x: list({item for sublist in x for item in sublist}),
            ),
            route_types=pd.NamedAgg(
                column="route_types",
                aggfunc=lambda x: list({item for sublist in x for item in sublist}),
            ),
            shape_ids=pd.NamedAgg(
                column="shape_ids",
                aggfunc=lambda x: list({item for sublist in x for item in sublist}),
            ),
            is_parent=pd.NamedAgg(column="is_parent", aggfunc=any),
            is_bus_stop=pd.NamedAgg(column="is_bus_stop", aggfunc=any),
        )
        .reset_index(drop=False)
    )
    feed_tables["stops"]["stop_id"] = feed_tables["stops"]["stop_id"].astype(int)

    # Update feed_tables['stop_times']
    feed_tables["stop_times"].rename(columns={"stop_id": "stop_id_GTFS"}, inplace=True)
    feed_tables["stop_times"]["stop_id"] = feed_tables["stop_times"]["stop_id_GTFS"].map(
        stop_id_to_model_node_id_dict
    )

    # Log all tables
    for table_name, table_data in feed_tables.items():
        WranglerLogger.debug(
            f"Before creating Feed object, feed_tables[{table_name}]:\n{table_data}"
        )

    # create Feed object from results of the above
    try:
        feed = Feed(**feed_tables)
        WranglerLogger.info(f"Successfully created Feed with {len(feed_tables)} tables")
        return feed
    except Exception as e:
        WranglerLogger.error(f"Error creating Feed: {e}")
        raise e


def get_original_shape_points_between_stops(
    feed_tables: dict, shape_id: str, from_stop_seq: int, to_stop_seq: int, trace: bool = False
):
    """Get original GTFS shape points between two consecutive stops.

    Uses stop_sequence information already added by add_additional_data_to_shapes().

    Args:
        feed_tables: GTFS feed tables dictionary
        shape_id: Shape identifier
        from_stop_seq: Starting stop sequence number
        to_stop_seq: Ending stop sequence number (should be from_stop_seq + 1)
        trace: If True, enable trace logging for debugging

    Returns:
        DataFrame of shape points between stops, or empty DataFrame if not found
    """
    try:
        if trace:
            WranglerLogger.debug(f"Getting shape points for shape_id={shape_id} between stop_seq {from_stop_seq} and {to_stop_seq}")

        # Get shape points for this shape_id
        shape_points = feed_tables['shapes'][feed_tables['shapes']['shape_id'] == shape_id].copy()

        if trace:
            WranglerLogger.debug(f"  trace Found {len(shape_points)} total shape points for shape_id={shape_id}")
            if not shape_points.empty and 'stop_sequence' in shape_points.columns:
                unique_stop_seqs = shape_points['stop_sequence'].dropna().unique()
                WranglerLogger.debug(f"  Unique stop_sequences in shape: {sorted(unique_stop_seqs)}")

        if shape_points.empty:
            if trace:
                WranglerLogger.debug(f"  No shape points found for shape_id={shape_id}")
            return shape_points

        # Sort by shape_pt_sequence
        shape_points = shape_points.sort_values('shape_pt_sequence')

        # Check if stop_sequence column exists
        if 'stop_sequence' not in shape_points.columns:
            if trace:
                WranglerLogger.debug(f"  WARNING: 'stop_sequence' column not found in shapes table")
                WranglerLogger.debug(f"  Available columns: {list(shape_points.columns)}")
            return pd.DataFrame()

        # Find the shape_pt_sequence values for the start and end stops
        # Only rows with stop_sequence values are actual stops
        stop_points = shape_points[shape_points['stop_sequence'].notna()]

        if trace:
            WranglerLogger.debug(f"  Found {len(stop_points)} stop points (vs {len(shape_points)} total shape points)")
            if not stop_points.empty:
                WranglerLogger.debug(f"  Stop sequences present: {sorted(stop_points['stop_sequence'].unique())}")

        # Find shape_pt_sequence range for the requested stop sequences
        from_stop_points = stop_points[stop_points['stop_sequence'] == from_stop_seq]
        to_stop_points = stop_points[stop_points['stop_sequence'] == to_stop_seq]

        if from_stop_points.empty or to_stop_points.empty:
            if trace:
                WranglerLogger.debug(f"  WARNING: Could not find stop sequences {from_stop_seq} or {to_stop_seq}")
                if from_stop_points.empty:
                    WranglerLogger.debug(f"    from_stop_seq {from_stop_seq} not found")
                if to_stop_points.empty:
                    WranglerLogger.debug(f"    to_stop_seq {to_stop_seq} not found")
            return pd.DataFrame()

        # Get the shape_pt_sequence values for these stops
        from_shape_seq = from_stop_points['shape_pt_sequence'].iloc[0]
        to_shape_seq = to_stop_points['shape_pt_sequence'].iloc[0]

        if trace:
            WranglerLogger.debug(f"  Stop sequence {from_stop_seq} is at shape_pt_sequence {from_shape_seq}")
            WranglerLogger.debug(f"  Stop sequence {to_stop_seq} is at shape_pt_sequence {to_shape_seq}")

        # Filter to get all shape points between these two stops (inclusive)
        shape_points = shape_points[
            (shape_points['shape_pt_sequence'] >= from_shape_seq) &
            (shape_points['shape_pt_sequence'] <= to_shape_seq)
        ]

        if trace:
            WranglerLogger.debug(f"  Filtered to {len(shape_points)} shape points between shape_pt_sequences {from_shape_seq} and {to_shape_seq}")
            if not shape_points.empty:
                num_with_stops = shape_points['stop_sequence'].notna().sum()
                WranglerLogger.debug(f"    Including {num_with_stops} stop points and {len(shape_points)-num_with_stops} intermediate shape points")

        return shape_points
    except Exception as e:
        if trace:
            WranglerLogger.debug(f"  ERROR in get_original_shape_points_between_stops: {type(e).__name__}: {e}")
            import traceback
            WranglerLogger.debug(f"  Traceback: {traceback.format_exc()}")
        return pd.DataFrame()


def calculate_path_deviation_from_shape(path_nodes: list, original_shape_points: pd.DataFrame, roadway_net, trace: bool = False) -> float:
    """Calculate total deviation of a path from original shape points.

    Creates a LineString from the path nodes and calculates the distance from each
    shape point to the nearest point on the line.

    Args:
        path_nodes: List of roadway node IDs in the path
        original_shape_points: DataFrame of original GTFS shape points
        roadway_net: RoadwayNetwork to get node coordinates
        trace: If True, enable trace logging for debugging

    Returns:
        Total deviation distance (sum of distances from shape points to path line)
    """
    if original_shape_points.empty or not path_nodes:
        return float('inf')

    try:
        from shapely.geometry import LineString, Point

        # Create LineString from path nodes
        path_coords = []
        for node_id in path_nodes:
            node_row = roadway_net.nodes_df[roadway_net.nodes_df['model_node_id'] == node_id]
            if not node_row.empty:
                path_coords.append((node_row.iloc[0]['X'], node_row.iloc[0]['Y']))

        if len(path_coords) < 2:
            return float('inf')

        path_line = LineString(path_coords)

        # Calculate total deviation - for each shape point, find distance to path line
        total_deviation = 0.0
        if trace:
            WranglerLogger.debug(f"Calculating path deviation for line with {len(path_nodes)} nodes against {len(original_shape_points)} shape points")
            # Get link names for first and last nodes for debugging
            first_node_row = roadway_net.nodes_df[roadway_net.nodes_df['model_node_id'] == path_nodes[0]]
            last_node_row = roadway_net.nodes_df[roadway_net.nodes_df['model_node_id'] == path_nodes[-1]]
            first_link_names = first_node_row.iloc[0].get('link_names', []) if not first_node_row.empty and 'link_names' in first_node_row.columns else []
            last_link_names = last_node_row.iloc[0].get('link_names', []) if not last_node_row.empty and 'link_names' in last_node_row.columns else []
            WranglerLogger.debug(f"  Path from node {path_nodes[0]} ({first_link_names}) to node {path_nodes[-1]} ({last_link_names})")

        for idx, shape_row in original_shape_points.iterrows():
            shape_point = Point(shape_row['shape_pt_lon'], shape_row['shape_pt_lat'])
            # Distance from point to nearest point on line
            dist = shape_point.distance(path_line)
            total_deviation += dist

            if trace and idx % max(1, len(original_shape_points) // 5) == 0:  # Log every ~20% of points
                WranglerLogger.debug(f"  Shape point {idx} distance to path: {dist:.6f}")

        if trace:
            WranglerLogger.debug(f"Total path deviation: {total_deviation:.6f} (avg per point: {total_deviation/len(original_shape_points):.6f})")

        return total_deviation
    except Exception as e:
        if trace:
            WranglerLogger.debug(f"Error calculating path deviation: {e}")
        return float('inf')


def find_shape_aware_shortest_path(
    G_bus: nx.DiGraph, start_node: int, end_node: int, original_shape_points: pd.DataFrame,
    roadway_net: RoadwayNetwork, tolerance: float = 1.10, trace: bool = False
) -> list:
    """Find shortest path that considers original shape points.

    Uses constrained shortest path approach:
    1. Find shortest distance
    2. Get all paths within tolerance of shortest distance
    3. Among those, select path with minimum deviation from original shape

    Args:
        G_bus: NetworkX DiGraph of bus network
        start_node: Starting node ID
        end_node: Ending node ID
        original_shape_points: DataFrame of original GTFS shape points
        roadway_net: RoadwayNetwork to get node coordinates
        tolerance: Maximum ratio of path distance to shortest distance (default 1.10 = 110%)
        trace: Whether to log trace information

    Returns:
        List of node IDs representing the best path
    """
    try:
        # First, get the absolute shortest path distance
        shortest_dist = nx.shortest_path_length(G_bus, start_node, end_node, weight="distance")
        max_allowed_dist = shortest_dist * tolerance

        # Get multiple shortest paths to evaluate
        from itertools import islice
        candidate_paths = list(islice(nx.shortest_simple_paths(G_bus, start_node, end_node, weight="distance"), MAX_SHAPE_CANDIDATE_PATHS))

        best_path = None
        best_deviation = float('inf')
        paths_within_tolerance = 0

        # Store path info for debug output
        debug_paths = []

        for path in candidate_paths:
            # Calculate path distance
            path_dist = 0
            for i in range(len(path) - 1):
                edge_data = G_bus[path[i]][path[i+1]]
                path_dist += edge_data.get('distance', 0)

            # Check if within tolerance
            if path_dist <= max_allowed_dist:
                paths_within_tolerance += 1

                # Calculate shape deviation for this path
                deviation = calculate_path_deviation_from_shape(
                    path, original_shape_points, roadway_net, trace=trace
                )

                # Store for debug output
                if trace:
                    debug_paths.append({
                        'path': path,
                        'distance': path_dist,
                        'deviation': deviation,
                        'is_best': False  # Will update later
                    })

                # Select path with minimum deviation
                if deviation < best_deviation:
                    best_deviation = deviation
                    best_path = path
                    if trace:
                        WranglerLogger.debug(f"  New best path with deviation {deviation:.6f}, distance {path_dist:.3f}")

        # Mark the best path
        if trace and debug_paths and best_path:
            for dp in debug_paths:
                if dp['path'] == best_path:
                    dp['is_best'] = True
                    break

        if trace:
            WranglerLogger.debug(f"Shape-aware routing: {paths_within_tolerance} paths within {tolerance:.1%} tolerance")
            if not original_shape_points.empty:
                WranglerLogger.debug(f"Original shape has {len(original_shape_points)} points between stops")
            if best_path:
                WranglerLogger.debug(f"Selected path with deviation {best_deviation:.6f}")


        return best_path if best_path else candidate_paths[0]  # Fallback to shortest

    except Exception as e:
        if trace:
            WranglerLogger.debug(f"Shape-aware routing failed: {type(e).__name__}: {e}, falling back to standard shortest path")
            import traceback
            WranglerLogger.debug(f"Traceback: {traceback.format_exc()}")
        # Fallback to standard shortest path
        return nx.shortest_path(G_bus, start_node, end_node, weight="distance")
