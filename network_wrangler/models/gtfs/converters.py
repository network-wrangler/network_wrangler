"""Utils for converting original gtfs to wrangler gtfs."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ...logger import WranglerLogger
from .._base.types import TimeString
from ...utils.time import time_to_seconds

def convert_gtfs_to_wrangler_gtfs(gtfs_path: Path, wrangler_path: Path) -> None:
    """Converts a GTFS feed to a Wrangler GTFS feed.

    Args:
        gtfs_path: Path to GTFS feed.
        wrangler_path: Path to save Wrangler GTFS feed.
    """
    gtfs_path = Path(gtfs_path)
    gtfs_stops_df = pd.read_csv(gtfs_path / "stops.txt")
    gtfs_stoptimes_df = pd.read_csv(gtfs_path / "stop_times.txt")

    wr_stoptimes_df = convert_stop_times_to_wrangler_stop_times(gtfs_stoptimes_df, gtfs_stops_df)
    wr_stops_df = convert_stops_to_wrangler_stops(gtfs_stops_df)

    wrangler_path = Path(wrangler_path)
    if not wrangler_path.exists():
        wrangler_path.mkdir(parents=True)
    wr_stops_df.to_csv(wrangler_path / "stops.txt", index=False)
    wr_stoptimes_df.to_csv(wrangler_path / "stop_times.txt", index=False)


def convert_stops_to_wrangler_stops(stops_df: pd.DataFrame) -> pd.DataFrame:
    """Converts a stops.txt file to a Wrangler stops.txt file.

    Creates table that is unique to each model_node_id.
    Takes first instance of value for all attributes except stop_id.
    Aggregates stop_id into a comma-separated string and renames to stop_id_GTFS.
    Renames model_node_id to stop_id.

    Example usage:

    ```python
    import pandas as pd
    from network_wrangler.models.gtfs.converters import convert_stops_to_wrangler_stops

    in_f = "network_wrangler/examples/stpaul/gtfs/stops.txt"
    stops_df = pd.read_csv(in_f)
    wr_stops_df = convert_stops_to_wrangler_stops(stops_df)
    wr_stops_df.to_csv("wr_stops.txt", index=False)
    ```
    Args:
        stops_df: stops.txt file as a pandas DataFrame.

    Returns:
        Wrangler stops.txt file as a pandas DataFrame.
    """
    wr_stops_df = stops_df.groupby("model_node_id").first().reset_index()
    wr_stops_df = wr_stops_df.drop(columns=["stop_id"])
    # if stop_id is an int, convert to string
    if stops_df["stop_id"].dtype == "int64":
        stops_df["stop_id"] = stops_df["stop_id"].astype(str)
    stop_id_GTFS = (
        stops_df.groupby("model_node_id").stop_id.apply(lambda x: ",".join(x)).reset_index()
    )
    wr_stops_df["stop_id_GTFS"] = stop_id_GTFS["stop_id"]
    wr_stops_df = wr_stops_df.rename(columns={"model_node_id": "stop_id"})
    return wr_stops_df


def convert_stop_times_to_wrangler_stop_times(
    gtfs_stop_times_df: pd.DataFrame, gtfs_stops_df: pd.DataFrame
) -> pd.DataFrame:
    """Converts a stop_times.txt file to a Wrangler stop_times.txt file.

    Replaces stop_id with model_node_id from stops.txt, making sure that if there are duplicate
    model_node_ids for each stop_id, the correct model_node_id is used.

    Args:
        gtfs_stop_times_df: stop_times.txt file as a pandas DataFrame.
        gtfs_stops_df: stops.txt file as a pandas DataFrame

    Returns:
        Wrangler stop_times.txt file as a pandas DataFrame.
    """
    wr_stop_times_df = gtfs_stop_times_df.merge(
        gtfs_stops_df[["stop_id", "model_node_id"]], on="stop_id", how="left"
    )
    wr_stop_times_df = wr_stop_times_df.drop(columns=["stop_id"])
    wr_stop_times_df = wr_stop_times_df.rename(columns={"model_node_id": "stop_id"})
    return wr_stop_times_df

def create_feed_frequencies(  # noqa: PLR0915
    feed_tables: dict[str, pd.DataFrame],
    timeperiods: dict[str, tuple[TimeString, TimeString]],
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
        msg = f"frequency_method must be one of {VALID_FREQUENCY_METHOD}; received {frequency_method}"
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
