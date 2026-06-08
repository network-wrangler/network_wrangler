"""Functions for editing the transit route shapes and stop patterns."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from ...configs import DefaultConfig
from ...errors import TransitRoutingChangeError
from ...logger import WranglerLogger
from ...models.gtfs.tables import (
    WranglerShapesTable,
    WranglerStopsTable,
    WranglerStopTimesTable,
    WranglerTripsTable,
)
from ...utils.data import concat_with_attr, segment_data_by_selection_min_overlap
from ...utils.ids import generate_new_id_from_existing
from ...utils.models import validate_df_to_model
from ..feed.shapes import (
    find_nearest_stops,
    node_pattern_for_shape_id,
    shape_ids_for_trip_ids,
    shapes_for_shape_id,
)
from ..feed.stop_times import stop_times_for_trip_id
from ..feed.stops import node_is_stop
from ..feed.trips import trip_ids_for_shape_id
from ..validate import (
    shape_links_without_road_links,
)

if TYPE_CHECKING:
    from ...roadway.network import RoadwayNetwork
    from ..feed.feed import Feed
    from ..network import TransitNetwork
    from ..selection import TransitSelection


def _create_stop_times(
    set_stops_node_ids: list[int], trip_id: str, project_name: str | None = None
) -> DataFrame[WranglerStopTimesTable]:
    """Modifies a list of nodes from project card routing key to a shape dataframe.

    Args:
        set_stops_node_ids: List of roadway node ids from project card to stop at.
        trip_id: trip_id to associate with the new shape
        project_name: Name of the project. Defaults to None.

    """
    WranglerLogger.debug(f"Creating new stop times for trip: {trip_id}")

    new_stoptime_rows = pd.DataFrame(
        {
            "trip_id": trip_id,
            "stop_id": set_stops_node_ids,
            "stop_sequence": np.arange(len(set_stops_node_ids)),
        }
    )
    if project_name is not None:
        new_stoptime_rows["projects"] = f"{project_name},"
    new_stoptime_rows = validate_df_to_model(new_stoptime_rows, WranglerStopTimesTable)
    return new_stoptime_rows


def _create_shapes(
    nodes_list: list[int],
    shape_id: str,
    road_net: RoadwayNetwork,
    project_name: str | None = None,
) -> DataFrame[WranglerShapesTable]:
    """Modifies a list of nodes from project card routing key to rows in a shapes.txt dataframe.

    Args:
        nodes_list (list[Union[str,int]]): List of nodes where stops are either integers with
            positive integers representing stops and negative integers representing waypoints
            or strings. e.g. ["-123","321",...] or [-123,321]
        shape_id: shape_id to associate with the new shape
        road_net: reference roadway network
        project_name: Name of the project. Defaults to None.

    Returns:
        pd.DataFrame: shapes.txt rows
    """
    WranglerLogger.debug(f"Creating new shape for shape_id: {shape_id}")
    new_shape_rows_df = pd.DataFrame(
        {
            "shape_id": shape_id,
            "shape_model_node_id": nodes_list,
            "shape_pt_sequence": np.arange(len(nodes_list)),
        }
    )

    missing_links = shape_links_without_road_links(new_shape_rows_df, road_net.links_df)
    if len(missing_links):
        WranglerLogger.error(f"!!! New shape links not in road links: \n {new_shape_rows_df}")
        # TODO: add a way to add missing links to the road network

    nodes_df = road_net.nodes_df[["model_node_id", "X", "Y"]]
    new_shape_rows_df = new_shape_rows_df.merge(
        nodes_df, how="left", left_on="shape_model_node_id", right_on="model_node_id"
    )
    new_shape_rows_df = new_shape_rows_df.rename(
        columns={
            "Y": "shape_pt_lat",
            "X": "shape_pt_lon",
        }
    )
    if project_name is not None:
        new_shape_rows_df["projects"] = f"{project_name},"
    new_shape_rows_df = new_shape_rows_df.drop(columns=["model_node_id"])
    # WranglerLogger.debug(f"New Shape Rows: \n{new_shape_rows_df.iloc[DISP_RECS]}")

    return new_shape_rows_df


def _add_new_shape_copy(
    old_shape_id: str,
    trip_ids: list[str],
    feed: Feed,
    id_scalar: int = DefaultConfig.IDS.TRANSIT_SHAPE_ID_SCALAR,
    project_name: str | None = None,
) -> tuple[DataFrame[WranglerShapesTable], DataFrame[WranglerTripsTable], str]:
    """Create an identical new shape_id from shape matching old_shape_id for the trip_ids.

    Args:
        old_shape_id (str): ID of shape to copy
        trip_ids (list[str]): list of trip IDS to associate with new shape
        feed: Input transit feed
        id_scalar (int): scalar value to add to old_shape_id to create new_shape_id. Defaults to
            SHAPE_ID_SCALAR.
        project_name (str, optional): Name of the project. Defaults to None.

    Returns:
        tuple[pd.DataFrame]: updated (shapes_df, trips_df, new_shape_id)
    """
    WranglerLogger.debug(
        f"Adding a new shape for shape_id: {old_shape_id} using scalar: {id_scalar}"
    )
    shapes = copy.deepcopy(feed.shapes)
    trips = copy.deepcopy(feed.trips)
    new_shape = copy.deepcopy(shapes[shapes.shape_id == old_shape_id])
    new_shape_id = generate_new_id_from_existing(old_shape_id, shapes["shape_id"], id_scalar)
    new_shape["shape_id"] = new_shape_id

    if project_name is not None:
        new_shape["projects"] = f"{project_name},"
    shapes = concat_with_attr([shapes, new_shape], ignore_index=True)

    trips.loc[trips.trip_id.isin(trip_ids), "shape_id"] = new_shape_id
    return shapes, trips, new_shape_id


def _replace_shapes_segment(
    routing_to_replace: list[int],
    shape_id: str,
    set_routing: list[int],
    feed: Feed,
    road_net: RoadwayNetwork,
    project_name: str | None = None,
) -> DataFrame[WranglerShapesTable]:
    """Returns shapes with a replaced segment for a given shape_id.

    Segment to replace is defined by existing_routing but will be updated based on:
    1. Expanding to shape start or shape end if it is the first or last stop respectfully
    2. Shrinking if replacement segment has overlap with existing_routing so that existing
        data can be preserved.

    Args:
        routing_to_replace: list of depicting start and end node ids for segment to replace
        shape_id: shape_id to be modified.
        set_routing (list): list of node ids to replace existing routing with
        feed: Feed object
        road_net: Reference roadway network
        project_name: Name of the project. Defaults to None.

    Returns:
        pd.DataFrame: Updated shape records
    """
    routing_to_replace = [int(abs(int(i))) for i in routing_to_replace]

    existing_shape_df = shapes_for_shape_id(feed.shapes, shape_id)

    _disp_col = ["shape_id", "shape_pt_sequence", "shape_model_node_id"]
    # WranglerLogger.debug(f"\nExisting Shape\n{existing_shape_df[_disp_col]}")

    (
        set_routing,
        (
            before_segment,
            _,
            after_segment,
        ),
    ) = segment_data_by_selection_min_overlap(
        routing_to_replace,
        existing_shape_df,
        "shape_model_node_id",
        set_routing,
    )
    # Create new segment
    updated_segment_shapes_df = _create_shapes(
        set_routing, shape_id, road_net, project_name=project_name
    )

    msg = f"\nShapes Segments: \nBefore: \n{before_segment[_disp_col]}\
                         \nReplacement: \n{updated_segment_shapes_df[_disp_col]}\
                         \nAfter: \n{after_segment[_disp_col]}"
    # WranglerLogger.debug(msg)

    # Only concatenate those that aren't empty bc NaN values will transfer integers to floats.
    dfs = [before_segment, updated_segment_shapes_df, after_segment]
    concat_dfs = [df for df in dfs if not df.empty]

    updated_shape = concat_with_attr(concat_dfs, ignore_index=True, sort=False)

    # Update shape_pt_sequence based on combined shape_df
    updated_shape["shape_pt_sequence"] = np.arange(len(updated_shape)) + 1

    # WranglerLogger.debug(f"\nShape w/Segment Replaced: \n {updated_shape[_disp_col]}")
    return updated_shape


def _replace_stop_times_segment_for_trip(
    existing_stop_nodes: list[int],
    trip_id: str,
    set_stops_nodes: list[int],
    feed: Feed,
    project_name: str | None = None,
) -> DataFrame[WranglerStopTimesTable]:
    """Replaces a segment of a specific set of stop_time records with the same shape_id.

    Args:
        existing_stop_nodes: list of roadway node ids for the existing segment to replace
        trip_id: selected trip_id to update
        set_stops_nodes: list of roadway node ids to make stops
        feed: transit feed
        project_name: Name of the project. Defaults to None.

    Returns:
        WranglerStopTimesTable: stop_time records for a trip_id with updated segment
    """
    # WranglerLogger.debug(f"Replacing existing nodes pattern: {existing_stop_nodes}")
    this_trip_stoptimes = stop_times_for_trip_id(feed.stop_times, trip_id)

    _disp_col = ["stop_id", "stop_sequence"]

    # if start or end nodes are not in stops, get closest ones from shapes
    start_n, end_n = abs(int(existing_stop_nodes[0])), abs(int(existing_stop_nodes[-1]))
    if start_n != 0 and not node_is_stop(feed.stops, feed.stop_times, start_n, trip_id):
        start_n, _ = find_nearest_stops(feed.shapes, feed.trips, feed.stop_times, trip_id, start_n)
        if start_n != 0:
            set_stops_nodes = [start_n, *set_stops_nodes]
    if end_n != 0 and not node_is_stop(feed.stops, feed.stop_times, end_n, trip_id):
        _, end_n = find_nearest_stops(feed.shapes, feed.trips, feed.stop_times, trip_id, end_n)
        if end_n != 0:
            set_stops_nodes = [*set_stops_nodes, end_n]

    WranglerLogger.debug(f"Start/End nodes w/stops: {start_n}/{end_n}")
    WranglerLogger.debug(f"Set stop nodes: {set_stops_nodes}")

    (
        set_stops_nodes,
        (
            before_segment,
            _,
            after_segment,
        ),
    ) = segment_data_by_selection_min_overlap(
        [start_n, end_n],
        this_trip_stoptimes,
        "stop_id",
        set_stops_nodes,
    )

    # Create new segment
    segment_stoptime_rows = _create_stop_times(set_stops_nodes, trip_id, project_name=project_name)
    # WranglerLogger.debug(f"Before Segment: \n{before_segment[_disp_col]}")
    # WranglerLogger.debug(f"Segment: \n{segment_stoptime_rows[_disp_col]}")
    # WranglerLogger.debug(f"After Segment: \n{after_segment[_disp_col]}")

    # Concatenate the dataframes

    # Only concatenate those that aren't empty bc NaN values will transfer integers to floats.
    dfs = [before_segment, segment_stoptime_rows, after_segment]
    concat_dfs = [df for df in dfs if not df.empty]

    updated_this_trip_stop_times = concat_with_attr(
        concat_dfs,
        ignore_index=True,
        sort=False,
    )

    updated_this_trip_stop_times["stop_sequence"] = (
        np.arange(len(updated_this_trip_stop_times)) + 1
    )
    # WranglerLogger.debug(f"Updated Stoptimes: \n{updated_this_trip_stop_times[_disp_col]}")
    return updated_this_trip_stop_times


def _consistent_routing(
    feed: Feed, shape_id: str, existing_routing: list[int], set_routing: list[int]
) -> bool:
    """Check if the routing is consistent with the existing routing."""
    # WranglerLogger.debug(f"Checking if routing is consistent for shape_id: {shape_id}")

    if not existing_routing:
        return False
    existing_pattern = node_pattern_for_shape_id(feed.shapes, shape_id)
    # WranglerLogger.debug(f"Existing pattern: {existing_pattern}")
    # WranglerLogger.debug(f"Existing routing: {existing_routing}")
    # WranglerLogger.debug(f"Set routing: {set_routing}")
    same_extents = (set_routing[0], set_routing[-1]) == (
        existing_routing[0],
        existing_routing[-1],
    )
    if not same_extents:
        return False
    same_route = "|".join(map(str, set_routing)) in "|".join(map(str, existing_pattern))
    return bool(same_route)


def _update_shapes_and_trips(
    feed: Feed,
    shape_id: str,
    trip_ids: list[str],
    routing_set: list[int],
    shape_id_scalar: int,
    road_net: RoadwayNetwork,
    routing_existing: list[int] | None = None,
    project_name: str | None = None,
) -> tuple[DataFrame[WranglerShapesTable], DataFrame[WranglerTripsTable]]:
    """Update shapes and trips for transit routing change.

    Args:
        feed: feed we are updating
        shape_id : shape id to update
        trip_ids: selected trip_ids to update
        routing_set: routing extents to replace as a list of model_node_ids
        routing_existing: existing routing extents to replace as a list of model_node_ids
        shape_id_scalar: scalar value to use when creating new shape_ids
        road_net: Reference roadway network to make sure shapes follow real links
        project_name: Name of the project. Defaults to None.

    Returns:
        Updated shapes and trips dataframes
    """
    WranglerLogger.debug(f"Updating shapes and trips for shape_id: {shape_id}")
    if routing_existing is None:
        routing_existing = []
    set_routing = [int(abs(int(i))) for i in routing_set]
    existing_routing = [int(abs(int(i))) for i in routing_existing]

    # ----- Don't need a new shape if its only the stops that change -----
    if _consistent_routing(feed, shape_id, existing_routing, set_routing):
        WranglerLogger.debug("No routing change, returning shapes and trips as-is.")
        return feed.shapes, feed.trips

    # --- Create new shape if `shape_id` is used by trips that are not in selected trip_ids --
    all_trips_using_shape_id = set(trip_ids_for_shape_id(feed.trips, shape_id))
    sel_trips_using_shape_id = set(trip_ids) & all_trips_using_shape_id
    if sel_trips_using_shape_id != all_trips_using_shape_id:
        # adds copied shape with new shape_id to feed.shapes + references it in feed.trips
        feed.shapes, feed.trips, shape_id = _add_new_shape_copy(
            old_shape_id=shape_id,
            trip_ids=list(sel_trips_using_shape_id),
            feed=feed,
            id_scalar=shape_id_scalar,
            project_name=project_name,
        )

    # If "existing" is specified, replace only that segment else, replace the whole thing
    if existing_routing:
        this_shape = _replace_shapes_segment(
            existing_routing, shape_id, set_routing, feed, road_net, project_name=project_name
        )
    else:
        this_shape = _create_shapes(set_routing, shape_id, road_net, project_name=project_name)

    # Add rows back into shapes
    unselected_shapes = feed.shapes[feed.shapes.shape_id != shape_id]
    feed.shapes = concat_with_attr(
        [unselected_shapes, this_shape],
        ignore_index=True,
        sort=False,
    )

    return feed.shapes, feed.trips


def _update_stops(
    feed: Feed,
    routing_set: list[int],
    road_net: RoadwayNetwork,
    project_name: str | None = None,
) -> DataFrame[WranglerStopsTable]:
    """Update stops for transit routing change.

    Args:
        feed: Feed object
        routing_set: List of model_node_ids to be stops
        road_net: Reference roadway network
        project_name: Name of the project. Defaults to None.

    Returns:
        pd.DataFrame: Updated stops.txt
    """
    set_stops_node_ids = [int(i) for i in routing_set if int(i) > 0]
    trn_net_stop_nodes = feed.stops["stop_id"].tolist()

    # Check if all stops are already in stops.txt and return as-is if so.
    missing_stops_node_ids = list(set(set_stops_node_ids) - set(trn_net_stop_nodes))
    if not missing_stops_node_ids:
        WranglerLogger.debug("Skipping updating stops for transit routing change.")
        return feed.stops

    WranglerLogger.debug("Updating stops for transit routing change.")

    # Create new stop records
    new_stops = pd.DataFrame(
        {
            "stop_id": missing_stops_node_ids,
        },
        index=range(len(missing_stops_node_ids)),
    )
    if project_name is not None:
        new_stops["projects"] = f"{project_name},"
    new_stops = new_stops.merge(
        road_net.nodes_df[["model_node_id", "X", "Y"]],
        how="left",
        left_on="stop_id",
        right_on="model_node_id",
    )
    new_stops = new_stops.drop(columns=["model_node_id"])
    new_stops = new_stops.rename(columns={"Y": "stop_lat", "X": "stop_lon"})
    stops = concat_with_attr([feed.stops, new_stops], ignore_index=True)
    return stops


def _delete_stop_times_for_nodes(
    trip_stoptimes: DataFrame[WranglerStopTimesTable], del_stops_nodes: list[int]
) -> DataFrame[WranglerStopTimesTable]:
    """Delete stop_times for specific nodes for a specific trip.

    Args:
        trip_stoptimes: stop_times for the trip
        del_stops_nodes: list of model_node_ids to delete

    Returns:
        WranglerStopTimesTable: Updated stop_times for the trip
    """
    WranglerLogger.debug(f"Deleting stop times for nodes: {del_stops_nodes}")
    trip_stoptimes = trip_stoptimes[~trip_stoptimes.stop_id.isin(del_stops_nodes)].copy()
    trip_stoptimes["stop_sequence"] = np.arange(len(trip_stoptimes)) + 1
    return trip_stoptimes


def _deletion_candidates(routing_set: list[int]) -> list[int]:
    """Identify stops that are at the beginning or end of the segment to be replaced."""
    first, last = int(routing_set[0]), int(routing_set[-1])
    deletion_candidate_nodes = []
    if first < 0:
        deletion_candidate_nodes.append(abs(first))
    if last < 0 and last != first:
        deletion_candidate_nodes.append(abs(last))

    return deletion_candidate_nodes


def _update_stop_times_for_trip(
    feed: Feed,
    trip_id: str,
    routing_set: list[int],
    routing_existing: list[int],
    project_name: str | None = None,
) -> DataFrame[WranglerStopTimesTable]:
    """Update stop_times for a specific trip with new stop_times.

    Args:
        feed: Feed object
        trip_id: trip_id to update
        routing_set: List of model_node_ids to be stops
        routing_existing: List of model_node_ids to replace
        project_name: Name of the project. Defaults to None.

    Returns:
        WranglerStopTimesTable: Updated stop_times.txt
    """
    WranglerLogger.debug(f"Updating stop times for trip: {trip_id}")

    existing_stops_nodes = [int(i) for i in routing_existing]
    set_stops_nodes = [int(i) for i in routing_set if int(i) > 0]
    del_stops_nodes = _deletion_candidates(routing_set)
    # WranglerLogger.debug(f"Existing stops: {existing_stops_nodes}")
    # WranglerLogger.debug(f"Set stops: {set_stops_nodes}")
    # WranglerLogger.debug(f"Delete stops: {del_stops_nodes}")

    # --------------- replace segment, delete stops, or replace whole thing ---------------
    this_trip_stop_times = stop_times_for_trip_id(feed.stop_times, trip_id)

    if existing_stops_nodes and set_stops_nodes:
        this_trip_stop_times = _replace_stop_times_segment_for_trip(
            existing_stops_nodes,
            trip_id,
            set_stops_nodes,
            feed,
            project_name=project_name,
        )

    if del_stops_nodes:
        this_trip_stop_times = _delete_stop_times_for_nodes(
            this_trip_stop_times,
            del_stops_nodes,
        )

    if not existing_stops_nodes:
        this_trip_stop_times = _create_stop_times(
            set_stops_nodes, trip_id, project_name=project_name
        )

    # --------------- Replace stop_times for this trip with updated stop times ---------------
    stop_times_not_this_trip = feed.stop_times[feed.stop_times.trip_id != trip_id]
    stop_times = concat_with_attr(
        [stop_times_not_this_trip, this_trip_stop_times],
        ignore_index=True,
        sort=False,
    )
    _show_col = [
        "trip_id",
        "stop_id",
        "stop_sequence",
        "departure_time",
        "arrival_time",
    ]
    msg = f"ST for trip: {stop_times.loc[stop_times.trip_id == trip_id, _show_col]}"
    # WranglerLogger.debug(msg)

    return stop_times


def apply_transit_routing_change(
    net: TransitNetwork,
    selection: TransitSelection,
    routing_change: dict,
    reference_road_net: RoadwayNetwork | None = None,
    project_name: str | None = None,
) -> TransitNetwork:
    """Apply a routing change to the transit network, including stop updates.

    Args:
        net (TransitNetwork): TransitNetwork object to apply routing change to.
        selection (Selection): TransitSelection object, created from a selection dictionary.
        routing_change (dict): Routing Change dictionary, e.g.
            ```python
            {
                "existing": [46665, 150855],
                "set": [-46665, 150855, 46665, 150855],
            }
            ```
        shape_id_scalar (int, optional): Initial scalar value to add to duplicated shape_ids to
            create a new shape_id. Defaults to SHAPE_ID_SCALAR.
        reference_road_net (RoadwayNetwork, optional): Reference roadway network to use for
            updating shapes and stops. Defaults to None.
        project_name (str, optional): Name of the project. Defaults to None.
    """
    WranglerLogger.debug("Applying transit routing change project.")
    WranglerLogger.debug(f"...selection: {selection.selection_dict}")
    WranglerLogger.debug(f"...routing: {routing_change}")

    # ---- Secure all inputs needed --------------
    updated_feed = copy.deepcopy(net.feed)
    trip_ids = selection.selected_trips
    if project_name:
        updated_feed.trips.loc[updated_feed.trips.trip_id.isin(trip_ids), "projects"] += (
            f"{project_name},"
        )

    road_net = net.road_net if reference_road_net is None else reference_road_net
    if road_net is None:
        WranglerLogger.error(
            "! Must have a reference road network set in order to update transit \
                         routin.  Either provide as an input to this function or set it for the \
                         transit network: >> transit_net.road_net = ..."
        )
        msg = "Must have a reference road network set in order to update transit routing."
        raise TransitRoutingChangeError(msg)

    # ---- update each shape that is used by selected trips to use new routing -------
    shape_ids = shape_ids_for_trip_ids(updated_feed.trips, trip_ids)
    # WranglerLogger.debug(f"shape_ids: {shape_ids}")
    for shape_id in shape_ids:
        updated_feed.shapes, updated_feed.trips = _update_shapes_and_trips(
            updated_feed,
            shape_id,
            trip_ids,
            routing_change["set"],
            net.config.IDS.TRANSIT_SHAPE_ID_SCALAR,
            road_net,
            routing_existing=routing_change.get("existing", []),
            project_name=project_name,
        )
    # WranglerLogger.debug(f"updated_feed.shapes: \n{updated_feed.shapes}")
    # WranglerLogger.debug(f"updated_feed.trips: \n{updated_feed.trips}")
    # ---- Check if any stops need adding to stops.txt and add if they do ----------
    updated_feed.stops = _update_stops(
        updated_feed, routing_change["set"], road_net, project_name=project_name
    )
    # WranglerLogger.debug(f"updated_feed.stops: \n{updated_feed.stops}")
    # ---- Update stop_times --------------------------------------------------------
    for trip_id in trip_ids:
        updated_feed.stop_times = _update_stop_times_for_trip(
            updated_feed,
            trip_id,
            routing_change["set"],
            routing_change.get("existing", []),
        )

    # ---- Check result -------------------------------------------------------------
    _show_col = [
        "trip_id",
        "stop_id",
        "stop_sequence",
        "departure_time",
        "arrival_time",
    ]
    _ex_stoptimes = updated_feed.stop_times.loc[
        updated_feed.stop_times.trip_id == trip_ids[0], _show_col
    ]
    # WranglerLogger.debug(f"stop_times for first updated trip: \n {_ex_stoptimes}")

    # ---- Update transit network with updated feed.
    net.feed = updated_feed
    # WranglerLogger.debug(f"net.feed.stops: \n {net.feed.stops}")
    return net
