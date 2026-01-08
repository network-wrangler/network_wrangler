"""Functions to filter transit feeds by various criteria.

Filtered transit feeds are subsets of the original feed based on selection criteria
like service_ids, route_types, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import geopandas as gpd
import pandas as pd

from ..logger import WranglerLogger
from ..models.gtfs.gtfs import GtfsModel
from ..models.gtfs.types import RouteType
from ..params import LAT_LON_CRS
from .feed.feed import Feed

MIN_ROUTE_SEGMENTS = 2
"""Minimum number of boundary segments before warning about complex route patterns.

Used in filter_transit_by_boundary() to detect routes that exit and re-enter the boundary.
"""

MAX_TRUNCATION_WARNING_STOPS = 10
"""Maximum number of removed stops to list individually in truncation warnings.

Used in truncate_route_at_stop() to control verbosity of warning messages.
If more stops are removed, only the count is shown instead of listing each stop.
"""


def filter_feed_by_service_ids(
    feed: Union[Feed, GtfsModel],
    service_ids: list[str],
) -> Union[Feed, GtfsModel]:
    """Filter a transit feed to only include trips for specified service_ids.
    
    Filters trips, stop_times, and stops to only include data related to the
    specified service_ids. Also ensures parent stations are retained if referenced.
    
    Args:
        feed: Feed or GtfsModel object to filter
        service_ids: List of service_ids to retain
        
    Returns:
        Union[Feed, GtfsModel]: Filtered copy of feed with only trips/stops/stop_times 
            for specified service_ids. Returns same type as input.
    """
    WranglerLogger.info(f"Filtering feed to {len(service_ids):,} service_ids")
    
    # Remember the input type to return the same type
    is_feed = isinstance(feed, Feed)
    
    # Extract dataframes to work with them directly (avoiding validation during filtering)
    feed_dfs = {}
    for table_name in feed.table_names:
        if hasattr(feed, table_name) and getattr(feed, table_name) is not None:
            feed_dfs[table_name] = getattr(feed, table_name).copy()
    
    # Filter trips for these service_ids
    original_trip_count = len(feed_dfs["trips"])
    feed_dfs["trips"]["service_id"] = feed_dfs["trips"]["service_id"].astype(str)
    
    # Create a DataFrame from the list for merging
    service_ids_df = pd.DataFrame({"service_id": service_ids})
    feed_dfs["trips"] = feed_dfs["trips"].merge(
        right=service_ids_df, on="service_id", how="left", indicator=True
    )
    WranglerLogger.debug(
        f"trips._merge.value_counts():\n{feed_dfs['trips']._merge.value_counts()}"
    )
    feed_dfs["trips"] = (
        feed_dfs["trips"]
        .loc[feed_dfs["trips"]._merge == "both"]
        .drop(columns=["_merge"])
        .reset_index(drop=True)
    )
    WranglerLogger.info(
        f"Filtered trips from {original_trip_count:,} to {len(feed_dfs['trips']):,}"
    )
    
    # Filter stop_times for these trip_ids
    feed_dfs["trips"]["trip_id"] = feed_dfs["trips"]["trip_id"].astype(str)
    trip_ids = feed_dfs["trips"][["trip_id"]].drop_duplicates().reset_index(drop=True)
    WranglerLogger.debug(
        f"After filtering trips to trip_ids (len={len(trip_ids):,})"
    )
    
    feed_dfs["stop_times"]["trip_id"] = feed_dfs["stop_times"]["trip_id"].astype(str)
    feed_dfs["stop_times"] = feed_dfs["stop_times"].merge(
        right=trip_ids, how="left", indicator=True
    )
    WranglerLogger.debug(
        f"stop_times._merge.value_counts():\n{feed_dfs['stop_times']._merge.value_counts()}"
    )
    feed_dfs["stop_times"] = (
        feed_dfs["stop_times"]
        .loc[feed_dfs["stop_times"]._merge == "both"]
        .drop(columns=["_merge"])
        .reset_index(drop=True)
    )
    
    # Filter stops for these stop_ids
    feed_dfs["stop_times"]["stop_id"] = feed_dfs["stop_times"]["stop_id"].astype(str)
    stop_ids = feed_dfs["stop_times"][["stop_id"]].drop_duplicates().reset_index(drop=True)
    stop_ids_set = set(stop_ids["stop_id"])
    WranglerLogger.debug(f"After filtering stop_times to stop_ids (len={len(stop_ids):,})")
    
    feed_dfs["stops"]["stop_id"] = feed_dfs["stops"]["stop_id"].astype(str)
    
    # Identify parent stations that should be kept
    parent_stations_to_keep = set()
    if "parent_station" in feed_dfs["stops"].columns:
        # Find all parent stations referenced by stops that will be kept
        stops_to_keep = feed_dfs["stops"][feed_dfs["stops"]["stop_id"].isin(stop_ids_set)]
        parent_stations = stops_to_keep["parent_station"].dropna().unique()
        parent_stations_to_keep = set([ps for ps in parent_stations if ps != ""])
        
        if len(parent_stations_to_keep) > 0:
            WranglerLogger.info(
                f"Preserving {len(parent_stations_to_keep)} parent stations referenced by kept stops"
            )
    
    # Create combined set of stop_ids to keep (original stops + parent stations)
    all_stop_ids_to_keep = stop_ids_set | parent_stations_to_keep
    
    # Filter stops to include both regular stops and their parent stations
    original_stop_count = len(feed_dfs["stops"])
    feed_dfs["stops"] = feed_dfs["stops"][
        feed_dfs["stops"]["stop_id"].isin(all_stop_ids_to_keep)
    ].reset_index(drop=True)
    
    WranglerLogger.debug(
        f"Filtered stops from {original_stop_count:,} to {len(feed_dfs['stops']):,} "
        f"(including {len(parent_stations_to_keep)} parent stations)"
    )
    
    # Check for stop_times with invalid stop_ids after all filtering is complete
    valid_stop_ids = set(feed_dfs["stops"]["stop_id"])
    invalid_mask = ~feed_dfs["stop_times"]["stop_id"].isin(valid_stop_ids)
    invalid_stop_times = feed_dfs["stop_times"][invalid_mask]
    
    if len(invalid_stop_times) > 0:
        WranglerLogger.warning(
            f"Found {len(invalid_stop_times):,} stop_times entries with invalid stop_ids after filtering"
        )
        
        # Join with trips to get route_id
        invalid_with_routes = invalid_stop_times.merge(
            feed_dfs["trips"][["trip_id", "route_id"]], on="trip_id", how="left"
        )
        
        # Log unique invalid stop_ids
        invalid_stop_ids = invalid_stop_times["stop_id"].unique()
        WranglerLogger.warning(
            f"Invalid stop_ids ({len(invalid_stop_ids)} unique): {invalid_stop_ids.tolist()}"
        )
        
        # Log sample of invalid entries with trip and route context
        sample_invalid = invalid_with_routes.head(10)
        WranglerLogger.warning(
            f"Sample invalid stop_times entries:\n{sample_invalid[['trip_id', 'route_id', 'stop_id', 'stop_sequence']]}"
        )
        
        # Log summary by route
        route_summary = (
            invalid_with_routes.groupby("route_id")["stop_id"]
            .agg(["count", "nunique"])
            .sort_values("count", ascending=False)
        )
        route_summary.columns = ["invalid_stop_times_count", "unique_invalid_stops"]
        WranglerLogger.warning(
            f"Invalid stop_times by route (top 20):\n{route_summary.head(20)}"
        )
        
        WranglerLogger.debug(
            f"All invalid stop_times entries with routes:\n{invalid_with_routes}"
        )
    else:
        WranglerLogger.info(
            "All stop_times entries have valid stop_ids after filtering"
        )
    
    # Filter other tables to match filtered trips
    if "shapes" in feed_dfs:
        shape_ids = feed_dfs["trips"]["shape_id"].dropna().unique()
        feed_dfs["shapes"] = feed_dfs["shapes"][
            feed_dfs["shapes"]["shape_id"].isin(shape_ids)
        ].reset_index(drop=True)
        WranglerLogger.debug(f"Filtered shapes to {len(feed_dfs['shapes']):,} records")
    
    if "routes" in feed_dfs:
        route_ids = feed_dfs["trips"]["route_id"].unique()
        feed_dfs["routes"] = feed_dfs["routes"][
            feed_dfs["routes"]["route_id"].isin(route_ids)
        ].reset_index(drop=True)
        WranglerLogger.debug(f"Filtered routes to {len(feed_dfs['routes']):,} records")
    
    # Feed has frequencies, GtfsModel doesn't
    if is_feed and "frequencies" in feed_dfs:
        feed_dfs["frequencies"]["trip_id"] = feed_dfs["frequencies"]["trip_id"].astype(str)
        feed_dfs["frequencies"] = feed_dfs["frequencies"][
            feed_dfs["frequencies"]["trip_id"].isin(feed_dfs["trips"]["trip_id"])
        ].reset_index(drop=True)
        WranglerLogger.debug(f"Filtered frequencies to {len(feed_dfs['frequencies']):,} records")
    
    # Create the appropriate object type with the filtered dataframes
    if is_feed:
        return Feed(**feed_dfs)
    else:
        return GtfsModel(**feed_dfs)


def filter_transit_by_boundary(  # noqa: PLR0912, PLR0915
    transit_data: Union[GtfsModel, Feed],
    boundary: Union[str, Path, gpd.GeoDataFrame],
    partially_include_route_type_action: Optional[dict[RouteType, str]] = None,
) -> None:
    """Filter transit routes based on whether they have stops within a boundary.

    Removes routes that are entirely outside the boundary shapefile. Routes that are
    partially within the boundary are kept by default, but can be configured per
    route type to be truncated at the boundary. Modifies transit_data in place.

    Args:
        transit_data: Either a GtfsModel or Feed object to filter. Modified in place.
        boundary: Path to boundary shapefile or a GeoDataFrame with boundary polygon(s)
        partially_include_route_type_action: Optional dictionary mapping RouteType enum to
            action for routes partially within boundary:
            - "truncate": Truncate route to only include stops within boundary
            Route types not specified in this dictionary will be kept entirely (default).

    Example:
        >>> from network_wrangler.models.gtfs.types import RouteType
        >>> # Remove routes entirely outside the Bay Area
        >>> filtered_gtfs = filter_transit_by_boundary(gtfs_model, "bay_area_boundary.shp")
        >>> # Truncate rail routes at boundary, keep other route types unchanged
        >>> filtered_gtfs = filter_transit_by_boundary(
        ...     gtfs_model,
        ...     "bay_area_boundary.shp",
        ...     partially_include_route_type_action={
        ...         RouteType.RAIL: "truncate",  # Rail - will be truncated at boundary
        ...         # Other route types not listed will be kept entirely
        ...     },
        ... )

    !!! todo
        This is similar to [`clip_feed_to_boundary`][network_wrangler.transit.clip.clip_feed_to_boundary] -- consolidate?

    """
    WranglerLogger.info("Filtering transit routes by boundary")

    # Log input parameters
    WranglerLogger.debug(
        f"partially_include_route_type_action: {partially_include_route_type_action}"
    )

    # Load boundary if it's a file path
    if isinstance(boundary, (str, Path)):
        WranglerLogger.debug(f"Loading boundary from file: {boundary}")
        boundary_gdf = gpd.read_file(boundary)
    else:
        WranglerLogger.debug("Using provided boundary GeoDataFrame")
        boundary_gdf = boundary

    WranglerLogger.debug(f"Boundary has {len(boundary_gdf)} polygon(s)")

    # Ensure boundary is in a geographic CRS for spatial operations
    if boundary_gdf.crs is None:
        WranglerLogger.warning("Boundary has no CRS, assuming EPSG:4326")
        boundary_gdf = boundary_gdf.set_crs(LAT_LON_CRS)
    else:
        WranglerLogger.debug(f"Boundary CRS: {boundary_gdf.crs}")

    # Get references to tables (not copies since we'll modify in place)
    is_gtfs = isinstance(transit_data, GtfsModel)
    stops_df = transit_data.stops
    routes_df = transit_data.routes
    trips_df = transit_data.trips
    stop_times_df = transit_data.stop_times

    if is_gtfs:
        WranglerLogger.debug("Processing GtfsModel data")
    else:
        WranglerLogger.debug("Processing Feed data")

    WranglerLogger.debug(
        f"Input data has {len(stops_df)} stops, {len(routes_df)} routes, {len(trips_df)} trips, {len(stop_times_df)} stop_times"
    )

    # Create GeoDataFrame from stops
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
        crs=LAT_LON_CRS,
    )

    # Reproject to match boundary CRS if needed
    if stops_gdf.crs != boundary_gdf.crs:
        WranglerLogger.debug(f"Reprojecting stops from {stops_gdf.crs} to {boundary_gdf.crs}")
        stops_gdf = stops_gdf.to_crs(boundary_gdf.crs)

    # Spatial join to find stops within boundary
    WranglerLogger.debug("Performing spatial join to find stops within boundary")
    stops_in_boundary = gpd.sjoin(stops_gdf, boundary_gdf, how="inner", predicate="within")
    stops_in_boundary_ids = set(stops_in_boundary.stop_id.unique())

    # Log some stops that are outside boundary for debugging
    stops_outside_boundary = set(stops_df.stop_id) - stops_in_boundary_ids
    if stops_outside_boundary:
        sample_outside = list(stops_outside_boundary)[:5]
        WranglerLogger.debug(f"Sample of stops outside boundary: {sample_outside}")

    WranglerLogger.info(
        f"Found {len(stops_in_boundary_ids):,} stops within boundary "
        f"out of {len(stops_df):,} total stops"
    )

    # Find which routes to keep
    # Get unique stop-route pairs from stop_times and trips
    stop_route_pairs = pd.merge(
        stop_times_df[["trip_id", "stop_id"]], trips_df[["trip_id", "route_id"]], on="trip_id"
    )[["stop_id", "route_id"]].drop_duplicates()

    # Group by route to find which stops each route serves
    route_stops = stop_route_pairs.groupby("route_id")["stop_id"].apply(set).reset_index()
    route_stops.columns = ["route_id", "stop_ids"]

    # Add route_type information
    route_stops = pd.merge(
        route_stops, routes_df[["route_id", "route_type"]], on="route_id", how="left"
    )

    # Initialize with default filters
    if partially_include_route_type_action is None:
        partially_include_route_type_action = {}

    # Convert RouteType enum keys to int values for comparison with dataframe
    normalized_route_type_action = {}
    for key, value in partially_include_route_type_action.items():
        if not isinstance(key, RouteType):
            msg = f"Keys in partially_include_route_type_action must be RouteType enum, got {type(key)}"
            raise TypeError(msg)
        normalized_route_type_action[key.value] = value
    partially_include_route_type_action = normalized_route_type_action

    # Track routes to truncate
    routes_to_truncate = {}

    # Determine which routes to keep and how to handle them
    def determine_route_handling(row):
        route_id = row["route_id"]
        route_type = row["route_type"]
        stop_ids = row["stop_ids"]

        # Check if route has stops both inside and outside boundary
        stops_inside = stop_ids.intersection(stops_in_boundary_ids)
        stops_outside = stop_ids - stops_in_boundary_ids

        # If all stops are outside, always remove
        if len(stops_inside) == 0:
            WranglerLogger.debug(
                f"Route {route_id} (type {route_type}): all {len(stop_ids)} stops outside boundary - REMOVE"
            )
            return "remove"

        # If all stops are inside, always keep
        if len(stops_outside) == 0:
            return "keep"

        # Route has stops both inside and outside - check partially_include_route_type_action
        WranglerLogger.debug(
            f"Route {route_id} (type {route_type}): {len(stops_inside)} stops inside, "
            f"{len(stops_outside)} stops outside boundary"
        )

        if route_type in partially_include_route_type_action:
            action = partially_include_route_type_action[route_type]
            WranglerLogger.debug(
                f"  - Applying configured action for route_type {route_type}: {action}"
            )
            if action == "truncate":
                return "truncate"

        # Default to keep if not specified
        WranglerLogger.debug(
            f"  - No action configured for route_type {route_type}, defaulting to KEEP"
        )
        return "keep"

    route_stops["handling"] = route_stops.apply(determine_route_handling, axis=1)
    WranglerLogger.debug(f"route_stops with handling set:\n{route_stops}")

    routes_to_keep = set(
        route_stops[route_stops["handling"].isin(["keep", "truncate"])]["route_id"]
    )
    routes_to_remove = set(route_stops[route_stops["handling"] == "remove"]["route_id"])
    routes_needing_truncation = set(route_stops[route_stops["handling"] == "truncate"]["route_id"])

    WranglerLogger.info(
        f"Keeping {len(routes_to_keep):,} routes out of {len(routes_df):,} total routes"
    )

    if routes_to_remove:
        WranglerLogger.info(f"Removing {len(routes_to_remove):,} routes entirely outside boundary")
        WranglerLogger.debug(f"Routes being removed: {sorted(routes_to_remove)[:10]}...")

    if routes_needing_truncation:
        WranglerLogger.info(f"Truncating {len(routes_needing_truncation):,} routes at boundary")
        WranglerLogger.debug(
            f"Routes being truncated: {sorted(routes_needing_truncation)[:10]}..."
        )

    # Filter data
    filtered_routes = routes_df[routes_df.route_id.isin(routes_to_keep)]
    filtered_trips = trips_df[trips_df.route_id.isin(routes_to_keep)]
    filtered_trip_ids = set(filtered_trips.trip_id)

    # Handle truncation by calling truncate_route_at_stop for each route needing truncation
    if routes_needing_truncation:
        WranglerLogger.debug(f"Processing truncation for {len(routes_needing_truncation)} routes")

        # Start with the current filtered data
        # Need to ensure stop_times only includes trips that are in filtered_trips
        filtered_stop_times_for_truncation = stop_times_df[
            stop_times_df.trip_id.isin(filtered_trip_ids)
        ]

        # First update transit_data with filtered data before truncation (in order to maintain validation)
        transit_data.stop_times = filtered_stop_times_for_truncation
        transit_data.trips = filtered_trips
        transit_data.routes = filtered_routes

        # Process each route that needs truncation
        for route_id in routes_needing_truncation:
            WranglerLogger.debug(f"Processing truncation for route {route_id}")

            # Get trips for this route
            route_trips = trips_df[trips_df.route_id == route_id]

            # Group by direction_id
            for direction_id in route_trips.direction_id.unique():
                dir_trips = route_trips[route_trips.direction_id == direction_id]
                if len(dir_trips) == 0:
                    continue

                # Analyze stop patterns for this route/direction
                # Get a representative trip (first one)
                sample_trip_id = dir_trips.iloc[0].trip_id
                sample_stop_times = transit_data.stop_times[
                    transit_data.stop_times.trip_id == sample_trip_id
                ].sort_values("stop_sequence")

                # Find which stops are inside/outside boundary
                stop_boundary_status = sample_stop_times["stop_id"].isin(stops_in_boundary_ids)

                # Check if route exits and re-enters boundary (complex case)
                boundary_changes = stop_boundary_status.ne(stop_boundary_status.shift()).cumsum()
                num_segments = boundary_changes.nunique()

                if num_segments > MIN_ROUTE_SEGMENTS:
                    # Complex case: route exits and re-enters boundary
                    route_info = routes_df[routes_df.route_id == route_id].iloc[0]
                    route_name = route_info.get("route_short_name", route_id)
                    msg = (
                        f"Route {route_name} ({route_id}) direction {direction_id} has a complex "
                        f"boundary crossing pattern (crosses boundary {num_segments - 1} times). "
                        f"Can only handle routes that exit boundary at beginning or end."
                    )
                    raise ValueError(msg)

                # Determine truncation type
                first_stop_inside = stop_boundary_status.iloc[0]
                last_stop_inside = stop_boundary_status.iloc[-1]

                if not first_stop_inside and not last_stop_inside:
                    # All stops outside - shouldn't happen as route would be removed
                    continue
                if first_stop_inside and last_stop_inside:
                    # All stops inside - no truncation needed
                    continue
                if not first_stop_inside and last_stop_inside:
                    # Starts outside, ends inside - truncate before first inside stop
                    # Find first True value (first stop inside boundary)
                    first_inside_pos = stop_boundary_status.tolist().index(True)
                    first_inside_stop = sample_stop_times.iloc[first_inside_pos]["stop_id"]

                    WranglerLogger.debug(
                        f"Route {route_id} dir {direction_id}: truncating before stop {first_inside_stop}"
                    )
                    truncate_route_at_stop(
                        transit_data, route_id, direction_id, first_inside_stop, "before"
                    )
                elif first_stop_inside and not last_stop_inside:
                    # Starts inside, ends outside - truncate after last inside stop
                    # Find last True value (last stop inside boundary)
                    reversed_list = stop_boundary_status.tolist()[::-1]
                    last_inside_pos = len(reversed_list) - 1 - reversed_list.index(True)
                    last_inside_stop = sample_stop_times.iloc[last_inside_pos]["stop_id"]

                    WranglerLogger.debug(
                        f"Route {route_id} dir {direction_id}: truncating after stop {last_inside_stop}"
                    )
                    truncate_route_at_stop(
                        transit_data, route_id, direction_id, last_inside_stop, "after"
                    )

        # After truncation, transit_data has been modified in place
        # Update references to current state (in order to maintain validation)
        filtered_stop_times = transit_data.stop_times
        filtered_trips = transit_data.trips
        filtered_routes = transit_data.routes
        filtered_stops = transit_data.stops
    else:
        # No truncation needed - update transit_data with filtered data
        filtered_stop_times = stop_times_df[stop_times_df.trip_id.isin(filtered_trip_ids)]
        filtered_stops = stops_df[stops_df.stop_id.isin(filtered_stop_times.stop_id.unique())]

        # Check if any of the filtered stops reference parent stations
        if "parent_station" in filtered_stops.columns:
            # Get parent stations that are referenced by kept stops
            parent_stations = filtered_stops["parent_station"].dropna().unique()
            parent_stations = [ps for ps in parent_stations if ps != ""]  # Remove empty strings

            if len(parent_stations) > 0:
                # Find parent stations that aren't already in our filtered stops
                existing_stop_ids = set(filtered_stops.stop_id)
                missing_parent_stations = [
                    ps for ps in parent_stations if ps not in existing_stop_ids
                ]

                if len(missing_parent_stations) > 0:
                    WranglerLogger.debug(
                        f"Adding back {len(missing_parent_stations)} parent stations referenced by kept stops"
                    )

                    # Get the parent station records
                    parent_station_records = stops_df[
                        stops_df.stop_id.isin(missing_parent_stations)
                    ]

                    # Append parent stations to filtered stops
                    filtered_stops = pd.concat(
                        [filtered_stops, parent_station_records], ignore_index=True
                    )

        transit_data.stop_times = filtered_stop_times
        transit_data.trips = filtered_trips
        transit_data.routes = filtered_routes
        transit_data.stops = filtered_stops

    # Log details about removed stops
    stops_still_used = set(filtered_stops.stop_id.unique())
    removed_stops = set(stops_df.stop_id) - stops_still_used
    if removed_stops:
        WranglerLogger.debug(f"Removed {len(removed_stops)} stops that are no longer referenced")

        # Get details of removed stops
        removed_stops_df = stops_df[stops_df["stop_id"].isin(removed_stops)][
            ["stop_id", "stop_name"]
        ]

        # Log up to 20 removed stops with their names
        sample_size = min(20, len(removed_stops_df))
        for _, stop in removed_stops_df.head(sample_size).iterrows():
            WranglerLogger.debug(f"  - Removed stop: {stop['stop_id']} ({stop['stop_name']})")

        if len(removed_stops) > sample_size:
            WranglerLogger.debug(f"  ... and {len(removed_stops) - sample_size} more stops")

    WranglerLogger.info(
        f"After filtering: {len(filtered_routes):,} routes, "
        f"{len(filtered_trips):,} trips, {len(filtered_stops):,} stops"
    )

    # Log summary of filtering by action type
    route_handling_summary = route_stops.groupby("handling").size()
    WranglerLogger.debug(f"Route handling summary:\n{route_handling_summary}")

    # Log route type distribution for routes with mixed stops
    mixed_routes = route_stops[
        (route_stops["handling"].isin(["keep", "truncate"]))
        & (
            route_stops["route_id"].isin(routes_needing_truncation) | route_stops["handling"]
            == "keep"
        )
    ]
    if len(mixed_routes) > 0:
        route_type_summary = mixed_routes.groupby("route_type")["handling"].value_counts()
        WranglerLogger.debug(f"Route types with partial stops:\n{route_type_summary}")

    # Update other tables in transit_data in place
    if is_gtfs:
        # For GtfsModel, also filter shapes and other tables if they exist
        if (hasattr(transit_data, "agency") and transit_data.agency is not None
            and "agency_id" in filtered_routes.columns):
                agency_ids = set(filtered_routes.agency_id.dropna().unique())
                transit_data.agency = transit_data.agency[
                    transit_data.agency.agency_id.isin(agency_ids)
                ]

        if (hasattr(transit_data, "shapes") and transit_data.shapes is not None
            and "shape_id" in filtered_trips.columns):
                shape_ids = set(filtered_trips.shape_id.dropna().unique())
                transit_data.shapes = transit_data.shapes[
                    transit_data.shapes.shape_id.isin(shape_ids)
                ]

        if hasattr(transit_data, "calendar") and transit_data.calendar is not None:
            # Keep only service_ids referenced by remaining trips
            service_ids = set(filtered_trips.service_id.unique())
            transit_data.calendar = transit_data.calendar[
                transit_data.calendar.service_id.isin(service_ids)
            ]

        if hasattr(transit_data, "calendar_dates") and transit_data.calendar_dates is not None:
            # Keep only service_ids referenced by remaining trips
            service_ids = set(filtered_trips.service_id.unique())
            transit_data.calendar_dates = transit_data.calendar_dates[
                transit_data.calendar_dates.service_id.isin(service_ids)
            ]

        if hasattr(transit_data, "frequencies") and transit_data.frequencies is not None:
            # Keep only frequencies for remaining trips
            transit_data.frequencies = transit_data.frequencies[
                transit_data.frequencies.trip_id.isin(filtered_trip_ids)
            ]

    else:  # Feed
        # For Feed, also handle frequencies and shapes
        if (hasattr(transit_data, "shapes") and transit_data.shapes is not None
            and "shape_id" in filtered_trips.columns):
                shape_ids = set(filtered_trips.shape_id.dropna().unique())
                transit_data.shapes = transit_data.shapes[
                    transit_data.shapes.shape_id.isin(shape_ids)
                ]

        if hasattr(transit_data, "frequencies") and transit_data.frequencies is not None:
            # Keep only frequencies for remaining trips
            transit_data.frequencies = transit_data.frequencies[
                transit_data.frequencies.trip_id.isin(filtered_trip_ids)
            ]


def drop_transit_agency(  # noqa: PLR0915
    transit_data: Union[GtfsModel, Feed],
    agency_id: Union[str, list[str]],
) -> None:
    """Remove all routes, trips, stops, etc. for a specific agency or agencies.

    Filters out all data associated with the specified agency_id(s), ensuring
    the resulting transit data remains valid by removing orphaned stops and
    maintaining referential integrity. Modifies transit_data in place.

    Args:
        transit_data: Either a GtfsModel or Feed object to filter. Modified in place.
        agency_id: Single agency_id string or list of agency_ids to remove

    Example:
        >>> # Remove a single agency
        >>> drop_transit_agency(gtfs_model, "SFMTA")
        >>> # Remove multiple agencies
        >>> drop_transit_agency(gtfs_model, ["SFMTA", "AC"])
    """
    # Convert single agency_id to list for uniform handling
    agency_ids_to_remove = [agency_id] if isinstance(agency_id, str) else agency_id

    WranglerLogger.info(f"Removing transit data for agency/agencies: {agency_ids_to_remove}")

    # Get data tables (references, not copies)
    routes_df = transit_data.routes
    trips_df = transit_data.trips
    stop_times_df = transit_data.stop_times
    stops_df = transit_data.stops
    is_gtfs = isinstance(transit_data, GtfsModel)

    # Find routes to keep (those NOT belonging to agencies being removed)
    if "agency_id" in routes_df.columns:
        routes_to_keep = routes_df[~routes_df.agency_id.isin(agency_ids_to_remove)]
        routes_removed = len(routes_df) - len(routes_to_keep)
    else:
        # If no agency_id column in routes, log warning and keep all routes
        WranglerLogger.warning(
            "No agency_id column found in routes table - cannot filter by agency"
        )
        routes_to_keep = routes_df
        routes_removed = 0

    route_ids_to_keep = set(routes_to_keep.route_id)

    # Filter trips based on remaining routes
    trips_to_keep = trips_df[trips_df.route_id.isin(route_ids_to_keep)]
    trips_removed = len(trips_df) - len(trips_to_keep)
    trip_ids_to_keep = set(trips_to_keep.trip_id)

    # Filter stop_times based on remaining trips
    stop_times_to_keep = stop_times_df[stop_times_df.trip_id.isin(trip_ids_to_keep)]
    stop_times_removed = len(stop_times_df) - len(stop_times_to_keep)

    # Find stops that are still referenced
    stops_still_used = set(stop_times_to_keep.stop_id.unique())
    stops_to_keep = stops_df[stops_df.stop_id.isin(stops_still_used)]

    # Check if any of these stops reference parent stations
    if "parent_station" in stops_to_keep.columns:
        # Get parent stations that are referenced by kept stops
        parent_stations = stops_to_keep["parent_station"].dropna().unique()
        parent_stations = [ps for ps in parent_stations if ps != ""]  # Remove empty strings

        if len(parent_stations) > 0:
            # Find parent stations that aren't already in our filtered stops
            existing_stop_ids = set(stops_to_keep.stop_id)
            missing_parent_stations = [ps for ps in parent_stations if ps not in existing_stop_ids]

            if len(missing_parent_stations) > 0:
                WranglerLogger.debug(
                    f"Adding back {len(missing_parent_stations)} parent stations referenced by kept stops"
                )

                # Get the parent station records
                parent_station_records = stops_df[stops_df.stop_id.isin(missing_parent_stations)]

                # Append parent stations to filtered stops
                stops_to_keep = pd.concat(
                    [stops_to_keep, parent_station_records], ignore_index=True
                )

    stops_removed = len(stops_df) - len(stops_to_keep)

    WranglerLogger.info(
        f"Removed: {routes_removed:,} routes, {trips_removed:,} trips, "
        f"{stop_times_removed:,} stop_times, {stops_removed:,} stops"
    )

    WranglerLogger.info(
        f"Remaining: {len(routes_to_keep):,} routes, {len(trips_to_keep):,} trips, "
        f"{len(stops_to_keep):,} stops"
    )
    WranglerLogger.debug(
        f"Stops removed:\n{stops_df.loc[~stops_df['stop_id'].isin(stops_still_used)]}"
    )

    # Update tables in place, in order so that validation is ok
    transit_data.stop_times = stop_times_to_keep
    transit_data.trips = trips_to_keep
    transit_data.routes = routes_to_keep
    transit_data.stops = stops_to_keep

    # Handle agency table
    if hasattr(transit_data, "agency") and transit_data.agency is not None:
        # Keep agencies that are NOT being removed
        filtered_agency = transit_data.agency[
            ~transit_data.agency.agency_id.isin(agency_ids_to_remove)
        ]
        WranglerLogger.info(
            f"Removed {len(transit_data.agency) - len(filtered_agency):,} agencies"
        )
        transit_data.agency = filtered_agency

    # Handle shapes table
    if (hasattr(transit_data, "shapes") and transit_data.shapes is not None 
        and "shape_id" in trips_to_keep.columns):
            shape_ids = set(trips_to_keep.shape_id.dropna().unique())
            filtered_shapes = transit_data.shapes[transit_data.shapes.shape_id.isin(shape_ids)]
            WranglerLogger.info(
                f"Removed {len(transit_data.shapes) - len(filtered_shapes):,} shape points"
            )
            transit_data.shapes = filtered_shapes

    # Handle calendar table
    if hasattr(transit_data, "calendar") and transit_data.calendar is not None:
        # Keep only service_ids referenced by remaining trips
        service_ids = set(trips_to_keep.service_id.unique())
        transit_data.calendar = transit_data.calendar[
            transit_data.calendar.service_id.isin(service_ids)
        ]

    # Handle calendar_dates table
    if hasattr(transit_data, "calendar_dates") and transit_data.calendar_dates is not None:
        # Keep only service_ids referenced by remaining trips
        service_ids = set(trips_to_keep.service_id.unique())
        transit_data.calendar_dates = transit_data.calendar_dates[
            transit_data.calendar_dates.service_id.isin(service_ids)
        ]

    # Handle frequencies table
    if hasattr(transit_data, "frequencies") and transit_data.frequencies is not None:
        # Keep only frequencies for remaining trips
        transit_data.frequencies = transit_data.frequencies[
            transit_data.frequencies.trip_id.isin(trip_ids_to_keep)
        ]


def truncate_route_at_stop(  # noqa: PLR0912, PLR0915
    transit_data: Union[GtfsModel, Feed],
    route_id: str,
    direction_id: int,
    stop_id: Union[str, int],
    truncate: Literal["before", "after"],
) -> None:
    """Truncate all trips of a route at a specific stop.

    Removes stops before or after the specified stop for all trips matching
    the given route_id and direction_id. This is useful for shortening routes
    at terminal stations or service boundaries. Modifies transit_data in place.

    Args:
        transit_data: Either a GtfsModel or Feed object to modify. Modified in place.
        route_id: The route_id to truncate
        direction_id: The direction_id of trips to truncate (0 or 1)
        stop_id: The stop where truncation occurs. For GtfsModel, this should be
                a string stop_id. For Feed, this should be an integer model_node_id.
        truncate: Either "before" to remove stops before stop_id, or
                 "after" to remove stops after stop_id

    Raises:
        ValueError: If truncate is not "before" or "after"
        ValueError: If stop_id is not found in any trips of the route/direction

    Example:
        >>> # Truncate outbound BART trips to end at Embarcadero (GtfsModel)
        >>> truncate_route_at_stop(
        ...     gtfs_model,
        ...     route_id="BART-01",
        ...     direction_id=0,
        ...     stop_id="EMBR",  # string stop_id
        ...     truncate="after",
        ... )
        >>> # Truncate outbound BART trips to end at node 12345 (Feed)
        >>> truncate_route_at_stop(
        ...     feed,
        ...     route_id="BART-01",
        ...     direction_id=0,
        ...     stop_id=12345,  # integer model_node_id
        ...     truncate="after",
        ... )
    """
    if truncate not in ["before", "after"]:
        msg = f"truncate must be 'before' or 'after', got '{truncate}'"
        raise ValueError(msg)

    WranglerLogger.info(
        f"Truncating route {route_id} direction {direction_id} {truncate} stop {stop_id}"
    )

    # Get data tables (references, not copies)
    routes_df = transit_data.routes
    trips_df = transit_data.trips
    stop_times_df = transit_data.stop_times
    stops_df = transit_data.stops
    is_gtfs = isinstance(transit_data, GtfsModel)

    # Find trips to truncate
    trips_to_truncate = trips_df[
        (trips_df.route_id == route_id) & (trips_df.direction_id == direction_id)
    ]

    if len(trips_to_truncate) == 0:
        WranglerLogger.warning(f"No trips found for route {route_id} direction {direction_id}")
        return  # No changes needed

    trip_ids_to_truncate = set(trips_to_truncate.trip_id)
    WranglerLogger.debug(f"Found {len(trip_ids_to_truncate)} trips to truncate")

    # Check if stop_id exists in any of these trips
    stop_times_for_route = stop_times_df[
        (stop_times_df.trip_id.isin(trip_ids_to_truncate)) & (stop_times_df.stop_id == stop_id)
    ]

    if len(stop_times_for_route) == 0:
        msg = f"Stop {stop_id} not found in any trips of route {route_id} direction {direction_id}"
        raise ValueError(msg)

    # Process stop_times to truncate trips
    truncated_stop_times = []
    trips_truncated = 0

    for trip_id in trip_ids_to_truncate:
        trip_stop_times = stop_times_df[stop_times_df.trip_id == trip_id]
        trip_stop_times = trip_stop_times.sort_values("stop_sequence")

        # Find the stop_sequence for the truncation stop
        stop_mask = trip_stop_times.stop_id == stop_id
        if not stop_mask.any():
            # This trip doesn't have the stop, keep all stops
            truncated_stop_times.append(trip_stop_times)
            continue

        stop_sequence_at_stop = trip_stop_times.loc[stop_mask, "stop_sequence"].iloc[0]

        # Truncate based on direction
        if truncate == "before":
            # Keep stops from stop_id onwards
            truncated_stops = trip_stop_times[
                trip_stop_times.stop_sequence >= stop_sequence_at_stop
            ].copy()  # Need copy here since we'll modify stop_sequence
        else:  # truncate == "after"
            # Keep stops up to and including stop_id
            truncated_stops = trip_stop_times[
                trip_stop_times.stop_sequence <= stop_sequence_at_stop
            ].copy()  # Need copy here since we'll modify stop_sequence

        # Renumber stop_sequence to be consecutive starting from 0
        if len(truncated_stops) > 0:
            truncated_stops["stop_sequence"] = range(len(truncated_stops))

        # Log truncation details
        original_count = len(trip_stop_times)
        truncated_count = len(truncated_stops)
        if truncated_count < original_count:
            trips_truncated += 1

            # Get removed stops details
            removed_stop_ids = set(trip_stop_times.stop_id) - set(truncated_stops.stop_id)
            if removed_stop_ids and len(removed_stop_ids) <= MAX_TRUNCATION_WARNING_STOPS:
                # Get stop names for removed stops
                removed_stops_info = stops_df[stops_df.stop_id.isin(removed_stop_ids)][
                    ["stop_id", "stop_name"]
                ]
                removed_stops_list = [
                    f"{row['stop_id']} ({row['stop_name']})"
                    for _, row in removed_stops_info.iterrows()
                ]

                WranglerLogger.debug(
                    f"Trip {trip_id}: truncated from {original_count} to {truncated_count} stops. "
                    f"Removed: {', '.join(removed_stops_list)}"
                )
            else:
                WranglerLogger.debug(
                    f"Trip {trip_id}: truncated from {original_count} to {truncated_count} stops"
                )

        truncated_stop_times.append(truncated_stops)

    WranglerLogger.info(f"Truncated {trips_truncated} trips")

    # Combine all stop times (truncated and non-truncated)
    other_stop_times = stop_times_df[~stop_times_df.trip_id.isin(trip_ids_to_truncate)]
    all_stop_times = pd.concat([other_stop_times, *truncated_stop_times], ignore_index=True)

    # Find stops that are still referenced
    stops_still_used = set(all_stop_times.stop_id.unique())
    filtered_stops = stops_df[stops_df.stop_id.isin(stops_still_used)]

    # Check if any of these stops reference parent stations
    if "parent_station" in filtered_stops.columns:
        # Get parent stations that are referenced by kept stops
        parent_stations = filtered_stops["parent_station"].dropna().unique()
        parent_stations = [ps for ps in parent_stations if ps != ""]  # Remove empty strings

        if len(parent_stations) > 0:
            # Find parent stations that aren't already in our filtered stops
            existing_stop_ids = set(filtered_stops.stop_id)
            missing_parent_stations = [ps for ps in parent_stations if ps not in existing_stop_ids]

            if len(missing_parent_stations) > 0:
                WranglerLogger.debug(
                    f"Adding back {len(missing_parent_stations)} parent stations referenced by kept stops"
                )

                # Get the parent station records
                parent_station_records = stops_df[stops_df.stop_id.isin(missing_parent_stations)]

                # Append parent stations to filtered stops
                filtered_stops = pd.concat(
                    [filtered_stops, parent_station_records], ignore_index=True
                )

    # Log removed stops
    removed_stops = set(stops_df.stop_id) - set(filtered_stops.stop_id)
    if removed_stops:
        WranglerLogger.debug(f"Removed {len(removed_stops)} stops that are no longer referenced")

        # Get details of removed stops
        removed_stops_df = stops_df[stops_df.stop_id.isin(removed_stops)][["stop_id", "stop_name"]]

        # Log up to 20 removed stops with their names
        sample_size = min(20, len(removed_stops_df))
        for _, stop in removed_stops_df.head(sample_size).iterrows():
            WranglerLogger.debug(f"  - Removed stop: {stop['stop_id']} ({stop['stop_name']})")

        if len(removed_stops) > sample_size:
            WranglerLogger.debug(f"  ... and {len(removed_stops) - sample_size} more stops")

    # Update transit_data in place
    transit_data.stop_times = all_stop_times
    transit_data.trips = trips_df
    transit_data.routes = routes_df
    transit_data.stops = filtered_stops

    # Note: shapes would need to be truncated to match truncated trips
    # TODO: truncate shapes to match truncated trips