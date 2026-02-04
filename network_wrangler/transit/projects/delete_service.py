"""Functions for adding a transit route to a TransitNetwork."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...logger import WranglerLogger
from ..feed.routes import route_ids_for_trip_ids
from ..feed.shapes import shape_ids_for_trip_ids

if TYPE_CHECKING:
    from ..feed.feed import Feed
    from ..network import TransitNetwork
    from ..selection import TransitSelection


def apply_transit_service_deletion(
    net: TransitNetwork,
    selection: TransitSelection,
    clean_shapes: bool | None = False,
    clean_routes: bool | None = False,
) -> TransitNetwork:
    """Delete transit service to TransitNetwork.

    Args:
        net (TransitNetwork): Network to modify.
        selection: TransitSelection object, created from a selection dictionary.
        clean_shapes (bool, optional): If True, remove shapes not used by any trips.
            Defaults to False.
        clean_routes (bool, optional): If True, remove routes not used by any trips.
            Defaults to False.

    Returns:
        TransitNetwork: Modified network.
    """
    WranglerLogger.debug("Applying delete transit service project.")

    trip_ids = selection.selected_trips
    net.feed = _delete_trips_from_feed(
        net.feed, trip_ids, clean_shapes=clean_shapes, clean_routes=clean_routes
    )

    return net


def _delete_trips_from_feed(
    feed: Feed,
    trip_ids: list,
    clean_shapes: bool | None = False,
    clean_routes: bool | None = False,
) -> Feed:
    """Delete transit service from feed based on trip_ids.

    Args:
        feed (Feed): Feed to modify.
        trip_ids (list): List of trip_ids to delete.
        clean_shapes (bool, optional): If True, remove shapes not used by any trips.
            Defaults to False.
        clean_routes (bool, optional): If True, remove routes not used by any trips.
            Defaults to False.

    Returns:
        Feed: Modified feed.
    """
    WranglerLogger.debug("Deleting service from feed.")

    trips_df = feed.trips.copy()
    stop_times_df = feed.stop_times.copy()
    frequencies_df = feed.frequencies.copy()

    trips_df = trips_df[~trips_df.trip_id.isin(trip_ids)]
    stop_times_df = stop_times_df[~stop_times_df.trip_id.isin(trip_ids)]
    frequencies_df = frequencies_df[~frequencies_df.trip_id.isin(trip_ids)]

    if clean_shapes:
        shapes_df = feed.shapes.copy()
        # don't delete shapes that are still used by other trips
        del_shape_ids = list(
            set(shape_ids_for_trip_ids(feed.trips, trip_ids)) - set(trips_df.shape_id.unique())
        )
        feed.shapes = shapes_df[~shapes_df.shapes_id.isin(del_shape_ids)]

    if clean_routes:
        routes_df = feed.routes.copy()
        # don't delete shapes that are still used by other trips
        del_route_ids = list(
            set(route_ids_for_trip_ids(feed.trips, trip_ids)) - set(trips_df.route_id.unique())
        )
        feed.routes = routes_df[~routes_df.route_id.isin(del_route_ids)]

    feed.stop_times = stop_times_df
    feed.frequencies = frequencies_df
    feed.trips = trips_df

    return feed
