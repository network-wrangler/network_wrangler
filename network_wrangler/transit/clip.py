"""Functions to clip a TransitNetwork object to a boundary.

Clipped transit is an independent transit network that is a subset of the original transit network.

Example usage:

```python
from network_wrangler.transit load_transit, write_transit
from network_wrangler.transit.clip import clip_transit

stpaul_transit = load_transit(example_dir / "stpaul")
boundary_file = test_dir / "data" / "ecolab.geojson"
clipped_network = clip_transit(stpaul_transit, boundary_file=boundary_file)
write_transit(clipped_network, out_dir, prefix="ecolab", format="geojson", true_shape=True)
```

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pandas as pd

from ..logger import WranglerLogger
from ..roadway.network import RoadwayNetwork
from ..roadway.nodes.io import get_nodes
from ..transit.io import load_transit
from ..utils.geo import get_bounding_polygon
from .feed.feed import (
    Feed,
)
from .feed.frequencies import frequencies_for_trips
from .feed.routes import routes_for_trips
from .feed.shapes import shapes_for_road_links, shapes_for_trips
from .feed.stop_times import (
    stop_times_for_min_stops,
    stop_times_for_shapes,
    stop_times_for_stops,
)
from .feed.stops import stops_for_stop_times
from .feed.trips import trips_for_stop_times
from .geo import shapes_to_shape_links_gdf
from .network import TransitNetwork

# minimum number of stops needed to retain a transit trip within clipped area
DEFAULT_MIN_STOPS: int = 2


def clip_feed_to_roadway(
    feed: Feed,
    roadway_net: RoadwayNetwork,
    min_stops: int = DEFAULT_MIN_STOPS,
) -> Feed:
    """Returns a copy of transit feed clipped to the roadway network.

    Args:
        feed (Feed): Transit Feed to clip.
        roadway_net: Roadway network to clip to.
        min_stops: minimum number of stops needed to retain a transit trip within clipped area.
            Defaults to DEFAULT_MIN_STOPS which is set to 2.

    Raises:
        ValueError: If no stops found within the roadway network.

    Returns:
        Feed: Clipped deep copy of feed limited to the roadway network.
    """
    WranglerLogger.info("Clipping transit network to roadway network.")

    clipped_feed = _remove_links_from_feed(feed, roadway_net.links_df, min_stops=min_stops)

    return clipped_feed


def _remove_links_from_feed(
    feed: Feed, links_df: pd.DataFrame, min_stops: int = DEFAULT_MIN_STOPS
) -> Feed:
    WranglerLogger.info("Clipping transit network to link A and Bs.")

    clipped_feed_dfs = {}
    # First find the shapes that are on the links; retaining only the longest segment of each shape
    _valid_shapes = shapes_for_road_links(feed.shapes, links_df)
    WranglerLogger.debug(
        f"_valid_shapes: \n\
        {_valid_shapes[['shape_id', 'shape_pt_sequence', 'shape_model_node_id']]}"
    )
    # Filter stop_times to relevant trips first so don't have to do complicated filtering on whole
    _trips_for_valid_shapes = feed.trips.loc[feed.trips.shape_id.isin(_valid_shapes.shape_id)]
    WranglerLogger.debug(
        f"_trips_for_valid_shapes: \n{_trips_for_valid_shapes[['trip_id', 'shape_id']]}"
    )
    _stop_times_for_valid_trips = feed.stop_times.loc[
        feed.stop_times.trip_id.isin(_trips_for_valid_shapes.trip_id)
    ]

    _valid_stop_times = stop_times_for_shapes(
        _stop_times_for_valid_trips, _valid_shapes, _trips_for_valid_shapes
    )
    clipped_feed_dfs["stop_times"] = stop_times_for_min_stops(_valid_stop_times, min_stops)
    WranglerLogger.debug(
        f"clipped_feed_dfs['stop_times']: \n\
        {clipped_feed_dfs['stop_times'][['trip_id', 'stop_id', 'stop_sequence']]}"
    )
    WranglerLogger.debug(
        f"Keeping {len(clipped_feed_dfs['stop_times'])}/{len(feed.stop_times)} stop_times."
    )

    # reselect trips + shapes so we aren't retaining ones that only had one stop in the stop_times
    clipped_feed_dfs["trips"] = trips_for_stop_times(feed.trips, clipped_feed_dfs["stop_times"])
    WranglerLogger.debug(f"Keeping {len(clipped_feed_dfs['trips'])}/{len(feed.trips)} trips.")

    clipped_feed_dfs["shapes"] = shapes_for_trips(_valid_shapes, clipped_feed_dfs["trips"])
    WranglerLogger.debug(f"Keeping {len(clipped_feed_dfs['shapes'])}/{len(feed.shapes)} shapes.")

    clipped_feed_dfs["stops"] = stops_for_stop_times(feed.stops, clipped_feed_dfs["stop_times"])
    WranglerLogger.debug(f"Keeping {len(clipped_feed_dfs['stops'])}/{len(feed.stops)} stops.")

    clipped_feed_dfs["routes"] = routes_for_trips(feed.routes, clipped_feed_dfs["trips"])
    WranglerLogger.debug(f"Keeping {len(clipped_feed_dfs['routes'])}/{len(feed.routes)} routes.")

    clipped_feed_dfs["frequencies"] = frequencies_for_trips(
        feed.frequencies, clipped_feed_dfs["trips"]
    )
    WranglerLogger.debug(
        f"Keeping {len(clipped_feed_dfs['frequencies'])}/{len(feed.frequencies)} frequencies."
    )

    return Feed(**clipped_feed_dfs)


def clip_feed_to_boundary(
    feed: Feed,
    ref_nodes_df: gpd.GeoDataFrame,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
    boundary_geocode: Optional[Union[str, dict]] = None,
    boundary_file: Optional[Union[str, Path]] = None,
    min_stops: int = DEFAULT_MIN_STOPS,
) -> Feed:
    """Clips a transit Feed object to a boundary and returns the resulting GeoDataFrames.

    Retains only the stops within the boundary and trips that traverse them subject to a minimum
    number of stops per trip as defined by `min_stops`.

    Args:
        feed: Feed object to be clipped.
        ref_nodes_df: geodataframe with node geometry to reference
        boundary_geocode (Union[str, dict], optional): A geocode string or dictionary
            representing the boundary. Defaults to None.
        boundary_file (Union[str, Path], optional): A path to the boundary file. Only used if
            boundary_geocode is None. Defaults to None.
        boundary_gdf (gpd.GeoDataFrame, optional): A GeoDataFrame representing the boundary.
            Only used if boundary_geocode and boundary_file are None. Defaults to None.
        min_stops: minimum number of stops needed to retain a transit trip within clipped area.
            Defaults to DEFAULT_MIN_STOPS which is set to 2.

    Returns: Feed object trimmed to the boundary.
    """
    WranglerLogger.info("Clipping transit network to boundary.")

    boundary_gdf = get_bounding_polygon(
        boundary_gdf=boundary_gdf,
        boundary_geocode=boundary_geocode,
        boundary_file=boundary_file,
    )

    shape_links_gdf = shapes_to_shape_links_gdf(feed.shapes, ref_nodes_df=ref_nodes_df)

    # make sure boundary_gdf.crs == network.crs
    if boundary_gdf.crs != shape_links_gdf.crs:
        boundary_gdf = boundary_gdf.to_crs(shape_links_gdf.crs)

    # get the boundary as a single polygon
    boundary = boundary_gdf.geometry.union_all()
    # get the shape_links that intersect the boundary
    clipped_shape_links = shape_links_gdf[shape_links_gdf.geometry.intersects(boundary)]

    # nodes within clipped_shape_links
    node_ids = list(set(clipped_shape_links.A.to_list() + clipped_shape_links.B.to_list()))
    WranglerLogger.debug(f"Clipping network to {len(node_ids)} nodes.")
    if not node_ids:
        msg = "No nodes found within the boundary."
        raise ValueError(msg)
    return _clip_feed_to_nodes(feed, node_ids, min_stops=min_stops)


def _clip_feed_to_nodes(
    feed: Feed,
    node_ids: list[str],
    min_stops: int = DEFAULT_MIN_STOPS,
) -> Feed:
    """Clip a transit feed object to a set of nodes.

    Retains only trips and stops that have at least two stops within the boundary are retained.

    Args:
        feed (Feed): Feed files
        node_ids (list[str]): list of stop_ids to clip to
        min_stops: min_stops: minimum number of stops needed to retain a transit trip within
            clipped area. Defaults to DEFAULT_MIN_STOPS which is set to 2.
    """
    WranglerLogger.info(f"Clipping transit network to {len(node_ids)} node_ids.")

    clipped_feed_dfs = {}

    clipped_feed_dfs["stops"] = feed.stops[feed.stops.stop_id.isin(node_ids)]
    WranglerLogger.debug(f"Keeping {len(clipped_feed_dfs['stops'])}/{len(feed.stops)} stops.")

    # don't retain stop_times unless they are more than min stops
    _clipped_stop_times = stop_times_for_stops(feed.stop_times, clipped_feed_dfs["stops"])
    clipped_feed_dfs["stop_times"] = stop_times_for_min_stops(_clipped_stop_times, min_stops)

    # reselect stops so we aren't retaining ones that only had one in the stop_times
    clipped_feed_dfs["stops"] = stops_for_stop_times(
        clipped_feed_dfs["stops"], clipped_feed_dfs["stop_times"]
    )

    clipped_feed_dfs["trips"] = trips_for_stop_times(feed.trips, clipped_feed_dfs["stop_times"])
    clipped_feed_dfs["routes"] = routes_for_trips(feed.routes, clipped_feed_dfs["trips"])

    # don't retain shapes unless trip is retained
    clipped_feed_dfs["shapes"] = shapes_for_trips(feed.shapes, clipped_feed_dfs["trips"])
    clipped_feed_dfs["shapes"] = clipped_feed_dfs["shapes"][
        clipped_feed_dfs["shapes"].shape_model_node_id.isin(node_ids)
    ]

    clipped_feed_dfs["frequencies"] = frequencies_for_trips(
        feed.frequencies, clipped_feed_dfs["trips"]
    )

    return Feed(**clipped_feed_dfs)


def clip_transit(
    network: Union[TransitNetwork, str, Path],
    node_ids: Optional[list[str]] = None,
    boundary_geocode: Optional[Union[str, dict]] = None,
    boundary_file: Optional[Union[str, Path]] = None,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
    ref_nodes_df: Optional[gpd.GeoDataFrame] = None,
    roadway_net: Optional[RoadwayNetwork] = None,
    min_stops: int = DEFAULT_MIN_STOPS,
) -> TransitNetwork:
    """Returns a new TransitNetwork clipped to a boundary as determined by arguments.

    Will clip based on which arguments are provided as prioritized below:

    1. If `node_ids` provided, will clip based on `node_ids`
    2. If `boundary_geocode` provided, will clip based on on search in OSM for that jurisdiction
        boundary using reference geometry from `ref_nodes_df`, `roadway_net`, or `roadway_path`
    3. If `boundary_file` provided, will clip based on that polygon  using reference geometry
        from `ref_nodes_df`, `roadway_net`, or `roadway_path`
    4. If `boundary_gdf` provided, will clip based on that geodataframe  using reference geometry
        from `ref_nodes_df`, `roadway_net`, or `roadway_path`
    5. If `roadway_net` provided, will clip based on that roadway network

    Args:
        network (TransitNetwork): TransitNetwork to clip.
        node_ids (list[str], optional): A list of node_ids to clip to. Defaults to None.
        boundary_geocode (Union[str, dict], optional): A geocode string or dictionary
            representing the boundary. Only used if node_ids are None. Defaults to None.
        boundary_file (Union[str, Path], optional): A path to the boundary file. Only used if
            node_ids and boundary_geocode are None. Defaults to None.
        boundary_gdf (gpd.GeoDataFrame, optional): A GeoDataFrame representing the boundary.
            Only used if node_ids, boundary_geocode and boundary_file are None. Defaults to None.
        ref_nodes_df: GeoDataFrame of geographic references for node_ids.  Only used if
            node_ids is None and one of boundary_* is not None.
        roadway_net: Roadway Network  instance to clip transit network to.  Only used if
            node_ids is None and allof boundary_* are None
        min_stops: minimum number of stops needed to retain a transit trip within clipped area.
            Defaults to DEFAULT_MIN_STOPS which is set to 2.
    """
    if not isinstance(network, TransitNetwork):
        network = load_transit(network)
    set_roadway_network = False
    feed = network.feed

    if node_ids is not None:
        clipped_feed = _clip_feed_to_nodes(feed, node_ids=node_ids, min_stops=min_stops)
    elif any(i is not None for i in [boundary_file, boundary_geocode, boundary_gdf]):
        if ref_nodes_df is None:
            ref_nodes_df = get_nodes(transit_net=network, roadway_net=roadway_net)

        clipped_feed = clip_feed_to_boundary(
            feed,
            ref_nodes_df,
            boundary_file=boundary_file,
            boundary_geocode=boundary_geocode,
            boundary_gdf=boundary_gdf,
            min_stops=min_stops,
        )
    elif roadway_net is not None:
        clipped_feed = clip_feed_to_roadway(feed, roadway_net=roadway_net)
        set_roadway_network = True
    else:
        msg = "Missing required arguments from clip_transit"
        raise ValueError(msg)

    # create a new TransitNetwork object with the clipped feed dataframes
    clipped_net = TransitNetwork(clipped_feed)

    if set_roadway_network:
        WranglerLogger.info("Setting roadway network for clipped transit network.")
        clipped_net.road_net = roadway_net
    return clipped_net
