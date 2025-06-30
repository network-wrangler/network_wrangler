"""Utilities for working with transit geodataframes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import geopandas as gpd
import pyarrow as pa
from pandera.typing import DataFrame
from shapely import LineString

from ..models.gtfs.tables import (
    WranglerShapesTable,
    WranglerStopsTable,
    WranglerStopTimesTable,
)
from ..params import LAT_LON_CRS
from ..utils.geo import linestring_from_lats_lons, update_point_geometry
from .feed import unique_shape_links, unique_stop_time_links

if TYPE_CHECKING:
    from ..models.roadway.tables import RoadNodesTable


def shapes_to_trip_shapes_gdf(
    shapes: DataFrame[WranglerShapesTable],
    # trips: WranglerTripsTable,
    ref_nodes_df: Optional[DataFrame[RoadNodesTable]] = None,
    crs: int = LAT_LON_CRS,
) -> gpd.GeoDataFrame:
    """Geodataframe with one polyline shape per shape_id.

    TODO: add information about the route and trips.

    Args:
        shapes: WranglerShapesTable
        trips: WranglerTripsTable
        ref_nodes_df: If specified, will use geometry from these nodes.  Otherwise, will use
            geometry in shapes file. Defaults to None.
        crs: int, optional, default 4326
    """
    if ref_nodes_df is not None:
        shapes = update_shapes_geometry(shapes, ref_nodes_df)

    shape_geom = (
        shapes[["shape_id", "shape_pt_lat", "shape_pt_lon"]]
        .groupby("shape_id")
        .agg(list)
        .apply(lambda x: LineString(zip(x[1], x[0])), axis=1)
    )

    route_shapes_gdf = gpd.GeoDataFrame(
        data=shape_geom.index, geometry=shape_geom.values, crs=crs
    ).set_crs(LAT_LON_CRS)

    return route_shapes_gdf


def update_stops_geometry(
    stops: DataFrame[WranglerStopsTable], ref_nodes_df: DataFrame[RoadNodesTable]
) -> DataFrame[WranglerStopsTable]:
    """Returns stops table with geometry updated from ref_nodes_df.

    NOTE: does not update "geometry" field if it exists.
    """
    return update_point_geometry(
        stops, ref_nodes_df, id_field="stop_id", lon_field="stop_lon", lat_field="stop_lat"
    )


def update_shapes_geometry(
    shapes: DataFrame[WranglerShapesTable], ref_nodes_df: DataFrame[RoadNodesTable]
) -> DataFrame[WranglerShapesTable]:
    """Returns shapes table with geometry updated from ref_nodes_df.

    NOTE: does not update "geometry" field if it exists.
    """
    return update_point_geometry(
        shapes,
        ref_nodes_df,
        id_field="shape_model_node_id",
        lon_field="shape_pt_lon",
        lat_field="shape_pt_lat",
    )


def shapes_to_shape_links_gdf(
    shapes: DataFrame[WranglerShapesTable],
    ref_nodes_df: Optional[DataFrame[RoadNodesTable]] = None,
    from_field: str = "A",
    to_field: str = "B",
    crs: int = LAT_LON_CRS,
) -> gpd.GeoDataFrame:
    """Translates shapes to shape links geodataframe using geometry from ref_nodes_df if provided.

    TODO: Add join to links and then shapes to get true geometry.

    Args:
        shapes: Feed shapes table
        ref_nodes_df: If specified, will use geometry from these nodes.  Otherwise, will use
            geometry in shapes file. Defaults to None.
        from_field: Field used for the link's from node `model_node_id`. Defaults to "A".
        to_field: Field used for the link's to node `model_node_id`. Defaults to "B".
        crs (int, optional): Coordinate reference system. SHouldn't be changed unless you know
            what you are doing. Defaults to LAT_LON_CRS which is WGS84 lat/long.

    Returns:
        gpd.GeoDataFrame: _description_
    """
    if ref_nodes_df is not None:
        shapes = update_shapes_geometry(shapes, ref_nodes_df)
    tr_links = unique_shape_links(shapes, from_field=from_field, to_field=to_field)
    # WranglerLogger.debug(f"tr_links :\n{tr_links }")

    geometry = linestring_from_lats_lons(
        tr_links,
        [f"shape_pt_lat_{from_field}", f"shape_pt_lat_{to_field}"],
        [f"shape_pt_lon_{from_field}", f"shape_pt_lon_{to_field}"],
    )
    # WranglerLogger.debug(f"geometry\n{geometry}")
    shapes_gdf = gpd.GeoDataFrame(tr_links, geometry=geometry, crs=crs).set_crs(LAT_LON_CRS)
    return shapes_gdf


def stop_times_to_stop_time_points_gdf(
    stop_times: DataFrame[WranglerStopTimesTable],
    stops: DataFrame[WranglerStopsTable],
    ref_nodes_df: Optional[DataFrame[RoadNodesTable]] = None,
) -> gpd.GeoDataFrame:
    """Stoptimes geodataframe as points using geometry from stops.txt or optionally another df.

    Args:
        stop_times (WranglerStopTimesTable): Feed stop times table.
        stops (WranglerStopsTable): Feed stops table.
        ref_nodes_df (pd.DataFrame, optional): If specified, will use geometry from these nodes.
            Otherwise, will use geometry in shapes file. Defaults to None.
    """
    if ref_nodes_df is not None:
        stops = update_stops_geometry(stops, ref_nodes_df)

    stop_times_geo = stop_times.merge(
        stops[["stop_id", "stop_lat", "stop_lon"]],
        right_on="stop_id",
        left_on="stop_id",
        how="left",
    )
    return gpd.GeoDataFrame(
        stop_times_geo,
        geometry=gpd.points_from_xy(stop_times_geo["stop_lon"], stop_times_geo["stop_lat"]),
        crs=LAT_LON_CRS,
    )


def stop_times_to_stop_time_links_gdf(
    stop_times: DataFrame[WranglerStopTimesTable],
    stops: DataFrame[WranglerStopsTable],
    ref_nodes_df: Optional[DataFrame[RoadNodesTable]] = None,
    from_field: str = "A",
    to_field: str = "B",
) -> gpd.GeoDataFrame:
    """Stop times geodataframe as links using geometry from stops.txt or optionally another df.

    Args:
        stop_times (WranglerStopTimesTable): Feed stop times table.
        stops (WranglerStopsTable): Feed stops table.
        ref_nodes_df (pd.DataFrame, optional): If specified, will use geometry from these nodes.
            Otherwise, will use geometry in shapes file. Defaults to None.
        from_field: Field used for the link's from node `model_node_id`. Defaults to "A".
        to_field: Field used for the link's to node `model_node_id`. Defaults to "B".
    """
    from ..utils.geo import linestring_from_lats_lons  # noqa: PLC0415

    if ref_nodes_df is not None:
        stops = update_stops_geometry(stops, ref_nodes_df)

    lat_fields = []
    lon_fields = []
    tr_links = unique_stop_time_links(stop_times, from_field=from_field, to_field=to_field)
    for f in (from_field, to_field):
        tr_links = tr_links.merge(
            stops[["stop_id", "stop_lat", "stop_lon"]],
            right_on="stop_id",
            left_on=f,
            how="left",
        )
        lon_f = f"{f}_X"
        lat_f = f"{f}_Y"
        tr_links = tr_links.rename(columns={"stop_lon": lon_f, "stop_lat": lat_f})
        lon_fields.append(lon_f)
        lat_fields.append(lat_f)

    geometry = linestring_from_lats_lons(tr_links, lat_fields, lon_fields)
    return gpd.GeoDataFrame(tr_links, geometry=geometry).set_crs(LAT_LON_CRS)
