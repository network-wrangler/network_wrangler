"""Helper geographic manipulation functions."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import geopandas as gpd
import pandas as pd
import pyarrow as pa
from geographiclib.geodesic import Geodesic
from pyproj import CRS, Proj, Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform

from ..errors import MissingNodesError
from ..logger import WranglerLogger
from ..models._base.geo import LatLongCoordinates
from ..models.roadway.types import LocationReference
from ..params import LAT_LON_CRS
from .data import update_df_by_col_value

if TYPE_CHECKING:
    from ..roadway.network import RoadwayNetwork

# key:value (from espg, to espg): pyproj transform object
transformers = {}


class InvalidCRSError(Exception):
    """Raised when a point is not valid for a given coordinate reference system."""


def get_bearing(lat1, lon1, lat2, lon2):
    """Calculate the bearing (forward azimuth) b/w the two points.

    returns: bearing in radians
    """
    # bearing in degrees
    brng = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)["azi1"]

    # convert bearing to radians
    brng = math.radians(brng)

    return brng


def offset_point_with_distance_and_bearing(
    lon: float, lat: float, distance: float, bearing: float
) -> list[float]:
    """Get the new lon-lat (in degrees) given current point (lon-lat), distance and bearing.

    Args:
        lon: longitude of original point
        lat: latitude of original point
        distance: distance in meters to offset point by
        bearing: direction to offset point to in radians

    returns: list of new offset lon-lat
    """
    # Earth's radius in meters
    radius = 6378137

    # convert the lat long from degree to radians
    lat_radians = math.radians(lat)
    lon_radians = math.radians(lon)

    # calculate the new lat long in radians
    out_lat_radians = math.asin(
        math.sin(lat_radians) * math.cos(distance / radius)
        + math.cos(lat_radians) * math.sin(distance / radius) * math.cos(bearing)
    )

    out_lon_radians = lon_radians + math.atan2(
        math.sin(bearing) * math.sin(distance / radius) * math.cos(lat_radians),
        math.cos(distance / radius) - math.sin(lat_radians) * math.sin(lat_radians),
    )
    # convert the new lat long back to degree
    out_lat = math.degrees(out_lat_radians)
    out_lon = math.degrees(out_lon_radians)

    return [out_lon, out_lat]


def length_of_linestring_miles(gdf: Union[gpd.GeoSeries, gpd.GeoDataFrame]) -> pd.Series:
    """Returns a Series with the linestring length in miles.

    Args:
        gdf: GeoDataFrame with linestring geometry.  If given a GeoSeries will attempt to convert
            to a GeoDataFrame.
    """
    # WranglerLogger.debug(f"length_of_linestring_miles.gdf input:\n{gdf}.")
    if isinstance(gdf, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=gdf)

    p_crs = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(p_crs)
    METERS_IN_MILES = 1609.34
    length_miles = gdf.geometry.length / METERS_IN_MILES
    length_s = pd.Series(length_miles, index=gdf.index)

    return length_s


def linestring_from_nodes(
    links_df: pd.DataFrame,
    nodes_df: gpd.GeoDataFrame,
    from_node: str = "A",
    to_node: str = "B",
    node_pk: str = "model_node_id",
) -> gpd.GeoSeries:
    """Creates a LineString geometry GeoSeries from a DataFrame of links and a DataFrame of nodes.

    Args:
        links_df: DataFrame with columns for from_node and to_node.
        nodes_df: GeoDataFrame with geometry column.
        from_node: column name in links_df for the from node. Defaults to "A".
        to_node: column name in links_df for the to node. Defaults to "B".
        node_pk: primary key column name in nodes_df. Defaults to "model_node_id".
    """
    assert "geometry" in nodes_df.columns, "nodes_df must have a 'geometry' column"

    idx_name = "index" if links_df.index.name is None else links_df.index.name
    # WranglerLogger.debug(f"Index name: {idx_name}")
    required_link_cols = [from_node, to_node]

    if not all(col in links_df.columns for col in required_link_cols):
        WranglerLogger.error(
            f"links_df.columns missing required columns.\n\
                            links_df.columns: {links_df.columns}\n\
                            required_link_cols: {required_link_cols}"
        )
        msg = "links_df must have columns {required_link_cols} to create linestring from nodes"
        raise ValueError(msg)

    links_geo_df = copy.deepcopy(links_df[required_link_cols])
    # need to continuously reset the index to make sure the index is the same as the link index
    links_geo_df = (
        links_geo_df.reset_index()
        .merge(
            nodes_df[[node_pk, "geometry"]],
            left_on=from_node,
            right_on=node_pk,
            how="left",
        )
        .set_index(idx_name)
    )

    links_geo_df = links_geo_df.rename(columns={"geometry": "geometry_A"})

    links_geo_df = (
        links_geo_df.reset_index()
        .merge(
            nodes_df[[node_pk, "geometry"]],
            left_on=to_node,
            right_on=node_pk,
            how="left",
        )
        .set_index(idx_name)
    )

    links_geo_df = links_geo_df.rename(columns={"geometry": "geometry_B"})

    # makes sure all nodes exist
    _missing_geo_links_df = links_geo_df[
        links_geo_df["geometry_A"].isnull() | links_geo_df["geometry_B"].isnull()
    ]
    if not _missing_geo_links_df.empty:
        missing_nodes = _missing_geo_links_df[[from_node, to_node]].values
        WranglerLogger.error(
            f"Cannot create link geometry from nodes because the nodes are\
                             missing from the network. Missing nodes: {missing_nodes}"
        )
        msg = "Cannot create link geometry from nodes because the nodes are missing from the network."
        raise MissingNodesError(msg)

    # create geometry from points
    links_geo_df["geometry"] = links_geo_df.apply(
        lambda row: LineString([row["geometry_A"], row["geometry_B"]]), axis=1
    )

    # convert to GeoDataFrame
    links_gdf = gpd.GeoDataFrame(links_geo_df["geometry"], geometry=links_geo_df["geometry"])
    return links_gdf["geometry"]


def linestring_from_lats_lons(df, lat_fields, lon_fields) -> gpd.GeoSeries:
    """Create a LineString geometry from a DataFrame with lon/lat fields.

    Args:
        df: DataFrame with columns for lon/lat fields.
        lat_fields: list of column names for the lat fields.
        lon_fields: list of column names for the lon fields.
    """
    if len(lon_fields) != len(lat_fields):
        msg = "lon_fields and lat_fields lists must have the same length"
        raise ValueError(msg)

    line_geometries = gpd.GeoSeries(
        [
            LineString([(row[lon], row[lat]) for lon, lat in zip(lon_fields, lat_fields)])
            for _, row in df.iterrows()
        ]
    )

    return gpd.GeoSeries(line_geometries)


def check_point_valid_for_crs(point: Point, crs: int):
    """Check if a point is valid for a given coordinate reference system.

    Args:
        point: Shapely Point
        crs: coordinate reference system in ESPG code

    raises: InvalidCRSError if point is not valid for the given crs
    """
    crs = CRS.from_user_input(crs)
    minx, miny, maxx, maxy = crs.area_of_use.bounds
    ok_bounds = True
    if not minx <= point.x <= maxx:
        WranglerLogger.error(f"Invalid X coordinate for CRS {crs}: {point.x}")
        ok_bounds = False
    if not miny <= point.y <= maxy:
        WranglerLogger.error(f"Invalid Y coordinate for CRS {crs}: {point.y}")
        ok_bounds = False

    if not ok_bounds:
        msg = f"Invalid coordinate for CRS {crs}: {point.x}, {point.y}"
        raise InvalidCRSError(msg)


def point_from_xy(x, y, xy_crs: int = LAT_LON_CRS, point_crs: int = LAT_LON_CRS):
    """Creates point geometry from x and y coordinates.

    Args:
        x: x coordinate, in xy_crs
        y: y coordinate, in xy_crs
        xy_crs: coordinate reference system in ESPG code for x/y inputs. Defaults to 4326 (WGS84)
        point_crs: coordinate reference system in ESPG code for point output.
            Defaults to 4326 (WGS84)

    Returns: Shapely Point in point_crs
    """
    point = Point(x, y)

    if xy_crs == point_crs:
        check_point_valid_for_crs(point, point_crs)
        return point

    if (xy_crs, point_crs) not in transformers:
        # store transformers in dictionary because they are an "expensive" operation
        transformers[(xy_crs, point_crs)] = Transformer.from_proj(
            Proj(init="epsg:" + str(xy_crs)),
            Proj(init="epsg:" + str(point_crs)),
            always_xy=True,  # required b/c Proj v6+ uses lon/lat instead of x/y
        )

    return transform(transformers[(xy_crs, point_crs)].transform, point)


def update_points_in_linestring(
    linestring: LineString, updated_coords: list[float], position: int
):
    """Replaces a point in a linestring with a new point.

    Args:
        linestring (LineString): original_linestring
        updated_coords (List[float]): updated poimt coordinates
        position (int): position in the linestring to update
    """
    coords = [c for c in linestring.coords]
    coords[position] = updated_coords
    return LineString(coords)


def update_nodes_in_linestring_geometry(
    original_df: gpd.GeoDataFrame,
    updated_nodes_df: gpd.GeoDataFrame,
    position: int,
) -> gpd.GeoSeries:
    """Updates the nodes in a linestring geometry and returns updated geometry.

    Args:
        original_df: GeoDataFrame with the `model_node_id` and linestring geometry
        updated_nodes_df: GeoDataFrame with updated node geometries.
        position: position in the linestring to update with the node.
    """
    LINK_FK_NODE = ["A", "B"]
    original_index = original_df.index

    updated_df = original_df.reset_index().merge(
        updated_nodes_df[["model_node_id", "geometry"]],
        left_on=LINK_FK_NODE[position],
        right_on="model_node_id",
        suffixes=("", "_node"),
    )

    updated_df["geometry"] = updated_df.apply(
        lambda row: update_points_in_linestring(
            row["geometry"], row["geometry_node"].coords[0], position
        ),
        axis=1,
    )

    updated_df = updated_df.reset_index().set_index(original_index.names)

    WranglerLogger.debug(f"updated_df - AFTER: \n {updated_df.geometry}")
    return updated_df["geometry"]


def get_point_geometry_from_linestring(polyline_geometry, pos: int = 0):
    """Get a point geometry from a linestring geometry.

    Args:
        polyline_geometry: shapely LineString instance
        pos: position in the linestring to get the point from. Defaults to 0.
    """
    # WranglerLogger.debug(
    #    f"get_point_geometry_from_linestring.polyline_geometry.coords[0]: \
    #    {polyline_geometry.coords[0]}."
    # )

    # Note: when upgrading to shapely 2.0, will need to use following command
    # _point_coords = get_coordinates(polyline_geometry).tolist()[pos]
    return point_from_xy(*polyline_geometry.coords[pos])


def location_ref_from_point(
    geometry: Point,
    sequence: int = 1,
    bearing: Optional[float] = None,
    distance_to_next_ref: Optional[float] = None,
) -> LocationReference:
    """Generates a shared street point location reference.

    Args:
        geometry (Point): Point shapely geometry
        sequence (int, optional): Sequence if part of polyline. Defaults to None.
        bearing (float, optional): Direction of line if part of polyline. Defaults to None.
        distance_to_next_ref (float, optional): Distnce to next point if part of polyline.
            Defaults to None.

    Returns:
        LocationReference: As defined by sharedStreets.io schema
    """
    lr = {
        "point": LatLongCoordinates(geometry.coords[0]),
    }

    for arg in ["sequence", "bearing", "distance_to_next_ref"]:
        if locals()[arg] is not None:
            lr[arg] = locals()[arg]

    return LocationReference(**lr)


def location_refs_from_linestring(geometry: LineString) -> list[LocationReference]:
    """Generates a shared street location reference from linestring.

    Args:
        geometry (LineString): Shapely LineString instance

    Returns:
        LocationReferences: As defined by sharedStreets.io schema
    """
    return [
        location_ref_from_point(
            point,
            sequence=i + 1,
            distance_to_next_ref=point.distance(geometry.coords[i + 1]),
            bearing=get_bearing(*point.coords[0], *geometry.coords[i + 1]),
        )
        for i, point in enumerate(geometry.coords[:-1])
    ]


def get_bounding_polygon(
    boundary_geocode: Optional[Union[str, dict]] = None,
    boundary_file: Optional[Union[str, Path]] = None,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
    crs: int = LAT_LON_CRS,  # WGS84
) -> gpd.GeoSeries:
    """Get the bounding polygon for a given boundary.

    Will return None if no arguments given. Will raise a ValueError if more than one given.

    This function retrieves the bounding polygon for a given boundary. The boundary can be provided
    as a GeoDataFrame, a geocode string or dictionary, or a boundary file. The resulting polygon
    geometry is returned as a GeoSeries.

    Args:
        boundary_geocode (Union[str, dict], optional): A geocode string or dictionary
            representing the boundary. Defaults to None.
        boundary_file (Union[str, Path], optional): A path to the boundary file. Only used if
            boundary_geocode is None. Defaults to None.
        boundary_gdf (gpd.GeoDataFrame, optional): A GeoDataFrame representing the boundary.
            Only used if boundary_geocode and boundary_file are None. Defaults to None.
        crs (int, optional): The coordinate reference system (CRS) code. Defaults to 4326 (WGS84).

    Returns:
        gpd.GeoSeries: The polygon geometry representing the bounding polygon.
    """
    import osmnx as ox  # noqa: PLC0415

    nargs = sum(x is not None for x in [boundary_gdf, boundary_geocode, boundary_file])
    if nargs == 0:
        return None
    if nargs != 1:
        msg = "Exactly one of boundary_gdf, boundary_geocode, or boundary_file must be provided."
        raise ValueError(msg)

    OK_BOUNDARY_SUFF = [".shp", ".geojson", ".parquet"]

    if boundary_geocode is not None:
        boundary_gdf = ox.geocode_to_gdf(boundary_geocode)
    elif boundary_file is not None:
        boundary_file = Path(boundary_file)
        if boundary_file.suffix not in OK_BOUNDARY_SUFF:
            msg = "Boundary file must have one of the following suffixes: {OK_BOUNDARY_SUFF}"
            raise ValueError(msg)
        if not boundary_file.exists():
            msg = f"Boundary file {boundary_file} does not exist"
            raise FileNotFoundError(msg)
        if boundary_file.suffix == ".parquet":
            boundary_gdf = gpd.read_parquet(boundary_file)
        else:
            boundary_gdf = gpd.read_file(boundary_file)
            if boundary_file.suffix == ".geojson":  # geojson standard is WGS84
                boundary_gdf.crs = crs

    if boundary_gdf is None:
        msg = "One of boundary_gdf, boundary_geocode or boundary_file must be provided."
        raise ValueError(msg)

    if boundary_gdf.crs is not None:
        boundary_gdf = boundary_gdf.to_crs(crs)
    # make sure boundary_gdf is a polygon
    if len(boundary_gdf.geom_type[boundary_gdf.geom_type != "Polygon"]) > 0:
        msg = "boundary_gdf must all be Polygons"
        raise ValueError(msg)
    # get the boundary as a single polygon
    boundary_gs = gpd.GeoSeries([boundary_gdf.geometry.union_all()], crs=crs)

    return boundary_gs


def _harmonize_crs(df: pd.DataFrame, crs: int = LAT_LON_CRS) -> pd.DataFrame:
    if isinstance(df, gpd.GeoDataFrame) and df.crs != crs:
        df = df.to_crs(crs)
    return df


def _id_utm_crs(gdf: Union[gpd.GeoSeries, gpd.GeoDataFrame]) -> int:
    """Returns the UTM CRS ESPG for the given GeoDataFrame.

    Args:
        gdf: GeoDataFrame to get UTM CRS for.
    """
    if isinstance(gdf, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=gdf)

    return gdf.estimate_utm_crs().to_epsg()


def offset_geometry_meters(geo_s: gpd.GeoSeries, offset_distance_meters: float) -> gpd.GeoSeries:
    """Offset a GeoSeries of LineStrings by a given distance in meters.

    Args:
        geo_s: GeoSeries of LineStrings to offset.
        offset_distance_meters: distance in meters to offset the LineStrings.
    """
    if geo_s.empty:
        return geo_s
    og_crs = geo_s.crs
    meters_crs = _id_utm_crs(geo_s)
    geo_s = geo_s.to_crs(meters_crs)
    offset_geo = geo_s.apply(lambda x: x.offset_curve(offset_distance_meters))
    offset_geo = gpd.GeoSeries(offset_geo)
    return offset_geo.to_crs(og_crs)


def to_points_gdf(
    table: pd.DataFrame,
    ref_nodes_df: Optional[gpd.GeoDataFrame] = None,
    ref_road_net: Optional[RoadwayNetwork] = None,
) -> gpd.GeoDataFrame:
    """Convert a table to a GeoDataFrame.

    If the table is already a GeoDataFrame, return it as is. Otherwise, attempt to convert the
    table to a GeoDataFrame using the following methods:
    1. If the table has a 'geometry' column, return a GeoDataFrame using that column.
    2. If the table has 'lat' and 'lon' columns, return a GeoDataFrame using those columns.
    3. If the table has a '*model_node_id' or 'stop_id' column, return a GeoDataFrame using that column and the
         nodes_df provided.
    If none of the above, raise a ValueError.

    Args:
        table: DataFrame to convert to GeoDataFrame.
        ref_nodes_df: GeoDataFrame of nodes to use to convert model_node_id to geometry.
        ref_road_net: RoadwayNetwork object to use to convert model_node_id to geometry.

    Returns:
        GeoDataFrame: GeoDataFrame representation of the table.
    """
    if table is gpd.GeoDataFrame:
        return table

    WranglerLogger.debug("Converting GTFS table to GeoDataFrame")
    if "geometry" in table.columns:
        return gpd.GeoDataFrame(table, geometry="geometry")

    lat_cols = list(filter(lambda col: "lat" in col, table.columns))
    lon_cols = list(filter(lambda col: "lon" in col, table.columns))
    model_node_id_cols = [
        c for c in ["model_node_id", "stop_id", "shape_model_node_id"] if c in table.columns
    ]

    if not (lat_cols and lon_cols) or not model_node_id_cols:
        WranglerLogger.error(
            "Needed either lat/long or *model_node_id columns to convert \
            to GeoDataFrame. Columns found: {table.columns}"
        )
        if not (lat_cols and lon_cols):
            WranglerLogger.error("No lat/long cols found.")
        if not model_node_id_cols:
            WranglerLogger.error("No *model_node_id cols found.")
        msg = "Could not find lat/long, geometry columns or *model_node_id column in \
                         table necessary to convert to GeoDataFrame"
        raise ValueError(msg)

    if lat_cols and lon_cols:
        # using first found lat and lon columns
        return gpd.GeoDataFrame(
            table,
            geometry=gpd.points_from_xy(table[lon_cols[0]], table[lat_cols[0]]),
            crs="EPSG:4326",
        )

    if model_node_id_cols:
        node_id_col = model_node_id_cols[0]

        if ref_nodes_df is None:
            if ref_road_net is None:
                msg = "Must provide either nodes_df or road_net to convert \
                                 model_node_id to geometry"
                raise ValueError(msg)
            ref_nodes_df = ref_road_net.nodes_df

        WranglerLogger.debug("Converting table to GeoDataFrame using model_node_id")

        _table = table.merge(
            ref_nodes_df[["model_node_id", "geometry"]],
            left_on=node_id_col,
            right_on="model_node_id",
        )
        return gpd.GeoDataFrame(_table, geometry="geometry")
    msg = "Could not find lat/long, geometry columns or *model_node_id column in table \
                        necessary to convert to GeoDataFrame"
    raise ValueError(msg)


def update_point_geometry(
    df: pd.DataFrame,
    ref_point_df: pd.DataFrame,
    lon_field: str = "X",
    lat_field: str = "Y",
    id_field: str = "model_node_id",
    ref_lon_field: str = "X",
    ref_lat_field: str = "Y",
    ref_id_field: str = "model_node_id",
) -> pd.DataFrame:
    """Returns copy of df with lat and long fields updated with geometry from ref_point_df.

    NOTE: does not update "geometry" field if it exists.
    """
    df = copy.deepcopy(df)

    ref_df = ref_point_df.rename(
        columns={
            ref_lon_field: lon_field,
            ref_lat_field: lat_field,
            ref_id_field: id_field,
        }
    )

    updated_df = update_df_by_col_value(
        df,
        ref_df[[id_field, lon_field, lat_field]],
        id_field,
        properties=[lat_field, lon_field],
        fail_if_missing=False,
    )
    return updated_df
