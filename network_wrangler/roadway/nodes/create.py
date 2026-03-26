"""Functions for creating nodes from data sources."""

import copy
import time

import geopandas as gpd
import pandas as pd
from pandera.typing import DataFrame
from pydantic import validate_call

from ...configs import DefaultConfig, WranglerConfig
from ...errors import NodeAddError
from ...logger import WranglerLogger
from ...models.roadway.tables import RoadLinksTable, RoadNodesAttrs, RoadNodesTable
from ...params import LAT_LON_CRS, SMALL_RECS
from ...utils.geo import get_point_geometry_from_linestring, point_from_xy
from ...utils.models import validate_df_to_model
from ..utils import set_df_index_to_pk


def _create_node_geometries_from_xy(
    nodes_df: pd.DataFrame | list[dict],
    in_crs: int = LAT_LON_CRS,
    net_crs: int = LAT_LON_CRS,
) -> gpd.GeoDataFrame:
    """Fixes geometries in nodes_df if necessary using X and Y columns.

    Args:
        nodes_df: nodes dataframe to fix geometries in.
        in_crs: coordinate system that X and Y are in. Defaults to LAT_LON_CRS.
        net_crs: coordinate system to convert geometries to. Defaults to
            LAT_LON_CRS.

    Returns:
        gpd.GeoDataFrame: nodes dataframe with fixed geometries.
    """
    if not isinstance(nodes_df, pd.DataFrame):
        nodes_df = pd.DataFrame(nodes_df)
    if "X" not in nodes_df.columns or "Y" not in nodes_df.columns:
        msg = "Must have X and Y properties to create geometries from."
        raise NodeAddError(msg)

    geo_start_time = time.time()
    if "geometry" in nodes_df:
        nodes_df["geometrys"] = nodes_df["geometry"].fillna(
            lambda x: point_from_xy(x["X"], x["Y"], xy_crs=in_crs, point_crs=net_crs),
        )
        WranglerLogger.debug(
            f"Filled missing geometry from X and Y in {round(time.time() - geo_start_time, 2)}."
        )
        return nodes_df

    node_geometries = nodes_df.apply(
        lambda x: point_from_xy(x["X"], x["Y"], xy_crs=in_crs, point_crs=net_crs),
        axis=1,
    )
    WranglerLogger.debug(
        f"Created node geometries from X and Y in {round(time.time() - geo_start_time, 2)}."
    )
    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=node_geometries, crs=in_crs)
    return nodes_gdf


@validate_call(config={"arbitrary_types_allowed": True})
def data_to_nodes_df(
    nodes_df: pd.DataFrame | gpd.GeoDataFrame | list[dict],
    config: WranglerConfig = DefaultConfig,  # noqa: ARG001
    in_crs: int = LAT_LON_CRS,
) -> DataFrame[RoadNodesTable]:
    """Turn nodes data into official nodes dataframe.

    Adds missing geometry.
    Makes sure X and Y are consistent with geometry GeoSeries.
    Converts to LAT_LON_CRS.
    Copies and sets idx to primary_key.
    Validates output to NodesSchema.

    Args:
        nodes_df : Nodes dataframe or list of dictionaries that can be converted to a dataframe.
        config: WranglerConfig instance. Defaults to DefaultConfig. NOTE: Not currently used.
        in_crs: Coordinate references system id incoming data xy is in, if it isn't already
            in a GeoDataFrame. Defaults to LAT_LON_CRS.

    Returns:
        gpd.GeoDataFrame: _description_
    """
    WranglerLogger.debug("Turning node data into official nodes_df")

    if isinstance(nodes_df, gpd.GeoDataFrame) and nodes_df.crs != LAT_LON_CRS:
        if nodes_df.crs is None:
            nodes_df.crs = in_crs
        nodes_df = nodes_df.to_crs(LAT_LON_CRS)

    if not isinstance(nodes_df, gpd.GeoDataFrame) or nodes_df.geometry.isnull().values.any():
        nodes_df = _create_node_geometries_from_xy(nodes_df, in_crs=in_crs, net_crs=LAT_LON_CRS)

    # Make sure values are consistent
    nodes_df["X"] = nodes_df["geometry"].apply(lambda g: g.x)
    nodes_df["Y"] = nodes_df["geometry"].apply(lambda g: g.y)

    if len(nodes_df) < SMALL_RECS:
        WranglerLogger.debug(f"nodes_df: \n{nodes_df[['model_node_id', 'geometry', 'X', 'Y']]}")

    # Validate and coerce to schema
    nodes_df = validate_df_to_model(nodes_df, RoadNodesTable)
    nodes_df.attrs.update(RoadNodesAttrs)
    nodes_df.gdf_name = nodes_df.attrs["name"]
    # lmz: why would we do this?
    nodes_df = set_df_index_to_pk(nodes_df)

    return nodes_df


def _create_nodes_from_link(
    links_df: DataFrame[RoadLinksTable], link_pos: int, node_key_field: str
) -> DataFrame[RoadNodesTable]:
    """Creates a basic list of node entries from links, their geometry, and a position.

    Useful for creating model network with simple geometry.
    TODO: Does not currently fill in additional values used in nodes.

    Args:
        links_df: subset of self.links_df or similar which needs nodes
            created
        link_pos (int): Position within geometry collection to use for geometry
        node_key_field (str): field name to use for generating index and node key

    Returns:
        DataFrame[RoadNodesTable]
    """
    nodes_df = copy.deepcopy(links_df[[node_key_field, "geometry"]].drop_duplicates())

    nodes_df = nodes_df.rename(columns={node_key_field: "model_node_id"})

    nodes_df["geometry"] = nodes_df["geometry"].apply(
        get_point_geometry_from_linestring, pos=link_pos
    )
    nodes_df["X"] = nodes_df.geometry.x
    nodes_df["Y"] = nodes_df.geometry.y
    nodes_df["model_node_id_idx"] = nodes_df["model_node_id"]
    nodes_df.set_index("model_node_id_idx", inplace=True)
    nodes_df = validate_df_to_model(nodes_df, RoadNodesTable)
    # WranglerLogger.debug(f"ct3: nodes_df:\n{nodes_df}")
    return nodes_df


def generate_node_ids(nodes_df: DataFrame[RoadNodesTable], range: tuple[int], n: int) -> list[int]:
    """Generate unique node ids for nodes_df.

    Args:
        nodes_df: nodes dataframe to generate unique ids for.
        range: range of ids to generate from.
        n: number of ids to generate.

    Returns:
        list[int]: list of unique node ids.
    """
    if n <= 0:
        return []
    existing_ids = set(nodes_df["model_node_id"].unique())
    new_ids = set(range) - existing_ids
    if len(new_ids) < n:
        msg = f"Only {len(new_ids)} new ids available, need {n}."
        raise NodeAddError(msg)

    return list(new_ids)[:n]
