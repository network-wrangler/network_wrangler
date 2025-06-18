"""Functions for creating RoadLinksTables."""

import copy
import time
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
from pandera.typing import DataFrame

from ...errors import LinkCreationError
from ...logger import WranglerLogger
from ...models.roadway.converters import (
    detect_v0_scoped_link_properties,
    translate_links_df_v0_to_v1,
)
from ...models.roadway.tables import RoadLinksAttrs, RoadLinksTable, RoadNodesTable
from ...params import LAT_LON_CRS, SMALL_RECS
from ...utils.data import coerce_gdf
from ...utils.geo import (
    _harmonize_crs,
    length_of_linestring_miles,
    linestring_from_nodes,
    offset_geometry_meters,
)
from ...utils.models import validate_call_pyd, validate_df_to_model
from ..utils import create_unique_shape_id, set_df_index_to_pk


def shape_id_from_link_geometry(
    links_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Create a unique shape_id from the geometry of the link."""
    shape_ids = links_df["geometry"].apply(create_unique_shape_id)
    return shape_ids


def _fill_missing_link_geometries_from_nodes(
    links_df: pd.DataFrame, nodes_df: Optional[DataFrame[RoadNodesTable]] = None
) -> gpd.GeoDataFrame:
    """Create location references and link geometries from nodes.

    If already has either, then will just fill NA.
    """
    if "geometry" not in links_df:
        links_df["geometry"] = None
    if links_df.geometry.isna().values.any():
        geo_start_t = time.time()
        if nodes_df is None:
            msg = "Must give nodes_df argument if don't have Geometry"
            raise LinkCreationError(msg)
        updated_links_df = linestring_from_nodes(links_df.loc[links_df.geometry.isna()], nodes_df)
        links_df.update(updated_links_df)
        # WranglerLogger.debug(f"links with updated geometries:\n{links_df}")
        msg = f"Created link geo from nodes in {round(time.time() - geo_start_t, 2)}."
        # WranglerLogger.debug(msg)
    return links_df


def _fill_missing_distance_from_geometry(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # WranglerLogger.debug(f"_fill_missing_distance_from_geometry.df input:\n{df}.")
    if "distance" not in df:
        df["distance"] = None
    if df["distance"].isna().values.any():
        df.loc[df["distance"].isna(), "distance"] = length_of_linestring_miles(
            df.loc[df["distance"].isna()]
        )
    return df


@validate_call_pyd
def data_to_links_df(
    links_df: Union[pd.DataFrame, list[dict]],
    in_crs: int = LAT_LON_CRS,
    nodes_df: Union[None, DataFrame[RoadNodesTable]] = None,
) -> DataFrame[RoadLinksTable]:
    """Create a links dataframe from list of link properties + link geometries or associated nodes.

    Sets index to be a copy of the primary key.
    Validates output dataframe using LinksSchema.

    Args:
        links_df (pd.DataFrame): df or list of dictionaries of link properties
        in_crs: coordinate reference system id for incoming links if geometry already exists.
            Defaults to LAT_LON_CRS. Will convert everything to LAT_LON_CRSif it doesn't match.
        nodes_df: Associated notes geodataframe to use if geometries or location references not
            present. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    WranglerLogger.debug(f"Creating {len(links_df)} links.")
    if not isinstance(links_df, pd.DataFrame):
        links_df = pd.DataFrame(links_df)
    # WranglerLogger.debug(f"data_to_links_df.links_df input: \n{links_df.head}.")

    v0_link_properties = detect_v0_scoped_link_properties(links_df)
    if v0_link_properties:
        links_df = translate_links_df_v0_to_v1(links_df, complex_properties=v0_link_properties)

    links_df = _fill_missing_link_geometries_from_nodes(links_df, nodes_df)
    # Now that have geometry, make sure is GDF
    links_df = coerce_gdf(links_df, in_crs=in_crs, geometry=links_df.geometry)

    links_df = _fill_missing_distance_from_geometry(links_df)

    links_df = _harmonize_crs(links_df, LAT_LON_CRS)
    nodes_df = _harmonize_crs(nodes_df, LAT_LON_CRS)

    links_df.attrs.update(RoadLinksAttrs)
    links_df = set_df_index_to_pk(links_df)
    links_df.gdf_name = links_df.attrs["name"]
    links_df = validate_df_to_model(links_df, RoadLinksTable)

    if len(links_df) < SMALL_RECS:
        WranglerLogger.debug(
            f"New Links: \n{links_df[links_df.attrs['display_cols'] + ['geometry']]}"
        )
    else:
        WranglerLogger.debug(f"{len(links_df)} new links.")

    return links_df


def copy_links(
    links_df: DataFrame[RoadLinksTable],
    link_id_lookup: dict[int, int],
    node_id_lookup: dict[int, int],
    updated_geometry_col: Optional[str] = None,
    nodes_df: Optional[DataFrame[RoadNodesTable]] = None,
    offset_meters: float = -5,
    copy_properties: Optional[list[str]] = None,
    rename_properties: Optional[dict[str, str]] = None,
    name_prefix: str = "copy of",
    validate: bool = True,
) -> DataFrame[RoadLinksTable]:
    """Copy links and optionally offset them.

    Will get geometry from another column if provided, otherwise will use nodes_df and then
    offset_meters to offset from previous geometry.

    Args:
        links_df (DataFrame[RoadLinksTable]): links dataframe of links to copy
        link_id_lookup (dict[int, int]): lookup of new link ID from old link id.
        node_id_lookup (dict[int, int]): lookup of new node ID from old node id.
        updated_geometry_col (str): name of the column to store the updated geometry.
            Will nodes_df for missing geometries if provided and offset_meters if not.
            Defaults to None.
        nodes_df (DataFrame[RoadNodesTable]): nodes dataframe of nodes to use for new
            link geometry. Defaults to None. If not provided, will use offset_meters.
        offset_meters (float): distance to offset links if nodes_df is not provided.
            Defaults to -5.
        copy_properties (list[str], optional): properties to keep. Defaults to [].
        rename_properties (dict[str, str], optional): properties to rename. Defaults to {}.
            Will default to REQUIRED_RENAMES if keys in that dict are not provided.
        name_prefix (str, optional): format string for new names. Defaults to "copy of".
        validate (bool, optional): whether to validate the output dataframe. Defaults to True.
            If set to false, you should validate the output dataframe before using it.

    Returns:
        DataFrame[RoadLinksTable]: offset links dataframe
    """
    copy_properties = copy_properties or []
    rename_properties = rename_properties or {}

    REQUIRED_KEEP = ["A", "B", "name", "distance", "geometry", "model_link_id"]

    # Should rename these columns to these columns - unless overriden by rename_properties
    REQUIRED_RENAMES = {
        "A": "source_A",
        "B": "source_B",
        "model_link_id": "source_model_link_id",
        "geometry": "source_geometry",
    }
    # cannot rename a column TO these fields
    FORBIDDEN_RENAMES = ["A", "B", "model_link_id", "geometry", "name"]
    WranglerLogger.debug(f"Copying {len(links_df)} links.")

    rename_properties = {k: v for k, v in rename_properties.items() if v not in FORBIDDEN_RENAMES}
    REQUIRED_RENAMES.update(rename_properties)
    # rename if different, otherwise copy
    rename_properties = {k: v for k, v in REQUIRED_RENAMES.items() if k != v}
    copy_properties += [
        k for k, v in REQUIRED_RENAMES.items() if k == v and k not in copy_properties
    ]

    _missing_copy_properties = set(copy_properties) - set(links_df.columns)
    if _missing_copy_properties:
        WranglerLogger.warning(
            f"Specified properties to copy not found in links_df.\
            Proceeding without copying: {_missing_copy_properties}"
        )
        copy_properties = [c for c in copy_properties if c not in _missing_copy_properties]

    _missing_rename_properties = set(rename_properties.keys()) - set(links_df.columns)
    if _missing_rename_properties:
        WranglerLogger.warning(
            f"Specified properties to rename not found in links_df.\
            Proceeding without renaming: {_missing_rename_properties}"
        )
        rename_properties = {
            k: v for k, v in rename_properties.items() if k not in _missing_rename_properties
        }

    offset_links = copy.deepcopy(links_df)
    drop_before_rename = [k for k in rename_properties.values() if k in offset_links.columns]
    offset_links = offset_links.drop(columns=drop_before_rename)
    offset_links = offset_links.rename(columns=rename_properties)

    offset_links["A"] = offset_links["source_A"].map(node_id_lookup)
    offset_links["B"] = offset_links["source_B"].map(node_id_lookup)
    offset_links["model_link_id"] = offset_links["source_model_link_id"].map(link_id_lookup)
    offset_links["name"] = name_prefix + " " + offset_links["name"]

    if updated_geometry_col is not None:
        offset_links = offset_links.rename(columns={updated_geometry_col: "geometry"})
    else:
        offset_links["geometry"] = None

    if nodes_df is None and offset_links.geometry.isna().values.any():
        WranglerLogger.debug(
            f"Adding node-based geometry with for {sum(offset_links.geometry.isna())} links."
        )
        offset_links.loc[[offset_links.geometry.isna(), "geometry"]] = offset_geometry_meters(
            offset_links["geometry"],
            offset_meters,
        )
    if offset_links.geometry.isna().values.any():
        WranglerLogger.debug(
            f"Adding offset geometry with for {sum(offset_links.geometry.isna())} links."
        )
        offset_links.loc[[offset_links.geometry.isna(), "geometry"]] = linestring_from_nodes(
            offset_links, nodes_df
        )

    offset_links = offset_links.set_geometry("geometry", inplace=False)
    offset_links.crs = links_df.crs
    offset_links["distance"] = length_of_linestring_miles(offset_links["geometry"])

    keep_properties = list(set(copy_properties + REQUIRED_KEEP + list(rename_properties.values())))
    offset_links = offset_links[keep_properties]

    # create and set index for new model_link_ids
    # offset_links.attrs.update(RoadLinksAttrs)
    offset_links = offset_links.reset_index(drop=True)
    offset_links = set_df_index_to_pk(offset_links)

    if validate:
        offset_links = validate_df_to_model(offset_links, RoadLinksTable)
    else:
        WranglerLogger.warning(
            "Skipping validation of offset links. Validate to RoadLinksTable before using."
        )
    return offset_links
