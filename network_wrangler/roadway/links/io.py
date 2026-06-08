"""Functions to read in and write out a RoadLinksTable."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from pandera.typing import DataFrame

from ...configs import DefaultConfig, WranglerConfig
from ...logger import WranglerLogger
from ...models._base.types import GeoFileTypes
from ...models.roadway.converters import translate_links_df_v1_to_v0
from ...models.roadway.tables import RoadLinksTable, RoadNodesTable
from ...params import LAT_LON_CRS
from ...utils.io_table import read_table, write_table
from ...utils.models import order_fields_from_data_model, validate_call_pyd
from .create import data_to_links_df


@validate_call_pyd
def read_links(
    filename: Path,
    in_crs: int = LAT_LON_CRS,
    config: WranglerConfig = DefaultConfig,
    nodes_df: DataFrame[RoadNodesTable] = None,
    filter_to_nodes: bool = False,
) -> DataFrame[RoadLinksTable]:
    """Reads links and returns a geodataframe of links conforming to RoadLinksTable.

    Sets index to be a copy of the primary key.
    Validates output dataframe using RoadLinksTable

    Args:
        filename (str): file to read links in from.
        in_crs: coordinate reference system number any link geometries are stored in.
            Defaults to 4323.
        config: WranglerConfig instance. Defaults to DefaultConfig.
        nodes_df: a RoadNodesTable to gather geometry from. Necesary if geometry is not
            provided. Defaults to None.
        filter_to_nodes: if True, will filter links to only those that connect to nodes. Requires
            nodes_df to be provided. Defaults to False.
    """
    WranglerLogger.info(f"Reading links from {filename}.")
    start_t = time.time()
    if filter_to_nodes is True and nodes_df is None:
        msg = "If filter_to_nodes is True, nodes_df must be provided."
        raise ValueError(msg)

    links_df = read_table(filename, read_speed=config.CPU.EST_PD_READ_SPEED)

    if filter_to_nodes:
        WranglerLogger.debug("Filtering links to only those that connect to nodes.")
        links_df = links_df[
            links_df["A"].isin(nodes_df.model_node_id) & links_df["B"].isin(nodes_df.model_node_id)
        ]

    WranglerLogger.debug(f"Read {len(links_df)} links in {round(time.time() - start_t, 2)}.")
    links_df = data_to_links_df(links_df, in_crs=in_crs, nodes_df=nodes_df)
    links_df.attrs["source_file"] = filename
    WranglerLogger.info(
        f"Read + transformed {len(links_df):,} links from "
        f"{filename} in {round(time.time() - start_t, 2)}."
    )
    return links_df


@validate_call_pyd
def write_links(
    links_df: DataFrame[RoadLinksTable],
    out_dir: str | Path = ".",
    convert_complex_properties_to_single_field: bool = False,
    prefix: str = "",
    file_format: GeoFileTypes = "json",
    overwrite: bool = False,
    include_geometry: bool = False,
) -> None:
    """Writes links to a file.

    Args:
        links_df: DataFrame[RoadLinksTable] to write out.
        convert_complex_properties_to_single_field: if True, will convert complex properties to a
            single column consistent with v0 format.  This format is NOT valid
            with parquet and many other softwares. Defaults to False.
        out_dir: directory to write files to. Defaults to ".".
        prefix: prefix to add to the filename. Defaults to "".
        file_format: file format to write out to. Defaults to "json".
        overwrite: if True, will overwrite existing files. Defaults to False.
        include_geometry: if True, will include geometry in the output. Defaults to False.
    """
    if not include_geometry and file_format == "geojson":
        file_format = "json"

    links_file = Path(out_dir) / f"{prefix}link.{file_format}"

    if convert_complex_properties_to_single_field:
        if file_format == "parquet":
            WranglerLogger.error(
                "convert_complex_properties_to_single_column is not supported with parquet. \
                Setting to False."
            )
            convert_complex_properties_to_single_field = False
        v1_links_df = links_df.copy()
        links_df = translate_links_df_v1_to_v0(v1_links_df)

    if not include_geometry:
        geo_cols = links_df.select_dtypes(include=["geometry"]).columns.tolist()
        links_df = pd.DataFrame(links_df)
        links_df = links_df.drop(columns=geo_cols)

    links_df = order_fields_from_data_model(links_df, RoadLinksTable)
    write_table(links_df, links_file, overwrite=overwrite)
