"""Functions for reading and writing nodes data."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geopandas import GeoDataFrame
from pandera.typing import DataFrame
from pydantic import validate_call

from ...configs import DefaultConfig, WranglerConfig
from ...logger import WranglerLogger
from ...models._base.types import GeoFileTypes
from ...models.roadway.tables import RoadNodesAttrs, RoadNodesTable
from ...params import LAT_LON_CRS
from ...utils.io_table import read_table, write_table
from ...utils.models import order_fields_from_data_model, validate_call_pyd, validate_df_to_model
from .create import data_to_nodes_df

if TYPE_CHECKING:
    from ...transit.network import TransitNetwork
    from ..network import RoadwayNetwork


@validate_call(config={"arbitrary_types_allowed": True})
def read_nodes(
    filename: Path,
    in_crs: int = LAT_LON_CRS,
    boundary_gdf: GeoDataFrame | None = None,
    boundary_geocode: str | None = None,
    boundary_file: Path | None = None,
    config: WranglerConfig = DefaultConfig,
) -> DataFrame[RoadNodesTable]:
    """Reads nodes and returns a geodataframe of nodes.

    Sets index to be a copy of the primary key.
    Validates output dataframe using NodesSchema.

    Args:
        filename (Path,str): file to read links in from.
        in_crs: coordinate reference system number that node data is in. Defaults to LAT_LON_CRS.
        boundary_gdf: GeoDataFrame to filter the input data to. Only used for geographic data.
            efaults to None.
        boundary_geocode: Geocode to filter the input data to. Only used for geographic data.
            Defaults to None.
        boundary_file: File to load as a boundary to filter the input data to. Only used for
            geographic data. Defaults to None.
        config: WranglerConfig instance. Defaults to DefaultConfig.
    """
    WranglerLogger.debug(f"Reading nodes from {filename}.")

    start_time = time.time()

    nodes_df = read_table(
        filename,
        boundary_gdf=boundary_gdf,
        boundary_geocode=boundary_geocode,
        boundary_file=boundary_file,
        read_speed=config.CPU.EST_PD_READ_SPEED,
    )
    WranglerLogger.debug(
        f"Read {len(nodes_df)} nodes from file in {round(time.time() - start_time, 2)}."
    )

    nodes_df = data_to_nodes_df(nodes_df, in_crs=in_crs, config=config)
    nodes_df.attrs["source_file"] = filename
    WranglerLogger.info(
        f"Read {len(nodes_df)} nodes from {filename} in {round(time.time() - start_time, 2)}."
    )
    nodes_df = validate_df_to_model(nodes_df, RoadNodesTable)
    return nodes_df


@validate_call_pyd
def nodes_df_to_geojson(nodes_df: DataFrame[RoadNodesTable], properties: list[str]):
    """Converts a nodes dataframe to a geojson.

    Attribution: Geoff Boeing:
    https://geoffboeing.com/2015/10/exporting-python-data-geojson/.
    """
    # TODO write wrapper on validate call so don't have to do this
    nodes_df.attrs.update(RoadNodesAttrs)
    geojson = {"type": "FeatureCollection", "features": []}
    for _, row in nodes_df.iterrows():
        feature: dict[str, Any] = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Point", "coordinates": []},
        }
        feature["geometry"]["coordinates"] = [row["geometry"].x, row["geometry"].y]
        feature["properties"][nodes_df.model_node_id] = row.name
        for prop in properties:
            feature["properties"][prop] = row[prop]
        geojson["features"].append(feature)
    return geojson


@validate_call_pyd
def write_nodes(
    nodes_df: DataFrame[RoadNodesTable],
    out_dir: str | Path,
    prefix: str,
    file_format: GeoFileTypes = "geojson",
    overwrite: bool = True,
) -> None:
    """Writes RoadNodesTable to file.

    Args:
        nodes_df: nodes dataframe
        out_dir: directory to write nodes to
        prefix: prefix to add to nodes file name
        file_format: format to write nodes in. e.g. "geojson" shp" "parquet" "csv" "txt". Defaults
            to "geojson".
        overwrite: whether to overwrite existing nodes file. Defaults to True.
    """
    nodes_file = Path(out_dir) / f"{prefix}node.{file_format}"
    nodes_df = order_fields_from_data_model(nodes_df, RoadNodesTable)
    write_table(nodes_df, nodes_file, overwrite=overwrite)


def get_nodes(
    transit_net: TransitNetwork | None = None,
    roadway_net: RoadwayNetwork | None = None,
    roadway_path: str | Path | None = None,
    config: WranglerConfig = DefaultConfig,
) -> GeoDataFrame:
    """Get nodes from a transit network, roadway network, or roadway file.

    Args:
        transit_net: TransitNetwork instance
        roadway_net: RoadwayNetwork instance
        roadway_path: path to a directory with roadway network
        config: WranglerConfig instance. Defaults to DefaultConfig.
    """
    if transit_net is not None and transit_net.road_net is not None:
        return transit_net.road_net.nodes_df
    if roadway_net is not None:
        return roadway_net.nodes_df
    if roadway_path is not None:
        nodes_path = Path(roadway_path)
        if nodes_path.is_dir():
            nodes_path = next(nodes_path.glob("*node*."))
        return read_nodes(nodes_path, config=config)
    msg = "nodes_df must either be given or provided via an associated road_net or by providing a roadway_net path or instance."
    raise ValueError(msg)
