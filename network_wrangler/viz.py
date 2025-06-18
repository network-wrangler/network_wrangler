"""Module for visualizing roadway and transit networks using Mapbox tiles.

This module provides a function `net_to_mapbox` that creates and serves Mapbox tiles on a local web server based on roadway and transit networks.

Example usage:
    net_to_mapbox(roadway, transit)
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pyarrow as pa

from .logger import WranglerLogger
from .roadway.network import RoadwayNetwork
from .transit.network import TransitNetwork


class MissingMapboxTokenError(Exception):
    """Raised when MAPBOX_ACCESS_TOKEN is not found in environment variables."""


def net_to_mapbox(
    roadway: Optional[Union[RoadwayNetwork, gpd.GeoDataFrame, str, Path]] = None,
    transit: Optional[Union[TransitNetwork, gpd.GeoDataFrame]] = None,
    roadway_geojson_out: Path = Path("roadway_shapes.geojson"),
    transit_geojson_out: Path = Path("transit_shapes.geojson"),
    mbtiles_out: Path = Path("network.mbtiles"),
    overwrite: bool = True,
    port: str = "9000",
):
    """Creates and serves mapbox tiles on local web server based on roadway and transit networks.

    Args:
        roadway: a RoadwayNetwork instance, geodataframe with roadway linetrings, or path to a
            geojson file. Defaults to empty GeoDataFrame.
        transit: a TransitNetwork instance or a geodataframe with roadway linetrings, or path to a
            geojson file. Defaults to empty GeoDataFrame.
        roadway_geojson_out: file path for roadway geojson which gets created if roadway is not
            a path to a geojson file. Defaults to roadway_shapes.geojson.
        transit_geojson_out: file path for transit geojson which gets created if transit is not
            a path to a geojson file. Defaults to transit_shapes.geojson.
        mbtiles_out: path to output mapbox tiles. Defaults to network.mbtiles
        overwrite: boolean indicating if can overwrite mbtiles_out and roadway_geojson_out and
            transit_geojson_out. Defaults to True.
        port: port to serve resulting tiles on. Defaults to 9000.
    """
    if roadway is None:
        roadway = gpd.GeoDataFrame()
    if transit is None:
        transit = gpd.GeoDataFrame()
    # test for mapbox token
    try:
        os.getenv("MAPBOX_ACCESS_TOKEN")
    except Exception as err:
        WranglerLogger.error(
            "NEED TO SET MAPBOX ACCESS TOKEN IN ENVIRONMENT VARIABLES/n \
                In command line: >>export MAPBOX_ACCESS_TOKEN='pk.0000.1111' # \
                replace value with your mapbox public access token"
        )
        raise MissingMapboxTokenError() from err

    if isinstance(transit, TransitNetwork):
        transit = transit.shape_links_gdf
        transit.to_file(transit_geojson_out, driver="GeoJSON")
    elif Path(transit).exists():
        transit_geojson_out = transit
    else:
        msg = f"Don't understand transit input: {transit}"
        raise ValueError(msg)

    if isinstance(roadway, RoadwayNetwork):
        roadway = roadway.link_shapes_df
        roadway.to_file(roadway_geojson_out, driver="GeoJSON")
    elif Path(roadway).exists():
        roadway_geojson_out = Path(roadway)
    else:
        msg = "Don't understand roadway input: {roadway}"
        raise ValueError(msg)

    tippe_options_list: list[str] = ["-zg", "-o", str(mbtiles_out)]
    if overwrite:
        tippe_options_list.append("--force")
    # tippe_options_list.append("--drop-densest-as-needed")
    tippe_options_list.append(str(roadway_geojson_out))
    tippe_options_list.append(str(transit_geojson_out))

    try:
        WranglerLogger.info(
            f"Running tippecanoe with following options: {' '.join(tippe_options_list)}"
        )
        subprocess.run(["tippecanoe", *tippe_options_list], check=False)
    except Exception as err:
        WranglerLogger.error(
            "If tippecanoe isn't installed, try `brew install tippecanoe` or \
                visit https://github.com/mapbox/tippecanoe"
        )
        raise ImportError() from err

    try:
        WranglerLogger.info(
            "Running mbview with following options: {}".format(" ".join(tippe_options_list))
        )
        subprocess.run(["mbview", "--port", port, f", /{mbtiles_out}"], check=False)
    except Exception as err:
        WranglerLogger.error(
            "If mbview isn't installed, try `npm install -g @mapbox/mbview` or \
                visit https://github.com/mapbox/mbview"
        )
        raise ImportError(msg) from err
