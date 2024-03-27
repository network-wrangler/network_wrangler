import os
from pathlib import Path
from typing import Union

import geopandas as gpd

from .roadwaynetwork import RoadwayNetwork
from .transitnetwork import TransitNetwork
from .logger import WranglerLogger


def net_to_mapbox(
    roadway: Union[RoadwayNetwork, gpd.GeoDataFrame, str, Path] = gpd.GeoDataFrame(),
    transit: Union[TransitNetwork, gpd.GeoDataFrame] = gpd.GeoDataFrame(),
    roadway_geojson_out: str = "roadway_shapes.geojson",
    transit_geojson_out: str = "transit_shapes.geojson",
    mbtiles_out: str = "network.mbtiles",
    overwrite: bool = True,
    port: str = "9000",
):
    """
    Creates and serves mapbox tiles on local web server based on roadway and transit networks.

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
    import subprocess

    # test for mapbox token
    try:
        os.getenv("MAPBOX_ACCESS_TOKEN")
    except:
        raise (
            "NEED TO SET MAPBOX ACCESS TOKEN IN ENVIRONMENT VARIABLES/n \
                In command line: >>export MAPBOX_ACCESS_TOKEN='pk.0000.1111' # \
                replace value with your mapbox public access token"
        )

    if isinstance(transit, TransitNetwork):
        transit = TransitNetwork.transit_net_to_gdf(transit)
    if isinstance(transit, RoadwayNetwork):
        roadway = RoadwayNetwork.roadway_net_to_gdf(roadway)

    if isinstance(transit, gpd.GeoDataFrame):
        transit.to_file(transit_geojson_out, driver="GeoJSON")
    elif Path(transit).exists():
        transit_geojson_out = transit
    else:
        raise ValueError(f"Don't understand transit input: {transit}")

    if isinstance(roadway, gpd.GeoDataFrame):
        roadway.to_file(roadway_geojson_out, driver="GeoJSON")
    elif Path(roadway).exists():
        roadway_geojson_out = roadway
    else:
        raise ValueError(f"Don't understand roadway input: {roadway}")

    tippe_options_list = ["-zg", "-o", mbtiles_out]
    if overwrite:
        tippe_options_list.append("--force")
    # tippe_options_list.append("--drop-densest-as-needed")
    tippe_options_list.append(roadway_geojson_out)
    tippe_options_list.append(transit_geojson_out)

    try:
        WranglerLogger.info(
            "Running tippecanoe with following options: {}".format(
                " ".join(tippe_options_list)
            )
        )
        subprocess.run(["tippecanoe"] + tippe_options_list)
    except:
        WranglerLogger.error()
        raise (
            "If tippecanoe isn't installed, try `brew install tippecanoe` or \
                visit https://github.com/mapbox/tippecanoe"
        )

    try:
        WranglerLogger.info(
            "Running mbview with following options: {}".format(
                " ".join(tippe_options_list)
            )
        )
        subprocess.run(["mbview", "--port", port, f",/{mbtiles_out}"])
    except:
        WranglerLogger.error()
        raise (
            "If mbview isn't installed, try `npm install -g @mapbox/mbview` or \
                visit https://github.com/mapbox/mbview"
        )