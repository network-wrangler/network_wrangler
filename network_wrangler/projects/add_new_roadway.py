import copy
from typing import Collection

import geopandas as gpd

from ..roadwaynetwork import RoadwayNetwork
from ..roadway.nodes import nodes_data_to_nodes_df
from ..roadway.links import links_data_to_links_df
from ..logger import WranglerLogger


def apply_new_roadway(
    roadway_net: RoadwayNetwork,
    add_links: Collection[dict] = [],
    add_nodes: Collection[dict] = [],
) -> RoadwayNetwork:
    """
    Add the new roadway features defined in the project card.

    New nodes are added first so that links can refer to any added nodes.

    args:
        roadway_net: input RoadwayNetwork to apply change to
        add_links: list of dictionaries
        add_nodes: list of dictionaries

    returns: updated network with new links and nodes and associated geometries
    """
    WranglerLogger.debug(
        f"Adding New Roadway Features:\n-Links:\n{add_links}\n-Nodes:\n{add_nodes}"
    )
    if add_nodes:
        _new_nodes_df = nodes_data_to_nodes_df(
            add_nodes, nodes_params=roadway_net.nodes_df.params
        )
        roadway_net.add_nodes(_new_nodes_df)

    if add_links:
        _new_links_df = links_data_to_links_df(
            add_links,
            links_params=roadway_net.links_df.params,
            nodes_df=roadway_net.nodes_df,
        )

        roadway_net.add_links(_new_links_df)

    return roadway_net


def _create_new_shapes_from_links(
    roadway_net: RoadwayNetwork, links_df: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    new_shapes_df = copy.deepcopy(
        links_df[[roadway_net.shapes_df.params.primary_key, "geometry"]]
    )
    return new_shapes_df
