"""Functions to convert RoadwayNetwork to osmnx graph and perform graph operations."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import networkx as nx
import osmnx as ox
from geopandas import GeoDataFrame
from pandas import DataFrame

from ..logger import WranglerLogger
from .nodes.filters import filter_nodes_to_links

if TYPE_CHECKING:
    from .network import RoadwayNetwork

ox_major_version = int(ox.__version__.split(".")[0])


"""Column to use for weights in shortest path.
"""
DEFAULT_GRAPH_WEIGHT_COL = "distance"

"""Factor to multiply sp_weight_col by to use for weights in shortest path.
"""
DEFAULT_GRAPH_WEIGHT_FACTOR = 1


def _drop_complex_df_columns(df: DataFrame) -> DataFrame:
    """Returns dataframe without columns with lists, tuples or dictionaries types."""
    _cols_to_exclude = ["geometry"]
    _cols_to_search = [c for c in df.columns if c not in _cols_to_exclude]
    _drop_types = (list, dict, tuple)

    _drop_cols = [c for c in _cols_to_search if df[c].apply(type).isin(_drop_types).any()]

    df = df.drop(_drop_cols, axis=1)

    return df


def _nodes_to_graph_nodes(nodes_df: GeoDataFrame) -> GeoDataFrame:
    """Transformes RoadwayNetwork nodes_df into format osmnx is expecting.

    OSMNX is expecting:
    - columns: id, x, y
    - gdf attribute:  gdf_name
    - property values which are simple int or strings

    Args:
        nodes_df (GeoDataFrame): nodes geodataframe from RoadwayNetwork instance
    """
    graph_nodes_df = copy.deepcopy(nodes_df)
    graph_nodes_df.gdf_name = "network_nodes"

    # drop column types which could have complex types (i.e. lists, dicts, etc)
    graph_nodes_df = _drop_complex_df_columns(graph_nodes_df)

    # the model_node_id is the index
    graph_nodes_df.set_index("model_node_id", inplace=True)

    # OSMNX is expecting id, x, y
    graph_nodes_df["id"] = graph_nodes_df.index
    graph_nodes_df = graph_nodes_df.rename(columns={"X": "x", "Y": "y"})

    return graph_nodes_df


def _links_to_graph_links(
    links_df: GeoDataFrame,
    sp_weight_col: str = DEFAULT_GRAPH_WEIGHT_COL,
    sp_weight_factor: float = DEFAULT_GRAPH_WEIGHT_FACTOR,
) -> GeoDataFrame:
    """Transformes RoadwayNetwork links_df into format osmnx is expecting.

    OSMNX is expecting:

    Args:
        links_df (GeoDataFrame): links geodataframe from RoadwayNetwork instance
        sp_weight_col: column to use for weights. Defaults to `distance`.
        sp_weight_factor: multiple to apply to the weights. Defaults to 1.
    """
    graph_links_df = copy.deepcopy(links_df)

    # drop column types which could have complex types (i.e. lists, dicts, etc)
    graph_links_df = _drop_complex_df_columns(graph_links_df)

    # have to add in weights to use for shortest paths before doing the conversion to a graph
    if sp_weight_col not in graph_links_df.columns:
        WranglerLogger.warning(
            f"{sp_weight_col} not in graph_links_df so adding and initializing to 1."
        )
        graph_links_df[sp_weight_col] = 1

    graph_links_df["weight"] = graph_links_df[sp_weight_col] * sp_weight_factor

    # osm-nx is expecting u and v instead of A B - but first have to drop existing u/v

    if "u" in graph_links_df.columns and (graph_links_df.u != graph_links_df.A).any():
        graph_links_df = graph_links_df.drop("u", axis=1)

    if "v" in graph_links_df.columns and (graph_links_df.v != graph_links_df.B).any():
        graph_links_df = graph_links_df.drop("v", axis=1)

    graph_links_df = graph_links_df.rename(columns={"A": "u", "B": "v"})

    # lmz: Why not use model_link_id as the key rather than index?

    graph_links_df["key"] = graph_links_df.index.copy()
    # Per osmnx u,v,key should be a multi-index;
    #     https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.utils_graph.graph_from_gdfs
    # However - if the index is set before hand in osmnx version <1.0 then it fails
    #     on the set_index line *within* osmnx.utils_graph.graph_from_gdfs():
    #           `for (u, v, k), row in gdf_edges.set_index(["u", "v", "key"]).iterrows()
    if ox_major_version >= 1:
        graph_links_df = graph_links_df.set_index(keys=["u", "v", "key"], drop=True)

    return graph_links_df


def links_nodes_to_ox_graph(
    links_df: GeoDataFrame,
    nodes_df: GeoDataFrame,
    sp_weight_col: str = "distance",
    sp_weight_factor: float = 1,
):
    """Create an osmnx-flavored network graph from nodes and links dfs.

    osmnx doesn't like values that are arrays, so remove the variables
    that have arrays.  osmnx also requires that certain variables
    be filled in, so do that too.

    Args:
        links_df: links_df from RoadwayNetwork
        nodes_df: nodes_df from RoadwayNetwork
        sp_weight_col: column to use for weights. Defaults to `distance`.
        sp_weight_factor: multiple to apply to the weights. Defaults to 1.

    Returns: a networkx multidigraph
    """
    # WranglerLogger.debug("links_nodes_to_ox_graph()")
    graph_nodes_df = _nodes_to_graph_nodes(nodes_df)
    # WranglerLogger.debug(f"graph_nodes_df:\n{graph_nodes_df}")
    graph_links_df = _links_to_graph_links(
        links_df,
        sp_weight_col=sp_weight_col,
        sp_weight_factor=sp_weight_factor,
    )
    # WranglerLogger.debug(f"graph_links_df:\n{graph_links_df}")

    try:
        WranglerLogger.debug(
            f"starting ox.graph_from_gdfs() with {len(graph_nodes_df)=:,} {len(graph_links_df)=:,}"
        )
        G = ox.graph_from_gdfs(graph_nodes_df, graph_links_df)

    except AttributeError as attr_error:
        if attr_error.args[0] == "module 'osmnx' has no attribute 'graph_from_gdfs'":
            # This is the only exception for which we have a workaround
            # Does this still work given the u,v,key multi-indexing?
            #
            WranglerLogger.warning(
                "Please upgrade your OSMNX package. For now, using deprecated "
                "osmnx.gdfs_to_graph because osmnx.graph_from_gdfs not found"
            )
            G = ox.gdfs_to_graph(graph_nodes_df, graph_links_df)
        else:
            # for other AttributeErrors, raise further
            raise attr_error
    except Exception as e:
        raise e

    WranglerLogger.debug("Created osmnx graph from RoadwayNetwork")
    return G


def net_to_graph(net: RoadwayNetwork, mode: str | None = None) -> nx.MultiDiGraph:
    """Converts a network to a MultiDiGraph.

    Args:
        net: RoadwayNetwork object
        mode: mode of the network, one of `drive`,`transit`,
            `walk`, `bike`

    Returns: networkx: osmnx: DiGraph  of network
    """
    _links_df = net.links_df.mode_query(mode)

    _nodes_df = filter_nodes_to_links(_links_df, net.nodes_df)

    G = links_nodes_to_ox_graph(_links_df, _nodes_df)

    return G


def shortest_path(G: nx.MultiDiGraph, O_id, D_id, sp_weight_property="weight") -> list | None:
    """Calculates the shortest path between two nodes in a network.

    Args:
        G: osmnx MultiDiGraph, created using links_nodes_to_ox_graph
        O_id: primary key for start node
        D_id: primary key for end node
        sp_weight_property: link property to use as weight in finding shortest path.
            Defaults to "weight".

    Returns: tuple with length of four
    - Boolean if shortest path found
    - nx Directed graph of graph links
    - route of shortest path nodes as List
    - links in shortest path selected from links_df
    """
    try:
        sp_route = nx.shortest_path(G, O_id, D_id, weight=sp_weight_property)
        WranglerLogger.debug("Shortest path successfully routed")
    except nx.NetworkXNoPath:
        WranglerLogger.debug(f"No SP from {O_id} to {D_id} Found.")
        return None
    except Exception as e:
        raise e

    return sp_route


def assess_connectivity(
    net: RoadwayNetwork,
    mode: str = "",
    ignore_end_nodes: bool = True,
):
    """Network graph and list of disconnected subgraphs described by a list of their member nodes.

    Args:
        net: RoadwayNetwork object
        mode:  mode of the network, one of `drive`,`transit`,
            `walk`, `bike`
        ignore_end_nodes: if True, ignores stray singleton nodes

    Returns: Tuple of
        Network Graph (osmnx flavored networkX DiGraph)
        List of disconnected subgraphs described by the list of their
            member nodes (as described by their `model_node_id`)
    """
    WranglerLogger.debug(f"Assessing network connectivity for mode: {mode}")

    G = net.get_modal_graph(mode)

    sub_graph_nodes = [
        list(s) for s in sorted(nx.strongly_connected_components(G), key=len, reverse=True)
    ]

    # sorted on decreasing length, dropping the main sub-graph
    disconnected_sub_graph_nodes = sub_graph_nodes[1:]

    # dropping the sub-graphs with only 1 node
    if ignore_end_nodes:
        disconnected_sub_graph_nodes = [
            list(s) for s in disconnected_sub_graph_nodes if len(s) > 1
        ]

    WranglerLogger.info(
        f"{net.nodes_df.model_node_id} for disconnected networks for mode = {mode}:\n"
        + "\n".join(list(map(str, disconnected_sub_graph_nodes))),
    )
    return G, disconnected_sub_graph_nodes
