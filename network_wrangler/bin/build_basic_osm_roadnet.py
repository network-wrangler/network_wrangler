#!/usr/bin/env python

"""This script builds a basic OpenStreetMap (OSM) road network for a specified place.

This script uses the network_wrangler library to build a roadway network from OSM data. It allows you to specify the place name, network type, output path, and file format for the resulting network.

Usage:
    `python build_basic_osm_roadnet.py <place_name> [--type <type>] [--path <path>] [--file_format <file_format>]`

Arguments:
    place_name (str): Name of the place to build the road network for.
    --type (str | None): Type of network to build Defaults to `drive`.
    --path (str | None): Path to write the network. Defaults to current working directory.
    --file_format (str | None):  File format for writing the network. Defaults to `geojson`.

Example:
    ```bash
    python build_basic_osm_roadnet.py "San Francisco, California" --type "drive" --path "/path/to/output" --file_format "geojson"
    ```
"""

import argparse
import hashlib
import logging
from pathlib import Path

import osmnx as ox

from network_wrangler import write_roadway
from network_wrangler.roadway.links.create import data_to_links_df
from network_wrangler.roadway.network import RoadwayNetwork
from network_wrangler.utils.utils import make_slug

ACCESS_LOOKUPS = {
    "rail_only": {"allow": ["railway"]},
    "walk_access": {"allow": ["footway", "path", "steps", "residential"]},
    "bike_access": {"allow": ["path", "cycleway", "service", "tertiary", "residential"]},
    "drive_access": {"deny": ["footway", "path", "steps", "cycleway"]},
    "truck_access": {"deny": ["footway", "path", "steps", "cycleway"]},
}

LANES_LOOKUPS = {
    "residential": 1,
    "tertiary": 1,
    "secondary": 2,
    "primary": 2,
    "trunk": 2,
    "motorway": 3,
    "motorway_link": 1,
    "primary_link": 1,
    "secondary_link": 1,
    "tertiary_link": 1,
    "trunk_link": 1,
    "service": 1,
    "unclassified": 1,
    "footpath": 0,
    "cycleway": 0,
    "busway": 1,
    "steps": 0,
}


def get_osm_as_gdf(place_name: str, net_type: str = "drive") -> tuple:
    """Get OSM data as GeoDataFrames."""
    g = ox.graph_from_place(place_name, network_type=net_type)
    points, edges = ox.graph_to_gdfs(g)
    edges = (
        edges.reset_index()
        .sort_values("geometry")
        .drop_duplicates(subset=["u", "v"], keep="first")
    )
    return points, edges


def _generate_unique_id(row):
    unique_string = f"{row['A']}_{row['B']}"
    return int(hashlib.sha256(unique_string.encode()).hexdigest(), 16) % 10**8


link_field_dict = {
    "u": "A",
    "v": "B",
    "name": "name",
    "geometry": "geometry",
    "lanes": "lanes",
    "highway": "roadway",
    "osmid": "osm_link_id",
}


def osm_edges_to_wr_links(edges, access_lookups=ACCESS_LOOKUPS, lanes_lookup=LANES_LOOKUPS):
    """Converts OSM edges to Wrangler links."""
    links_df = edges.reset_index()
    # remove dupes
    links_df = links_df.loc[:, list(link_field_dict.keys())].rename(columns=link_field_dict)
    for access_field, allow_deny in access_lookups.items():
        if allow_deny.get("allow"):
            access_values = allow_deny["allow"]
            links_df[access_field] = links_df.roadway.isin(access_values)
        elif allow_deny.get("deny"):
            noaccess_values = allow_deny["deny"]
            links_df[access_field] = ~links_df.roadway.isin(noaccess_values)
    links_df["lanes"] = links_df["roadway"].map(lanes_lookup)
    links_df.A = links_df.A.astype(int)
    links_df.B = links_df.B.astype(int)
    links_df["model_link_id"] = links_df.apply(_generate_unique_id, axis=1)
    assert links_df["model_link_id"].is_unique, "model_link_id values are not unique"
    wr_links_df = data_to_links_df(links_df)
    return wr_links_df


node_field_dict = {
    "model_node_id": "model_node_id",
    "osmid": "osm_node_id",
    "x": "X",
    "y": "Y",
    "geometry": "geometry",
}


def osm_points_to_wr_nodes(nodes_df):
    """Converts OSM nodes to Wrangler nodes."""
    nodes_df = nodes_df.reset_index()
    nodes_df["model_node_id"] = nodes_df.osmid.astype(int)
    nodes_df = nodes_df.loc[:, list(node_field_dict.keys())].rename(columns=node_field_dict)
    return nodes_df


def roadnet_from_osm(
    place_name: str,
    net_type: str = "drive",
    access_lookups: dict = ACCESS_LOOKUPS,
    lanes_lookups: dict = LANES_LOOKUPS,
) -> RoadwayNetwork:
    """Build a roadway network from OSM."""
    nodes, edges = get_osm_as_gdf(place_name, net_type)
    wr_links_df = osm_edges_to_wr_links(edges, access_lookups, lanes_lookups)
    wr_nodes_df = osm_points_to_wr_nodes(nodes)
    return RoadwayNetwork(nodes_df=wr_nodes_df, links_df=wr_links_df)


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build a basic OSM road network.")
    parser.add_argument(
        "place_name", type=str, help="Name of the place to build the road network for."
    )
    parser.add_argument(
        "--type", type=str, default="drive", help="Type of network to build (default: 'drive')."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Path to write the network (default: current working directory).",
    )
    parser.add_argument(
        "--file_format",
        type=str,
        default="geojson",
        help="File format for writing the network (default: 'geojson').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()

    place_name = args.place_name
    network_type = args.type
    output_path = args.path
    file_format = args.file_format
    msg = f"Building network for {place_name} with type {network_type}"
    logging.info(msg)
    net = roadnet_from_osm(place_name, network_type)

    msg = f"Writing network to {output_path} in {file_format} format"
    logging.info(msg)
    file_prefix = make_slug(place_name[:10])
    write_roadway(net, output_path, prefix=file_prefix, file_format=file_format)
