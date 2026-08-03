"""Functions to create centroid connectors."""

import math
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely.geometry

from ..logger import WranglerLogger
from ..params import LAT_LON_CRS, MODES_TO_NETWORK_LINK_VARIABLES
from .network import RoadwayNetwork


class FitForCentroidConnection(IntEnum):
    """Indicates the fitness of this link to be connected to a centroid connector.

    Since the connector will connect to a node, the highest (worst) of this value
    for the links will apply to the node. So if one link is DO_NOT_USE, it won't be used.

    Used by add_centroid_connectors().
    """

    NA_IS_CONNECTOR = 0
    BEST = 1
    GOOD = 2
    OKAY = 3
    DO_NOT_USE = 100


def calculate_angle_from_centroid(
    gdf: gpd.GeoDataFrame,
    centroid_col: str = "geometry_centroid",
    angle_col: str = "angle_from_north",
) -> gpd.GeoDataFrame:
    """Calculate the angle from centroid to point, measured clockwise from true north.

    Adds a new column to the GeoDataFrame containing the angle in degrees from true north
    (0-360) of the line from each zone centroid to its corresponding point geometry.

    Args:
        gdf: GeoDataFrame with point geometries in the geometry column
        centroid_col: Name of column containing the centroid geometry (default: 'geometry_centroid')
        angle_col: Name of the new column to create with angle values (default: 'angle_from_north')

    Returns:
        GeoDataFrame with new angle column added

    Note:
        - Angle is measured clockwise from north (0° = north, 90° = east, 180° = south, 270° = west)
        - Returns values in range [0, 360)
        - Assumes geometries are in a projected coordinate system or handles lat/lon appropriately
    """

    def calculate_bearing(centroid, point):
        """Calculate bearing angle from centroid to point."""
        if not centroid:
            return None
        if not point:
            return None

        # Calculate difference
        dx = point.x - centroid.x
        dy = point.y - centroid.y

        # Calculate angle in radians from east (standard math convention)
        angle_rad = math.atan2(dy, dx)

        # Convert to degrees
        angle_deg = math.degrees(angle_rad)

        # Convert from east-based to north-based (geographic convention)
        # and from counter-clockwise to clockwise
        bearing = (90 - angle_deg) % 360

        return bearing

    # Calculate angles for all rows
    gdf[angle_col] = gdf.apply(
        lambda row: calculate_bearing(row[centroid_col], row["geometry"]), axis=1
    )

    return gdf


def add_centroid_nodes(
    road_net: RoadwayNetwork,
    zones_gdf: gpd.GeoDataFrame,
    zone_id: str,
    default_node_attribute_dict: dict[str, any] | None = None,
):
    """Adds the given centroid nodes to the roadway network.

    Args:
        road_net: the RoadwayNetwork to update by adding centroids
        zones_gdf: zones definition which must have two geometry columns:
            'geometry', which the geometry boundary, and 'centroid_geometry',
            which should contain the centroid point location
        zone_id: the zone id field in zones_gdf; this will be used as the
            model_node_id
        default_node_attribute_dict: node attributes to set for the new centroid nodes.
            Defaults to None.
    """
    centroid_nodes_gdf = (
        zones_gdf[[zone_id, "geometry_centroid"]]
        .rename(columns={"geometry_centroid": "geometry", zone_id: "model_node_id"})
        .set_geometry("geometry", crs=LAT_LON_CRS)
    )
    centroid_nodes_gdf["X"] = centroid_nodes_gdf["geometry"].x
    centroid_nodes_gdf["Y"] = centroid_nodes_gdf["geometry"].y
    centroid_nodes_gdf["osm_node_id"] = f"{zone_id}:" + centroid_nodes_gdf["model_node_id"].astype(
        str
    )

    # set default node attributes
    centroid_nodes_gdf = centroid_nodes_gdf.assign(**(default_node_attribute_dict or {}))

    # assume the model_node_id
    len_road_net_nodes = len(road_net.nodes_df)
    WranglerLogger.debug(f"centroid_nodes_gdf:\n{centroid_nodes_gdf}")
    road_net.add_nodes(centroid_nodes_gdf)
    WranglerLogger.info(
        f"Added node centroids for {zone_id}: "
        f"increased size of nodes_df from {len_road_net_nodes:,} to {len(road_net.nodes_df):,}"
    )


def add_centroid_connectors(  # noqa: PLR0912, PLR0915
    road_net: RoadwayNetwork,
    zones_gdf: gpd.GeoDataFrame,
    zone_id: str,
    mode: str,
    local_crs: str,
    zone_buffer_distance: int,
    num_centroid_connectors: int,
    max_mode_graph_degrees: int,
    default_link_attribute_dict: dict[str, any] | None = None,
) -> gpd.GeoDataFrame:
    """Creates centroid connector links between zone centroids and roadway network nodes.

    This function identifies suitable roadway nodes for each zone and creates connector links
    from the zone centroid to those nodes. The selection process prioritizes nodes based on
    their fitness for centroid connections and ensures good spatial distribution.

    Selection Algorithm:
        1. **Node Evaluation**: For each node in the modal graph, calculates:
            - Outgoing degree (number of outbound links)
            - Fitness for centroid connection (worst fitness of connected links)

        2. **Spatial Filtering**: Identifies nodes within each zone boundary and filters out:
            - Nodes with `{mode}_centroid_fit` = DO_NOT_USE
            - Nodes outside zone boundaries
            - Nodes with outgoing degree > max_mode_graph_degrees

        3. **Connector Selection** (per zone):
            - First connector: Node with best fitness and closest to centroid
            - Additional connectors: For each subsequent connector (up to num_centroid_connectors):
                * Among nodes with the best available fitness level
                * Select the one with maximum angular separation from existing connectors
                * This ensures spatial distribution while prioritizing network suitability

        4. **Link Creation**: Creates bidirectional links between zone centroid and selected nodes

    Centroid Connector Link Attributes:
        - **model_link_id**: Auto-incremented from max existing link ID
        - **A, B**: Origin and destination node IDs (bidirectional, so both directions created)
        - **name**: Set to "node to {zone_id}" or "{zone_id} to node"
        - **length**: Euclidean distance between centroid and node (in local_crs units)
        - **geometry**: LineString from origin to destination
        - **highway**: Set to zone_id value (if highway column exists in network)
        - **Mode access variables**:
            * All mode variables set to False by default
            * Variables for specified mode set to True (from MODES_TO_NETWORK_LINK_VARIABLES)
            * Example: For mode='drive', sets drive_access=True, bike_access=False, etc.
        - **Custom attributes**: Any attributes from default_link_attribute_dict parameter

    Args:
        road_net: the RoadwayNetwork to update by adding centroid connectors.
            Assumes centroids exist as nodes already. Also assumes links have
            an attribute, `{mode}_centroid_fit`, set to one of the FitForCentroidConnect values.
        zones_gdf: zones definition which must have two geometry columns:
            'geometry', which the geometry boundary, and 'centroid_geometry',
            which should contain the centroid point location (in LAT_LON_CRS)
        zone_id: the zone id field in zones_gdf; this will be used as the
            model_node_id for the centroid
        mode: one of the keys in [`MODES_TO_NETWORK_LINK_VARIABLES`][network_wrangler.params.MODES_TO_NETWORK_LINK_VARIABLES]
        local_crs: CRS to use for distance calculations
        zone_buffer_distance: buffer distance from zone shape to consider node for centroid connector.
            This should be in the units of the local_crs.
        num_centroid_connectors: maximum number of centroid connectors per zone
        max_mode_graph_degrees: maximum outgoing degree for a node to be eligible
        default_link_attribute_dict: link attributes to set for the new centroid connector links.
            Defaults to None.

    Returns:
        A copy of zones_gdf with an additional column, `num_connectors`. The road_net is
            updated in place with new centroid connector links, and the nodes table has an
            additional column: `{zone_id}_num_connectors`.

    """
    WranglerLogger.info(f"Adding centroid connectors for zone:{zone_id} and mode:{mode}")
    WranglerLogger.debug(f"zones_gdf:\n{zones_gdf}")

    G = road_net.get_modal_graph(mode)
    WranglerLogger.debug(f"Created road_net modal_graph for {mode}:")

    node_dict = {}  # node -> *outgoing* degree to all nodes, centroid_fit
    for node in G.nodes():
        # calculate each nodes fitness for centroid based on max of links
        edge_fit_values = [data[f"{mode}_centroid_fit"] for u, v, data in G.edges(node, data=True)]
        # WranglerLogger.debug(f"node: {node} edge_fit_values:{edge_fit_values}")
        node_dict[node] = [G.out_degree(node), max(edge_fit_values) if edge_fit_values else None]

    mode_node_df = pd.DataFrame.from_dict(
        node_dict, orient="index", columns=[f"{mode}_graph_degrees", f"{mode}_centroid_fit"]
    )
    mode_node_df.reset_index(drop=False, names="model_node_id", inplace=True)
    WranglerLogger.debug(f"Created mode_node_df:\n{mode_node_df}")

    # filter to usable nodes
    mode_node_df = mode_node_df.loc[
        mode_node_df[f"{mode}_centroid_fit"] != FitForCentroidConnection.DO_NOT_USE
    ]

    # get node information for these nodes
    mode_node_df = gpd.GeoDataFrame(
        pd.merge(
            left=mode_node_df,
            right=road_net.nodes_df[["model_node_id", "geometry", "osm_node_id", "street_count"]],
            how="left",
            validate="one_to_one",
        ),
        geometry="geometry",
        crs=LAT_LON_CRS,
    )

    mode_node_df.fillna({f"{mode}_graph_degrees": 0}, inplace=True)
    mode_node_df.fillna(
        {f"{mode}_centroid_fit": FitForCentroidConnection.DO_NOT_USE}, inplace=True
    )
    mode_node_df[f"{mode}_graph_degrees"] = mode_node_df[f"{mode}_graph_degrees"].astype(int)
    mode_node_df[f"{mode}_centroid_fit"] = mode_node_df[f"{mode}_centroid_fit"].astype(int)
    WranglerLogger.debug(
        f"Added columns from road_net.nodes_df: mode_node_df type={type(mode_node_df)}:\n{mode_node_df}"
    )

    # Convert if not
    assert isinstance(mode_node_df, gpd.GeoDataFrame)
    mode_node_df.to_crs(local_crs, inplace=True)
    zones_gdf.to_crs(local_crs, inplace=True)

    # spatial intersect nodes with zones
    mode_node_df = gpd.sjoin(
        left_df=mode_node_df,
        right_df=zones_gdf[[zone_id, "geometry", "geometry_centroid"]],
        how="left",
        predicate="dwithin",  # give zones a little buffer because of edge cases
        distance=zone_buffer_distance,
    )
    # This means that model_node_id is not unique -- but that's ok, a node can
    # be connected to multiple centroid connectors
    WranglerLogger.debug(
        f"After spatial join, non-unique model_node_ids:\n"
        f"{mode_node_df.loc[mode_node_df['model_node_id'].duplicated(keep=False)]}"
    )

    # calculate distance from centroid
    gs = gpd.GeoSeries(mode_node_df["geometry_centroid"], crs=LAT_LON_CRS)  # source CRS
    mode_node_df["geometry_centroid"] = gs.to_crs(local_crs).values  # target CRS
    mode_node_df["distance_from_centroid"] = mode_node_df.apply(
        lambda row: row["geometry"].distance(row["geometry_centroid"]), axis=1
    )
    WranglerLogger.debug(
        f"After spatial join, mode_node_df type={type(mode_node_df)}:\n{mode_node_df}"
    )

    # add angle fron centroid
    mode_node_df = calculate_angle_from_centroid(
        mode_node_df, "geometry_centroid", "centroid_angle"
    )
    WranglerLogger.debug(
        f"After adding angle from centroid, mode_node_df type={type(mode_node_df)}:\n{mode_node_df}"
    )

    # Filter to nodes within the given zones
    mode_node_df = mode_node_df.loc[mode_node_df[zone_id].isna() == False]
    mode_node_df[zone_id] = mode_node_df[zone_id].astype(int)
    # and mode_graph_degress <= max_mode_graph_degress
    mode_node_df = mode_node_df.loc[
        mode_node_df[f"{mode}_graph_degrees"] <= max_mode_graph_degrees
    ]

    # sort by drive_centroid_fit, centroid_angle
    mode_node_df.sort_values(
        by=[zone_id, f"{mode}_centroid_fit", "distance_from_centroid"], inplace=True
    )
    mode_node_df.reset_index(drop=True, inplace=True)
    mode_node_df["connector_num"] = 0

    def calculate_min_angle_separation(candidate_angle, selected_angles):
        """Calculate the minimum angular separation between candidate and all selected angles."""
        if len(selected_angles) == 0:
            return 360  # Maximum possible separation if no angles selected yet
        # Vectorized angle difference calculation
        angle_diffs = np.abs(candidate_angle - selected_angles)
        # Handle wraparound
        angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)
        return np.min(angle_diffs)

    WranglerLogger.debug(
        f"Before choosing centroid connector nodes, mode_node_df:\n{mode_node_df}"
    )

    # Process each zone and select connectors
    for zone_num in mode_node_df[zone_id].unique():
        zone_mask = mode_node_df[zone_id] == zone_num
        zone_data = mode_node_df[zone_mask].copy()  # Copy to avoid SettingWithCopyWarning

        if len(zone_data) == 0:
            WranglerLogger.warning(f"No centroid connectors for {zone_id} {zone_num}")
            continue

        # 1: choose the connector with the lowest f"{mode}_centroid_fit" and lowest distance_from_centroid
        # Since data is already sorted by fit and distance, first row is the best
        first_idx = zone_data.index[0]
        mode_node_df.loc[first_idx, "connector_num"] = 1
        zone_data.loc[first_idx, "connector_num"] = 1

        if zone_num == 650:  # noqa: PLR2004
            WranglerLogger.debug(f"A {zone_id} {zone_num}:\n{zone_data}")

        # 2-n: select additional connectors, choosing the one with maximum angular separation
        # from existing connectors, prioritizing by lowest {mode}_centroid_fit first
        for connector_num in range(2, num_centroid_connectors + 1):
            # Get already selected connectors and candidates
            selected_mask = zone_data["connector_num"] > 0
            candidate_mask = zone_data["connector_num"] == 0

            if not candidate_mask.any():
                break  # No more candidates for this zone

            selected_angles = zone_data.loc[selected_mask, "centroid_angle"].values
            candidates = zone_data[candidate_mask].copy()

            # Calculate minimum angular separation for each candidate
            candidates["min_angle_sep"] = candidates["centroid_angle"].apply(
                lambda angle, _sa=selected_angles: calculate_min_angle_separation(angle, _sa)
            )

            # Group by centroid_fit level and find the one with max angular separation within each level
            # Since data is already sorted by fit, we can process in order
            best_fit = candidates[f"{mode}_centroid_fit"].min()
            best_fit_candidates = candidates[candidates[f"{mode}_centroid_fit"] == best_fit]

            if zone_num == 650:  # noqa: PLR2004
                WranglerLogger.debug(
                    f"B {zone_id} {zone_num} {connector_num=}:\n{best_fit_candidates}"
                )

            # Among candidates with the best fit, choose the one with maximum angular separation
            if len(best_fit_candidates) > 0:
                selected_idx = best_fit_candidates["min_angle_sep"].idxmax()
                mode_node_df.loc[selected_idx, "connector_num"] = connector_num
                # Update zone_data to reflect this selection for next iteration
                zone_data.loc[selected_idx, "connector_num"] = connector_num
            else:
                break  # No more candidates

        if zone_num == 650:  # noqa: PLR2004
            WranglerLogger.debug(f"B {zone_id} {zone_num}:\n{zone_data}")

    # Filter to only selected connectors
    mode_node_df = mode_node_df[mode_node_df["connector_num"] > 0]
    mode_node_df.sort_values(by=[zone_id, "connector_num"], inplace=True)
    mode_node_df.reset_index(drop=True, inplace=True)

    WranglerLogger.info(
        f"Selected {len(mode_node_df):,} centroid connectors for {len(zones_gdf):,} {zone_id}s"
    )
    WranglerLogger.debug(f"mode_node_df:\n{mode_node_df}")
    # create centroid connector links: zone to node
    links_taz_to_node_df = mode_node_df.copy()
    links_taz_to_node_df.rename(
        columns={zone_id: "A", "model_node_id": "B", "distance_from_centroid": "length"},
        inplace=True,
    )
    links_taz_to_node_df["name"] = f"{zone_id} to node"
    links_taz_to_node_df["geometry"] = links_taz_to_node_df.apply(
        lambda row: shapely.geometry.LineString([row["geometry_centroid"], row["geometry"]]),
        axis=1,
    )
    # create centroid connector links: node to zone
    links_node_to_taz_df = mode_node_df.copy()
    links_node_to_taz_df.rename(
        columns={"model_node_id": "A", zone_id: "B", "distance_from_centroid": "length"},
        inplace=True,
    )
    links_node_to_taz_df["name"] = f"node to {zone_id}"
    links_node_to_taz_df["geometry"] = links_node_to_taz_df.apply(
        lambda row: shapely.geometry.LineString([row["geometry"], row["geometry_centroid"]]),
        axis=1,
    )

    # Put together zone to node and node to zone
    centroid_links_df = pd.concat([links_taz_to_node_df, links_node_to_taz_df])
    centroid_links_df.reset_index(drop=False, inplace=True)

    # select minimal columns
    centroid_links_df = centroid_links_df[["A", "B", "name", "length", "geometry"]]
    WranglerLogger.debug(f"centroid_links_df:\n{centroid_links_df}")

    max_model_link_id = road_net.links_df.model_link_id.max()
    centroid_links_df["model_link_id"] = centroid_links_df.index + max_model_link_id + 1
    centroid_links_df["shape_id"] = "sh" + centroid_links_df["model_link_id"].astype("str")
    # default to False
    link_mode_variables = set()
    for _mode, link_vars in MODES_TO_NETWORK_LINK_VARIABLES.items():
        link_mode_variables.update(link_vars)
    WranglerLogger.debug(f"link_mode_variables:{link_mode_variables}")

    for link_var in link_mode_variables:
        centroid_links_df[link_var] = False
    # but set the ones for this mode to True
    for link_var in MODES_TO_NETWORK_LINK_VARIABLES[mode]:
        centroid_links_df[link_var] = True

    if "highway" in road_net.links_df.columns:
        centroid_links_df["highway"] = zone_id

    # set default link attributes
    if default_link_attribute_dict is None:
        default_link_attribute_dict = {}
    for colname, default_value in default_link_attribute_dict.items():
        centroid_links_df[colname] = default_value

    road_net.add_links(centroid_links_df)
    road_net.add_shapes(centroid_links_df)
    WranglerLogger.info("Added centroid connectors to roadway network")

    # summarize number of connectors per zone
    summary_df = (
        mode_node_df.groupby(by=zone_id)
        .aggregate(num_connectors=pd.NamedAgg(column="model_node_id", aggfunc="nunique"))
        .reset_index(drop=False)
    )
    # join to zones_gdf to see if we missed any zones
    summary_df = pd.merge(left=zones_gdf, right=summary_df, how="left", validate="one_to_one")
    summary_df["num_connectors"] = summary_df["num_connectors"].fillna(0)
    summary_df["num_connectors"] = summary_df["num_connectors"].astype(int)
    WranglerLogger.debug(f"summary_df:\n{summary_df}")
    WranglerLogger.info(
        f"num_connectors added per {zone_id} (target:{num_centroid_connectors}):\n"
        f"{summary_df['num_connectors'].value_counts()}"
    )

    # add column {zone_id}_num_connectors to nodes
    road_net.nodes_df = pd.merge(
        left=road_net.nodes_df,
        right=summary_df[[zone_id, "num_connectors"]].rename(
            columns={zone_id: "model_node_id", "num_connectors": f"{zone_id}_num_connectors"}
        ),
        how="left",
        validate="one_to_one",
    )
    WranglerLogger.debug(f"road_net.nodes_df:\n{road_net.nodes_df}")
    return summary_df
