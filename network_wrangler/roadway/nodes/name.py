"""Functions for handling link names associated with roadway nodes."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from ...logger import WranglerLogger

if TYPE_CHECKING:
    from ..network import RoadwayNetwork


def add_roadway_link_names_to_nodes(roadway_network: RoadwayNetwork) -> None:  # noqa: PLR0912, PLR0915
    """Add lists of unique incoming and outgoing link names to each roadway network node.

    For each node in the roadway network, this function adds three new columns:
    - 'incoming_link_names': List of unique names from links ending at this node (B = node_id)
    - 'outgoing_link_names': List of unique names from links starting at this node (A = node_id)
    - 'link_names': List of all unique link names connected to this node (union of incoming and outgoing)

    All lists are sorted alphabetically.

    Args:
        roadway_network: RoadwayNetwork object to modify in place
    """
    # Initialize dictionaries to store link names for each node
    incoming_names = defaultdict(set)
    outgoing_names = defaultdict(set)

    # Iterate through all links to collect names
    WranglerLogger.debug(f"Processing {len(roadway_network.links_df)} links to extract names")

    # Log the types of values in the name field
    name_types = roadway_network.links_df['name'].apply(lambda x: type(x).__name__).value_counts()
    WranglerLogger.debug("Value counts of 'name' field types:")
    for type_name, count in name_types.items():
        WranglerLogger.debug(f"  - {type_name}: {count}")

    # Also check for string representations of lists
    string_names = roadway_network.links_df[roadway_network.links_df['name'].apply(lambda x: isinstance(x, str))]
    if not string_names.empty:
        list_like_strings = string_names['name'].apply(
            lambda x: x.startswith('[') and x.endswith(']') if isinstance(x, str) else False
        ).sum()
        WranglerLogger.debug(f"  - Strings that look like lists: {list_like_strings}")

    unique_link_names = set()

    for _, link in roadway_network.links_df.iterrows():
        node_a = link['A']
        node_b = link['B']
        link_name = link.get('name', 'unknown')

        # Handle case where link_name might be a string representation of a list
        if isinstance(link_name, str) and link_name.startswith('[') and link_name.endswith(']'):
            # Try to parse string representation of list
            try:
                import ast
                parsed_names = ast.literal_eval(link_name)
                if isinstance(parsed_names, list):
                    for name in parsed_names:
                        if name != 'unknown':
                            unique_link_names.add(name)
                            outgoing_names[node_a].add(name)
                            incoming_names[node_b].add(name)
                # If parsing didn't give us a list, treat as regular string
                elif link_name != 'unknown':
                    unique_link_names.add(link_name)
                    outgoing_names[node_a].add(link_name)
                    incoming_names[node_b].add(link_name)
            except (ValueError, SyntaxError):
                # If parsing fails, treat as regular string
                if link_name != 'unknown':
                    unique_link_names.add(link_name)
                    outgoing_names[node_a].add(link_name)
                    incoming_names[node_b].add(link_name)
        elif isinstance(link_name, list):
            # If it's an actual list, process each name individually
            for name in link_name:
                if name != 'unknown':
                    unique_link_names.add(name)
                    outgoing_names[node_a].add(name)
                    incoming_names[node_b].add(name)
        elif link_name != 'unknown':
            # Single string name
            unique_link_names.add(link_name)
            # Add to outgoing for node A and incoming for node B
            outgoing_names[node_a].add(link_name)
            incoming_names[node_b].add(link_name)

    WranglerLogger.debug(f"Found {len(unique_link_names)} unique link names")

    # Log sample of unique link names (first 10)
    sample_names = sorted(unique_link_names)[:10]
    for name in sample_names:
        WranglerLogger.debug(f"  - Link name: '{name}'")
    if len(unique_link_names) > 10:  # noqa: PLR2004
        WranglerLogger.debug(f"  ... and {len(unique_link_names) - 10} more unique names")

    # Initialize columns with empty lists
    roadway_network.nodes_df['incoming_link_names'] = [[] for _ in range(len(roadway_network.nodes_df))]
    roadway_network.nodes_df['outgoing_link_names'] = [[] for _ in range(len(roadway_network.nodes_df))]
    roadway_network.nodes_df['link_names'] = [[] for _ in range(len(roadway_network.nodes_df))]

    # Set values directly in dataframe using iterrows to get actual index
    for idx, row in roadway_network.nodes_df.iterrows():
        node_id = row['model_node_id']
        if node_id in incoming_names:
            roadway_network.nodes_df.at[idx, 'incoming_link_names'] = sorted(incoming_names[node_id])
        if node_id in outgoing_names:
            roadway_network.nodes_df.at[idx, 'outgoing_link_names'] = sorted(outgoing_names[node_id])

        # Combine for all link names
        combined = incoming_names.get(node_id, set()) | outgoing_names.get(node_id, set())
        if combined:
            roadway_network.nodes_df.at[idx, 'link_names'] = sorted(combined)

    # Debug logging for specific node 1011109
    if 1011109 in roadway_network.nodes_df['model_node_id'].values:  # noqa: PLR2004
        node_row = roadway_network.nodes_df[roadway_network.nodes_df['model_node_id'] == 1011109].iloc[0]  # noqa: PLR2004
        WranglerLogger.debug("Debug info for node 1011109:")
        WranglerLogger.debug(f"  - Incoming link names: {node_row['incoming_link_names']}")
        WranglerLogger.debug(f"  - Outgoing link names: {node_row['outgoing_link_names']}")
        WranglerLogger.debug(f"  - All link names: {node_row['link_names']}")
    else:
        WranglerLogger.debug("Node 1011109 not found in the network")

    # Log nodes with most connections
    nodes_with_counts = roadway_network.nodes_df[['model_node_id', 'link_names']].copy()
    nodes_with_counts['count'] = nodes_with_counts['link_names'].apply(len)
    top_nodes = nodes_with_counts.nlargest(5, 'count')

    WranglerLogger.debug("Top 5 nodes by number of unique connected link names:")
    for _, row in top_nodes.iterrows():
        WranglerLogger.debug(f"  - Node {row['model_node_id']}: {row['count']} unique link names")
        if row['link_names']:
            sample = row['link_names'][:3]
            names_str = ", ".join([f"'{n}'" for n in sample])
            if len(row['link_names']) > 3:  # noqa: PLR2004
                names_str += f", ... ({len(row['link_names']) - 3} more)"
            WranglerLogger.debug(f"    Names: {names_str}")

    WranglerLogger.debug("Successfully added link name columns to nodes_df")