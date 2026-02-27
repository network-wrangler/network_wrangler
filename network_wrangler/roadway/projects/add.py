"""Functions for applying roadway link or node addition project cards to the roadway network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ...errors import NewRoadwayError
from ...logger import WranglerLogger
from ..links.create import data_to_links_df
from ..nodes.create import data_to_nodes_df

if TYPE_CHECKING:
    from ..network import RoadwayNetwork


def apply_new_roadway(
    roadway_net: RoadwayNetwork,
    roadway_addition: dict,
    project_name: str | None = None,
) -> RoadwayNetwork:
    """Add the new roadway features defined in the project card.

    New nodes are added first so that links can refer to any added nodes.

    Args:
        roadway_net: input RoadwayNetwork to apply change to
        roadway_addition: dictionary conforming to RoadwayAddition model such as:

        ```json
            {
                "links": [
                    {
                        "model_link_id": 1000,
                        "A": 100,
                        "B": 101,
                        "lanes": 2,
                        "name": "Main St"
                    }
                ],
                "nodes": [
                    {
                        "model_node_id": 100,
                        "X": 0,
                        "Y": 0
                    },
                    {
                        "model_node_id": 101,
                        "X": 0,
                        "Y": 100
                    }
                ],
            }
        ```
        project_name: optional name of the project to be applied

    returns: updated network with new links and nodes and associated geometries
    """
    add_links, add_nodes = roadway_addition.get("links", []), roadway_addition.get("nodes", [])
    if not add_links and not add_nodes:
        msg = "No links or nodes given to add."
        raise NewRoadwayError(msg)

    WranglerLogger.debug(
        f"Adding New Roadway Features: \n-Links: \n{add_links}\n-Nodes: \n{add_nodes}"
    )
    if add_nodes:
        _new_nodes_df = data_to_nodes_df(pd.DataFrame(add_nodes), config=roadway_net.config)
        if project_name:
            _new_nodes_df["projects"] = f"{project_name},"
        roadway_net.add_nodes(_new_nodes_df)

    if add_links:
        # make sure links refer to nodes in network
        _missing_nodes = _node_ids_from_set_links(add_links) - set(roadway_net.nodes_df.index)
        if _missing_nodes:
            msg = "Link additions use nodes not found in network."
            WranglerLogger.error(msg + f" Missing nodes for new links: {_missing_nodes}")
            raise NewRoadwayError(msg)
        _new_links_df = data_to_links_df(
            add_links,
            nodes_df=roadway_net.nodes_df,
        )
        if project_name:
            _new_links_df["projects"] = f"{project_name},"

        roadway_net.add_links(_new_links_df)

    return roadway_net


def _node_ids_from_set_links(set_links: list[dict]) -> set[int]:
    """Get the nodes from a set of links."""
    return set([link["A"] for link in set_links] + [link["B"] for link in set_links])
