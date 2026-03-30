"""Functions to filter nodes dataframe."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pandera.typing import DataFrame

from ...logger import WranglerLogger
from ..links.links import node_ids_in_link_ids, node_ids_in_links

if TYPE_CHECKING:
    from ...models.roadway.tables import RoadLinksTable, RoadNodesTable


def filter_nodes_to_ids(
    nodes_df: DataFrame[RoadNodesTable], node_ids: list[int]
) -> DataFrame[RoadNodesTable]:
    """Filters nodes dataframe by node_ids.

    Args:
        nodes_df (pd.DataFrame): nodes dataframe
        node_ids (List[int]): list of node_ids to filter by.

    Returns:
        pd.DataFrame: filtered nodes dataframe
    """
    return nodes_df.loc[nodes_df["model_node_id"].isin(node_ids)]


def filter_nodes_to_link_ids(
    link_ids: list[int],
    links_df: DataFrame[RoadLinksTable],
    nodes_df: DataFrame[RoadNodesTable] | None = None,
) -> DataFrame[RoadNodesTable]:
    """Filters nodes dataframe to those used by given link_ids.

    Args:
        link_ids (List[int]): list of link_ids
        links_df (RoadLinksTable): links dataframe
        nodes_df (RoadNodesTable): nodes dataframe

    Returns:
        pd.DataFrame: nodes dataframe
    """
    _node_ids = node_ids_in_link_ids(link_ids, links_df, nodes_df)
    return filter_nodes_to_ids(nodes_df, _node_ids)


def filter_nodes_to_links(
    links_df: DataFrame[RoadLinksTable], nodes_df: DataFrame[RoadNodesTable]
) -> DataFrame[RoadNodesTable]:
    """Filters RoadNodesTable to those used by given links dataframe.

    Args:
        links_df (RoadLinksTable): links dataframe
        nodes_df (RoadNodesTable): nodes dataframe
    """
    _node_ids = node_ids_in_links(links_df, nodes_df)
    nodes_in_links = nodes_df.loc[nodes_df.model_node_id.isin(_node_ids)]
    WranglerLogger.debug(
        f"filter_nodes_to_links(): Selected {len(nodes_in_links):,} of {len(nodes_df):,} nodes."
    )
    return nodes_in_links
