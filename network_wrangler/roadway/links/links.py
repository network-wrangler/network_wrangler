"""Functions for querying RoadLinksTable."""


import pandas as pd
from pandera.typing import DataFrame

from ...logger import WranglerLogger
from ...models.roadway.tables import RoadLinksTable, RoadNodesTable, RoadShapesTable
from ...utils.data import fk_in_pk
from .filters import filter_links_not_in_ids, filter_links_to_ids
from .validate import validate_links_have_nodes


def node_ids_in_links(
    links_df: DataFrame[RoadLinksTable], nodes_df: DataFrame[RoadNodesTable] | None = None
) -> pd.Series:
    """Returns the unique node_ids in a links dataframe.

    Args:
        links_df (RoadLinksTable): links dataframe
        nodes_df (RoadNodesTable): Optional. nodes dataframe. If provided, will check that
            all node_ids are in nodes_df.

    Returns:
        List[int]: list of unique node_ids
    """
    _node_ids = pd.concat([links_df["A"], links_df["B"]]).unique()

    if nodes_df is not None:
        validate_links_have_nodes(links_df, nodes_df)
    return pd.Series(_node_ids)


def node_ids_in_link_ids(
    link_ids: list[int],
    links_df: DataFrame[RoadLinksTable],
    nodes_df: DataFrame[RoadNodesTable] | None = None,
) -> pd.Series:
    """Returns the unique node_ids in a list of link_ids.

    Args:
        link_ids (List[int]): list of link_ids
        links_df (RoadLinksTable): links dataframe
        nodes_df (RoadNodesTable): Optional. nodes dataframe. If provided, will check that
            all node_ids are in nodes_df.
    """
    _links_df = filter_links_to_ids(links_df, link_ids)
    _node_ids = node_ids_in_links(_links_df, nodes_df=nodes_df)
    return _node_ids


def node_ids_unique_to_link_ids(
    link_ids: list[int],
    links_df: DataFrame[RoadLinksTable],
    nodes_df: DataFrame[RoadNodesTable] | None = None,
) -> list[int]:
    """Returns the unique node_ids in a list of link_ids that are not in other links."""
    selected_link_node_ids = node_ids_in_link_ids(link_ids, links_df, nodes_df=nodes_df)
    _unselected_links_df = filter_links_not_in_ids(links_df, link_ids)
    unselected_link_node_ids = node_ids_in_links(_unselected_links_df, nodes_df=nodes_df)

    return selected_link_node_ids[~selected_link_node_ids.isin(unselected_link_node_ids)].tolist()


def shape_ids_in_links(
    links_df: DataFrame[RoadLinksTable], shapes_df: DataFrame[RoadShapesTable] | None = None
) -> list[int]:
    """Returns the unique shape_ids in a links dataframe.

    Args:
        links_df (RoadLinksTable): links dataframe
        shapes_df (RoadNodesTable): Optional. shapes dataframe. If provided, will check that
            all shape_ids are in shapes_df.
    """
    shape_ids = links_df["shape_id"].unique().tolist()

    if shapes_df is not None:
        fk_valid, fk_missing = fk_in_pk(shapes_df.index, shape_ids)
        if not fk_valid:
            WranglerLogger.error(f"Shapes missing that are specified in links: {fk_missing}")
    return shape_ids


def shape_ids_in_link_ids(
    link_ids: list[int],
    links_df: DataFrame[RoadLinksTable],
    shapes_df: DataFrame[RoadShapesTable] | None = None,
) -> list[int]:
    """Returns the unique shape_ids in a list of link_ids."""
    _links_df = filter_links_to_ids(links_df, link_ids)
    return shape_ids_in_links(_links_df, shapes_df=shapes_df)


def shape_ids_unique_to_link_ids(
    link_ids: list[int],
    links_df: DataFrame[RoadLinksTable],
    shapes_df: DataFrame[RoadShapesTable] | None = None,
) -> list[int]:
    """Returns the unique shape_ids in a list of link_ids."""
    selected_link_shape_ids = shape_ids_in_link_ids(link_ids, links_df, shapes_df=shapes_df)
    _unselected_links_df = filter_links_not_in_ids(links_df, link_ids)
    unselected_link_shape_ids = shape_ids_in_links(_unselected_links_df, shapes_df=shapes_df)

    return list(set(selected_link_shape_ids) - set(unselected_link_shape_ids))


def calc_lane_miles(links_df: DataFrame[RoadLinksTable]) -> pd.Series:
    """Calculates default lane miles for each link in the links dataframe.

    For non road links, calculates the lane miles as the distance of the link.
    """
    return links_df.distance * links_df.lanes.clip(lower=1)
