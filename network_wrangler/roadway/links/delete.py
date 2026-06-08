"""Deletes links from RoadLinksTable."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pandera.typing import DataFrame

from ...errors import LinkDeletionError
from ...logger import WranglerLogger
from ...models.roadway.tables import RoadLinksTable
from ...transit.validate import shape_links_without_road_links

if TYPE_CHECKING:
    from ...transit.network import TransitNetwork


def delete_links_by_ids(
    links_df: DataFrame[RoadLinksTable],
    del_link_ids: list[int],
    ignore_missing: bool = False,
    transit_net: TransitNetwork | None = None,
) -> DataFrame[RoadLinksTable]:
    """Delete links from a links table.

    Args:
        links_df: DataFrame[RoadLinksTable] to delete links from.
        del_link_ids: list of link ids to delete.
        ignore_missing: if True, will not raise an error if a link id to delete is not in
            the network. Defaults to False.
        transit_net: If provided, will check TransitNetwork and warn if deletion breaks transit shapes. Defaults to None.
    """
    WranglerLogger.debug(f"Deleting links with ids: \n{del_link_ids}")
    _missing = set(del_link_ids) - set(links_df.index)
    if _missing:
        WranglerLogger.warning(f"Links in network not there to delete: \n{_missing}")
        if not ignore_missing:
            msg = "Links to delete are not in the network."
            raise LinkDeletionError(msg)

    if transit_net is not None:
        check_deletion_breaks_transit_shapes(links_df, del_link_ids, transit_net)
    return links_df.drop(labels=del_link_ids, errors="ignore")


def check_deletion_breaks_transit_shapes(
    links_df: DataFrame[RoadLinksTable], del_link_ids: list[int], transit_net: TransitNetwork
) -> bool:
    """Check if any transit shapes go on the deleted links.

    Args:
        links_df: DataFrame[RoadLinksTable] to delete links from.
        del_link_ids: list of link ids to delete.
        transit_net: input TransitNetwork

    returns: true if there are broken shapes, false otherwise
    """
    missing_links = shape_links_without_road_links(
        transit_net.feed.shapes, links_df[~links_df.index.isin(del_link_ids)]
    )
    if not missing_links.empty:
        msg = f"Deletion breaks transit shapes:\n{missing_links}"
        WranglerLogger.warning(msg)
        return True
    return False
