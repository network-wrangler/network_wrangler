"""Functions for applying roadway property change project cards to the roadway network."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from network_wrangler.models.projects.roadway_changes import RoadPropertyChange

from ...errors import RoadwayPropertyChangeError
from ...logger import WranglerLogger
from ..links.edit import edit_link_properties
from ..nodes.edit import NodeGeometryChange, NodeGeometryChangeTable, edit_node_property
from ..selection import RoadwayLinkSelection, RoadwayNodeSelection

if TYPE_CHECKING:
    from ..network import RoadwayNetwork


def _node_geo_change_from_property_changes(
    property_changes: dict[str, RoadPropertyChange],
    node_idx: list[int],
) -> None | NodeGeometryChangeTable:
    """Return NodeGeometryChangeTable if property_changes includes gometry change else None."""
    geo_change_present = any(f in property_changes for f in ["X", "Y"])
    if not geo_change_present:
        return None
    if len(node_idx) > 1:
        msg = "Shouldn't move >1 node to the same geography."
        WranglerLogger.error(msg + f" Selected {len(node_idx)}")
        raise RoadwayPropertyChangeError(msg)

    if not all(f in property_changes for f in ["X", "Y"]):
        msg = "Must provide both X and Y to move node to new location."
        WranglerLogger.error(msg + f" Got {property_changes}")
        raise RoadwayPropertyChangeError(msg)

    geo_changes = {
        k: v["set"] for k, v in property_changes.items() if k in NodeGeometryChange.model_fields
    }
    geo_changes["model_node_id"] = node_idx[0]
    # Don't set in_crs to None - let pandera's add_missing_columns use the default

    return NodeGeometryChangeTable(pd.DataFrame(geo_changes, index=[0]))


def apply_roadway_property_change(
    roadway_net: RoadwayNetwork,
    selection: RoadwayNodeSelection | RoadwayLinkSelection,
    property_changes: dict[str, RoadPropertyChange],
    project_name: str | None = None,
) -> RoadwayNetwork:
    """Changes roadway properties for the selected features based on the project card.

    Args:
        roadway_net: input RoadwayNetwork to apply change to
        selection : roadway selection object
        property_changes : dictionary of roadway properties to change.
            e.g.

            ```yml
            #changes number of lanes 3 to 2 (reduction of 1) and adds a bicycle lane
            lanes:
                existing: 3
                change: -1
            bicycle_facility:
                set: 2
            ```
        project_name: optional name of the project to be applied
    """
    WranglerLogger.debug("Applying roadway property change project.")

    if isinstance(selection, RoadwayLinkSelection):
        roadway_net.links_df = edit_link_properties(
            roadway_net.links_df,
            selection.selected_links,
            property_changes,
            project_name=project_name,
        )
        roadway_net._mark_modified()

    elif isinstance(selection, RoadwayNodeSelection):
        non_geo_changes = {
            k: v for k, v in property_changes.items() if k not in NodeGeometryChange.model_fields
        }
        for property, property_dict in non_geo_changes.items():
            prop_change = RoadPropertyChange(**property_dict)
            roadway_net.nodes_df = edit_node_property(
                roadway_net.nodes_df,
                selection.selected_nodes,
                property,
                prop_change,
                project_name=project_name,
            )
        if non_geo_changes:
            roadway_net._mark_modified()

        geo_changes_df = _node_geo_change_from_property_changes(
            property_changes, selection.selected_nodes
        )
        if geo_changes_df is not None:
            # move_nodes already calls _mark_modified()
            roadway_net.move_nodes(geo_changes_df)

    else:
        msg = "geometry_type must be either 'links' or 'nodes'"
        raise RoadwayPropertyChangeError(msg)

    return roadway_net
