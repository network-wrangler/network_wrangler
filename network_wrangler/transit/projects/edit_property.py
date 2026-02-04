"""Functions for editing transit properties in a TransitNetwork."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from ...errors import ProjectCardError, TransitPropertyChangeError
from ...logger import WranglerLogger
from ...utils.data import validate_existing_value_in_df

if TYPE_CHECKING:
    from ...transit.network import TransitNetwork
    from ...transit.selection import TransitSelection

TABLE_TO_APPLY_BY_PROPERTY: dict[str, str] = {
    "headway_secs": "frequencies",
}

IMPLEMENTED_TABLES = ["trips", "frequencies", "stop_times"]


def apply_transit_property_change(
    net: TransitNetwork,
    selection: TransitSelection,
    property_changes: dict,
    project_name: str | None = None,
) -> TransitNetwork:
    """Apply changes to transit properties.

    Args:
        net (TransitNetwork): Network to modify.
        selection (TransitSelection): Selection of trips to modify.
        property_changes (dict): Dictionary of properties to change.
        project_name (str, optional): Name of the project. Defaults to None.

    Returns:
        TransitNetwork: Modified network.
    """
    WranglerLogger.debug("Applying transit property change project.")
    for property, property_change in property_changes.items():
        net = _apply_transit_property_change_to_table(
            net,
            selection,
            property,
            property_change,
            project_name=project_name,
        )
    return net


def _get_table_name_for_property(net: TransitNetwork, property: str) -> str:
    table_name = TABLE_TO_APPLY_BY_PROPERTY.get(property)
    if table_name is None:
        possible_table_names = net.feed.table_names_with_field(property)
        if len(possible_table_names) != 1:
            msg = f"Found property {property} in multiple tables: {possible_table_names}"
            raise NotImplementedError(msg)
        table_name = possible_table_names[0]
    if table_name not in IMPLEMENTED_TABLES:
        msg = f"{table_name} table changes not currently implemented."
        raise NotImplementedError(msg)
    return table_name


def _apply_transit_property_change_to_table(
    net: TransitNetwork,
    selection: TransitSelection,
    prop_name: str,
    prop_change: dict,
    project_name: str | None = None,
) -> TransitNetwork:
    table_name = _get_table_name_for_property(net, prop_name)
    WranglerLogger.debug(f"...modifying {prop_name} in {table_name}.")

    # Allow the project card to override the default behavior of raising an error
    existing_value_conflict = prop_change.get(
        "existing_value_conflict", net.config.EDITS.EXISTING_VALUE_CONFLICT
    )

    table_df = net.feed.get_table(table_name)

    # Update records matching trip_ids or matching frequencies
    if table_name in ["trips", "stop_times"]:
        update_idx = table_df[table_df.trip_id.isin(selection.selected_trips)].index
    elif table_name == "frequencies":
        update_idx = selection.selected_frequencies_df.index
    else:
        msg = f"Changes in table {table_name} not implemented."
        raise NotImplementedError(msg)

    if not _check_existing_value_conflict(
        table_df, update_idx, prop_name, prop_change, existing_value_conflict
    ):
        return net

    set_df = copy.deepcopy(table_df)

    # Calculate build value
    if "set" in prop_change:
        set_df.loc[update_idx, prop_name] = prop_change["set"]
    elif "change" in prop_change:
        set_df.loc[update_idx, prop_name] = (
            set_df.loc[update_idx, prop_name] + prop_change["change"]
        )
    else:
        msg = "Property change must include 'set' or 'change'."
        raise ProjectCardError(msg)

    if project_name is not None:
        set_df.loc[update_idx, "projects"] += f"{project_name},"

    # Update in feed
    net.feed.__dict__[table_name] = set_df

    return net


def _check_existing_value_conflict(
    table_df, update_idx, prop_name, prop_change, existing_value_conflict
) -> bool:
    if "existing" not in prop_change:
        return True

    if validate_existing_value_in_df(table_df, update_idx, prop_name, prop_change["existing"]):
        return True

    WranglerLogger.warning(f"Existing {prop_name} != {prop_change['existing']}.")
    if existing_value_conflict == "error":
        msg = f"Existing {prop_name} does not match {prop_change['existing']}."
        raise TransitPropertyChangeError(msg)
    if existing_value_conflict == "skip":
        WranglerLogger.warning(f"Skipping {prop_name} change due to existing value conflict.")
        return False
    WranglerLogger.warning(f"Changing {prop_name} despite conflict with existing value.")
    return True
