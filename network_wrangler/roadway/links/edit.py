"""Edits RoadLinksTable properties.

NOTE: Each public method will return a new, whole copy of the RoadLinksTable with associated edits.
Private methods may return mutated originals.

Usage:
    # Returns copy of links_df with lanes set to 2 for links in link_idx
    links_df = edit_link_property(links_df, link_idx, "lanes", {"set": 2})

    # Returns copy of links_df with price reduced by 50 for links in link_idx and raises error
    # if existing value doesn't match 100
    links_df = edit_link_properties(
        links_df, link_idx, "price",
        {"existing": 100,"change":-50},
    )

    # Returns copy of links_df with geometry of links with node_ids updated based on nodes_df
    links_df = edit_link_geometry_from_nodes(links_df, nodes_df, node_ids)
"""

from __future__ import annotations

import copy
from typing import Any, Literal, Optional, Union

import geopandas as gpd
import numpy as np
from pandera.typing import DataFrame
from pydantic import validate_call

from ...configs import DefaultConfig, WranglerConfig
from ...errors import InvalidScopedLinkValue, LinkChangeError
from ...logger import WranglerLogger
from ...models.projects.roadway_changes import (
    IndivScopedPropertySetItem,
    RoadPropertyChange,
    ScopedPropertySetList,
)
from ...models.roadway.tables import RoadLinksAttrs, RoadLinksTable, RoadNodesAttrs, RoadNodesTable
from ...models.roadway.types import ScopedLinkValueItem
from ...utils.data import validate_existing_value_in_df
from ...utils.geo import offset_geometry_meters, update_nodes_in_linestring_geometry
from ...utils.models import default_from_datamodel, validate_call_pyd, validate_df_to_model
from .scopes import (
    _filter_to_conflicting_scopes,
    _filter_to_matching_scope,
)


def _initialize_links_as_managed_lanes(
    links_df: DataFrame[RoadLinksTable],
    link_idx: list[int],
    geometry_offset_meters: float = DefaultConfig.MODEL_ROADWAY.ML_OFFSET_METERS,
) -> DataFrame[RoadLinksTable]:
    """Initialize links as managed lanes if they are not already."""
    # TODO write wrapper on validate call so don't have to do this
    links_df.attrs.update(RoadLinksAttrs)
    links_df.loc[link_idx, "managed"] = 1
    initialize_if_missing = ["ML_geometry", "ML_access_point", "ML_egress_point", "ML_projects"]
    for f in initialize_if_missing:
        if f not in links_df:
            links_df[f] = default_from_datamodel(RoadLinksTable, f)
    _ml_wo_geometry = links_df.loc[links_df["ML_geometry"].isna() & links_df["managed"] == 1].index
    links_df.loc[_ml_wo_geometry, "ML_geometry"] = offset_geometry_meters(
        links_df.loc[_ml_wo_geometry, "geometry"], geometry_offset_meters
    )
    links_df["ML_geometry"] = gpd.GeoSeries(links_df["ML_geometry"])

    return links_df


@validate_call(config={"arbitrary_types_allowed": True}, validate_return=True)
def _resolve_conflicting_scopes(
    scoped_values: list[ScopedLinkValueItem],
    scoped_item: IndivScopedPropertySetItem,
    overwrite_conflicting: bool = True,
) -> list[ScopedLinkValueItem]:
    conflicting_existing = _filter_to_conflicting_scopes(
        scoped_values,
        timespan=scoped_item.timespan,
        category=scoped_item.category,
    )
    if conflicting_existing:
        if overwrite_conflicting:
            return [s for s in scoped_values if s not in conflicting_existing]
        WranglerLogger.error(
            f"""Existing link value conflicts with change.  Either update to
                set overwrite_conflicting = True to overwrite with set value
                or update the scoped value to not conflict.\n
                Conflicting existing value(s): {conflicting_existing}\n
                Set value: {scoped_item} """
        )
    return scoped_values


def _valid_default_value_for_change(value: Any) -> bool:
    if isinstance(value, (int, np.integer)):
        return True
    return bool(isinstance(value, float))


@validate_call(config={"arbitrary_types_allowed": True}, validate_return=True)
def _update_property_for_scope(
    scoped_prop_set: IndivScopedPropertySetItem, existing_value: Any
) -> ScopedLinkValueItem:
    """Update property for a single scope."""
    if scoped_prop_set.set is not None:
        scoped_item = ScopedLinkValueItem(
            category=scoped_prop_set.category,
            timespan=scoped_prop_set.timespan,
            value=scoped_prop_set.set,
        )
        return scoped_item
    if scoped_prop_set.change is not None:
        if not _valid_default_value_for_change(existing_value):
            msg = f"Cannot implement change from default_value of type {type(existing_value)}."
            WranglerLogger.error(msg)
            raise InvalidScopedLinkValue(msg)
        scoped_item = ScopedLinkValueItem(
            category=scoped_prop_set.category,
            timespan=scoped_prop_set.timespan,
            value=existing_value + scoped_prop_set.change,
        )
        return scoped_item
    msg = f"Scoped property change must have set or change."
    WranglerLogger.error(msg + f" Found: {scoped_prop_set}")
    raise InvalidScopedLinkValue(msg)


@validate_call(config={"arbitrary_types_allowed": True}, validate_return=True)
def _edit_scoped_link_property(
    scoped_prop_value_list: Union[None, list[ScopedLinkValueItem]],
    scoped_prop_set: ScopedPropertySetList,
    default_value: Any = None,
    overwrite_scoped: Literal["conflicting", "all", "error"] = "error",
) -> list[ScopedLinkValueItem]:
    """Edit scoped property on a single link.

    Args:
        scoped_prop_value_list: List of scoped property values for the link.
        scoped_prop_set: ScopedPropertySetList of changes to make.
        default_value: Default value for the property if not set.
        overwrite_scoped: If 'all', will overwrite all scoped properties. If 'conflicting',
            will overwrite conflicting scoped properties. If 'error', will raise an error on
            conflicting scoped properties. Defaults to 'error'.
    """
    msg = f"Setting scoped link property.\n\
            - Current value:{scoped_prop_value_list}\n\
            - Set value: {scoped_prop_set}\n\
            - Default value: {default_value}\n\
            - Overwrite scoped: {overwrite_scoped}"
    # WranglerLogger.debug(msg)
    # If None, or asked to overwrite all scopes, and return all set items
    if overwrite_scoped == "all" or not scoped_prop_value_list:
        scoped_prop_value_list = [
            _update_property_for_scope(i, default_value) for i in scoped_prop_set
        ]
        # WranglerLogger.debug(f"Scoped link property:\n{scoped_prop_value_list}")
        return scoped_prop_value_list

    # Copy so not iterating over something that is changing
    updated_scoped_prop_value_list = copy.deepcopy(scoped_prop_value_list)

    for set_item in scoped_prop_set:
        WranglerLogger.debug(f"Editing link for scoped item: {set_item}")

        # delete or error on conflicting scopes
        updated_scoped_prop_value_list = _resolve_conflicting_scopes(
            updated_scoped_prop_value_list,
            set_item,
            overwrite_conflicting=overwrite_scoped == "conflicting",
        )

        # filter matching scopes from updated scopes
        matching_existing, _ = _filter_to_matching_scope(
            updated_scoped_prop_value_list,
            timespan=set_item.timespan,
            category=set_item.category,
        )

        # Remove matching scopes from the list
        updated_scoped_prop_value_list = [
            s for s in updated_scoped_prop_value_list if s not in matching_existing
        ]

        # if no matching scope, append the set-value; barf on change on wrong type
        if not matching_existing:
            updated_scoped_prop_value_list.append(
                _update_property_for_scope(set_item, default_value)
            )
        else:
            # for each matching scope, replace with the new value
            for match_i in matching_existing:
                new_value = _update_property_for_scope(set_item, match_i.value)
                updated_scoped_prop_value_list.append(new_value)

    # WranglerLogger.debug(f"Updated scoped link property:\n{updated_scoped_prop_value_list}")
    return updated_scoped_prop_value_list


def _edit_ml_access_egress_points(
    links_df: DataFrame[RoadLinksTable],
    prop_name: str,
    prop_change: RoadPropertyChange,
    link_idx: list[int],
):
    """Edit ML access or egress points on links."""
    prop_to_node_col = {"ML_access_point": "A", "ML_egress_point": "B"}
    node_col = prop_to_node_col[prop_name]
    if prop_change.set == "all":
        WranglerLogger.debug(f"Setting all {prop_name} to True")
        links_df.loc[link_idx, prop_name] = True
    elif isinstance(prop_change.set, list):
        mask = (links_df[node_col].isin(prop_change.set)) & (links_df.index.isin(link_idx))

        WranglerLogger.debug(
            f"Setting {prop_name} to True for {len(links_df.loc[mask])} links: \n\
                            {links_df.loc[mask, ['model_link_id', 'A', 'B']]}"
        )
        links_df.loc[mask, prop_name] = True
    else:
        msg = f"Invalid value for {prop_name}. Must be list of ints or 'all': {prop_change.set}."
        WranglerLogger.error(msg + f" Must be list of ints or 'all': {prop_change.set}")
        raise ValueError(msg)

    return links_df


def _edit_link_property(
    links_df: DataFrame[RoadLinksTable],
    link_idx: list[int],
    prop_name: str,
    prop_change: dict,
    project_name: Optional[str] = None,
    config: WranglerConfig = DefaultConfig,
) -> DataFrame[RoadLinksTable]:
    """Return edited (in place) RoadLinksTable with property changes for a list of links.

    Args:
        links_df: links to edit
        link_idx: list of link indices to change
        prop_name: name of the property to change
        prop_change: dict conforming to RoadPropertyChange instance with changes to make
        project_name: optional name of the project to be applied
        config: WranglerConfig instance. Defaults to DefaultConfig.
    """
    WranglerLogger.debug(f"Editing {prop_name} on links {link_idx}")
    prop_change = RoadPropertyChange(**prop_change)

    existing_value_conflict = (
        prop_change.existing_value_conflict or config.EDITS.EXISTING_VALUE_CONFLICT
    )
    overwrite_scoped = prop_change.overwrite_scoped or config.EDITS.OVERWRITE_SCOPED

    if not _check_existing_value_conflict(
        links_df, link_idx, prop_name, prop_change, existing_value_conflict
    ):
        return links_df

    project_fields = ["projects"]
    if prop_name.startswith("ML_"):
        links_df = _initialize_links_as_managed_lanes(
            links_df, link_idx, config.MODEL_ROADWAY.ML_OFFSET_METERS
        )
        project_fields.append("ML_projects")

    if project_name is not None:
        links_df.loc[link_idx, project_fields] += f"{project_name},"

    if prop_name not in links_df:
        links_df[prop_name] = default_from_datamodel(RoadLinksTable, prop_name)

    WranglerLogger.debug(f"Editing {prop_name} to {prop_change}")

    if prop_name in ["ML_access_point", "ML_egress_point"]:
        links_df = _edit_ml_access_egress_points(links_df, prop_name, prop_change, link_idx)
    elif prop_change.set is not None:
        WranglerLogger.debug(f"Setting {prop_name} to {prop_change.set}")
        links_df.loc[link_idx, prop_name] = prop_change.set
    elif prop_change.change is not None:
        WranglerLogger.debug(f"Changing {prop_name} by {prop_change.change}")
        links_df.loc[link_idx, prop_name] += prop_change.change
    if prop_change.scoped is not None:
        links_df = _edit_scoped_property(
            links_df, link_idx, prop_name, prop_change, overwrite_scoped
        )

    msg = f"links_df.loc[link_idx,prop_name] After:\n {links_df.loc[link_idx, prop_name]}"
    # WranglerLogger.debug(msg)
    return links_df


def _check_existing_value_conflict(
    links_df: DataFrame[RoadLinksTable],
    link_idx: list[int],
    prop_name: str,
    prop_change: RoadPropertyChange,
    existing_value_conflict: str,
) -> bool:
    """Handle existing value conflict by skipping, changing, or raising an error.

    Returns: True if the change should proceed, False if it should be skipped.
    """
    if prop_change.existing is not None:
        exist_ok = validate_existing_value_in_df(
            links_df, link_idx, prop_name, prop_change.existing
        )
        if not exist_ok:
            if existing_value_conflict == "error":
                msg = (
                    f"Existing value doesn't match specified value in project card for {prop_name}"
                )
                raise LinkChangeError(msg)
            if existing_value_conflict == "skip":
                WranglerLogger.warning(
                    f"Skipping change for {prop_name} because of conflict with existing value."
                )
                return False
            WranglerLogger.warning(f"Changing {prop_name} despite conflict with existing value.")
    return True


def _edit_scoped_property(
    links_df: DataFrame[RoadLinksTable],
    link_idx: list[int],
    prop_name: str,
    prop_change: RoadPropertyChange,
    overwrite_scoped: Literal["conflicting", "all", "error"],
) -> DataFrame[RoadLinksTable]:
    """Handle scoped property change."""
    sc_prop_name = f"sc_{prop_name}"
    WranglerLogger.debug(f"Setting {sc_prop_name} to {prop_change.scoped}")
    if sc_prop_name not in links_df:
        links_df[sc_prop_name] = default_from_datamodel(RoadLinksTable, sc_prop_name)
    for idx in link_idx:
        links_df.at[idx, sc_prop_name] = _edit_scoped_link_property(
            links_df.at[idx, sc_prop_name],
            prop_change.scoped,
            links_df.at[idx, prop_name],
            overwrite_scoped=overwrite_scoped,
        )
        msg = f"idx:\n   {idx}\n\
                type: \n   {type(links_df.at[idx, sc_prop_name])}\n\
                value:\n   {links_df.at[idx, sc_prop_name]}"
        # WranglerLogger.debug(msg)
    return links_df


@validate_call_pyd
def edit_link_properties(
    links_df: DataFrame[RoadLinksTable],
    link_idx: list,
    property_changes: dict[str, dict],
    project_name: Optional[str] = None,
    config: WranglerConfig = DefaultConfig,
) -> DataFrame[RoadLinksTable]:
    """Return copy of RoadLinksTable with edited link properties for a list of links.

    Args:
        links_df: links to edit
        link_idx: list of link indices to change
        property_changes: dictionary of property changes
        project_name: optional name of the project to be applied
        config: WranglerConfig instance. Defaults to DefaultConfig.
    """
    links_df = copy.deepcopy(links_df)
    # TODO write wrapper on validate call so don't have to do this
    links_df.attrs.update(RoadLinksAttrs)
    ml_property_changes = bool([k for k in property_changes if k.startswith("ML_")])
    existing_managed_lanes = len(links_df.loc[link_idx].of_type.managed) == 0
    flag_create_managed_lane = existing_managed_lanes & ml_property_changes

    # WranglerLogger.debug(f"property_changes: \n{property_changes}")
    for property, prop_change in property_changes.items():
        WranglerLogger.debug(f"prop_dict: \n{prop_change}")
        links_df = _edit_link_property(
            links_df,
            link_idx,
            property,
            prop_change,
            config=config,
        )

    # Only want to set this once per project.
    if project_name is not None:
        links_df.loc[link_idx, "projects"] += f"{project_name},"

    # if a managed lane created without access or egress, set it to True for all selected links
    if flag_create_managed_lane:
        if links_df.loc[link_idx].ML_access_point.sum() == 0:
            WranglerLogger.warning(
                "Access point not set in project card for a new managed lane.\
                                   \nSetting ML_access_point to True for selected links."
            )
            links_df.loc[link_idx, "ML_access_point"] = True
        if links_df.loc[link_idx].ML_egress_point.sum() == 0:
            WranglerLogger.warning(
                "Egress point not set in project card for a new managed lane.\
                                   \nSetting ML_egress_point to True for selected links."
            )
            links_df.loc[link_idx, "ML_egress_point"] = True

    links_df = validate_df_to_model(links_df, RoadLinksTable)
    return links_df


@validate_call_pyd
def edit_link_geometry_from_nodes(
    links_df: DataFrame[RoadLinksTable],
    nodes_df: DataFrame[RoadNodesTable],
    node_ids: list[int],
) -> DataFrame[RoadLinksTable]:
    """Returns a copy of links with updated geometry for given links for a given list of nodes.

    Should be called by any function that changes a node location.

    Args:
        links_df: RoadLinksTable to update
        nodes_df: RoadNodesTable to get updated node geometry from
        node_ids: list of node PKs with updated geometry
    """
    # WranglerLogger.debug(f"nodes_df.loc[node_ids]:\n {nodes_df.loc[node_ids]}")
    # TODO write wrapper on validate call so don't have to do this
    links_df.attrs.update(RoadLinksAttrs)
    nodes_df.attrs.update(RoadNodesAttrs)
    links_df = copy.deepcopy(links_df)

    updated_a_geometry = update_nodes_in_linestring_geometry(
        links_df.loc[links_df.A.isin(node_ids)], nodes_df, 0
    )
    links_df.update(updated_a_geometry)

    updated_b_geometry = update_nodes_in_linestring_geometry(
        links_df.loc[links_df.B.isin(node_ids)], nodes_df, -1
    )
    links_df.update(updated_b_geometry)

    _a_or_b_mask = links_df.A.isin(node_ids) | links_df.B.isin(node_ids)
    WranglerLogger.debug(f"links_df: \n{links_df.loc[_a_or_b_mask, ['A', 'B', 'geometry']]}")
    return links_df
