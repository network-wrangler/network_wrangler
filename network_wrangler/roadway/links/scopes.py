"""Utilities for filtering and querying scoped properties based on scoping dimensions.

This module provides various utility functions for filtering and querying scoped properties based
on scoping dimensions such as category and timespan. It includes functions for filtering scoped
values based on non-overlapping or overlapping timespans, non-overlapping or overlapping
categories, and matching exact category and timespan. It also includes functions for creating
exploded dataframes for scoped properties and filtering them based on scope.

Public Functions:
- prop_for_scope: Creates a dataframe with the value of a property for a given category and
    timespan. Can return maximum overlapping timespan value given a minimum number of overlapping
    minutes, or strictly enforce timespans.

Internal function terminology for scopes:

- `matching` scope value: a scope that could be applied for a given category/timespan combination.
    This includes the default scopes as well as scopes that are contained within the given
    category AND timespan combination.
- `overlapping` scope value: a scope that fully or partially overlaps a given category OR timespan
    combination.  This includes the default scopes, all `matching` scopes and all scopes where
    at lest one minute of timespan or one category overlap.
- `conflicting` scope value: a scope that is overlapping but not matching for a given category/
    timespan. By definition default scope values are not conflicting.
- `independent` scope value: a scope value that is not overlapping.


Usage:

```python
model_links_df["lanes_AM_sov"] = prop_for_scope(links_df, ["6:00":"9:00"], category="sov")
```

"""

import copy
from typing import Any, TypeGuard, Union

import pandas as pd
from pandera.typing import DataFrame
from pydantic import validate_call
from typing_extensions import TypeGuard

from ...errors import InvalidScopedLinkValue
from ...logger import WranglerLogger
from ...models._base.types import TimeString
from ...models.projects.roadway_changes import IndivScopedPropertySetItem
from ...models.roadway.tables import (
    ExplodedScopedLinkPropertyTable,
    RoadLinksAttrs,
    RoadLinksTable,
)
from ...models.roadway.types import ScopedLinkValueItem
from ...params import DEFAULT_CATEGORY, DEFAULT_TIMESPAN
from ...utils.models import validate_call_pyd, validate_df_to_model
from ...utils.time import (
    convert_timespan_to_start_end_dt,
    dt_contains,
    dt_list_overlaps,
    filter_df_to_max_overlapping_timespans,
    str_to_time,
)


def _convert_to_scoped_items(
    scoped_values: list[Union[ScopedLinkValueItem, dict]],
) -> list[ScopedLinkValueItem]:
    """Convert dictionaries to ScopedLinkValueItem objects if needed."""
    converted = []
    for item in scoped_values:
        if isinstance(item, dict):
            converted.append(ScopedLinkValueItem(**item))
        else:
            converted.append(item)
    return converted


def _filter_to_matching_timespan_scopes(
    scoped_values: list[Union[ScopedLinkValueItem, dict]], timespan: list[TimeString]
) -> list[ScopedLinkValueItem]:
    """Filters list of ScopedLinkValueItems to list of those that match.

    `matching` scope value: a scope that could be applied for a given category/timespan
        combination. This includes the default scopes as well as scopes that are contained within
        the given category AND timespan combination.
    """
    scoped_values = _convert_to_scoped_items(scoped_values)

    if timespan == DEFAULT_TIMESPAN:
        return scoped_values
    times_dt = list(map(str_to_time, timespan))
    # typeguard - mypy suggested this b/c we cannot guarantee we got rid of all the Nones
    return [
        s
        for s in scoped_values
        if (
            _islist(s.timespan)
            and dt_contains([str_to_time(i) for i in _islist(s.timespan)], times_dt)
        )
        or s.timespan == DEFAULT_TIMESPAN
    ]


def _filter_to_matching_category_scopes(
    scoped_values: list[Union[ScopedLinkValueItem, dict]], category: Union[str, list]
) -> list[ScopedLinkValueItem]:
    """Filters list of ScopedLinkValueItems to list of those that match.

    `matching` scope value: a scope that could be applied for a given category/timespan
        combination. This includes the default scopes as well as scopes that are contained within
        the given category AND timespan combination.
    """
    scoped_values = _convert_to_scoped_items(scoped_values)

    return [s for s in scoped_values if s.category in category or s.category == DEFAULT_CATEGORY]


def _filter_to_matching_scope(
    scoped_values: list[Union[ScopedLinkValueItem, dict]],
    category: Union[str, list] = DEFAULT_CATEGORY,
    timespan: list[TimeString] = DEFAULT_TIMESPAN,
) -> tuple[list[ScopedLinkValueItem], list[ScopedLinkValueItem]]:
    """Filters list of ScopedLinkValueItems to list of those that match.

    `matching` scope value: a scope that could be applied for a given category/timespan
        combination. This includes the default scopes as well as scopes that are contained within
        the given category AND timespan combination.
    """
    scoped_values = _convert_to_scoped_items(scoped_values)

    potential_match = _filter_to_matching_category_scopes(scoped_values, category)
    match = _filter_to_matching_timespan_scopes(potential_match, timespan)
    return match, potential_match


def _filter_to_overlapping_timespan_scopes(
    scoped_values: list[Union[ScopedLinkValueItem, dict]], timespan: list[TimeString]
) -> list[ScopedLinkValueItem]:
    """Filters list of ScopedLinkValueItems to list of those that overlap.

    `overlapping` scope value: a scope that fully or partially overlaps a given category OR
        timespan combination.  This includes the default scopes, all `matching` scopes and
        all scopes where at least one minute of timespan or one category overlap.
    """
    scoped_values = _convert_to_scoped_items(scoped_values)

    if timespan == DEFAULT_TIMESPAN:
        return scoped_values
    q_timespan_dt = list(map(str_to_time, timespan))
    # typeguard - mypy suggested this b/c we cannot guarantee we got rid of all the Nones
    return [
        s
        for s in scoped_values
        if (
            _islist(s.timespan)
            and dt_list_overlaps([q_timespan_dt, [str_to_time(i) for i in _islist(s.timespan)]])
        )
        or s.timespan == DEFAULT_TIMESPAN
    ]


def _islist(s: Any) -> TypeGuard[list]:
    """Typeguard for list to make mypy not complain."""
    if s is list:
        return s
    if isinstance(s, list):
        return s  # type: ignore  # noqa: PGH003
    is_list = bool(issubclass(type(s), list))
    if is_list:
        return s
    msg = f"{s} is not a list but is required to be one."
    raise TypeError(msg)


def _filter_to_overlapping_scopes(
    scoped_prop_list: list[Union[ScopedLinkValueItem, IndivScopedPropertySetItem, dict]],
    category: Union[str, list] = DEFAULT_CATEGORY,
    timespan: list[TimeString] = DEFAULT_TIMESPAN,
) -> list[Union[ScopedLinkValueItem, IndivScopedPropertySetItem]]:
    """Filter a list of IndivScopedPropertySetItem and ScopedLinkValueItems to a specific scope.

    Defaults are considered to overlap everything in their scope dimension.

    `overlapping` scope value: a scope that fully or partially overlaps a given category OR
        timespan combination.  This includes the default scopes, all `matching` scopes and
        all scopes where at least one minute of timespan or one category overlap.
    """
    # Convert dictionaries to ScopedLinkValueItem objects
    converted_list = []
    for item in scoped_prop_list:
        if isinstance(item, dict):
            converted_list.append(ScopedLinkValueItem(**item))
        else:
            converted_list.append(item)

    scoped_prop_list = _filter_to_matching_category_scopes(converted_list, category)
    scoped_prop_list = _filter_to_overlapping_timespan_scopes(scoped_prop_list, timespan)
    return scoped_prop_list


def _filter_to_conflicting_timespan_scopes(
    scoped_values: list[Union[ScopedLinkValueItem, dict]], timespan: list[TimeString]
) -> list[ScopedLinkValueItem]:
    """Filters scoped values to only include those that conflict with the timespan.

    Default timespan does not conflict.

    `matching` scope value: a scope that could be applied for a given category/timespan
        combination. This includes the default scopes as well as scopes that are contained within
        the given category AND timespan combination.
    `overlapping` scope value: a scope that fully or partially overlaps a given category OR
        timespan combination.  This includes the default scopes, all `matching` scopes and
        all scopes where at least one minute of timespan or one category overlap.
    `conflicting` scope value: a scope that is overlapping but not matching for a given category/
        timespan. By definition default scope values are not conflicting.
    """
    scoped_values = _convert_to_scoped_items(scoped_values)

    overlaps = _filter_to_overlapping_timespan_scopes(scoped_values, timespan)
    matches = _filter_to_matching_timespan_scopes(scoped_values, timespan)

    return [s for s in overlaps if s not in matches]


def _filter_to_conflicting_scopes(
    scoped_values: list[Union[ScopedLinkValueItem, dict]],
    timespan: list[TimeString],
    category: Union[str, list[str]],
) -> list[ScopedLinkValueItem]:
    """Filters scoped values to only include those that conflict with the timespan.

    Default timespan and categoies do not conflict.
    NOTE: Categories cannot conflict b/c we are limiting to one per item.

    `matching` scope value: a scope that could be applied for a given category/timespan
        combination. This includes the default scopes as well as scopes that are contained within
        the given category AND timespan combination.
    `overlapping` scope value: a scope that fully or partially overlaps a given category OR
        timespan combination.  This includes the default scopes, all `matching` scopes and
        all scopes where at least one minute of timespan or one category overlap.
    `conflicting` scope value: a scope that is overlapping but not matching for a given category/
        timespan. By definition default scope values are not conflicting.
    """
    scoped_values = _convert_to_scoped_items(scoped_values)

    if category == DEFAULT_CATEGORY and timespan == DEFAULT_TIMESPAN:
        return scoped_values

    conflicting_scopes = _filter_to_conflicting_timespan_scopes(scoped_values, timespan)
    return conflicting_scopes


def _create_exploded_df_for_scoped_prop(
    links_df: DataFrame[RoadLinksTable],
    prop_name: str,
    default_category=DEFAULT_CATEGORY,
) -> DataFrame[ExplodedScopedLinkPropertyTable]:
    """Creates a tidy df of `default`,`category`,`timespan`,`start_time`,`end_time`,`scoped`.

    Steps:

        1. Filter to records that have scoped values
        2. Create a record for each scope (explode records)
        3. Normalize dictionary to columns for category, timespan, value (explode columns)
        4. Fill default category
        5. Separate timespan to start_time and end_time of dt type to facilitate queries
        6. Tidy up and align with data model for export
    """
    # 1. Filter to records that have scoped values
    keep_cols = ["model_link_id", prop_name, f"sc_{prop_name}"]
    scoped_values_df = links_df.loc[links_df[f"sc_{prop_name}"].notna(), keep_cols]

    # 2. Create a record for each scope
    exp_df = scoped_values_df.explode(f"sc_{prop_name}")
    WranglerLogger.debug(f"Exploded Records: \n{exp_df}")

    # 3. normalize dictionary to columns for each dictionary key: category, timespan, value
    #       convert to dictionary from data model
    exp_df[f"sc_{prop_name}"] = exp_df[f"sc_{prop_name}"].apply(
        lambda x: x.model_dump() if hasattr(x, "model_dump") else x
    )
    normalized_scope_df = pd.json_normalize(exp_df.pop(f"sc_{prop_name}")).set_index(exp_df.index)
    exp_df = scoped_values_df[["model_link_id"]].join(normalized_scope_df)
    WranglerLogger.debug(f"Exploded columns: \n{exp_df}")

    # 4. Fill default category (timespan query should take care of this itself)
    if "category" not in exp_df.columns:
        exp_df["category"] = default_category
    else:
        exp_df.loc[exp_df["category"].isna(), "category"] = default_category

    # 5. Separate timespan to start_time and end_time of dt type to facilitate queries
    exp_df[["start_time", "end_time"]] = convert_timespan_to_start_end_dt(exp_df.timespan)

    # 6. Tidy up and align with data model for export
    exp_df = exp_df.rename(columns={"value": "scoped"})
    exp_df = ExplodedScopedLinkPropertyTable(exp_df)
    WranglerLogger.debug(f"exp_df: \n{exp_df}")

    return exp_df


# @validate_call(config={"arbitrary_types_allowed": True})
def _filter_exploded_df_to_scope(
    exp_scoped_prop_df: DataFrame[ExplodedScopedLinkPropertyTable],
    timespan: list[TimeString] = DEFAULT_TIMESPAN,
    category: Union[str, int] = DEFAULT_CATEGORY,
    strict_timespan_match: bool = False,
    min_overlap_minutes: int = 60,
) -> pd.DataFrame:
    """Queries an exploded dataframe of prop_name to determine value for scope for each link.

    Args:
        exp_scoped_prop_df: `ExplodedScopedLinkPropertyTable` for property `prop_name `
        timespan: TimespanString of format ['HH:MM','HH:MM'] to query orig_df for overlapping
            records. Defaults to DEFAULT_TIMESPAN.
        category: category to query orig_df for overlapping records. Defaults to DEFAULT_CATEGORY.
        strict_timespan_match: boolean indicating if the returned df should only contain
            records that fully contain the query timespan. If set to True, min_overlap_minutes
            does not apply. Defaults to False.
        min_overlap_minutes: minimum number of minutes the timespans need to overlap to keep.
            Defaults to 0.

    Returns:
        pd.DataFrame with `model_link_id` and `prop_name`
    """
    match_df = exp_scoped_prop_df
    # Filter dataframe based on matching category and timespan
    if category != DEFAULT_CATEGORY:
        match_df = match_df[
            (match_df["category"] == category) | (match_df["category"] == DEFAULT_CATEGORY)
        ]

    if timespan != DEFAULT_TIMESPAN:
        match_df = filter_df_to_max_overlapping_timespans(
            match_df,
            timespan,
            strict_match=strict_timespan_match,
            min_overlap_minutes=min_overlap_minutes,
        )
    WranglerLogger.debug(f"match_df: \n{match_df}")
    return match_df


@validate_call_pyd
def prop_for_scope(
    links_df: DataFrame[RoadLinksTable],
    prop_name: str,
    timespan: Union[None, list[TimeString]] = DEFAULT_TIMESPAN,
    category: Union[str, int, None] = DEFAULT_CATEGORY,
    strict_timespan_match: bool = False,
    min_overlap_minutes: int = 60,
    allow_default: bool = True,
) -> pd.DataFrame:
    """Creates a df with the value of a property for a given category and timespan.

    Args:
        links_df:(RoadLinksTable
        prop_name: name of property to query
        timespan: TimespanString of format ['HH:MM','HH:MM'] to query orig_df for overlapping
            records.
        category: category to query orig_df for overlapping records. Defaults to None.
        strict_timespan_match: boolean indicating if the returned df should only contain
            records that fully contain the query timespan. If set to True, min_overlap_minutes
            does not apply. Defaults to False.
        min_overlap_minutes: minimum number of minutes the timespans need to overlap to keep.
            Defaults to 0.
        allow_default: boolean indicating if the default value should be returned if no scoped
            values are found. Defaults to True.

    Returns:
        pd.DataFrame with `model_link_id` and `prop_name`
    """
    links_df = validate_df_to_model(links_df, RoadLinksTable)
    timespan = timespan if timespan is not None else DEFAULT_TIMESPAN
    category = category if category is not None else DEFAULT_CATEGORY

    if prop_name not in links_df.columns:
        msg = f"{prop_name} not in dataframe."
        raise ValueError(msg)

    # Check if scoped values even exist and if can just return the default.
    if f"sc_{prop_name}" not in links_df.columns or links_df[f"sc_{prop_name}"].isna().all():
        if not allow_default:
            msg = f"{prop_name} does not have a scoped property column or it is null."
            WranglerLogger.error(
                msg + " Set `allow_default = True` or fill column `sc_{prop_name}`."
            )
            raise ValueError(msg)
        WranglerLogger.debug(f"No scoped values {prop_name}. Returning default.")
        return copy.deepcopy(links_df[["model_link_id", prop_name]])

    # All possible scopings
    candidate_scoped_prop_df = _create_exploded_df_for_scoped_prop(links_df, prop_name)

    # Find scopes that apply
    scoped_prop_df = _filter_exploded_df_to_scope(
        candidate_scoped_prop_df,
        timespan=timespan,
        category=category,
        strict_timespan_match=strict_timespan_match,
        min_overlap_minutes=min_overlap_minutes,
    )

    # Attach them back to all links and update default.
    result_df = copy.deepcopy(links_df[["model_link_id", prop_name]])
    result_df.loc[scoped_prop_df.index, prop_name] = scoped_prop_df["scoped"]
    WranglerLogger.debug(
        f"result_df[prop_name]: \n{result_df.loc[scoped_prop_df.index, prop_name]}"
    )
    return result_df
