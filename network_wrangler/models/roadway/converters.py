"""Converters for roadway networks between data models."""

from __future__ import annotations

from typing import Optional

from pandas import DataFrame, Series

from ...logger import WranglerLogger
from ...params import DEFAULT_CATEGORY
from ...utils.time import seconds_from_midnight_to_str, str_to_seconds_from_midnight

POTENTIAL_COMPLEX_PROPERTIES = ["lanes", "price", "ML_lanes", "ML_price"]


def translate_links_df_v0_to_v1(
    links_df: DataFrame, complex_properties: Optional[list[str]] = None
) -> DataFrame:
    """Translates a links dataframe from v0 to v1 format.

    Args:
        links_df (DataFrame): _description_
        complex_properties (Optional[list[str]], optional): List of complex properties to
            convert from v0 to v1 link data model. Defaults to None. If None, will detect
            complex properties.
    """
    if complex_properties is None:
        complex_properties = detect_v0_scoped_link_properties(links_df)
    if not complex_properties:
        WranglerLogger.warning(
            "No complex properties detected to convert in links_df.\
                               Returning links_df as-is."
        )
        return links_df

    for prop in complex_properties:
        links_df = _translate_scoped_link_property_v0_to_v1(links_df, prop)
    return links_df


def translate_links_df_v1_to_v0(
    links_df: DataFrame, complex_properties: Optional[list[str]] = None
) -> DataFrame:
    """Translates a links dataframe from v1 to v0 format.

    Args:
        links_df (DataFrame): _description_
        complex_properties (Optional[list[str]], optional): List of complex properties to
            convert from v0 to v1 link data model. Defaults to None. If None, will detect
            complex properties.
    """
    if complex_properties is None:
        complex_properties = [p for p in links_df.columns if p.startswith("sc_")]
    if not complex_properties:
        WranglerLogger.warning(
            "No complex properties detected to convert in links_df.\
                               Returning links_df as-is."
        )
        return links_df

    for prop in complex_properties:
        links_df = _translate_scoped_link_property_v1_to_v0(links_df, prop)
    return links_df


# TRANSLATION 0 to 1 ###
def _v0_to_v1_scoped_link_property_list(v0_item_list: list[dict]) -> list[dict]:
    """Translates a scoped link property item from v0 to v1 format.

    v0 format:
        `value`
        `category` and or `time`

        where `time` is a timespan represented as a tuple(integer) in seconds from midnight

    v1 format:
        `value`
        `category` and or `timespan`

        where `timespan` is represented as a list(str) in 'HH:MM'

    Args:
        v0_item_list (dict):in v0 format

    Returns:
        list[dict]: in v1 format
    """
    import pprint  # noqa: PLC0415

    v1_item_list = []

    for v0_item in v0_item_list.get("timeofday", []):
        WranglerLogger.debug(f"v0_item: {pprint.pformat(v0_item)}")
        v1_item = {"value": v0_item["value"]}
        if "time" in v0_item:
            v1_item["timespan"] = [seconds_from_midnight_to_str(t) for t in v0_item["time"]]

        if "category" in v0_item:
            if not isinstance(v0_item["category"], list):
                v1_item["category"] = v0_item["category"]
                v1_item_list.append(v1_item)
                WranglerLogger.debug(f"v1_item: {pprint.pformat(v1_item)}")
            else:
                for cat in v0_item["category"]:
                    v1_item_c = v1_item.copy()
                    v1_item_c["category"] = cat
                    v1_item_list.append(v1_item_c)
                    WranglerLogger.debug(f"v1_item: {pprint.pformat(v1_item_c)}")
        else:
            WranglerLogger.debug(f"v1_item: {pprint.pformat(v1_item)}")
            v1_item_list.append(v1_item)

    return v1_item_list


def _translate_scoped_link_property_v0_to_v1(links_df: DataFrame, prop: str) -> DataFrame:
    """Translates a scoped link property from v0 to v1 format.

    Args:
        links_df: DataFrame
        prop: str, the property to translate

    Returns:
        DataFrame: the links dataframe with the property translated.
    """
    if prop not in links_df.columns:
        return links_df
    complex_idx = links_df[prop].apply(lambda x: isinstance(x, dict))
    WranglerLogger.debug(f"Translating {sum(complex_idx)} records in {prop} from v0 to v1 format.")
    WranglerLogger.debug(
        f"links_df.loc[complex_idx, prop]:\n\
                         {links_df.loc[complex_idx, prop].head()}"
    )

    links_df.loc[complex_idx, f"sc_{prop}"] = links_df.loc[complex_idx, prop].apply(
        lambda x: _v0_to_v1_scoped_link_property_list(x)
    )
    links_df.loc[complex_idx, prop] = links_df.loc[complex_idx, prop].apply(lambda x: x["default"])
    return links_df


# TRANSLATION 1 to 0 ###
def _v1_to_v0_scoped_link_property(v1_row: Series, prop: str) -> dict:
    """Translates a scoped link property item from v1 to v0 format.

    v0 format:
        `value`
        `category` and or `time`

        where `time` is a timespan represented as a tuple(integer) in seconds from midnight

    v1 format:
        `value`
        `category` and or `timespan`

        where `timespan` is represented as a list(str) in 'HH:MM'

    Args:
        v1_row: links_df in v1 format
        prop: str, the property to translate
    """
    v0_item_list = []
    for v1_item in v1_row[prop]:
        v0_item = {"value": v1_item["value"] if isinstance(v1_item, dict) else v1_item.value}
        if isinstance(v1_item, dict):
            if "timespan" in v1_item:
                # time is a tuple of seconds from midnight from a tuple of "HH:MM"
                v0_item["time"] = tuple(
                    [str_to_seconds_from_midnight(t) for t in v1_item["timespan"]]
                )
            if "category" in v1_item and v1_item["category"] != DEFAULT_CATEGORY:
                v0_item["category"] = [v1_item["category"]]
        else:  # v1_item is a class instance
            if hasattr(v1_item, "timespan"):
                v0_item["time"] = tuple(
                    [str_to_seconds_from_midnight(t) for t in v1_item.timespan]
                )
            if hasattr(v1_item, "category") and v1_item.category != DEFAULT_CATEGORY:
                v0_item["category"] = [v1_item.category]
        v0_item_list.append(v0_item)
    default_prop = prop[3:]
    v0_prop = {"default": v1_row[default_prop], "timeofday": v0_item_list}
    return v0_prop


def _translate_scoped_link_property_v1_to_v0(links_df: DataFrame, prop: str) -> DataFrame:
    """Translates a scoped link property from v1 to v0 format.

    Args:
        links_df: DataFrame
        prop: str, the property to translate

    Returns:
        DataFrame: the links dataframe with the property translated.
    """
    if prop not in links_df.columns:
        return links_df

    default_prop = prop[3:]
    complex_idx = links_df[prop].apply(lambda x: isinstance(x, list))

    v0_prop_s = links_df.loc[complex_idx].apply(_v1_to_v0_scoped_link_property, prop=prop, axis=1)
    links_df[default_prop] = links_df[default_prop].astype(object)
    links_df.loc[complex_idx, default_prop] = v0_prop_s
    links_df = links_df.drop(columns=[prop])

    return links_df


# SNIFFS ###
def detect_v0_scoped_link_properties(links_df: DataFrame) -> list[str]:
    """Detects if a links dataframe has scoped properties in v0 format.

    If finds and columns that start with "sc_" will return [].
    If it finds any dictionaries in potential complex properties, it will return a list of them.
    Otherwise, it will return []

    Args:
        links_df: DataFrame

    Returns:
        list[str]: True if the links dataframe is in the v0 format, False otherwise.
    """
    explicit_scoped_columns = [c for c in links_df.columns if c.startswith("sc_")]
    if explicit_scoped_columns:
        return []

    complex_properties = []

    for prop in POTENTIAL_COMPLEX_PROPERTIES:
        if prop not in links_df.columns:
            continue
        if links_df[prop].apply(lambda x: isinstance(x, dict)).any():
            complex_properties.append(prop)

    return complex_properties
