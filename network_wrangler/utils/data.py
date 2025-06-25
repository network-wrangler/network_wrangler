"""Utility functions for pandas data manipulation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from geopandas import GeoDataFrame, GeoSeries
from numpy import ndarray
from shapely import wkt

from ..errors import DataframeSelectionError
from ..logger import WranglerLogger
from ..params import LAT_LON_CRS, STRICT_MATCH_FIELDS


class DataSegmentationError(Exception):
    """Raised when there is an error segmenting data."""


class MissingPropertiesError(Exception):
    """Raised when properties are missing from the dataframe."""


class InvalidJoinFieldError(Exception):
    """Raised when the join field is not unique."""


def dict_to_query(
    selection_dict: Mapping[str, Any],
) -> str:
    """Generates the query of from selection_dict.

    Args:
        selection_dict: selection dictionary

    Returns:
        _type_: Query value
    """
    WranglerLogger.debug("Building selection query")

    def _kv_to_query_part(k, v, _q_part=""):
        if isinstance(v, list):
            _q_part += "(" + " or ".join([_kv_to_query_part(k, i) for i in v]) + ")"
            return _q_part
        if isinstance(v, str):
            return k + '.str.contains("' + v + '")'
        return k + "==" + str(v)

    query = "(" + " and ".join([_kv_to_query_part(k, v) for k, v in selection_dict.items()]) + ")"
    WranglerLogger.debug(f"Selection query: \n{query}")
    return query


def _df_missing_cols(df, cols):
    return [col for col in cols if col not in df.columns]


def _s1_missing_s2_vals(s1, s2):
    return set(s1) - set(s2)


def _common_df_cols(df1, df2):
    return [col for col in df1.columns if col in df2.columns]


def update_df_by_col_value(
    destination_df: pd.DataFrame,
    source_df: pd.DataFrame,
    join_col: str,
    properties: Optional[list[str]] = None,
    fail_if_missing: bool = True,
) -> pd.DataFrame:
    """Updates destination_df with ALL values in source_df for specified props with same join_col.

    Source_df can contain a subset of IDs of destination_df.
    If fail_if_missing is true, destination_df must have all
    the IDS in source DF - ensuring all source_df values are contained in resulting df.

    ```
    >> destination_df
    trip_id  property1  property2
    1         10      100
    2         20      200
    3         30      300
    4         40      400

    >> source_df
    trip_id  property1  property2
    2         25      250
    3         35      350

    >> updated_df
    trip_id  property1  property2
    0        1       10      100
    1        2       25      250
    2        3       35      350
    3        4       40      400
    ```

    Args:
        destination_df (pd.DataFrame): Dataframe to modify.
        source_df (pd.DataFrame): Dataframe with updated columns
        join_col (str): column to join on
        properties (list[str]): List of properties to use. If None, will default to all
            in source_df.
        fail_if_missing (bool): If True, will raise an error if there are missing IDs in
            destination_df that exist in source_df.
    """
    # 1. Identify which properties should be updated; and if they exist in both DFs.
    if properties is None:
        properties = [
            c for c in source_df.columns if c in destination_df.columns and c != join_col
        ]
    else:
        _dest_miss = _df_missing_cols(destination_df, [*properties, join_col])
        if _dest_miss:
            msg = f"Properties missing from destination_df: {_dest_miss}"
            raise MissingPropertiesError(msg)
        _source_miss = _df_missing_cols(source_df, [*properties, join_col])
        if _source_miss:
            msg = f"Properties missing from source_df: {_source_miss}"
            raise MissingPropertiesError(msg)

    # 2. Identify if there are IDs missing from destination_df that exist in source_df
    if fail_if_missing:
        missing_ids = set(source_df[join_col]) - set(destination_df[join_col])
        if missing_ids:
            msg = f"IDs missing from source_df: \n{missing_ids}"
            raise InvalidJoinFieldError(msg)

    WranglerLogger.debug(f"Updating properties for {len(source_df)} records: {properties}.")

    if not source_df[join_col].is_unique:
        msg = f"Can't join from source_df when join_col: {join_col} is not unique."
        raise InvalidJoinFieldError(msg)

    if not destination_df[join_col].is_unique:
        return _update_props_from_one_to_many(destination_df, source_df, join_col, properties)

    return _update_props_for_common_idx(destination_df, source_df, join_col, properties)


def _update_props_from_one_to_many(
    destination_df: pd.DataFrame,
    source_df: pd.DataFrame,
    join_col: str,
    properties: list[str],
) -> pd.DataFrame:
    """Update value of destination_df[properties] with source_df[properties].

    Allows 1: many between source and destination relationship via `join_col`.
    """
    destination_df.set_index(join_col, inplace=True)
    source_df.set_index(join_col, inplace=True)

    merged_df = destination_df.merge(
        source_df[properties],
        left_index=True,
        right_index=True,
        how="left",
        suffixes=("", "_new"),
    )
    for prop in properties:
        update_idx = merged_df[f"{prop}_new"].notna()
        update_vals = merged_df.loc[update_idx, f"{prop}_new"]
        update_vals = update_vals.astype(destination_df[prop].dtype)
        # annoying but necessary to ensure compatable types
        if len(update_idx) == 1:
            update_vals = update_vals.values[0]
        destination_df.loc[update_idx, prop] = update_vals

    destination_df.reset_index(inplace=True)
    source_df.reset_index(inplace=True)
    return destination_df


def _update_props_for_common_idx(
    destination_df: pd.DataFrame,
    source_df: pd.DataFrame,
    join_col: str,
    properties: list[str],
) -> pd.DataFrame:
    """Quicker update operation w/out merge but requires 1:1 indices."""
    # 1. Set the join_col as the index for both DataFrames
    original_index = destination_df.index

    destination_df = destination_df.reset_index().set_index(join_col)
    source_df = source_df.reset_index().set_index(join_col)

    # 2. Update the destination_df with the source_df values for specified properties
    overlapping_idx = destination_df.index.intersection(source_df.index)

    for prop in properties:
        update_idx = overlapping_idx[source_df.loc[overlapping_idx, prop].notna()]
        update_vals = source_df.loc[update_idx, prop]
        update_vals = update_vals.astype(destination_df[prop].dtype)
        # annoying but necessary to ensure compatable types
        if len(update_idx) == 1:
            update_vals = update_vals.values[0]

        destination_df.loc[update_idx, prop] = update_vals

    # 3. Reset the index to bring back the join_col
    if isinstance(original_index, pd.RangeIndex):
        updated_df = destination_df.reset_index().set_index(original_index)
        updated_df = updated_df.drop(columns=["index"])
    else:
        updated_df = destination_df.reset_index().set_index(original_index.names)

    return updated_df


def list_like_columns(df, item_type: Optional[type] = None) -> list[str]:
    """Find columns in a dataframe that contain list-like items that can't be json-serialized.

    Args:
        df: dataframe to check
        item_type: if not None, will only return columns where all items are of this type by
            checking **only** the first item in the column.  Defaults to None.
    """
    list_like_columns = []

    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, (list, ndarray))).any():
            if item_type is not None and not isinstance(df[column].iloc[0], item_type):
                continue
            list_like_columns.append(column)
    return list_like_columns


def compare_df_values(
    df1, df2, join_col: Optional[str] = None, ignore: Optional[list[str]] = None, atol=1e-5
):
    """Compare overlapping part of dataframes and returns where there are differences."""
    if ignore is None:
        ignore = []
    comp_c = [
        c
        for c in df1.columns
        if c in df2.columns and c not in ignore and not isinstance(df1[c], GeoSeries)
    ]
    if join_col is None:
        comp_df = df1[comp_c].merge(
            df2[comp_c],
            how="inner",
            right_index=True,
            left_index=True,
            suffixes=["_a", "_b"],
        )
    else:
        comp_df = df1[comp_c].merge(df2[comp_c], how="inner", on=join_col, suffixes=["_a", "_b"])

    # Filter columns by data type
    numeric_cols = [col for col in comp_c if np.issubdtype(df1[col].dtype, np.number)]
    ll_cols = list(set(list_like_columns(df1) + list_like_columns(df2)))
    other_cols = [col for col in comp_c if col not in numeric_cols and col not in ll_cols]

    # For numeric columns, use np.isclose
    if numeric_cols:
        numeric_a = comp_df[[f"{col}_a" for col in numeric_cols]]
        numeric_b = comp_df[[f"{col}_b" for col in numeric_cols]]
        is_close = np.isclose(numeric_a, numeric_b, atol=atol, equal_nan=True)
        comp_df[numeric_cols] = ~is_close

    if ll_cols:
        for ll_c in ll_cols:
            comp_df[ll_c] = diff_list_like_series(comp_df[ll_c + "_a"], comp_df[ll_c + "_b"])

    # For non-numeric columns, use direct comparison
    if other_cols:
        for col in other_cols:
            comp_df[col] = (comp_df[f"{col}_a"] != comp_df[f"{col}_b"]) & ~(
                comp_df[f"{col}_a"].isna() & comp_df[f"{col}_b"].isna()
            )

    # Filter columns and rows where no differences
    cols_w_diffs = [col for col in comp_c if comp_df[col].any()]
    out_cols = [col for subcol in cols_w_diffs for col in (f"{subcol}_a", f"{subcol}_b", subcol)]
    comp_df = comp_df[out_cols]
    comp_df = comp_df.loc[comp_df[cols_w_diffs].any(axis=1)]

    return comp_df


def diff_dfs(df1, df2, ignore: Optional[list[str]] = None) -> bool:
    """Returns True if two dataframes are different and log differences."""
    if ignore is None:
        ignore = []
    diff = False
    if set(df1.columns) != set(df2.columns):
        WranglerLogger.warning(
            f" Columns are different 1vs2 \n    {set(df1.columns) ^ set(df2.columns)}"
        )
        common_cols = [col for col in df1.columns if col in df2.columns]
        df1 = df1[common_cols]
        df2 = df2[common_cols]
        diff = True

    cols_to_compare = [col for col in df1.columns if col not in ignore]
    df1 = df1[cols_to_compare]
    df2 = df2[cols_to_compare]

    if len(df1) != len(df2):
        WranglerLogger.warning(f" Length is different /DF1: {len(df1)} vs /DF2: {len(df2)}\n /")
        diff = True

    diff_df = compare_df_values(df1, df2)

    if not diff_df.empty:
        WranglerLogger.error(f"!!! Differences dfs: \n{diff_df}")
        return True

    if not diff:
        WranglerLogger.info("...no differences in df found.")
    return diff


def convert_numpy_to_list(item):
    """Function to recursively convert numpy arrays to lists."""
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, list):
        return [convert_numpy_to_list(sub_item) for sub_item in item]
    if isinstance(item, dict):
        return {key: convert_numpy_to_list(value) for key, value in item.items()}
    return item


def compare_lists(list1, list2) -> bool:
    """Compare two lists."""
    list1 = convert_numpy_to_list(list1)
    list2 = convert_numpy_to_list(list1)
    return list1 != list2


def diff_list_like_series(s1, s2) -> bool:
    """Compare two series that contain list-like items as strings."""
    diff_df = concat_with_attr([s1, s2], axis=1, keys=["s1", "s2"])
    # diff_df["diff"] = diff_df.apply(lambda x: str(x["s1"]) != str(x["s2"]), axis=1)
    diff_df["diff"] = diff_df.apply(lambda x: compare_lists(x["s1"], x["s2"]), axis=1)
    if diff_df["diff"].any():
        WranglerLogger.info("List-Like differences:")
        WranglerLogger.info(diff_df)
        return True
    return False


def segment_data_by_selection(
    item_list: list,
    data: Union[list, pd.DataFrame, pd.Series],
    field: Optional[str] = None,
    end_val=0,
) -> tuple[
    Union[pd.Series, list, pd.DataFrame],
    Union[pd.Series, list, pd.DataFrame],
    Union[pd.Series, list, pd.DataFrame],
]:
    """Segment a dataframe or series into before, middle, and end segments based on item_list.

    selected segment = everything from the first to last item in item_list inclusive of the first
        and last items.
    Before segment = everything before
    After segment = everything after

    Args:
        item_list (list): List of items to segment data by. If longer than two, will only
            use the first and last items.
        data (Union[pd.Series, pd.DataFrame]): Data to segment into before, middle, and after.
        field (str, optional): If a dataframe, specifies which field to reference.
            Defaults to None.
        end_val (int, optional): Notation for util the end or from the begining. Defaults to 0.

    Raises:
        DataSegmentationError: If item list isn't found in data in correct order.

    Returns:
        tuple: data broken out by beofore, selected segment, and after.
    """
    ref_data = data
    if isinstance(data, pd.DataFrame):
        ref_data = data[field].tolist()
    elif isinstance(data, pd.Series):
        ref_data = data.tolist()

    # ------- Replace "to the end" indicators with first or last value --------
    start_item, end_item = item_list[0], item_list[-1]
    if start_item == end_val:
        start_item = ref_data[0]
    if end_item == end_val:
        end_item = ref_data[-1]

    # --------Find the start and end indices -----------------------------------
    start_idxs = list({i for i, item in enumerate(ref_data) if item == start_item})
    if not start_idxs:
        msg = f"Segment start item: {start_item} not in data."
        raise DataSegmentationError(msg)
    if len(start_idxs) > 1:
        WranglerLogger.warning(
            f"Found multiple starting locations for data segment: {start_item}.\
                                Choosing first ... largest segment being selected."
        )
    start_idx = min(start_idxs)

    # find the end node starting from the start index.
    end_idxs = [i + start_idx for i, item in enumerate(ref_data[start_idx:]) if item == end_item]
    # WranglerLogger.debug(f"End indexes: {end_idxs}")
    if not end_idxs:
        msg = f"Segment end item: {end_item} not in data after starting idx."
        raise DataSegmentationError(msg)
    if len(end_idxs) > 1:
        WranglerLogger.warning(
            f"Found multiple ending locations for data segment: {end_item}.\
                                Choosing last ... largest segment being selected."
        )
    end_idx = max(end_idxs) + 1
    # WranglerLogger.debug(
    # f"Segmenting data fr {start_item} idx:{start_idx} to {end_item} idx:{end_idx}.\n{ref_data}")
    # -------- Extract the segments --------------------------------------------
    if isinstance(data, pd.DataFrame):
        before_segment = data.iloc[:start_idx]
        selected_segment = data.iloc[start_idx:end_idx]
        after_segment = data.iloc[end_idx:]
    else:
        before_segment = data[:start_idx]
        selected_segment = data[start_idx:end_idx]
        after_segment = data[end_idx:]

    if isinstance(data, (pd.DataFrame, pd.Series)):
        before_segment = before_segment.reset_index(drop=True)
        selected_segment = selected_segment.reset_index(drop=True)
        after_segment = after_segment.reset_index(drop=True)

    # WranglerLogger.debug(f"Segmented data into before, selected, and after.\n \
    #    Before:\n{before_segment}\nSelected:\n{selected_segment}\nAfter:\n{after_segment}")

    return before_segment, selected_segment, after_segment


def segment_data_by_selection_min_overlap(
    selection_list: list,
    data: pd.DataFrame,
    field: str,
    replacements_list: list,
    end_val=0,
) -> tuple[
    list,
    tuple[
        Union[pd.Series, pd.DataFrame],
        Union[pd.Series, pd.DataFrame],
        Union[pd.Series, pd.DataFrame],
    ],
]:
    """Segments data based on item_list reducing overlap with replacement list.

    *selected segment*: everything from the first to last item in item_list inclusive of the first
        and last items but not if first and last items overlap with replacement list.
    Before segment = everything before
    After segment = everything after

    Example:
    selection_list = [2,5]
    data = pd.DataFrame({"i":[1,2,3,4,5,6]})
    field = "i"
    replacements_list = [2,22,33]

    Returns:
        [22,33]
        [1], [2,3,4,5], [6]

    Args:
        selection_list (list): List of items to segment data by. If longer than two, will only
            use the first and last items.
        data (Union[pd.Series, pd.DataFrame]): Data to segment into before, middle, and after.
        field (str): Specifies which field to reference.
        replacements_list (list): List of items to eventually replace the selected segment with.
        end_val (int, optional): Notation for util the end or from the begining. Defaults to 0.

    Returns: tuple containing:
        - updated replacement_list
        - tuple of before, selected segment, and after data
    """
    before_segment, segment_df, after_segment = segment_data_by_selection(
        selection_list, data, field=field, end_val=end_val
    )
    if not isinstance(segment_df, pd.DataFrame):
        msg = "segment_df should be a DataFrame - something is wrong."
        raise ValueError(msg)

    if replacements_list and replacements_list[0] == segment_df[field].iat[0]:
        # move first item from selected segment to the before_segment df
        replacements_list = replacements_list[1:]
        before_segment = concat_with_attr(
            [before_segment, segment_df.iloc[:1]], ignore_index=True, sort=False
        )
        segment_df = segment_df.iloc[1:]
        # WranglerLogger.debug(f"item start overlaps with replacement. Repl: {replacements_list}")
    if replacements_list and replacements_list[-1] == data[field].iat[-1]:
        # move last item from selected segment to the after_segment df
        replacements_list = replacements_list[:-1]
        after_segment = concat_with_attr(
            [data.iloc[-1:], after_segment], ignore_index=True, sort=False
        )
        segment_df = segment_df.iloc[:-1]
        # WranglerLogger.debug(f"item end overlaps with replacement. Repl: {replacements_list}")

    return replacements_list, (before_segment, segment_df, after_segment)


def coerce_gdf(
    df: pd.DataFrame, geometry: GeoSeries = None, in_crs: int = LAT_LON_CRS
) -> GeoDataFrame:
    """Coerce a DataFrame to a GeoDataFrame, optionally with a new geometry."""
    if isinstance(df, GeoDataFrame):
        if df.crs is None:
            df.crs = in_crs
        return df
    p = None

    if "geometry" not in df and geometry is None:
        msg = "Must give geometry argument if don't have Geometry in dataframe"
        raise ValueError(msg)

    geometry = geometry if geometry is not None else df["geometry"]
    if not isinstance(geometry, GeoSeries):
        try:
            geometry = GeoSeries(geometry)
        except Exception:
            geometry = geometry.apply(wkt.loads)
    df = GeoDataFrame(df, geometry=geometry, crs=in_crs)

    return df


def validate_existing_value_in_df(df: pd.DataFrame, idx: list[int], field: str, expected_value):
    """Validate if df[field]==expected_value for all indices in idx."""
    if field not in df.columns:
        WranglerLogger.warning(f"!! {field} Not an existing field.")
        return False
    if not df.loc[idx, field].eq(expected_value).all():
        WranglerLogger.warning(
            f"Existing value defined for {field} in project card \
            does not match the value in the selection links. \n\
            Specified Existing: {expected_value}\n\
            Actual Existing: \n {df.loc[idx, field]}."
        )
        return False
    return True


CoerceTypes = Union[str, int, float, bool, list[Union[str, int, float, bool]]]


def coerce_val_to_df_types(  # noqa: PLR0911
    field: str,
    val: CoerceTypes,
    df: pd.DataFrame,
) -> CoerceTypes:
    """Coerce field value to match the type of a matching dataframe columns.

    Args:
        field: field to lookup
        val: value or list of values to coerce
        df (pd.DataFrame): dataframe to get types from

    Returns: coerced value or list of values
    """
    if field not in df.columns:
        msg = f"Field {field} not in dataframe columns."
        raise ValueError(msg)
    if pd.api.types.infer_dtype(df[field]) == "integer":
        if isinstance(val, list):
            return [int(float(v)) for v in val]
        return int(float(val))
    if pd.api.types.infer_dtype(df[field]) == "floating":
        if isinstance(val, list):
            return [float(v) for v in val]
        return float(val)
    if pd.api.types.infer_dtype(df[field]) == "boolean":
        if isinstance(val, list):
            return [bool(v) for v in val]
        return bool(val)
    if isinstance(val, list):
        return [str(v) for v in val]
    return str(val)


def coerce_dict_to_df_types(
    d: dict[str, CoerceTypes],
    df: pd.DataFrame,
    skip_keys: Optional[list] = None,
    return_skipped: bool = False,
) -> dict[str, CoerceTypes]:
    """Coerce dictionary values to match the type of a dataframe columns matching dict keys.

    Will also coerce a list of values.

    Args:
        d (dict): dictionary to coerce with singleton or list values
        df (pd.DataFrame): dataframe to get types from
        skip_keys: list of dict keys to skip. Defaults to []/
        return_skipped: keep the uncoerced, skipped keys/vals in the resulting dict.
            Defaults to False.

    Returns:
        dict: dict with coerced types
    """
    if skip_keys is None:
        skip_keys = []
    coerced_dict: dict[str, CoerceTypes] = {}
    for k, vals in d.items():
        if k in skip_keys:
            if return_skipped:
                coerced_dict[k] = vals
            continue
        if k not in df.columns:
            msg = f"Key {k} not in dataframe columns."
            raise ValueError(msg)
        if pd.api.types.infer_dtype(df[k]) == "integer":
            if isinstance(vals, list):
                coerced_v: CoerceTypes = [int(float(v)) for v in vals]
            else:
                coerced_v = int(float(vals))
        elif pd.api.types.infer_dtype(df[k]) == "floating":
            coerced_v = [float(v) for v in vals] if isinstance(vals, list) else float(vals)
        elif pd.api.types.infer_dtype(df[k]) == "boolean":
            coerced_v = [bool(v) for v in vals] if isinstance(vals, list) else bool(vals)
        elif isinstance(vals, list):
            coerced_v = [str(v) for v in vals]
        else:
            coerced_v = str(vals)
        coerced_dict[k] = coerced_v
    return coerced_dict


def coerce_val_to_series_type(val, s: pd.Series) -> Union[float, str, bool]:
    """Coerces a value to match type of pandas series.

    Will try not to fail so if you give it a value that can't convert to a number, it will
    return a string.

    Args:
        val: Any type of singleton value
        s (pd.Series): series to match the type to
    """
    # WranglerLogger.debug(f"Input val: {val} of type {type(val)} to match with series type \
    #    {pd.api.types.infer_dtype(s)}.")
    if pd.api.types.infer_dtype(s) in ["integer", "floating"]:
        try:
            v: Union[float, str, bool] = float(val)
        except:
            v = str(val)
    elif pd.api.types.infer_dtype(s) == "boolean":
        v = bool(val)
    else:
        v = str(val)
    # WranglerLogger.debug(f"Return value: {v}")
    return v


def fk_in_pk(
    pk: Union[pd.Series, list], fk: Union[pd.Series, list], ignore_nan: bool = True
) -> tuple[bool, list]:
    """Check if all foreign keys are in the primary keys, optionally ignoring NaN."""
    if isinstance(fk, list):
        fk = pd.Series(fk)

    if ignore_nan:
        fk = fk.dropna()

    missing_flag = ~fk.isin(pk)

    if missing_flag.any():
        WranglerLogger.warning(
            f"Following keys referenced in {fk.name} but missing in\
            primary key table: \n{fk[missing_flag]} "
        )
        return False, fk[missing_flag].tolist()

    return True, []


def dict_fields_in_df(d: dict, df: pd.DataFrame) -> bool:
    """Check if all fields in dict are in dataframe."""
    missing_fields = [f for f in d if f not in df.columns]
    if missing_fields:
        msg = f"Fields in dictionary missing from dataframe: {missing_fields}."
        WranglerLogger.error(msg)
        raise ValueError(msg)
    return True


def concat_with_attr(dfs: list[pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Concatenate a list of dataframes and retain the attributes of the first dataframe."""
    import copy  # noqa: PLC0415

    if not dfs:
        msg = "No dataframes to concatenate."
        raise ValueError(msg)
    attrs = copy.deepcopy(dfs[0].attrs)
    df = pd.concat(dfs, **kwargs)
    df.attrs = attrs
    return df


def isin_dict(
    df: pd.DataFrame, d: dict, ignore_missing: bool = True, strict_str: bool = False
) -> pd.DataFrame:
    """Filter the dataframe using a dictionary - faster than using isin.

    Uses merge to filter the dataframe by the dictionary keys and values.

    Args:
        df: dataframe to filter
        d: dictionary with keys as column names and values as values to filter by
        ignore_missing: if True, will ignore missing values in the selection dict.
        strict_str: if True, will not allow partial string matches and will force case-matching.
            Defaults to False. If False, will be overridden if key is in STRICT_MATCH_FIELDS or if
            ignore_missing is False.
    """
    sel_links_mask = np.zeros(len(df), dtype=bool)
    missing = {}
    for col, vals in d.items():
        if vals is None:
            continue
        if col not in df.columns:
            msg = f"Key {col} not in dataframe columns."
            raise DataframeSelectionError(msg)
        _strict_str = strict_str or col in STRICT_MATCH_FIELDS or not ignore_missing
        vals_list = [vals] if not isinstance(vals, list) else vals

        index_name = df.index.name if df.index.name is not None else "index"
        _df = df[[col]].reset_index(names=index_name)

        if isinstance(vals_list[0], str) and not _strict_str:
            vals_list = [val.lower() for val in vals_list]
            _df[col] = _df[col].str.lower()

            # Use str.contains for partial matching
            mask = np.zeros(len(_df), dtype=bool)
            for val in vals_list:
                mask |= _df[col].str.contains(val, case=False, na=False)
            selected = _df[mask].set_index(index_name)
        else:
            vals_df = pd.DataFrame({col: vals_list}, index=range(len(vals_list)))
            merged_df = _df.merge(vals_df, on=col, how="outer", indicator=True)
            selected = merged_df[merged_df["_merge"] == "both"].set_index(index_name)
            _missing_vals = merged_df[merged_df["_merge"] == "right_only"][col].tolist()
            if _missing_vals:
                missing[col] = _missing_vals
                WranglerLogger.warning(f"Missing values in selection dict for {col}: {missing}")

        sel_links_mask |= df.index.isin(selected.index)

    if not ignore_missing and any(missing):
        msg = "Missing values in selection dict."
        raise DataframeSelectionError(msg)

    return df.loc[sel_links_mask]
