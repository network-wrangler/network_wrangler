"""Tests for /utils/data.

Run just these tests using `pytest tests/test_utils/test_data.py`
"""

import numpy as np
import pandas as pd
import pytest
from pandas import testing as tm

from network_wrangler.errors import DataframeSelectionError
from network_wrangler.logger import WranglerLogger
from network_wrangler.utils.data import (
    DataSegmentationError,
    InvalidJoinFieldError,
    MissingPropertiesError,
    dict_to_query,
    diff_dfs,
    isin_dict,
    list_like_columns,
    segment_data_by_selection,
    segment_data_by_selection_min_overlap,
    update_df_by_col_value,
    validate_existing_value_in_df,
)


def test_update_df_by_col_value():
    # Create destination_df
    destination_df = pd.DataFrame(
        {
            "trip_id": [1, 2, 3, 4],
            "property1": [10, 20, 30, 40],
            "property2": [100, 200, 300, 400],
        }
    )

    # Create source_df
    source_df = pd.DataFrame(
        {"trip_id": [2, 3], "property1": [25, pd.NA], "property2": [None, 350]}
    )

    # Expected updated_df
    expected_df = pd.DataFrame(
        {
            "trip_id": [1, 2, 3, 4],
            "property1": [10, 25, 30, 40],
            "property2": [100, 200, 350, 400],
        }
    )

    # Call the function
    updated_df = update_df_by_col_value(destination_df, source_df, "trip_id")

    WranglerLogger.debug(f"expected_df: \n{expected_df}")
    WranglerLogger.debug(f"updated_df: \n{updated_df}")
    # Check if the updated_df matches the expected_df
    pd.testing.assert_frame_equal(updated_df, expected_df)


def test_update_df_by_col_value_missing_ids():
    # Create destination_df
    destination_df = pd.DataFrame(
        {
            "trip_id": [1, 2, 3, 4],
            "property1": [10, 20, 30, 40],
            "property2": [100, 200, 300, 400],
        }
    )

    # Create source_df with missing IDs
    source_df = pd.DataFrame(
        {
            "trip_id": [2, 3, 5],  # ID 5 is missing in destination_df
            "property1": [25, 35, 50],
            "property2": [250, 350, 500],
        }
    )

    # Call the function with fail_if_missing=True
    with pytest.raises(InvalidJoinFieldError):
        update_df_by_col_value(destination_df, source_df, "trip_id", fail_if_missing=True)


def test_update_df_by_col_value_missing_properties():
    # Create destination_df
    destination_df = pd.DataFrame(
        {
            "trip_id": [1, 2, 3, 4],
            "property1": [10, 20, 30, 40],
            "property2": [100, 200, 300, 400],
        }
    )

    # Create source_df with missing properties
    source_df = pd.DataFrame(
        {
            "trip_id": [2, 3],
            "property1": [25, 35],
            "property3": [250, 350],  # property3 is missing in destination_df
        }
    )

    # Call the function
    with pytest.raises(MissingPropertiesError):
        update_df_by_col_value(
            destination_df, source_df, "trip_id", properties=["property1", "property3"]
        )


def test_update_df_by_col_value_non_unique_join_col():
    # Create destination_df with non-unique join_col
    destination_df = pd.DataFrame(
        {
            "trip_id": [1, 2, 2, 3, 4],
            "property1": [10, 20, 21, 30, 40],
            "property2": [100, 200, 201, 300, 400],
        }
    )

    # Create source_df
    source_df = pd.DataFrame(
        {"trip_id": [2, 3], "property1": [25, pd.NA], "property2": [None, 350]}
    )

    # Expected updated_df
    expected_df = pd.DataFrame(
        {
            "trip_id": [1, 2, 2, 3, 4],
            "property1": [10, 25, 25, 30, 40],
            "property2": [100, 200, 201, 350, 400],
        }
    )
    WranglerLogger.debug(f"expected_df\n{expected_df}")
    # Call the function
    updated_df = update_df_by_col_value(destination_df, source_df, "trip_id")
    WranglerLogger.debug(f"updated_df: \n{updated_df}")
    # Check if the updated_df matches the expected_df
    pd.testing.assert_frame_equal(updated_df, expected_df)


def test_dict_to_query_single_key_value():
    # Single key-value pair
    selection_dict = {"property1": 10}
    expected_query = "(property1==10)"
    assert dict_to_query(selection_dict) == expected_query


def test_dict_to_query_multiple_key_values():
    # Multiple key-value pairs
    selection_dict = {"property1": 10, "property2": "abc"}
    expected_query = '(property1==10 and property2.str.contains("abc"))'
    assert dict_to_query(selection_dict) == expected_query


def test_dict_to_query_list_values():
    # List values
    selection_dict = {"property1": [10, 20, 30]}
    expected_query = "((property1==10 or property1==20 or property1==30))"
    assert dict_to_query(selection_dict) == expected_query


def test_list_like_columns_no_item_type():
    # Create a dataframe with list-like columns
    df = pd.DataFrame(
        {
            "column1": [[1, 2, 3], [4, 5], [6]],
            "column2": [[7, 8], [9], [10, 11, 12]],
            "column3": [13, 14, 15],
        }
    )

    # Expected list of list-like columns
    expected_columns = ["column1", "column2"]

    # Call the function
    result = list_like_columns(df)

    # Check if the result matches the expected list of columns
    assert result == expected_columns


def test_list_like_columns_with_item_type():
    # Create a dataframe with list-like columns
    df = pd.DataFrame(
        {
            "column1": [[1, 2, 3], [4, 5], [6]],
            "column2": [[7, 8], [9], [10, 11, 12]],
            "column3": [13, 14, 15],
        }
    )

    # Expected list of list-like columns with items of type list
    expected_columns = ["column1", "column2"]

    # Call the function with item_type=list
    result = list_like_columns(df, item_type=list)

    # Check if the result matches the expected list of columns
    assert result == expected_columns


def test_list_like_columns_with_item_type_numpy_array():
    # Create a dataframe with list-like columns
    df = pd.DataFrame(
        {
            "column1": [[1, 2, 3], [4, 5], [6]],
            "column2": [[7, 8], [9], [10, 11, 12]],
            "column3": [13, 14, 15],
        }
    )

    # Expected list of list-like columns with items of type numpy.ndarray
    expected_columns = []

    # Call the function with item_type=numpy.ndarray
    result = list_like_columns(df, item_type=np.ndarray)

    # Check if the result matches the expected list of columns
    assert result == expected_columns


def test_list_like_columns_no_list_like_columns():
    # Create a dataframe without any list-like columns
    df = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6], "column3": [7, 8, 9]})

    # Expected empty list
    expected_columns = []

    # Call the function
    result = list_like_columns(df)

    # Check if the result matches the expected empty list
    assert result == expected_columns


def test_diff_dfs_same_data():
    # Create test dataframes
    df1 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    df2 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # Call the function
    result = diff_dfs(df1, df2)

    # Check if the result is False
    assert not result


def test_diff_dfs_different_columns():
    # Create test dataframes with different columns
    df1 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    df2 = pd.DataFrame({"col1": [1, 2, 3], "col3": ["x", "y", "z"]})

    # Call the function
    result = diff_dfs(df1, df2)

    # Check if the result is True
    assert result


def test_diff_dfs_different_length():
    # Create test dataframes with different lengths
    df1 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    df2 = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

    # Call the function
    result = diff_dfs(df1, df2)

    # Check if the result is True
    assert result


def test_diff_dfs_different_values():
    # Create test dataframes with different values
    df1 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    df2 = pd.DataFrame({"col1": [1, 2, 4], "col2": ["a", "b", "d"]})

    # Call the function
    result = diff_dfs(df1, df2)

    # Check if the result is True
    assert result


def test_segment_data_by_selection_dataframe():
    # Create a sample dataframe
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "value": [10, 20, 30, 40, 50, 60]})

    # Define the item list
    item_list = [2, 4]

    # Call the function
    before_segment, selected_segment, after_segment = segment_data_by_selection(
        item_list, df, field="id"
    )

    # Define the expected segments
    expected_before_segment = pd.DataFrame({"id": [1], "value": [10]})
    expected_selected_segment = pd.DataFrame({"id": [2, 3, 4], "value": [20, 30, 40]})
    expected_after_segment = pd.DataFrame({"id": [5, 6], "value": [50, 60]})

    # Check if the segments match the expected segments
    pd.testing.assert_frame_equal(before_segment, expected_before_segment)
    pd.testing.assert_frame_equal(selected_segment, expected_selected_segment)
    pd.testing.assert_frame_equal(after_segment, expected_after_segment)


def test_segment_data_by_selection_series():
    # Create a sample series
    series = pd.Series([10, 20, 30, 40, 50, 60])

    # Define the item list
    item_list = [20, 40]

    # Call the function
    before_segment, selected_segment, after_segment = segment_data_by_selection(item_list, series)

    # Define the expected segments
    expected_before_segment = pd.Series([10])
    expected_selected_segment = pd.Series([20, 30, 40])
    expected_after_segment = pd.Series([50, 60])

    # Check if the segments match the expected segments
    pd.testing.assert_series_equal(before_segment, expected_before_segment)
    pd.testing.assert_series_equal(selected_segment, expected_selected_segment)
    pd.testing.assert_series_equal(after_segment, expected_after_segment)


def test_segment_data_by_selection_missing_start_item():
    # Create a sample dataframe
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "value": [10, 20, 30, 40, 50, 60]})

    # Define the item list with a missing start item
    item_list = [10, 4]

    # Call the function and expect a ValueError
    with pytest.raises(DataSegmentationError):
        segment_data_by_selection(item_list, df, field="id")


def test_segment_data_by_selection_missing_end_item():
    # Create a sample dataframe
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "value": [10, 20, 30, 40, 50, 60]})

    # Define the item list with a missing end item
    item_list = [2, 60]

    # Call the function and expect a ValueError
    with pytest.raises(DataSegmentationError):
        segment_data_by_selection(item_list, df, field="id")


def test_segment_data_by_selection_min_overlap():
    data = pd.DataFrame({"field": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    selection_list = [3, 6]
    replacements_list = [99, 88]

    # Expected result
    expected_replacements_list = [99, 88]
    expected_data = (
        pd.DataFrame({"field": [1, 2]}),
        pd.DataFrame({"field": [3, 4, 5, 6]}),
        pd.DataFrame({"field": [7, 8, 9, 10]}),
    )

    # Call the function
    updated_replacements_list, updated_data = segment_data_by_selection_min_overlap(
        selection_list, data, "field", replacements_list
    )

    # Check if the updated_replacements_list matches the expected_replacements_list
    assert updated_replacements_list == expected_replacements_list

    # Check if the updated_data matches the expected_data
    pd.testing.assert_frame_equal(updated_data[0].reset_index(drop=True), expected_data[0])
    pd.testing.assert_frame_equal(updated_data[1].reset_index(drop=True), expected_data[1])
    pd.testing.assert_frame_equal(updated_data[2].reset_index(drop=True), expected_data[2])

    # now test where i needs to udpat the replacement list
    selection_list = [2, 5]
    data = pd.DataFrame({"i": [1, 2, 3, 4, 5, 6]})
    replacements_list = [2, 22, 33]

    updated_replacements_list, updated_data = segment_data_by_selection_min_overlap(
        selection_list, data, "i", replacements_list
    )

    # Expected result
    expected_replacements_list = [22, 33]
    expected_data = (
        pd.DataFrame({"i": [1, 2]}),
        pd.DataFrame({"i": [3, 4, 5]}),
        pd.DataFrame({"i": [6]}),
    )

    assert updated_replacements_list == expected_replacements_list

    WranglerLogger.debug(f"Expected data: \n{expected_data}")
    WranglerLogger.debug(f"Returned Data: \n{updated_data}")
    # Check if the updated_data matches the expected_data
    pd.testing.assert_frame_equal(updated_data[0].reset_index(drop=True), expected_data[0])
    pd.testing.assert_frame_equal(updated_data[1].reset_index(drop=True), expected_data[1])
    pd.testing.assert_frame_equal(updated_data[2].reset_index(drop=True), expected_data[2])


def test_validate_existing_value_in_df_existing_field():
    # Create a test DataFrame
    df = pd.DataFrame(
        {
            # IDX    0   1   2   3
            "name": ["Alice", "Bob", "Charlie", "Dave"],
            "age": [25, 30, 30, 40],
        }
    )

    # Call the function with an existing field and expected value
    result = validate_existing_value_in_df(df, [1, 2], "age", 30)

    # Check if the result is True
    assert result is True

    # Call the function with an existing field and expected value
    result = validate_existing_value_in_df(df, [2, 3], "age", 30)

    # Check if the result is True
    assert result is False


def test_validate_existing_value_in_df_non_existing_field():
    # Create a test DataFrame
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "Dave"],
            "age": [25, 30, 35, 40],
        }
    )

    # Call the function with a non-existing field and expected value
    result = validate_existing_value_in_df(df, [1, 2, 3], "height", 170)

    # Check if the result is False
    assert not result


def test_validate_existing_value_in_df_mismatched_value():
    # Create a test DataFrame
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "Dave"],
            "age": [25, 30, 35, 40],
        }
    )

    # Call the function with an existing field and mismatched expected value
    result = validate_existing_value_in_df(df, [1, 2, 3], "age", 45)

    # Check if the result is False
    assert not result


def test_segment_series_by_list(request):
    from network_wrangler.utils.data import segment_data_by_selection

    WranglerLogger.info(f"--Starting: {request.node.name}")
    s = pd.Series([1, 2, 3, 4, 5], dtype="int64")
    item_list = [1, 2]
    exp_answer = (
        pd.Series([], dtype="int64"),
        pd.Series([1, 2], dtype="int64"),
        pd.Series([3, 4, 5], dtype="int64"),
    )

    calc_answer = segment_data_by_selection(item_list, s)
    for calc, exp in zip(calc_answer, exp_answer, strict=False):
        WranglerLogger.debug(f"\ncalc: \n{calc}")
        WranglerLogger.debug(f"\nexp: \n{exp}")
        tm.assert_series_equal(calc, exp)


def test_segment_df_by_list(request):
    from network_wrangler.utils.data import segment_data_by_selection

    WranglerLogger.info(f"--Starting: {request.node.name}")
    s = pd.DataFrame({"mynodes": [1, 2, 3, 4, 3, 2, 5]})
    item_list = [2, 3]
    exp_answer = ([1], [2, 3, 4, 3], [2, 5])

    calc_answer = segment_data_by_selection(item_list, s, field="mynodes")
    for calc, exp in zip(calc_answer, exp_answer, strict=False):
        # WranglerLogger.debug(f"\ncalc:\n{calc['mynodes']}")
        # WranglerLogger.debug(f"\nexp:\n{exp}")
        assert exp == calc["mynodes"].to_list()


SPLT_DF_TEST_PARAMS = [
    (
        [1, 2, 3, 4, 5],
        [1, 2],
        ([], [1, 2], [3, 4, 5]),
    ),
    (
        [1, 2, 3, 4, 5],
        [0, 2],
        ([], [1, 2], [3, 4, 5]),
    ),
    ([1, 2, 3, 4, 5], [1, 6], DataSegmentationError),
    (
        [1, 2, 3, 4, 5, 6, 7],
        [2, 5],
        ([1], [2, 3, 4, 5], [6, 7]),
    ),
    (
        [1, 2, 3, 4, 5],
        [5, 0],
        ([1, 2, 3, 4], [5], []),
    ),
    (
        [1, 2, 3, 2],
        [3, 2],
        ([1, 2], [3, 2], []),
    ),
    ([1, 2, 3, 2], [2, 1], DataSegmentationError),
]


@pytest.mark.parametrize(
    ("ref_list", "item_list", "expected_result"),
    SPLT_DF_TEST_PARAMS,
)
def test_segment_list_by_list(request, ref_list, item_list, expected_result):
    from network_wrangler.utils.data import segment_data_by_selection

    WranglerLogger.info(f"--Starting: {request.node.name}")
    if expected_result in [DataSegmentationError]:
        with pytest.raises(expected_result):
            segment_data_by_selection(ref_list, item_list)
    else:
        calc_answer = segment_data_by_selection(item_list, ref_list)
        assert expected_result == calc_answer


def test_update_props_from_one_to_many():
    # Create destination_df
    from network_wrangler.utils.data import _update_props_from_one_to_many

    destination_df = pd.DataFrame(
        {
            "trip_id": [2, 2, 3, 4],
            "property1": [10, 20, 30, 40],
            "property2": [100, 200, 300, 400],
        }
    )
    # Create source_df
    source_df = pd.DataFrame(
        {"trip_id": [2, 3], "property1": [25, pd.NA], "property2": [None, 350]}
    )
    # Expected updated_df
    expected_df = pd.DataFrame(
        {
            "trip_id": [2, 2, 3, 4],
            "property1": [25, 25, 30, 40],
            "property2": [100, 200, 350, 400],
        }
    )
    # Call the function
    updated_df = _update_props_from_one_to_many(
        destination_df, source_df, "trip_id", ["property1", "property2"]
    )
    # Check if the updated_df matches the expected_df
    pd.testing.assert_frame_equal(updated_df, expected_df)


def test_isin_dict_basic():
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})
    d = {"col1": [2, 4], "col2": ["c", "d"]}
    expected_df = pd.DataFrame({"col1": [2, 3, 4], "col2": ["b", "c", "d"]})
    result_df = isin_dict(df, d)
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)


def test_isin_dict_caps_diff():
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})
    d = {"col1": [2, 4], "col2": ["C", "d"]}
    expected_df = pd.DataFrame({"col1": [2, 3, 4], "col2": ["b", "c", "d"]})
    result_df = isin_dict(df, d)
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)


def test_isin_dict_str_within():
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c ave", "d", "e"]})
    d = {"col1": [2, 4], "col2": ["c", "d"]}
    expected_df = pd.DataFrame({"col1": [2, 3, 4], "col2": ["b", "c ave", "d"]})
    result_df = isin_dict(df, d)
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)


def test_isin_dict_missing_values():
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})
    d = {"col1": [2, 6], "col2": ["b", "e"]}
    expected_df = pd.DataFrame({"col1": [2, 5], "col2": ["b", "e"]})
    result_df = isin_dict(df, d)
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)


def test_isin_dict_ignore_missing_false():
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})
    d = {"col1": [2, 6], "col2": ["b", "f"]}
    with pytest.raises(DataframeSelectionError):
        isin_dict(df, d, ignore_missing=False)


def test_isin_dict_non_existing_column():
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})
    d = {"col1": [2, 4], "col3": ["x", "y"]}
    with pytest.raises(DataframeSelectionError):
        isin_dict(df, d)


def test_isin_dict_empty_dataframe():
    df = pd.DataFrame(columns=["col1", "col2"])
    d = {"col1": [2, 4], "col2": ["b", "d"]}
    expected_df = pd.DataFrame(columns=["col1", "col2"])
    result_df = isin_dict(df, d)
    pd.testing.assert_frame_equal(result_df, expected_df)
