"""Functions for translating transit tables into visualizable links relatable to roadway network."""

from __future__ import annotations

import pandas as pd
from pandera.typing import DataFrame

from ...models.gtfs.tables import (
    WranglerShapesTable,
    WranglerStopTimesTable,
)
from ...utils.net import point_seq_to_links


def shapes_to_shape_links(shapes: DataFrame[WranglerShapesTable]) -> pd.DataFrame:
    """Converts shapes DataFrame to shape links DataFrame.

    Args:
        shapes (DataFrame[WranglerShapesTable]): The input shapes DataFrame.

    Returns:
        pd.DataFrame: The resulting shape links DataFrame.
    """
    return point_seq_to_links(
        shapes,
        id_field="shape_id",
        seq_field="shape_pt_sequence",
        node_id_field="shape_model_node_id",
    )


def unique_shape_links(
    shapes: DataFrame[WranglerShapesTable], from_field: str = "A", to_field: str = "B"
) -> pd.DataFrame:
    """Returns a DataFrame containing unique shape links based on the provided shapes DataFrame.

    Parameters:
        shapes (DataFrame[WranglerShapesTable]): The input DataFrame containing shape information.
        from_field (str, optional): The name of the column representing the 'from' field.
            Defaults to "A".
        to_field (str, optional): The name of the column representing the 'to' field.
            Defaults to "B".

    Returns:
        pd.DataFrame: DataFrame containing unique shape links based on the provided shapes df.
    """
    shape_links = shapes_to_shape_links(shapes)
    # WranglerLogger.debug(f"Shape links: \n {shape_links[['shape_id', from_field, to_field]]}")

    _agg_dict: dict[str, type | str] = {"shape_id": list}
    _opt_fields = [f"shape_pt_{v}_{t}" for v in ["lat", "lon"] for t in [from_field, to_field]]
    for f in _opt_fields:
        if f in shape_links:
            _agg_dict[f] = "first"

    unique_shape_links = shape_links.groupby([from_field, to_field]).agg(_agg_dict).reset_index()
    return unique_shape_links


def stop_times_to_stop_times_links(
    stop_times: DataFrame[WranglerStopTimesTable],
    from_field: str = "A",
    to_field: str = "B",
) -> pd.DataFrame:
    """Converts stop times to stop times links.

    Args:
        stop_times (DataFrame[WranglerStopTimesTable]): The stop times data.
        from_field (str, optional): The name of the field representing the 'from' stop.
            Defaults to "A".
        to_field (str, optional): The name of the field representing the 'to' stop.
            Defaults to "B".

    Returns:
        pd.DataFrame: The resulting stop times links.
    """
    return point_seq_to_links(
        stop_times,
        id_field="trip_id",
        seq_field="stop_sequence",
        node_id_field="stop_id",
        from_field=from_field,
        to_field=to_field,
    )


def unique_stop_time_links(
    stop_times: DataFrame[WranglerStopTimesTable],
    from_field: str = "A",
    to_field: str = "B",
) -> pd.DataFrame:
    """Returns a DataFrame containing unique stop time links based on the given stop times DataFrame.

    Parameters:
        stop_times (DataFrame[WranglerStopTimesTable]): The DataFrame containing stop times data.
        from_field (str, optional): The name of the column representing the 'from' field in the
            stop times DataFrame. Defaults to "A".
        to_field (str, optional): The name of the column representing the 'to' field in the stop
            times DataFrame. Defaults to "B".

    Returns:
        pd.DataFrame: A DataFrame containing unique stop time links with columns 'from_field',
            'to_field', and 'trip_id'.
    """
    links = stop_times_to_stop_times_links(stop_times, from_field=from_field, to_field=to_field)
    unique_links = links.groupby([from_field, to_field])["trip_id"].apply(list).reset_index()
    return unique_links
