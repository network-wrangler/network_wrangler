"""Validation for roadway nodes dataframes and files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa

from ...logger import WranglerLogger
from ...models.roadway.tables import RoadNodesTable
from ...utils.models import TableValidationError, validate_df_to_model


def validate_nodes_file(
    nodes_filename: Path, strict: bool = False, errors_filename: Path = Path("node_errors.csv")
) -> bool:
    """Validates a nodes file to RoadNodesTable.

    Args:
        nodes_filename (Path): The nodes file.
        strict (bool): If True, will validate to nodes_df without trying to parse it first.
        errors_filename (Path): The output file for the validation errors. Defaults
            to "node_errors.csv".

    Returns:
        bool: True if the nodes file is valid.
    """
    nodes_df = pd.read_csv(nodes_filename)
    return validate_nodes_df(nodes_df, strict=strict, errors_filename=errors_filename)


def validate_nodes_df(
    nodes_df: pd.DataFrame, strict: bool = False, errors_filename: Path = Path("node_errors.csv")
) -> bool:
    """Validates a shapes df to RoadNodessTables.

    Args:
        nodes_df (pd.DataFrame): The nodes dataframe.
        strict (bool): If True, will validate to nodes_df without trying to parse it first.
        errors_filename (Path): The output file for the validation errors. Defaults
            to "node_errors.csv".

    Returns:
        bool: True if the nodes dataframe is valid.
    """
    is_valid = True

    if not strict:
        from .create import data_to_nodes_df  # noqa: PLC0415

        try:
            nodes_df = data_to_nodes_df(nodes_df)
        except Exception as e:
            WranglerLogger.error(f"!!! [Nodes invalid] - Failed to parse nodes_df\n{e}")
            is_valid = False

    try:
        validate_df_to_model(nodes_df, RoadNodesTable, output_file=errors_filename)
    except TableValidationError as e:
        WranglerLogger.error(f"!!! [Nodes invalid] - Failed Schema validation\n{e}")
        is_valid = False

    return is_valid
