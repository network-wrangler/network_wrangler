"""Validation for roadway nodes dataframes and files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa

from ...logger import WranglerLogger
from ...models.roadway.tables import RoadShapesTable
from ...utils.models import TableValidationError, validate_df_to_model


def validate_shapes_file(
    shapes_filename: Path, strict: bool = False, errors_filename: Path = Path("shape_errors.csv")
) -> bool:
    """Validates a shapes file to RoadShapesTable.

    Args:
        shapes_filename (Path): The shapes file.
        strict (bool): If True, will validate to shapes_df without trying to parse it first.
        errors_filename (Path): The output file for the validation errors. Defaults
            to "shape_errors.csv".

    Returns:
        bool: True if the nodes file is valid.
    """
    shapes_df = pd.read_csv(shapes_filename)
    return validate_shapes_df(shapes_df, strict=strict, errors_filename=errors_filename)


def validate_shapes_df(
    shapes_df: pd.DataFrame, strict: bool = False, errors_filename: Path = Path("shape_errors.csv")
) -> bool:
    """Validates a Shapes df to RoadShapesTables.

    Args:
        shapes_df (pd.DataFrame): The shapes dataframe.
        strict (bool): If True, will validate to shapes_df without trying to parse it first.
        errors_filename (Path): The output file for the validation errors. Defaults
            to "shape_errors.csv".

    Returns:
        bool: True if the shapes dataframe is valid.
    """
    is_valid = True

    if not strict:
        from .create import df_to_shapes_df  # noqa: PLC0415

        try:
            shapes_df = df_to_shapes_df(shapes_df)
        except Exception as e:
            WranglerLogger.error(f"!!! [Shapes invalid] - Failed to parse shapes_df\n{e}")
            is_valid = False

    try:
        validate_df_to_model(shapes_df, RoadShapesTable, output_file=errors_filename)
    except TableValidationError as e:
        WranglerLogger.error(f"!!! [Shapes invalid] - Failed Schema validation\n{e}")
        is_valid = False

    return is_valid
