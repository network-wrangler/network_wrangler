"""Utilities for validating a RoadLinksTable beyond its data model."""

from pathlib import Path
from typing import Optional

import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ...errors import NodesInLinksMissingError
from ...logger import WranglerLogger
from ...utils.data import fk_in_pk


def validate_links_have_nodes(links_df: pd.DataFrame, nodes_df: pd.DataFrame) -> bool:
    """Checks if links have nodes and returns a boolean.

    raises: NodesInLinksMissingError if nodes_df is missing and A or B node
    """
    nodes_in_links = list(set(links_df["A"]).union(set(links_df["B"])))
    node_idx_in_links = nodes_df[nodes_df["model_node_id"].isin(nodes_in_links)].index

    fk_valid, fk_missing = fk_in_pk(nodes_df.index, node_idx_in_links)
    if not fk_valid:
        msg = "Links are missing len{fk_missing} nodes."
        WranglerLogger.error(msg + f"\n  Missing: {fk_missing}")
        raise NodesInLinksMissingError(msg)
    return True


def validate_links_file(
    links_filename: Path,
    nodes_df: Optional[pd.DataFrame] = None,
    strict: bool = False,
    errors_filename: Path = Path("link_errors.csv"),
) -> bool:
    """Validates a links file to RoadLinksTable and optionally checks if nodes are in the links.

    Args:
        links_filename (Path): The links file.
        nodes_df (pd.DataFrame): The nodes dataframe. Defaults to None.
        strict (bool): If True, will validate to links_df without trying to parse it first.
        errors_filename (Path): The output file for the validation errors. Defaults
            to "link_errors.csv".

    Returns:
        bool: True if the links file is valid.
    """
    links_df = pd.read_csv(links_filename)
    return validate_links_df(
        links_df, nodes_df=nodes_df, strict=strict, errors_filename=errors_filename
    )


def validate_links_df(
    links_df: pd.DataFrame,
    nodes_df: Optional[pd.DataFrame] = None,
    strict: bool = False,
    errors_filename: Path = Path("link_errors.csv"),
) -> bool:
    """Validates a links df to RoadLinksTable and optionally checks if nodes are in the links.

    Args:
        links_df (pd.DataFrame): The links dataframe.
        nodes_df (pd.DataFrame): The nodes dataframe. Defaults to None.
        strict (bool): If True, will validate to links_df without trying to parse it first.
        errors_filename (Path): The output file for the validation errors. Defaults
            to "link_errors.csv".

    Returns:
        bool: True if the links dataframe is valid.
    """
    from ...models.roadway.tables import RoadLinksTable  # noqa: PLC0415
    from ...utils.models import TableValidationError, validate_df_to_model  # noqa: PLC0415

    is_valid = True

    if not strict:
        from .create import data_to_links_df  # noqa: PLC0415

        try:
            links_df = data_to_links_df(links_df)
        except Exception as e:
            WranglerLogger.error(f"!!! [Links invalid] - Failed to parse links_df\n{e}")
            is_valid = False

    try:
        validate_df_to_model(links_df, RoadLinksTable, output_file=errors_filename)
    except TableValidationError as e:
        WranglerLogger.error(f"!!! [Links invalid] - Failed Schema validation\n{e}")
        is_valid = False

    try:
        validate_links_have_nodes(links_df, nodes_df)
    except NodesInLinksMissingError as e:
        WranglerLogger.error(f"!!! [Links invalid] - Nodes missing in links\n{e}")
        is_valid = False
    return is_valid
