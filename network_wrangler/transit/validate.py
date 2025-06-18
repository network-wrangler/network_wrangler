"""Functions to check for transit network validity and consistency with roadway network."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandera.errors import SchemaErrors
from pandera.typing import DataFrame

from ..logger import WranglerLogger
from ..models.gtfs.tables import (
    WranglerShapesTable,
    WranglerStopTimesTable,
)
from ..transit.feed.feed import Feed
from ..transit.feed.transit_links import unique_shape_links, unique_stop_time_links
from ..utils.data import concat_with_attr

if TYPE_CHECKING:
    from ..models._base.types import RoadwayFileTypes, TransitFileTypes
    from ..models.roadway.tables import RoadLinksTable, RoadNodesTable
    from ..roadway.network import RoadwayNetwork


def transit_nodes_without_road_nodes(
    feed: Feed,
    nodes_df: DataFrame[RoadNodesTable],
    rd_field: str = "model_node_id",
) -> list[int]:
    """Validate all of a transit feeds node foreign keys exist in referenced roadway nodes.

    Args:
        feed: Transit Feed to query.
        nodes_df (pd.DataFrame, optional): Nodes dataframe from roadway network to validate
            foreign key to. Defaults to self.roadway_net.nodes_df
        rd_field: field in roadway nodes to check against. Defaults to "model_node_id"

    Returns:
        boolean indicating if relationships are all valid
    """
    feed_nodes_series = [
        feed.stops["stop_id"],
        feed.shapes["shape_model_node_id"],
        feed.stop_times["stop_id"],
    ]
    tr_nodes = set(concat_with_attr(feed_nodes_series).unique())
    rd_nodes = set(nodes_df[rd_field].unique().tolist())
    # nodes in tr_nodes but not rd_nodes
    missing_tr_nodes = list(tr_nodes - rd_nodes)

    if missing_tr_nodes:
        WranglerLogger.error(
            f"! Transit nodes in missing in roadway network: \n {missing_tr_nodes}"
        )
    return missing_tr_nodes


def shape_links_without_road_links(
    tr_shapes: DataFrame[WranglerShapesTable],
    rd_links_df: DataFrame[RoadLinksTable],
) -> pd.DataFrame:
    """Validate that links in transit shapes exist in referenced roadway links.

    Args:
        tr_shapes: transit shapes from shapes.txt to validate foreign key to.
        rd_links_df: Links dataframe from roadway network to validate

    Returns:
        df with shape_id and A, B
    """
    tr_shape_links = unique_shape_links(tr_shapes)
    # WranglerLogger.debug(f"Unique shape links: \n {tr_shape_links}")
    rd_links_transit_ok = rd_links_df[
        (rd_links_df["drive_access"]) | (rd_links_df["bus_only"]) | (rd_links_df["rail_only"])
    ]

    merged_df = tr_shape_links.merge(
        rd_links_transit_ok[["A", "B"]],
        how="left",
        on=["A", "B"],
        indicator=True,
    )

    missing_links_df = merged_df.loc[merged_df._merge == "left_only", ["shape_id", "A", "B"]]
    if len(missing_links_df):
        WranglerLogger.error(
            f"! Transit shape links missing in roadway network: \n {missing_links_df}"
        )
    return missing_links_df[["shape_id", "A", "B"]]


def stop_times_without_road_links(
    tr_stop_times: DataFrame[WranglerStopTimesTable],
    rd_links_df: DataFrame[RoadLinksTable],
) -> pd.DataFrame:
    """Validate that links in transit shapes exist in referenced roadway links.

    Args:
        tr_stop_times: transit stop_times from stop_times.txt to validate foreign key to.
        rd_links_df: Links dataframe from roadway network to validate

    Returns:
        df with shape_id and A, B
    """
    tr_links = unique_stop_time_links(tr_stop_times)

    rd_links_transit_ok = rd_links_df[
        (rd_links_df["drive_access"]) | (rd_links_df["bus_only"]) | (rd_links_df["rail_only"])
    ]

    merged_df = tr_links.merge(
        rd_links_transit_ok[["A", "B"]],
        how="left",
        on=["A", "B"],
        indicator=True,
    )

    missing_links_df = merged_df.loc[merged_df._merge == "left_only", ["trip_id", "A", "B"]]
    if len(missing_links_df):
        WranglerLogger.error(
            f"! Transit stop_time links missing in roadway network: \n {missing_links_df}"
        )
    return missing_links_df[["trip_id", "A", "B"]]


def transit_road_net_consistency(feed: Feed, road_net: RoadwayNetwork) -> bool:
    """Checks foreign key and network link relationships between transit feed and a road_net.

    Args:
        feed: Transit Feed.
        road_net (RoadwayNetwork): Roadway network to check relationship with.

    Returns:
        bool: boolean indicating if road_net is consistent with transit network.
    """
    _missing_links = shape_links_without_road_links(feed.shapes, road_net.links_df)
    _missing_nodes = transit_nodes_without_road_nodes(feed, road_net.nodes_df)
    _consistency = _missing_links.empty and not _missing_nodes
    return _consistency


def validate_transit_in_dir(
    dir: Path,
    file_format: TransitFileTypes = "txt",
    road_dir: Optional[Path] = None,
    road_file_format: RoadwayFileTypes = "geojson",
) -> bool:
    """Validates a roadway network in a directory to the wrangler data model specifications.

    Args:
        dir (Path): The transit network file directory.
        file_format (str): The format of roadway network file name. Defaults to "txt".
        road_dir (Path): The roadway network file directory. Defaults to None.
        road_file_format (str): The format of roadway network file name. Defaults to "geojson".
        output_dir (str): The output directory for the validation report. Defaults to ".".
    """
    from .io import load_transit  # noqa: PLC0415

    try:
        t = load_transit(dir, file_format=file_format)
    except SchemaErrors as e:
        WranglerLogger.error(f"!!! [Transit Network invalid] - Failed Loading to Feed object\n{e}")
        return False
    if road_dir is not None:
        from ..roadway import load_roadway_from_dir  # noqa: PLC0415
        from .network import TransitRoadwayConsistencyError  # noqa: PLC0415

        try:
            r = load_roadway_from_dir(road_dir, file_format=road_file_format)
        except FileNotFoundError:
            WranglerLogger.error(f"! Roadway network not found in {road_dir}")
            return False
        except Exception as e:
            WranglerLogger.error(
                f"! Error loading roadway network. \
                                 Skipping validation of road to transit network.\n{e}"
            )
        try:
            t.road_net = r
        except TransitRoadwayConsistencyError as e:
            WranglerLogger.error(
                f"!!! [Tranit Network inconsistent] Error in road to transit \
                                 network consistency.\n{e}"
            )
            return False

    return True
