"""Functions for reading and writing transit feeds and networks."""

from pathlib import Path
from typing import Literal, Optional, Union

import geopandas as gpd
import pandas as pd
import pyarrow as pa

from ..configs import DefaultConfig, WranglerConfig
from ..errors import FeedReadError
from ..logger import WranglerLogger
from ..models._base.db import RequiredTableError
from ..models._base.types import TransitFileTypes
from ..models.gtfs.gtfs import GtfsModel
from ..utils.geo import to_points_gdf
from ..utils.io_table import unzip_file, write_table
from .feed.feed import Feed
from .network import TransitNetwork


def _feed_path_ref(path: Path) -> Path:
    if path.suffix == ".zip":
        path = unzip_file(path)
    if not path.exists():
        msg = f"Feed path does not exist: {path}"
        raise FileExistsError(msg)

    return path


def load_feed_from_path(
    feed_path: Union[Path, str], file_format: TransitFileTypes = "txt"
) -> Feed:
    """Create a Feed object from the path to a GTFS transit feed.

    Args:
        feed_path (Union[Path, str]): The path to the GTFS transit feed.
        file_format: the format of the files to read. Defaults to "txt"

    Returns:
        Feed: The TransitNetwork object created from the GTFS transit feed.
    """
    feed_path = _feed_path_ref(Path(feed_path))  # unzips if needs to be unzipped

    if not feed_path.is_dir():
        msg = f"Feed path not a directory: {feed_path}"
        raise NotADirectoryError(msg)

    WranglerLogger.info(f"Reading GTFS feed tables from {feed_path}")

    feed_possible_files = {
        table: list(feed_path.glob(f"*{table}.{file_format}")) for table in Feed.table_names
    }

    # make sure we have all the tables we need
    _missing_files = [t for t, v in feed_possible_files.items() if not v]

    if _missing_files:
        WranglerLogger.debug(f"!!! Missing transit files: {_missing_files}")
        msg = f"Required GTFS Feed table(s) not in {feed_path}: \n  {_missing_files}"
        raise RequiredTableError(msg)

    # but don't want to have more than one file per search
    _ambiguous_files = [t for t, v in feed_possible_files.items() if len(v) > 1]
    if _ambiguous_files:
        WranglerLogger.warning(
            f"! More than one file matches following tables. \
                               Using the first on the list: {_ambiguous_files}"
        )

    feed_files = {t: f[0] for t, f in feed_possible_files.items()}
    feed_dfs = {table: _read_table_from_file(table, file) for table, file in feed_files.items()}

    return load_feed_from_dfs(feed_dfs)


def _read_table_from_file(table: str, file: Path) -> pd.DataFrame:
    WranglerLogger.debug(f"...reading {file}.")
    try:
        if file.suffix in [".csv", ".txt"]:
            return pd.read_csv(file)
        if file.suffix == ".parquet":
            return pd.read_parquet(file)
    except Exception as e:
        msg = f"Error reading table {table} from file: {file}.\n{e}"
        WranglerLogger.error(msg)
        raise FeedReadError(msg) from e


def load_feed_from_dfs(feed_dfs: dict) -> Feed:
    """Create a TransitNetwork object from a dictionary of DataFrames representing a GTFS feed.

    Args:
        feed_dfs (dict): A dictionary containing DataFrames representing the tables of a GTFS feed.

    Returns:
        Feed: A Feed object representing the transit network.

    Raises:
        ValueError: If the feed_dfs dictionary does not contain all the required tables.

    Example:
        >>> feed_dfs = {
        ...     "agency": agency_df,
        ...     "routes": routes_df,
        ...     "stops": stops_df,
        ...     "trips": trips_df,
        ...     "stop_times": stop_times_df,
        ... }
        >>> feed = load_feed_from_dfs(feed_dfs)
    """
    if not all(table in feed_dfs for table in Feed.table_names):
        msg = f"feed_dfs must contain the following tables: {Feed.table_names}"
        raise ValueError(msg)

    feed = Feed(**feed_dfs)

    return feed


def load_transit(
    feed: Union[Feed, GtfsModel, dict[str, pd.DataFrame], str, Path],
    file_format: TransitFileTypes = "txt",
    config: WranglerConfig = DefaultConfig,
) -> "TransitNetwork":
    """Create a TransitNetwork object.

    This function takes in a `feed` parameter, which can be one of the following types:
    - `Feed`: A Feed object representing a transit feed.
    - `dict[str, pd.DataFrame]`: A dictionary of DataFrames representing transit data.
    - `str` or `Path`: A string or a Path object representing the path to a transit feed file.

    Args:
        feed: Feed boject, dict of transit data frames, or path to transit feed data
        file_format: the format of the files to read. Defaults to "txt"
        config: WranglerConfig object. Defaults to DefaultConfig.

    Returns:
    A TransitNetwork object representing the loaded transit network.

    Raises:
    ValueError: If the `feed` parameter is not one of the supported types.

    Example usage:
    ```
    transit_network_from_zip = load_transit("path/to/gtfs.zip")

    transit_network_from_unzipped_dir = load_transit("path/to/files")

    transit_network_from_parquet = load_transit("path/to/files", file_format="parquet")

    dfs_of_transit_data = {"routes": routes_df, "stops": stops_df, "trips": trips_df...}
    transit_network_from_dfs = load_transit(dfs_of_transit_data)
    ```

    """
    if isinstance(feed, (Path, str)):
        feed = Path(feed)
        feed_obj = load_feed_from_path(feed, file_format=file_format)
        feed_obj.feed_path = feed
    elif isinstance(feed, dict):
        feed_obj = load_feed_from_dfs(feed)
    elif isinstance(feed, GtfsModel):
        feed_obj = Feed(**feed.__dict__)
    else:
        if not isinstance(feed, Feed):
            msg = f"TransitNetwork must be seeded with a Feed, dict of dfs or Path. Found {type(feed)}"
            raise ValueError(msg)
        feed_obj = feed

    return TransitNetwork(feed_obj, config=config)


def write_transit(
    transit_net,
    out_dir: Union[Path, str] = ".",
    prefix: Optional[Union[Path, str]] = None,
    file_format: Literal["txt", "csv", "parquet"] = "txt",
    overwrite: bool = True,
) -> None:
    """Writes a network in the transit network standard.

    Args:
        transit_net: a TransitNetwork instance
        out_dir: directory to write the network to
        file_format: the format of the output files. Defaults to "txt" which is csv with txt
            file format.
        prefix: prefix to add to the file name
        overwrite: if True, will overwrite the files if they already exist. Defaults to True
    """
    out_dir = Path(out_dir)
    prefix = f"{prefix}_" if prefix else ""
    for table in transit_net.feed.table_names:
        df = transit_net.feed.get_table(table)
        outpath = out_dir / f"{prefix}{table}.{file_format}"
        write_table(df, outpath, overwrite=overwrite)
    WranglerLogger.info(f"Wrote {len(transit_net.feed.tables)} files to {out_dir}")


def convert_transit_serialization(
    input_path: Union[str, Path],
    output_format: TransitFileTypes,
    out_dir: Union[Path, str] = ".",
    input_file_format: TransitFileTypes = "csv",
    out_prefix: str = "",
    overwrite: bool = True,
):
    """Converts a transit network from one serialization to another.

    Args:
        input_path: path to the input network
        output_format: the format of the output files. Should be txt, csv, or parquet.
        out_dir: directory to write the network to. Defaults to current directory.
        input_file_format: the file_format of the files to read. Should be txt, csv, or parquet.
            Defaults to "txt"
        out_prefix: prefix to add to the file name. Defaults to ""
        overwrite: if True, will overwrite the files if they already exist. Defaults to True
    """
    WranglerLogger.info(
        f"Loading transit net from {input_path} with input type {input_file_format}"
    )
    net = load_transit(input_path, file_format=input_file_format)
    WranglerLogger.info(f"Writing transit network to {out_dir} in {output_format} format.")
    write_transit(
        net,
        prefix=out_prefix,
        out_dir=out_dir,
        file_format=output_format,
        overwrite=overwrite,
    )


def write_feed_geo(
    feed: Feed,
    ref_nodes_df: gpd.GeoDataFrame,
    out_dir: Union[str, Path],
    file_format: Literal["geojson", "shp", "parquet"] = "geojson",
    out_prefix=None,
    overwrite: bool = True,
) -> None:
    """Write a Feed object to a directory in a geospatial format.

    Args:
        feed: Feed object to write
        ref_nodes_df: Reference nodes dataframe to use for geometry
        out_dir: directory to write the network to
        file_format: the format of the output files. Defaults to "geojson"
        out_prefix: prefix to add to the file name
        overwrite: if True, will overwrite the files if they already exist. Defaults to True
    """
    from .geo import shapes_to_shape_links_gdf  # noqa: PLC0415

    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        if out_dir.parent.is_dir():
            out_dir.mkdir()
        else:
            msg = f"Output directory {out_dir} ands its parent path does not exist"
            raise FileNotFoundError(msg)

    prefix = f"{out_prefix}_" if out_prefix else ""
    shapes_outpath = out_dir / f"{prefix}trn_shapes.{file_format}"
    shapes_gdf = shapes_to_shape_links_gdf(feed.shapes, ref_nodes_df=ref_nodes_df)
    write_table(shapes_gdf, shapes_outpath, overwrite=overwrite)

    stops_outpath = out_dir / f"{prefix}trn_stops.{file_format}"
    stops_gdf = to_points_gdf(feed.stops, ref_nodes_df=ref_nodes_df)
    write_table(stops_gdf, stops_outpath, overwrite=overwrite)
