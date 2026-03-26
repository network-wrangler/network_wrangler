"""Functions for reading and writing transit feeds and networks."""

import os
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
from .filter import filter_feed_by_service_ids
from .network import TransitNetwork


def _feed_path_ref(path: Path) -> Path:
    if path.suffix == ".zip":
        path = unzip_file(path)
    if not path.exists():
        msg = f"Feed path does not exist: {path}"
        raise FileExistsError(msg)

    return path


def load_feed_from_path(
    feed_path: Path | str,
    file_format: TransitFileTypes = "txt",
    wrangler_flavored: bool = True,
    service_ids_filter: list[str] | None = None,
    **read_kwargs,
) -> Feed | GtfsModel:
    """Create a Feed or GtfsModel object from the path to a GTFS transit feed.

    Args:
        feed_path (Union[Path, str]): The path to the GTFS transit feed.
        file_format: the format of the files to read. Defaults to "txt"
        wrangler_flavored: If True, creates a Wrangler-enhanced Feed object.
                          If False, creates a pure GtfsModel object. Defaults to True.
        service_ids_filter (Optional[list[str]]): If not None, filter to these service_ids. Assumes service_id is a str.
        **read_kwargs: Additional keyword arguments to pass to the file reader (e.g., low_memory, dtype)

    Returns:
        Union[Feed, GtfsModel]: The Feed or GtfsModel object created from the GTFS transit feed.
    """
    feed_path = _feed_path_ref(Path(feed_path))  # unzips if needs to be unzipped

    if not feed_path.is_dir():
        msg = f"Feed path not a directory: {feed_path}"
        raise NotADirectoryError(msg)

    WranglerLogger.info(f"Reading GTFS feed tables from {feed_path}")

    # Use the appropriate table names based on the model type
    model_class = Feed if wrangler_flavored else GtfsModel
    feed_possible_files = {
        table: list(feed_path.glob(f"*{table}.{file_format}")) for table in model_class.table_names + model_class.optional_table_names
    }
    WranglerLogger.debug(f"model_class={model_class}  feed_possible_files={feed_possible_files}")

    # make sure we have all the tables we need -- missing optional is ok
    _missing_files = []
    for table_name in list(feed_possible_files.keys()):
        if not feed_possible_files[table_name]:
            # remove those that don't have files
            del feed_possible_files[table_name]

            # missiong optional is ok
            if table_name in model_class.table_names:
                _missing_files.append(table_name)

        

    if _missing_files:
        WranglerLogger.debug(f"!!! Missing transit files: {_missing_files}")
        model_name = "Feed" if wrangler_flavored else "GtfsModel"
        msg = f"Required GTFS {model_name} table(s) not in {feed_path}: \n  {_missing_files}"
        raise RequiredTableError(msg)

    # but don't want to have more than one file per search
    _ambiguous_files = [t for t, v in feed_possible_files.items() if len(v) > 1]
    if _ambiguous_files:
        WranglerLogger.warning(
            f"! More than one file matches following tables. "
            + f"Using the first on the list: {_ambiguous_files}"
        )

    feed_files = {t: f[0] for t, f in feed_possible_files.items()}
    feed_dfs = {table: _read_table_from_file(table, file, **read_kwargs) for table, file in feed_files.items()}
    
    # Create the feed object first
    feed_obj = load_feed_from_dfs(feed_dfs, wrangler_flavored=wrangler_flavored)
    WranglerLogger.debug(f"loaded {type(feed_obj)} from dfs:\n{feed_obj}")
    
    # Apply service_ids filter if provided
    if service_ids_filter is not None:
        feed_obj = filter_feed_by_service_ids(feed_obj, service_ids_filter)
    
    return feed_obj


def _read_table_from_file(table: str, file: Path, **kwargs) -> pd.DataFrame:
    """Read a table from a file with support for additional kwargs.
    
    Args:
        table: Name of the table being read (for error messages)
        file: Path to the file to read
        **kwargs: Additional keyword arguments to pass to the appropriate reader
        
    Returns:
        pd.DataFrame: The loaded dataframe
    """
    WranglerLogger.debug(f"...reading {file}.")
    try:
        if file.suffix in [".csv", ".txt"]:
            return pd.read_csv(file, **kwargs)
        if file.suffix == ".parquet":
            return pd.read_parquet(file, **kwargs)
    except Exception as e:
        msg = f"Error reading table {table} from file: {file}.\n{e}"
        WranglerLogger.error(msg)
        raise FeedReadError(msg) from e


def load_feed_from_dfs(feed_dfs: dict, wrangler_flavored: bool = True) -> Feed | GtfsModel:
    """Create a Feed or GtfsModel object from a dictionary of DataFrames representing a GTFS feed.

    Args:
        feed_dfs (dict): A dictionary containing DataFrames representing the tables of a GTFS feed.
        wrangler_flavored: If True, creates a Wrangler-enhanced Feed] object.
                           If False, creates a pure GtfsModel object. Defaults to True.

    Returns:
        Union[Feed, GtfsModel]: A Feed or GtfsModel object representing the transit network.

    Raises:
        ValueError: If the feed_dfs dictionary does not contain all the required tables.

    Example usage:
    ```python
    feed_dfs = {
        "agency": agency_df,
        "routes": routes_df,
        "stops": stops_df,
        "trips": trips_df,
        "stop_times": stop_times_df,
    }
    feed = load_feed_from_dfs(feed_dfs)  # Creates Feed by default
    gtfs_model = load_feed_from_dfs(feed_dfs, wrangler_flavored=False)  # Creates GtfsModel
    ```
    """
    # Use the appropriate model class based on the parameter
    model_class = Feed if wrangler_flavored else GtfsModel

    if not all(table in feed_dfs for table in model_class.table_names):
        model_name = "Feed" if wrangler_flavored else "GtfsModel"
        msg = f"feed_dfs must contain the following tables for {model_name}: {model_class.table_names}"
        raise ValueError(msg)

    feed = model_class(**feed_dfs)

    return feed


def load_transit(
    feed: Feed | GtfsModel | dict[str, pd.DataFrame] | str | Path,
    file_format: TransitFileTypes = "txt",
    config: WranglerConfig = DefaultConfig,
) -> TransitNetwork:
    """Create a [`TransitNetwork`][network_wrangler.transit.network.TransitNetwork] object.

    This function takes in a `feed` parameter, which can be one of the following types:

    - `Feed`: A Feed object representing a transit feed.
    - `dict[str, pd.DataFrame]`: A dictionary of DataFrames representing transit data.
    - `str` or `Path`: A string or a Path object representing the path to a transit feed file.

    Args:
        feed: Feed boject, dict of transit data frames, or path to transit feed data
        file_format: the format of the files to read. Defaults to "txt"
        config: WranglerConfig object. Defaults to DefaultConfig.

    Returns:
        (TransitNetwork): object representing the loaded transit network.

    Raises:
    ValueError: If the `feed` parameter is not one of the supported types.

    Example usage:
    ```python
    transit_network_from_zip = load_transit("path/to/gtfs.zip")

    transit_network_from_unzipped_dir = load_transit("path/to/files")

    transit_network_from_parquet = load_transit("path/to/files", file_format="parquet")

    dfs_of_transit_data = {"routes": routes_df, "stops": stops_df, "trips": trips_df...}
    transit_network_from_dfs = load_transit(dfs_of_transit_data)
    ```

    """
    if isinstance(feed, Path | str):
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
    transit_obj: TransitNetwork | Feed | GtfsModel,
    out_dir: Path | str = ".",
    prefix: Path | str | None = None,
    file_format: Literal["txt", "csv", "parquet"] = "txt",
    overwrite: bool = True,
) -> None:
    """Writes a transit network, feed, or GTFS model to files.

    Args:
        transit_obj: a TransitNetwork, Feed, or GtfsModel instance
        out_dir: directory to write the network to
        file_format: the format of the output files. Defaults to "txt" which is csv with txt
            file format.
        prefix: prefix to add to the file name
        overwrite: if True, will overwrite the files if they already exist. Defaults to True
    """
    out_dir = Path(out_dir)
    prefix = f"{prefix}_" if prefix else ""

    # Determine the data source based on input type
    if isinstance(transit_obj, TransitNetwork):
        data_source = transit_obj.feed
        source_type = "TransitNetwork"
    elif isinstance(transit_obj, Feed | GtfsModel):
        data_source = transit_obj
        source_type = type(transit_obj).__name__
    else:
        msg = (
            f"transit_obj must be a TransitNetwork, Feed, or GtfsModel instance, "
            f"not {type(transit_obj).__name__}"
        )
        raise TypeError(msg)

    # Write tables
    tables_written = 0
    for table in data_source.table_names:
        df = data_source.get_table(table)
        if df is not None and len(df) > 0:  # Only write non-empty tables
            outpath = out_dir / f"{prefix}{table}.{file_format}"
            write_table(df, outpath, overwrite=overwrite)
            tables_written += 1

    WranglerLogger.info(f"Wrote {tables_written} {source_type} tables to {out_dir}")


def convert_transit_serialization(
    input_path: str | Path,
    output_format: TransitFileTypes,
    out_dir: Path | str = ".",
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
    out_dir: str | Path,
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
    from .geo import shapes_to_shape_links_gdf

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
