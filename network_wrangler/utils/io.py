"""Helper functions for reading and writing files to reduce boilerplate."""

import json
import tempfile
import shutil
import tempfile
import weakref

from datetime import datetime
from pathlib import Path
from typing import Union, Optional

import geopandas as gpd
import pandas as pd

from ..params import EST_PD_READ_SPEED

from ..utils.geo import get_bounding_polygon
from ..utils.data import convert_numpy_to_list
from ..logger import WranglerLogger
from .time import format_time

try:
    gpd.options.io_engine = "pyogrio"
except:  # noqa: E722
    if gpd.__version__ < "0.12.0":
        WranglerLogger.warning(
            f"Installed Geopandas version {gpd.__version__} isn't recent enough to support\
                pyogrio. Falling back to default engine (fiona).\
                Update geopandas and install pyogrio to benefit."
        )
    else:
        WranglerLogger.warning(
            "Geopandas is not using pyogrio as the I/O engine.\
                Install pyogrio to benefit from faster I/O."
        )


class FileReadError(Exception):
    """Raised when there is an error reading a file."""

    pass


class FileWriteError(Exception):
    """Raised when there is an error writing a file."""

    pass


def write_table(
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    filename: Path,
    overwrite: bool = False,
    **kwargs,
) -> None:
    """Write a dataframe or geodataframe to a file.

    Args:
        df (pd.DataFrame): dataframe to write.
        filename (Path): filename to write to.
        overwrite (bool): whether to overwrite the file if it exists. Defaults to False.
        kwargs: additional arguments to pass to the writer.

    """
    filename = Path(filename)
    if filename.exists() and not overwrite:
        raise FileExistsError(f"File {filename} already exists and overwrite is False.")

    if filename.parent.is_dir() and not filename.parent.exists():
        filename.parent.mkdir(parents=True)

    WranglerLogger.debug(f"Writing to {filename}.")

    if "shp" in filename.suffix:
        df.to_file(filename, index=False, **kwargs)
    elif "parquet" in filename.suffix:
        df.to_parquet(filename, index=False, **kwargs)
    elif "csv" in filename.suffix:
        df.to_csv(filename, index=False, date_format="%H:%M:%S", **kwargs)
    elif "txt" in filename.suffix:
        df.to_csv(filename, index=False, date_format="%H:%M:%S", **kwargs)
    elif "geojson" in filename.suffix:
        # required due to issues with list-like columns
        if isinstance(df, gpd.GeoDataFrame):
            data = df.to_json(drop_id=True)
        else:
            data = df.to_json(orient="records", index=False)
        with open(filename, "w", encoding="utf-8") as file:
            file.write(data)
    elif "json" in filename.suffix:
        with open(filename, "w") as f:
            f.write(df.to_json(orient="records"))
    else:
        raise NotImplementedError(f"Filetype {filename.suffix} not implemented.")


def _estimate_read_time_of_file(filepath: Union[str, Path]):
    """Estimates read time in seconds based on a given file size and speed factor.

    The speed factor is MB per second which you can adjust based on empirical data.

    TODO: implement based on file type and empirical
    """
    # MB per second
    filepath = Path(filepath)
    file_size_mb = filepath.stat().st_size / (1024 * 1024)  # Convert bytes to MB
    filetype = filepath.suffix[1:]
    if filetype in EST_PD_READ_SPEED:
        return format_time(file_size_mb * EST_PD_READ_SPEED[filetype])
    else:
        return "unknown"


def read_table(
    filename: Path,
    sub_filename: str = None,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
    boundary_geocode: Optional[str] = None,
    boundary_file: Optional[Path] = None,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Read file and return a dataframe or geodataframe.

    If filename is a zip file, will unzip to a temporary directory.

    If filename is a geojson or shapefile, will filter the data
    to the boundary_gdf, boundary_geocode, or boundary_file if provided. Note that you can only
    provide one of these boundary filters.

    If filename is a geoparquet file, will filter the data to the *bounding box* of the
    boundary_gdf, boundary_geocode, or boundary_file if provided. Note that you can only
    provide one of these boundary filters.

    NOTE:  if you are accessing multiple files from this zip file you will want to unzip it first
    and THEN access the table files so you don't create multiple duplicate unzipped tmp dirs.

    Args:
        filename (Path): filename to load.
        sub_filename: if the file is a zip, the sub_filename to load.
        boundary_gdf: GeoDataFrame to filter the input data to. Only used for geographic data.
            Defaults to None.
        boundary_geocode: Geocode to filter the input data to. Only used for geographic data.
            Defaults to None.
        boundary_file: File to load as a boundary to filter the input data to. Only used for
            geographic data. Defaults to None.
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"Input file {filename} does not exist.")
    if filename.stat().st_size == 0:
        raise FileExistsError(f"File {filename} is empty.")
    if filename.suffix == ".zip":
        filename = unzip_file(filename) / sub_filename
    WranglerLogger.debug(f"Estimated read time: {_estimate_read_time_of_file(filename)}.")

    # will result in None if no boundary is provided
    mask_gdf = get_bounding_polygon(
        boundary_gdf=boundary_gdf,
        boundary_geocode=boundary_geocode,
        boundary_file=boundary_file,
    )

    if any([x in filename.suffix for x in ["geojson", "shp", "csv"]]):
        try:
            # masking only supported by fiona engine, which is slower.
            if mask_gdf is None:
                return gpd.read_file(filename, engine="pyogrio")
            else:
                return gpd.read_file(filename, mask=mask_gdf, engine="fiona")
        except:  # noqa: E722
            if "csv" in filename.suffix:
                return pd.read_csv(filename)
            raise FileReadError
    elif "parquet" in filename.suffix:
        return _read_parquet_table(filename, mask_gdf)
    elif "json" in filename.suffix:
        with open(filename) as f:
            return pd.read_json(f, orient="records")
    raise NotImplementedError(f"Filetype {filename.suffix} not implemented.")


def _read_parquet_table(filename, mask_gdf) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
    """Read a parquet file and filter to a bounding box if provided.

    Converts numpy arrays to lists.

    Tries first to use geopandas and see if geoparquet file. If not, will use pandas.

    Tries to filter to a bounding box if a mask_gdf is provided. If the geopandas version is
    less than 1.0, will return the unfiltered data.
    """
    try:
        if mask_gdf is None:
            df = gpd.read_parquet(filename)
        else:
            try:
                df = gpd.read_parquet(filename, bbox=mask_gdf.total_bounds)
            except TypeError:
                WranglerLogger.warning(f"Could not filter to bounding box {mask_gdf}.\
                                        Try upgrading to geopandas > 1.0.\
                                        Returning unfiltered data.")
                df = gpd.read_parquet(filename)
    except:  # noqa: E722
        df = pd.read_parquet(filename)

    _cols = [col for col in df.columns if col.startswith("sc_")]
    for col in _cols:
        df[col] = df[col].apply(convert_numpy_to_list)
    return df


def convert_file_serialization(
    input_file: Path,
    output_file: Path,
    overwrite: bool = True,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
    boundary_geocode: Optional[str] = None,
    boundary_file: Optional[Path] = None,
    node_filter_s: Optional[pd.Series] = None,
    chunk_size: Optional[int] = None,
):
    """Convert a file serialization format to another and optionally filter to a boundary.

    If the input file is a JSON file that is larger than a reasonable portion of available
    memory, *and* the output file is a Parquet file the JSON file will be read in chunks.

    If the input file is a Geographic data type (shp, geojon, geoparquet) and a boundary is
    provided, the data will be filtered to the boundary.

    Args:
        input_file: Path to the input JSON or GEOJSON file.
        output_file: Path to the output Parquet file.
        overwrite: If True, overwrite the output file if it exists.
        boundary_gdf: GeoDataFrame to filter the input data to. Only used for geographic data.
            Defaults to None.
        boundary_geocode: Geocode to filter the input data to. Only used for geographic data.
            Defaults to None.
        boundary_file: File to load as a boundary to filter the input data to. Only used for
            geographic data. Defaults to None.
        node_filter_s: If provided, will filter links in .json file to only those that connect to
            nodes. Defaults to None.
        chunk_size: Number of JSON objects to process in each chunk. Only works for
            JSON to Parquet. If None, will determine if chunking needed and what size.
    """
    WranglerLogger.debug(f"Converting {input_file} to {output_file}.")

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"File {output_file} already exists and overwrite is False.")

    if Path(input_file).suffix == ".json" and Path(output_file).suffix == ".parquet":
        if chunk_size is None:
            chunk_size = _suggest_json_chunk_size(input_file)
        if chunk_size is None:
            df = read_table(input_file)
            if node_filter_s is not None and "A" in df.columns and "B" in df.columns:
                df = df[df["A"].isin(node_filter_s) | df["B"].isin(node_filter_s)]
            write_table(df, output_file, overwrite=overwrite)
        else:
            _json_to_parquet_in_chunks(input_file, output_file, chunk_size)

    df = read_table(
        input_file,
        boundary_gdf=boundary_gdf,
        boundary_geocode=boundary_geocode,
        boundary_file=boundary_file,
    )
    if node_filter_s is not None and "A" in df.columns and "B" in df.columns:
        df = df[df["A"].isin(node_filter_s) | df["B"].isin(node_filter_s)]
    write_table(df, output_file, overwrite=overwrite)


def _available_memory():
    """Return the available memory in bytes."""
    import psutil

    return psutil.virtual_memory().available


def _estimate_bytes_per_json_object(json_path: Path) -> int:
    """Estimate the size of a JSON object in bytes based on sample."""
    SAMPLE_SIZE = 50
    with open(json_path, "r") as f:
        json_objects = []
        for _ in range(SAMPLE_SIZE):
            line = f.readline()
            if not line:
                break
            json_objects.append(json.loads(line))

    # Calculate the average size of the JSON objects
    total_size = sum(len(json.dumps(obj)) for obj in json_objects)
    return total_size / len(json_objects)


def _suggest_json_chunk_size(json_path: Path, memory_fraction: float = 0.6) -> Union[None, int]:
    """Ascertain if a file should be processed in chunks and how large the chunks should be in mb.

    Args:
        json_path: Path to the input JSON file.
        memory_fraction: Fraction of available memory to use. Defaults to 0.6.
    """
    file_size_bytes = json_path.stat().st_size
    max_memory_use_bytes = _available_memory() * memory_fraction
    if file_size_bytes < max_memory_use_bytes:
        return None

    avg_bytes_per_object = _estimate_bytes_per_json_object(json_path)

    # Determine the number of objects that fit into the max memory use
    chunk_size = int(max_memory_use_bytes / avg_bytes_per_object)
    WranglerLogger.debug(f"Suggested chunk size: {chunk_size}\n...for file: {json_path}.")
    return chunk_size


def _append_parquet_table(
    new_data: pd.DataFrame,
    file_counter=1,
    base_filename: Optional[Path] = None,
    directory: Optional[Path] = None,
) -> Path:
    """Append new data to a Parquet dataset directory.

    If no directory is provided, it will create one based on the base filename
    and the current date-time in the current working directory.

    Args:
        new_data: DataFrame containing new data to append.
        file_counter: An integer used to name the new Parquet file uniquely.
        base_filename: The base name for the directory and files.
        directory: The directory to write the new Parquet file to.

    Returns:
        Path: The path to the output directory.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if directory is None:
        temp_dir = tempfile.mkdtemp()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if base_filename is None:
            base_filename = "output"
        output_directory = Path(temp_dir) / f"{base_filename}_{current_time}"
    else:
        output_directory = directory

    output_directory.mkdir(parents=True, exist_ok=True)
    chunk_file = output_directory / f"part-{file_counter}.parquet"
    pq.write_table(pa.Table.from_pandas(new_data), chunk_file)

    return output_directory


def _json_to_parquet_in_chunks(input_file: Path, output_file: Path, chunk_size: int = 100000):
    """Process a large JSON in chunks and convert it to a single Parquet file.

    Args:
        input_file: Path to the input JSON file.
        output_file: Path to the output Parquet file.
        chunk_size: Number of JSON objects to process in each chunk.
    """
    try:
        import ijson
    except ModuleNotFoundError:
        raise ModuleNotFoundError("ijson is required for chunked JSON processing.")

    import pyarrow.parquet as pq

    base_filename = Path(output_file).stem
    directory = None
    file_counter = 0

    buffer = []
    with open(input_file, "r") as f:
        parser = ijson.items(f, "item")
        for item in parser:
            buffer.append(item)

            if len(buffer) >= chunk_size:
                df = pd.DataFrame(buffer)
                directory = _append_parquet_table(
                    df, file_counter=file_counter, base_filename=base_filename, directory=directory
                )
                buffer = []
                file_counter += 1

        if buffer:
            df = pd.DataFrame(buffer)
            directory = _append_parquet_table(
                df, file_counter=file_counter, base_filename=base_filename, directory=directory
            )

    # Combine all the chunks into a single Parquet file
    combined_table = pq.ParquetDataset(directory).read()
    pq.write_table(combined_table, output_file)

    # Clean up the temporary chunk files
    shutil.rmtree(directory)

    WranglerLogger.debug(f"Wrote combined data to {output_file} and cleaned up temporary files.")


def unzip_file(path: Path) -> Path:
    """Unzips a file to a temporary directory and returns the directory path."""
    tmpdir = tempfile.mkdtemp()
    shutil.unpack_archive(path, tmpdir)

    def finalize() -> None:
        shutil.rmtree(tmpdir)

    # Lazy cleanup
    weakref.finalize(tmpdir, finalize)

    return tmpdir