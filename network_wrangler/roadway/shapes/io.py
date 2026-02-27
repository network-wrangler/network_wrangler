"""Functions to read and write RoadShapesTable."""

from __future__ import annotations

import time
from pathlib import Path

from geopandas import GeoDataFrame
from pandera.typing import DataFrame

from ...configs import DefaultConfig, WranglerConfig
from ...logger import WranglerLogger
from ...models.roadway.tables import RoadShapesTable
from ...params import LAT_LON_CRS
from ...utils.io_table import read_table, write_table
from ...utils.models import (
    empty_df_from_datamodel,
    order_fields_from_data_model,
    validate_call_pyd,
    validate_df_to_model,
)
from .create import df_to_shapes_df


@validate_call_pyd
def read_shapes(
    filename: Path,
    in_crs: int = LAT_LON_CRS,
    boundary_gdf: GeoDataFrame | None = None,
    boundary_geocode: str | None = None,
    boundary_file: Path | None = None,
    filter_to_shape_ids: list | None = None,
    config: WranglerConfig = DefaultConfig,
) -> DataFrame[RoadShapesTable]:
    """Reads shapes and returns a geodataframe of shapes if filename is found.

    Otherwise, returns empty GeoDataFrame conforming to ShapesSchema.

    Sets index to be a copy of the primary key.
    Validates output dataframe using ShapesSchema.

    Args:
        filename (str): file to read shapes in from.
        in_crs: coordinate reference system number file is in. Defaults to LAT_LON_CRS.
        boundary_gdf: GeoDataFrame to filter the input data to. Only used for geographic data.
            Defaults to None.
        boundary_geocode: Geocode to filter the input data to. Only used for geographic data.
            Defaults to None.
        boundary_file: File to load as a boundary to filter the input data to. Only used for
            geographic data. Defaults to None.
        filter_to_shape_ids: List of shape_ids to filter the input data to. Defaults to None.
        config: WranglerConfig instance. Defaults to DefaultConfig.
    """
    if not Path(filename).exists():
        WranglerLogger.warning(
            f"Shapes file {filename} not found, but is optional. \
                               Returning empty shapes dataframe."
        )
        return empty_df_from_datamodel(RoadShapesTable, crs=LAT_LON_CRS).set_index(
            "shape_id_idx", inplace=True
        )

    start_time = time.time()
    WranglerLogger.debug(f"Reading shapes from {filename}.")

    shapes_df = read_table(
        filename,
        boundary_gdf=boundary_gdf,
        boundary_geocode=boundary_geocode,
        boundary_file=boundary_file,
        read_speed=config.CPU.EST_PD_READ_SPEED,
    )
    if filter_to_shape_ids:
        shapes_df = shapes_df[shapes_df["shape_id"].isin(filter_to_shape_ids)]
    WranglerLogger.debug(
        f"Read {len(shapes_df)} shapes from file in {round(time.time() - start_time, 2)}."
    )
    shapes_df = df_to_shapes_df(shapes_df, in_crs=in_crs)
    shapes_df.attrs["source_file"] = filename
    WranglerLogger.info(
        f"Read {len(shapes_df)} shapes from {filename} in {round(time.time() - start_time, 2)}."
    )
    shapes_df = validate_df_to_model(shapes_df, RoadShapesTable)
    return shapes_df


@validate_call_pyd
def write_shapes(
    shapes_df: DataFrame[RoadShapesTable],
    out_dir: str | Path,
    prefix: str,
    format: str,
    overwrite: bool,
) -> None:
    """Writes shapes to file.

    Args:
        shapes_df: DataFrame of shapes to write.
        out_dir: directory to write shapes to.
        prefix: prefix to add to file name.
        format: format to write shapes in.
        overwrite: whether to overwrite file if it exists.
    """
    shapes_file = Path(out_dir) / f"{prefix}shape.{format}"
    shapes_df = order_fields_from_data_model(shapes_df, RoadShapesTable)
    write_table(shapes_df, shapes_file, overwrite=overwrite)
