"""Functions to read and write RoadShapesTable."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union, Optional

from geopandas import GeoDataFrame

from pydantic import validate_call
from pandera.typing import DataFrame

from ...logger import WranglerLogger
from ...models.roadway.tables import RoadShapesTable
from ...params import ShapesParams, LAT_LON_CRS
from ...utils.io import read_table, write_table
from ...utils.models import empty_df_from_datamodel
from .create import df_to_shapes_df


@validate_call(config=dict(arbitrary_types_allowed=True), validate_return=True)
def read_shapes(
    filename: Union[str, Path],
    in_crs: int = LAT_LON_CRS,
    shapes_params: Union[dict, ShapesParams, None] = None,
    boundary_gdf: Optional[GeoDataFrame] = None,
    boundary_geocode: Optional[str] = None,
    boundary_file: Optional[Path] = None,
    filter_to_shape_ids: Optional[list] = None,
) -> DataFrame[RoadShapesTable]:
    """Reads shapes and returns a geodataframe of shapes if filename is found.

    Otherwise, returns empty GeoDataFrame conforming to ShapesSchema.

    Sets index to be a copy of the primary key.
    Validates output dataframe using ShapesSchema.

    Args:
        filename (str): file to read shapes in from.
        in_crs: coordinate reference system number file is in. Defaults to LAT_LON_CRS.
        shapes_params: a ShapesParams instance. Defaults to a default ShapesParams instance.
        boundary_gdf: GeoDataFrame to filter the input data to. Only used for geographic data.
            Defaults to None.
        boundary_geocode: Geocode to filter the input data to. Only used for geographic data.
            Defaults to None.
        boundary_file: File to load as a boundary to filter the input data to. Only used for
            geographic data. Defaults to None.
        filter_to_shape_ids: List of shape_ids to filter the input data to. Defaults to None.
    """
    if not Path(filename).exists():
        WranglerLogger.warning(
            f"Shapes file {filename} not found, but is optional. \
                               Returning empty shapes dataframe."
        )
        return empty_df_from_datamodel(RoadShapesTable).set_index("shape_id_idx", inplace=True)

    start_time = time.time()
    WranglerLogger.debug(f"Reading shapes from {filename}.")

    shapes_df = read_table(
        filename,
        boundary_gdf=boundary_gdf,
        boundary_geocode=boundary_geocode,
        boundary_file=boundary_file,
    )
    if filter_to_shape_ids:
        shapes_df = shapes_df[shapes_df["shape_id"].isin(filter_to_shape_ids)]
    WranglerLogger.debug(
        f"Read {len(shapes_df)} shapes from file in {round(time.time() - start_time, 2)}."
    )
    shapes_df = df_to_shapes_df(shapes_df, in_crs=in_crs, shapes_params=shapes_params)
    shapes_df.params.source_file = filename
    WranglerLogger.info(
        f"Read {len(shapes_df)} shapes from {filename} in {round(time.time() - start_time, 2)}."
    )
    return shapes_df


def write_shapes(
    shapes_df: DataFrame[RoadShapesTable],
    out_dir: Union[str, Path],
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
    write_table(shapes_df, shapes_file, overwrite=overwrite)