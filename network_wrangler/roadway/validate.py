"""Validates a roadway network to the wrangler data model specifications."""

from pathlib import Path

from ..logger import WranglerLogger
from ..models._base.types import RoadwayFileTypes
from ..utils.io_table import read_table
from .io import id_roadway_file_paths_in_dir
from .links.validate import validate_links_df
from .network import RoadwayNetwork
from .nodes.validate import validate_nodes_df
from .shapes.validate import validate_shapes_df


def validate_roadway_in_dir(
    directory: Path,
    file_format: RoadwayFileTypes = "geojson",
    strict: bool = False,
    output_dir: Path = Path(),
):
    """Validates a roadway network in a directory to the wrangler data model specifications.

    Args:
        directory (str): The roadway network file directory.
        file_format(str): The formats of roadway network file name.
        strict (bool): If True, will validate the roadway network strictly without
            parsing and filling in data.
        output_dir (str): The output directory for the validation report. Defaults to ".".
    """
    links_file, nodes_file, shapes_file = id_roadway_file_paths_in_dir(directory, file_format)
    validate_roadway_files(
        links_file, nodes_file, shapes_file, strict=strict, output_dir=output_dir
    )


def validate_roadway_files(
    links_file: Path,
    nodes_file: Path,
    shapes_file: Path | None = None,
    strict: bool = False,
    output_dir: Path = Path(),
):
    """Validates the roadway network files strictly to the wrangler data model specifications.

    Args:
        links_file (str): The path to the links file.
        nodes_file (str): The path to the nodes file.
        shapes_file (str): The path to the shapes file.
        strict (bool): If True, will validate the roadway network strictly without
            parsing and filling in data.
        output_dir (str): The output directory for the validation report. Defaults to ".".
    """
    valid = {"net": True, "links": True, "nodes": True}

    nodes_df = read_table(nodes_file)
    valid["links"] = validate_nodes_df(
        nodes_df, strict=strict, errors_filename=Path(output_dir) / "node_errors.csv"
    )

    links_df = read_table(links_file)
    valid["links"] = validate_links_df(
        links_df,
        nodes_df=nodes_df,
        strict=strict,
        errors_filename=Path(output_dir) / "link_errors.csv",
    )

    if shapes_file:
        valid["shapes"] = True
        shapes_df = read_table(shapes_file)
        valid["shapes"] = validate_shapes_df(
            shapes_df, strict=strict, errors_filename=Path(output_dir) / "shape_errors.csv"
        )

    try:
        RoadwayNetwork(links_df=links_df, nodes_df=nodes_df, _shapes_df=shapes_df)
    except Exception as e:
        WranglerLogger.error(f"!!! [Network invalid] - Failed Loading to object\n{e}")
        valid["net"] = False
