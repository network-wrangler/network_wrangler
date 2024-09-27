"""Tests roadway input output."""

import time
import os

import pytest

from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from network_wrangler import (
    write_roadway,
    load_roadway_from_dir,
    WranglerLogger,
)
from network_wrangler.roadway import diff_nets
from network_wrangler.roadway.io import (
    convert_roadway_file_serialization,
    id_roadway_file_paths_in_dir,
)
from network_wrangler.roadway.network import RoadwayNetwork


def test_id_roadway_file_paths_in_dir(tmpdir):
    # Create mock files in the temporary directory
    links_file = tmpdir / "test_links.json"
    nodes_file = tmpdir / "test_nodes.geojson"
    shapes_file = tmpdir / "test_shapes.geojson"
    links_file.write("")
    nodes_file.write("")
    shapes_file.write("")

    # Test Case 1: All files are present
    links_path, nodes_path, shapes_path = id_roadway_file_paths_in_dir(
        tmpdir, file_format="geojson"
    )
    assert links_path == links_file
    assert nodes_path == nodes_file
    assert shapes_path == shapes_file

    # Test Case 2: Links file is missing
    os.remove(links_file)
    with pytest.raises(FileNotFoundError):
        id_roadway_file_paths_in_dir(tmpdir, file_format="geojson")

    # Test Case 3: Nodes file is missing
    links_file.write("")
    os.remove(nodes_file)
    with pytest.raises(FileNotFoundError):
        id_roadway_file_paths_in_dir(tmpdir, file_format="geojson")

    # Test Case 4: Shapes file is missing (optional)
    nodes_file.write("")
    os.remove(shapes_file)
    links_path, nodes_path, shapes_path = id_roadway_file_paths_in_dir(
        tmpdir, file_format="geojson"
    )
    assert links_path == links_file
    assert nodes_path == nodes_file
    assert shapes_path is None


def test_convert(example_dir, tmpdir):
    """Test that the convert function works for both geojson and parquet.

    Also makes sure that the converted network is the same as the original when the original
    is geographically complete (it will have added information when it is not geographically
    complete).
    """
    out_dir = tmpdir

    # convert EX from geojson to parquet
    convert_roadway_file_serialization(
        example_dir / "small",
        "geojson",
        out_dir,
        "parquet",
        "simple",
        True,
    )

    output_files_parq = [
        out_dir / "simple_links.parquet",
        out_dir / "simple_nodes.parquet",
    ]

    missing_parq = [i for i in output_files_parq if not i.exists()]
    if missing_parq:
        WranglerLogger.error(f"Missing {len(missing_parq)} parquet output files: {missing_parq})")
        raise FileNotFoundError("Missing converted parquet files.")

    # convert parquet to geojson
    convert_roadway_file_serialization(
        out_dir,
        "parquet",
        out_dir,
        "geojson",
        "simple",
        True,
    )

    output_files_geojson = [
        out_dir / "simple_links.json",
        out_dir / "simple_nodes.geojson",
    ]

    missing_geojson = [i for i in output_files_geojson if not i.exists()]
    if missing_geojson:
        WranglerLogger.error(
            f"Missing {len(missing_geojson)} geojson output files: {missing_geojson})"
        )
        raise FileNotFoundError("Missing converted geojson files.")

    WranglerLogger.debug("Reading in og network to test that it is equal.")
    in_net = load_roadway_from_dir(example_dir / "small", file_format="geojson")

    WranglerLogger.debug("Reading in converted network to test that it is equal.")
    out_net_parq = load_roadway_from_dir(out_dir, file_format="parquet")
    out_net_geojson = load_roadway_from_dir(out_dir, file_format="geojson")

    WranglerLogger.info("Evaluating original vs parquet network.")
    assert not diff_nets(in_net, out_net_parq), "The original and parquet networks differ."
    WranglerLogger.info("Evaluating parquet vs geojson network.")
    assert not diff_nets(out_net_parq, out_net_geojson), "The parquet and geojson networks differ."


def test_roadway_model_coerce(small_net):
    assert isinstance(small_net, RoadwayNetwork)
    WranglerLogger.debug(f"small_net.nodes_df.cols: \n{small_net.nodes_df.columns}")
    assert "osm_node_id" in small_net.nodes_df.columns
    WranglerLogger.debug(f"small_net.links_df.cols: \n{small_net.links_df.columns}")
    assert "osm_link_id" in small_net.links_df.columns


@pytest.mark.parametrize("io_format", ["geojson", "parquet"])
@pytest.mark.parametrize("ex", ["stpaul", "small"])
def test_roadway_geojson_read_write_read(example_dir, test_out_dir, ex, io_format):
    read_dir = example_dir / ex
    net = load_roadway_from_dir(read_dir)
    test_io_dir = test_out_dir / ex
    t_0 = time.time()
    write_roadway(net, file_format=io_format, out_dir=test_io_dir, overwrite=True)
    t_write = time.time() - t_0
    WranglerLogger.info(
        f"{int(t_write // 60): 02d}:{int(t_write % 60): 02d} – {ex} write to {io_format}"  # noqa: E231, E501
    )
    t_0 = time.time()
    net = load_roadway_from_dir(test_io_dir, file_format=io_format)
    t_read = time.time() - t_0
    WranglerLogger.info(
        f"{int(t_read // 60): 02d}:{int(t_read % 60): 02d} – {ex} read from {io_format}"  # noqa: E231, E501
    )
    assert isinstance(net, RoadwayNetwork)


def test_load_roadway_no_shapes(tmpdir, example_dir):
    # Test Case 2: Without Shapes File
    roadway_network = load_roadway_from_dir(example_dir / "small")
    assert isinstance(roadway_network, RoadwayNetwork)
    assert not roadway_network.links_df.empty
    assert not roadway_network.nodes_df.empty
    assert roadway_network._shapes_df is None


def test_load_roadway_within_boundary(tmpdir, example_dir):
    one_block = Polygon(
        [
            [-93.09424891687992, 44.950667556032386],
            [-93.09318302493314, 44.949458919295751],
            [-93.09110424119152, 44.950413327659845],
            [-93.09238213374682, 44.951563597873246],
            [-93.09424891687992, 44.950667556032386],
        ]
    )
    boundary_gdf = GeoDataFrame({"geometry": [one_block]}, crs="EPSG:4326")
    roadway_network = load_roadway_from_dir(example_dir / "small", boundary_gdf=boundary_gdf)

    assert isinstance(roadway_network, RoadwayNetwork)

    expected_node_ids = [2, 3, 6, 7]
    assert set(roadway_network.nodes_df.index) == set(expected_node_ids)
    assert set(roadway_network.links_df["A"]).issubset(set(expected_node_ids))
    assert set(roadway_network.links_df["B"]).issubset(set(expected_node_ids))
