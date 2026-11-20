"""Characterize Pandera behavior for roadway table schemas."""

from datetime import datetime

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from network_wrangler.models.roadway.tables import (
    ExplodedScopedLinkPropertyTable,
    RoadLinksTable,
    RoadNodesTable,
    RoadShapesTable,
    ZonesTable,
)
from network_wrangler.roadway.nodes.edit import NodeGeometryChangeTable
from network_wrangler.utils.models import (
    TableValidationError,
    empty_df_from_datamodel,
    validate_df_to_model,
)


@pytest.mark.parametrize(
    ("attribute", "schema"),
    [
        ("links_df", RoadLinksTable),
        ("nodes_df", RoadNodesTable),
        ("shapes_df", RoadShapesTable),
    ],
)
def test_loaded_roadway_tables_validate_without_losing_metadata(small_net, attribute, schema):
    source = getattr(small_net, attribute)
    source.attrs["characterization"] = {"schema": schema.__name__}

    result = validate_df_to_model(source, schema)

    assert isinstance(result, gpd.GeoDataFrame)
    assert result is not source
    assert result.index.equals(source.index)
    assert result.crs == source.crs
    assert result.geometry.name == source.geometry.name
    assert result.attrs == source.attrs
    assert list(result.columns[: len(source.columns)]) == list(source.columns)


def test_road_links_defaults_coercion_extras_and_column_order(small_net):
    source = small_net.links_df.iloc[[0]].copy()
    source["model_link_id"] = source["model_link_id"].astype(str)
    source["user_defined"] = ["preserve-me"]
    source = source.drop(
        columns=[
            "name",
            "rail_only",
            "bus_only",
            "drive_access",
            "bike_access",
            "walk_access",
            "roadway",
            "projects",
            "managed",
            "price",
        ]
    )
    result = validate_df_to_model(source, RoadLinksTable)

    assert result.loc[result.index[0], "model_link_id"] == int(source.iloc[0]["model_link_id"])
    assert result.loc[result.index[0], "name"] == "unknown"
    assert not bool(result.loc[result.index[0], "rail_only"])
    assert bool(result.loc[result.index[0], "drive_access"])
    assert result.loc[result.index[0], "roadway"] == "road"
    assert result.loc[result.index[0], "projects"] == ""
    assert result.loc[result.index[0], "managed"] == 0
    assert result.loc[result.index[0], "price"] == 0
    assert result.loc[result.index[0], "user_defined"] == "preserve-me"
    assert list(result.columns) == [
        "model_link_id",
        "A",
        "B",
        "geometry",
        "name",
        "rail_only",
        "bus_only",
        "drive_access",
        "bike_access",
        "walk_access",
        "distance",
        "roadway",
        "projects",
        "managed",
        "shape_id",
        "lanes",
        "price",
        "sc_lanes",
        "sc_price",
        "ML_projects",
        "ML_geometry",
        "osm_link_id",
        "county",
        "centroidconnect",
        "assign_group",
        "roadway_class",
        "area_type",
        "bike",
        "walk",
        "user_defined",
    ]


@pytest.mark.parametrize(
    ("mutator", "expected_column"),
    [
        (lambda df: df.assign(model_link_id=df["model_link_id"].iloc[0]), "model_link_id"),
        (lambda df: df.assign(A=df["A"].iloc[0], B=df["B"].iloc[0]), "A"),
        (lambda df: df.assign(distance=None), "distance"),
        (lambda df: df.assign(model_link_id="not-an-integer"), "model_link_id"),
        (lambda df: df.assign(geometry=None), "geometry"),
    ],
)
def test_road_links_reject_unique_composite_and_null_failures(small_net, mutator, expected_column):
    source = small_net.links_df.iloc[:2].copy()
    invalid = mutator(source)

    with pytest.raises(TableValidationError) as exc_info:
        validate_df_to_model(invalid, RoadLinksTable)

    failures = exc_info.value.__cause__.failure_cases
    assert expected_column in failures["column"].astype(str).tolist()


def test_road_links_scoped_fields_use_pydantic_models(small_net):
    source = small_net.links_df.iloc[[0]].copy()
    source["sc_lanes"] = pd.Series(
        [[{"timespan": ["06:00", "09:00"], "category": "hov", "value": 2}]],
        index=source.index,
        dtype=object,
    )

    valid = validate_df_to_model(source, RoadLinksTable)
    assert valid.iloc[0]["sc_lanes"][0]["value"] == 2

    source.at[source.index[0], "sc_lanes"] = [{"timespan": ["not-a-time", "09:00"], "value": 2}]
    with pytest.raises(TableValidationError):
        validate_df_to_model(source, RoadLinksTable)


def test_road_links_nullable_integer_coercion(small_net):
    source = small_net.links_df.iloc[:2].copy()
    source["ML_lanes"] = pd.Series(["2", None], index=source.index, dtype=object)

    result = validate_df_to_model(source, RoadLinksTable)

    assert str(result["ML_lanes"].dtype) == "Int64"
    assert result["ML_lanes"].tolist() == [2, pd.NA]


def test_road_nodes_defaults_and_unique_id(small_net):
    source = small_net.nodes_df.iloc[:2].copy().drop(columns=["osm_node_id", "projects"])
    result = validate_df_to_model(source, RoadNodesTable)

    assert result["osm_node_id"].tolist() == ["", ""]
    assert result["projects"].tolist() == ["", ""]

    duplicate = source.copy()
    duplicate["model_node_id"] = duplicate["model_node_id"].iloc[0]
    with pytest.raises(TableValidationError):
        validate_df_to_model(duplicate, RoadNodesTable)


def test_road_shapes_reject_duplicate_shape_ids():
    duplicate = gpd.GeoDataFrame(
        {
            "shape_id": ["duplicate", "duplicate"],
            "shape_id_idx": [1, 2],
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(0, 0), (2, 2)]),
            ],
            "ref_shape_id": [None, None],
        },
        geometry="geometry",
        crs=4326,
    )

    with pytest.raises(TableValidationError):
        validate_df_to_model(duplicate, RoadShapesTable)


@pytest.mark.parametrize("schema", [RoadLinksTable, RoadShapesTable, ZonesTable])
def test_empty_geometry_tables_preserve_schema_columns_and_crs(schema):
    result = empty_df_from_datamodel(schema, crs=26915)

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.empty
    assert result.crs.to_epsg() == 26915
    assert list(result.columns) == list(schema.__fields__)
    assert str(result["geometry"].dtype) == "geometry"


@pytest.mark.xfail(
    strict=True,
    reason="empty RoadNodesTable validation gives optional list columns float dtypes",
)
def test_empty_road_nodes_table_preserves_list_column_dtypes():
    result = empty_df_from_datamodel(RoadNodesTable)

    assert result.empty
    assert str(result["inboundReferenceIds"].dtype) == "object"
    assert str(result["outboundReferenceIds"].dtype) == "object"


def test_zones_table_geometry_and_coercion():
    zones = gpd.GeoDataFrame(
        {
            "zone_id": ["7"],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
            "geometry_centroid": [Point(0.5, 0.5)],
        },
        geometry="geometry",
        crs=4326,
    )

    result = validate_df_to_model(zones, ZonesTable)

    assert result["zone_id"].tolist() == [7]
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == zones.crs
    assert result["geometry_centroid"].iloc[0].equals(Point(0.5, 0.5))


def test_exploded_scoped_link_table_coercion_and_nullable_default():
    source = pd.DataFrame(
        {
            "model_link_id": ["1"],
            "category": ["hov"],
            "timespan": [["06:00", "09:00"]],
            "start_time": [datetime(2026, 1, 1, 6)],
            "end_time": [datetime(2026, 1, 1, 9)],
            "scoped": [None],
        }
    )

    result = validate_df_to_model(source, ExplodedScopedLinkPropertyTable)

    assert result["model_link_id"].tolist() == [1]
    assert result["scoped"].tolist() == [None]
    assert pd.api.types.is_datetime64_ns_dtype(result["start_time"].dtype)


def test_node_geometry_change_adds_default_crs_and_coerces_coordinates():
    source = pd.DataFrame({"model_node_id": [1], "X": ["-122.4"], "Y": ["37.8"]})

    result = validate_df_to_model(source, NodeGeometryChangeTable)

    assert result["X"].tolist() == [-122.4]
    assert result["Y"].tolist() == [37.8]
    assert result["in_crs"].tolist() == [4326]
