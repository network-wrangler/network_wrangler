"""Characterize Pandera behavior for GTFS table schemas."""

from datetime import date, timedelta

import pandas as pd
import pytest

from network_wrangler.models.gtfs.tables import (
    AgenciesTable,
    FrequenciesTable,
    RoutesTable,
    ShapesTable,
    StopsTable,
    StopTimesTable,
    TripsTable,
    WranglerFrequenciesTable,
    WranglerShapesTable,
    WranglerStopsTable,
    WranglerStopTimesTable,
    WranglerTripsTable,
)
from network_wrangler.utils.models import TableValidationError, validate_df_to_model


def _base_gtfs_cases():
    """Return compact, hand-readable valid inputs for every base GTFS schema."""
    return {
        AgenciesTable: pd.DataFrame(
            {
                "agency_id": ["agency"],
                "agency_name": ["Example Transit"],
                "agency_url": ["https://example.com"],
                "agency_timezone": ["America/New_York"],
                "agency_lang": ["en"],
                "agency_phone": ["555-0100"],
                "agency_fare_url": ["https://example.com/fares"],
                "agency_email": ["info@example.com"],
            }
        ),
        StopsTable: pd.DataFrame(
            {"stop_id": ["stop"], "stop_lat": ["37.0"], "stop_lon": ["-122.0"]}
        ),
        RoutesTable: pd.DataFrame(
            {
                "route_id": ["route"],
                "route_short_name": ["1"],
                "route_long_name": ["Example Route"],
                "route_type": [3],
            }
        ),
        ShapesTable: pd.DataFrame(
            {
                "shape_id": ["shape"],
                "shape_pt_lat": ["37.0"],
                "shape_pt_lon": ["-122.0"],
                "shape_pt_sequence": ["0"],
            }
        ),
        TripsTable: pd.DataFrame({"trip_id": ["trip"], "route_id": ["route"]}),
        FrequenciesTable: pd.DataFrame({"trip_id": ["trip"], "headway_secs": ["600"]}),
        StopTimesTable: pd.DataFrame(
            {"trip_id": ["trip"], "stop_id": ["stop"], "stop_sequence": ["0"]}
        ),
    }


@pytest.mark.parametrize("schema", list(_base_gtfs_cases()))
def test_base_gtfs_schema_valid_minimal_inputs(schema):
    source = _base_gtfs_cases()[schema]
    source["extension_field"] = [f"extra-{schema.__name__}"]

    result = validate_df_to_model(source, schema)

    assert result is not source
    assert result["extension_field"].tolist() == [f"extra-{schema.__name__}"]
    assert result.index.equals(source.index)


@pytest.mark.parametrize(
    ("table_name", "schema"),
    [
        ("frequencies", WranglerFrequenciesTable),
        ("routes", RoutesTable),
        ("shapes", WranglerShapesTable),
        ("stops", WranglerStopsTable),
        ("trips", WranglerTripsTable),
        ("stop_times", WranglerStopTimesTable),
    ],
)
def test_loaded_wrangler_gtfs_tables_validate(small_transit_net, table_name, schema):
    source = small_transit_net.feed.get_table(table_name)
    source.attrs["characterization"] = table_name

    result = validate_df_to_model(source, schema)

    assert result is not source
    assert result.index.equals(source.index)
    assert result.attrs == source.attrs
    assert list(result.columns) == list(source.columns)


def test_gtfs_missing_columns_add_only_declared_defaults_in_schema_order():
    stops = validate_df_to_model(
        pd.DataFrame({"stop_id": ["s"], "stop_lat": [1], "stop_lon": [2]}),
        StopsTable,
    )
    trips = validate_df_to_model(pd.DataFrame({"trip_id": ["t"], "route_id": ["r"]}), TripsTable)
    frequencies = validate_df_to_model(
        pd.DataFrame({"trip_id": ["t"], "headway_secs": [600]}),
        FrequenciesTable,
    )

    # Pandera does not insert missing ``Series[...] | None`` columns, even when
    # the Field declares a default.
    assert list(stops.columns) == ["stop_id", "stop_lat", "stop_lon"]
    assert trips["direction_id"].tolist() == [0]
    assert trips["service_id"].tolist() == ["1"]
    assert "wheelchair_accessible" not in trips
    assert "bikes_allowed" not in trips
    assert frequencies["start_time"].tolist() == ["00:00"]
    assert frequencies["end_time"].tolist() == ["24:00"]


@pytest.mark.parametrize(
    ("schema", "source", "column"),
    [
        (
            AgenciesTable,
            pd.DataFrame(
                {
                    "agency_id": ["agency"],
                    "agency_name": ["Example"],
                    "agency_url": ["not-a-url"],
                    "agency_timezone": ["America/New_York"],
                    "agency_lang": ["en"],
                    "agency_phone": ["555"],
                    "agency_fare_url": ["https://example.com/fares"],
                    "agency_email": ["info@example.com"],
                }
            ),
            "agency_url",
        ),
        (
            StopsTable,
            pd.DataFrame({"stop_id": ["s"], "stop_lat": [91], "stop_lon": [0]}),
            "stop_lat",
        ),
        (
            RoutesTable,
            pd.DataFrame(
                {
                    "route_id": ["r"],
                    "route_short_name": ["1"],
                    "route_long_name": ["Route"],
                    "route_type": [999],
                }
            ),
            "route_type",
        ),
        (
            ShapesTable,
            pd.DataFrame(
                {
                    "shape_id": ["sh"],
                    "shape_pt_lat": [0],
                    "shape_pt_lon": [0],
                    "shape_pt_sequence": [-1],
                }
            ),
            "shape_pt_sequence",
        ),
        (
            TripsTable,
            pd.DataFrame({"trip_id": ["t"], "route_id": ["r"], "direction_id": [3]}),
            "direction_id",
        ),
        (FrequenciesTable, pd.DataFrame({"trip_id": ["t"], "headway_secs": [0]}), "headway_secs"),
        (
            StopTimesTable,
            pd.DataFrame({"trip_id": ["t"], "stop_id": ["s"], "stop_sequence": [-1]}),
            "stop_sequence",
        ),
    ],
)
def test_gtfs_constraints_reject_invalid_values(schema, source, column):
    with pytest.raises(TableValidationError) as exc_info:
        validate_df_to_model(source, schema)

    cause = exc_info.value.__cause__
    if hasattr(cause, "failure_cases"):
        assert column in cause.failure_cases["column"].astype(str).tolist()


@pytest.mark.parametrize(
    ("schema", "source"),
    [
        (
            StopsTable,
            pd.DataFrame(
                {"stop_id": ["duplicate", "duplicate"], "stop_lat": [0, 1], "stop_lon": [0, 1]}
            ),
        ),
        (
            ShapesTable,
            pd.DataFrame(
                {
                    "shape_id": ["shape", "shape"],
                    "shape_pt_lat": [0, 1],
                    "shape_pt_lon": [0, 1],
                    "shape_pt_sequence": [0, 0],
                }
            ),
        ),
        (
            FrequenciesTable,
            pd.DataFrame(
                {
                    "trip_id": ["trip", "trip"],
                    "start_time": ["06:00", "06:00"],
                    "end_time": ["07:00", "08:00"],
                    "headway_secs": [600, 900],
                }
            ),
        ),
        (
            StopTimesTable,
            pd.DataFrame(
                {
                    "trip_id": ["trip", "trip"],
                    "stop_id": ["a", "b"],
                    "stop_sequence": [0, 0],
                }
            ),
        ),
    ],
)
def test_gtfs_single_and_composite_uniqueness(schema, source):
    with pytest.raises(TableValidationError):
        validate_df_to_model(source, schema)


def test_wrangler_stop_times_parse_strings_and_nulls_to_timestamps():
    source = pd.DataFrame(
        {
            "trip_id": ["trip", "trip"],
            "stop_id": ["1", "2"],
            "stop_sequence": ["0", "1"],
            "arrival_time": ["23:59", None],
            "departure_time": ["23:59:30", None],
        }
    )

    result = validate_df_to_model(source, WranglerStopTimesTable)

    assert pd.api.types.is_datetime64_any_dtype(result["arrival_time"])
    assert pd.api.types.is_datetime64_any_dtype(result["departure_time"])
    assert result.loc[0, "arrival_time"].hour == 23
    assert result.loc[0, "departure_time"].second == 30
    assert pd.isna(result.loc[1, "arrival_time"])
    assert pd.isna(result.loc[1, "departure_time"])
    assert result["stop_id"].tolist() == [1, 2]
    assert result["projects"].tolist() == ["", ""]


def test_wrangler_stop_times_parse_midnight_boundaries():
    source = pd.DataFrame(
        {
            "trip_id": ["trip", "trip"],
            "stop_id": [1, 2],
            "stop_sequence": [0, 1],
            "arrival_time": ["00:00:00", "23:59:59"],
            "departure_time": ["00:00:01", "23:59:00"],
        }
    )

    result = validate_df_to_model(source, WranglerStopTimesTable)

    assert result["arrival_time"].dt.strftime("%H:%M:%S").tolist() == [
        "00:00:00",
        "23:59:59",
    ]
    assert result["departure_time"].dt.strftime("%H:%M:%S").tolist() == [
        "00:00:01",
        "23:59:00",
    ]


def test_wrangler_frequencies_parse_times_and_project_default():
    source = pd.DataFrame(
        {
            "trip_id": ["trip"],
            "start_time": ["00:00"],
            "end_time": ["24:00"],
            "headway_secs": [600],
        }
    )

    result = validate_df_to_model(source, WranglerFrequenciesTable)

    assert pd.api.types.is_datetime64_any_dtype(result["start_time"])
    assert pd.api.types.is_datetime64_any_dtype(result["end_time"])
    assert result.loc[0, "start_time"].strftime("%H:%M") == "00:00"
    assert result.loc[0, "end_time"].strftime("%H:%M") == "00:00"
    assert result["projects"].tolist() == [""]


@pytest.mark.xfail(
    strict=True,
    reason="WranglerFrequenciesTable has untyped time columns whose missing defaults cannot be coerced",
)
def test_wrangler_frequencies_insert_missing_time_defaults():
    source = pd.DataFrame({"trip_id": ["trip"], "headway_secs": [600]})

    result = validate_df_to_model(source, WranglerFrequenciesTable)

    assert result.loc[0, "start_time"].strftime("%H:%M") == "00:00"
    assert result.loc[0, "end_time"].strftime("%H:%M") == "00:00"


@pytest.mark.xfail(
    strict=True,
    reason="str_to_time_series cannot parse mixed HH:MM and HH:MM:SS values in one Series",
)
def test_wrangler_stop_times_accept_mixed_time_precision():
    source = pd.DataFrame(
        {
            "trip_id": ["trip", "trip"],
            "stop_id": [1, 2],
            "stop_sequence": [0, 1],
            "arrival_time": ["00:00", "23:59:59"],
            "departure_time": ["00:00:01", "23:59"],
        }
    )

    result = validate_df_to_model(source, WranglerStopTimesTable)

    assert pd.api.types.is_datetime64_any_dtype(result["arrival_time"])


@pytest.mark.xfail(
    strict=True,
    reason="FrequenciesTable annotates times as plain strings and accepts malformed values",
)
def test_frequencies_reject_malformed_time_strings():
    source = pd.DataFrame(
        {
            "trip_id": ["trip"],
            "start_time": ["not-a-time"],
            "end_time": ["also-not-a-time"],
            "headway_secs": [600],
        }
    )

    with pytest.raises(TableValidationError):
        validate_df_to_model(source, FrequenciesTable)


@pytest.mark.xfail(
    strict=True,
    reason="str_to_time_series wraps 24-47 hour GTFS values without advancing the date",
)
def test_wrangler_stop_times_preserve_extended_gtfs_day_offset():
    source = pd.DataFrame(
        {
            "trip_id": ["trip"],
            "stop_id": [1],
            "stop_sequence": [0],
            "arrival_time": ["25:10"],
            "departure_time": ["47:59"],
        }
    )

    result = validate_df_to_model(source, WranglerStopTimesTable)

    assert result.loc[0, "arrival_time"].date() == date.today() + timedelta(days=1)
    assert result.loc[0, "departure_time"].date() == date.today() + timedelta(days=1)


def test_gtfs_schema_key_metadata():
    assert StopsTable.Config._pk == ["stop_id"]
    assert StopsTable.Config._fk == {"parent_station": ("stops", "stop_id")}
    assert ShapesTable.Config._pk == ["shape_id", "shape_pt_sequence"]
    assert ShapesTable.Config.unique == ["shape_id", "shape_pt_sequence"]
    assert TripsTable.Config._fk == {"route_id": ("routes", "route_id")}
    assert StopTimesTable.Config._fk == {
        "trip_id": ("trips", "trip_id"),
        "stop_id": ("stops", "stop_id"),
    }
