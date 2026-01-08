"""Data Model for Pure GTFS Feed (not wrangler-flavored)."""

from typing import ClassVar

import geopandas as gpd

from ...models._base.db import DBModelMixin
from .tables import (
    AgenciesTable,
    FrequenciesTable,
    RoutesTable,
    ShapesTable,
    StopsTable,
    StopTimesTable,
    TripsTable,
)
from .types import RouteType

# Constants for display
MAX_AGENCIES_DISPLAY = 3

# Route type categorizations
MIXED_TRAFFIC_ROUTE_TYPES = [
    RouteType.TRAM,
    RouteType.BUS,
    RouteType.CABLE_TRAM,
    RouteType.TROLLEYBUS,
]
"""GTFS route types that operate in mixed traffic so stops are nodes that are drive-accessible.

See [GTFS routes.txt](https://gtfs.org/documentation/schedule/reference/#routestxt)

- TRAM = Tram, Streetcar, Light rail, operates in mixed traffic AND at stations
- CABLE_TRAM = street-level rail with underground cable
- TROLLEYBUS = electric buses with overhead wires
"""

STATION_ROUTE_TYPES = [
    RouteType.TRAM,  # TODO: This is partial...
    RouteType.SUBWAY,
    RouteType.RAIL,
    RouteType.FERRY,
    RouteType.CABLE_TRAM,  # TODO: This is partial...
    RouteType.AERIAL_LIFT,
    RouteType.FUNICULAR,
    RouteType.MONORAIL,
]
"""GTFS route types that operate at stations.
"""

RAIL_ROUTE_TYPES = [
    RouteType.TRAM,
    RouteType.SUBWAY,
    RouteType.RAIL,
    RouteType.CABLE_TRAM,
    RouteType.AERIAL_LIFT,
    RouteType.FUNICULAR,
    RouteType.MONORAIL,
]
"""GTFS route types which trigger 'rail_only' link creation in add_stations_and_links_to_roadway_network()
"""

FERRY_ROUTE_TYPES = [RouteType.FERRY]
"""GTFS route types which trigger 'ferry_only' link creation in add_stations_and_links_to_roadway_network()
"""


class GtfsValidationError(Exception):
    """Exception raised for errors in the GTFS feed."""


class GtfsModel(DBModelMixin):
    """Wrapper class around GTFS feed.

    This is the pure GTFS model version of [Feed][network_wrangler.transit.feed.feed.Feed]

    Most functionality derives from mixin class
    [`DBModelMixin`][network_wrangler.models._base.db.DBModelMixin] which provides:

    - validation of tables to schemas when setting a table attribute (e.g. self.trips = trips_df)
    - validation of fks when setting a table attribute (e.g. self.trips = trips_df)
    - hashing and deep copy functionality
    - overload of __eq__ to apply only to tables in table_names.
    - convenience methods for accessing tables

    Attributes:
        table_names (list[str]): list of table names in GTFS feed.
        tables (list[DataFrame]): list tables as dataframes.
        agency (DataFrame[AgenciesTable]): agency dataframe
        stop_times (DataFrame[StopTimesTable]): stop_times dataframe
        stops (DataFrame[WranglerStopsTable]): stops dataframe
        shapes (DataFrame[ShapesTable]): shapes dataframe
        trips (DataFrame[TripsTable]): trips dataframe
        frequencies (Optional[DataFrame[FrequenciesTable]]): frequencies dataframe
        routes (DataFrame[RoutesTable]): route dataframe
        net (Optional[TransitNetwork]): TransitNetwork object
    """

    # the ordering here matters because the stops need to be added before stop_times if
    # stop times needs to be converted
    _table_models: ClassVar[dict] = {
        "agency": AgenciesTable,
        "frequencies": FrequenciesTable,
        "routes": RoutesTable,
        "shapes": ShapesTable,
        "stops": StopsTable,
        "trips": TripsTable,
        "stop_times": StopTimesTable,
    }

    table_names: ClassVar[list[str]] = [
        "agency",
        "routes",
        "shapes",
        "stops",
        "trips",
        "stop_times",
    ]

    optional_table_names: ClassVar[list[str]] = ["frequencies"]

    def __init__(self, **kwargs):
        """Initialize GTFS model."""
        self.initialize_tables(**kwargs)

        # Set extra provided attributes.
        extra_attr = {k: v for k, v in kwargs.items() if k not in self.table_names}
        for k, v in extra_attr.items():
            self.__setattr__(k, v)

    @property
    def summary(self) -> dict:
        """A high level summary of the GTFS model object and public attributes"""

        summary_dict = {}
        for table_name in self._table_models:
            if hasattr(self, table_name):
                table = getattr(self, table_name)
                table_type = type(table)
                summary_dict[table_name] = f"{len(getattr(self, table_name)):,} {table_name} (type={table_type})"
            else:
                summary_dict[table_name] = "not set"

        return summary_dict

    def __repr__(self) -> str:
        """Return a string representation of the GtfsModel with table summaries."""
        lines = ["GtfsModel:"]

        for k,v in self.summary.items():
            lines.append(f"  {k}: {v}")

        return "\n".join(lines)
