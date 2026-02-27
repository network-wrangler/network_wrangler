"""Main functionality for GTFS tables including Feed object."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
from pandera.typing import DataFrame

from ...logger import WranglerLogger
from ...models._base.db import DBModelMixin
from ...models.gtfs.tables import (
    AgenciesTable,
    RoutesTable,
    WranglerFrequenciesTable,
    WranglerShapesTable,
    WranglerStopsTable,
    WranglerStopTimesTable,
    WranglerTripsTable,
)
from ...utils.data import update_df_by_col_value


class Feed(DBModelMixin):
    """Wrapper class around Wrangler flavored GTFS feed.

    Most functionality derives from mixin class DBModelMixin which provides:

    - validation of tables to schemas when setting a table attribute (e.g. self.trips = trips_df)
    - validation of fks when setting a table attribute (e.g. self.trips = trips_df)
    - hashing and deep copy functionality
    - overload of __eq__ to apply only to tables in table_names.
    - convenience methods for accessing tables

    Attributes:
        table_names (list[str]): list of table names in GTFS feed.
        tables (list[DataFrame]):: list tables as dataframes.
        stop_times (DataFrame[WranglerStopTimesTable]):: stop_times dataframe with roadway node_ids
        stops (DataFrame[WranglerStopsTable]):stops dataframe
        shapes(DataFrame[WranglerShapesTable]): shapes dataframe
        trips (DataFrame[WranglerTripsTable]): trips dataframe
        frequencies (DataFrame[WranglerFrequenciesTable]): frequencies dataframe
        routes (DataFrame[RoutesTable]): route dataframe
        agencies (Optional[DataFrame[AgenciesTable]]): agencies dataframe
        net (Optional[TransitNetwork]): TransitNetwork object
    """

    # the ordering here matters because the stops need to be added before stop_times if
    # stop times needs to be converted
    _table_models: ClassVar[dict] = {
        "agencies": AgenciesTable,
        "frequencies": WranglerFrequenciesTable,
        "routes": RoutesTable,
        "shapes": WranglerShapesTable,
        "stops": WranglerStopsTable,
        "trips": WranglerTripsTable,
        "stop_times": WranglerStopTimesTable,
    }

    # Define the converters if the table needs to be converted to a Wrangler table.
    # Format: "table_name": converter_function
    _converters: ClassVar[dict[str, Callable]] = {}

    table_names: ClassVar[list[str]] = [
        "frequencies",
        "routes",
        "shapes",
        "stops",
        "trips",
        "stop_times",
    ]

    optional_table_names: ClassVar[list[str]] = ["agencies"]

    def __init__(self, **kwargs):
        """Create a Feed object from a dictionary of DataFrames representing a GTFS feed.

        Args:
            kwargs: A dictionary containing DataFrames representing the tables of a GTFS feed.
        """
        self._net = None
        self.feed_path: Path = None
        self.initialize_tables(**kwargs)

        # Set extra provided attributes but just FYI in logger.
        extra_attr = {k: v for k, v in kwargs.items() if k not in self.table_names}
        if extra_attr:
            WranglerLogger.info(f"Adding additional attributes to Feed: {extra_attr.keys()}")
        for k, v in extra_attr.items():
            self.__setattr__(k, v)

    def set_by_id(
        self,
        table_name: str,
        set_df: pd.DataFrame,
        id_property: str = "index",
        properties: list[str] | None = None,
    ):
        """Set one or more property values based on an ID property for a given table.

        Args:
            table_name (str): Name of the table to modify.
            set_df (pd.DataFrame): DataFrame with columns `<id_property>` and `value` containing
                values to set for the specified property where `<id_property>` is unique.
            id_property: Property to use as ID to set by. Defaults to "index".
            properties: List of properties to set which are in set_df. If not specified, will set
                all properties.
        """
        if not set_df[id_property].is_unique:
            msg = f"{id_property} must be unique in set_df."
            _dupes = set_df[id_property][set_df[id_property].duplicated()]
            WranglerLogger.error(msg + f"Found duplicates: {_dupes.sum()}")

            raise ValueError(msg)
        table_df = self.get_table(table_name)
        updated_df = update_df_by_col_value(table_df, set_df, id_property, properties=properties)
        self.__dict__[table_name] = updated_df


PickupDropoffAvailability = Literal["either", "both", "pickup_only", "dropoff_only", "any"]


def stop_count_by_trip(
    stop_times: DataFrame[WranglerStopTimesTable],
) -> pd.DataFrame:
    """Returns dataframe with trip_id and stop_count from stop_times."""
    stops_count = stop_times.groupby("trip_id").size()
    return stops_count.reset_index(name="stop_count")


def merge_shapes_to_stop_times(
    stop_times: DataFrame[WranglerStopTimesTable],
    shapes: DataFrame[WranglerShapesTable],
    trips: DataFrame[WranglerTripsTable],
) -> DataFrame[WranglerStopTimesTable]:
    """Add shape_id and shape_pt_sequence to stop_times dataframe.

    Args:
        stop_times: stop_times dataframe to add shape_id and shape_pt_sequence to.
        shapes: shapes dataframe to add to stop_times.
        trips: trips dataframe to link stop_times to shapes

    Returns:
        stop_times dataframe with shape_id and shape_pt_sequence added.
    """
    stop_times_w_shape_id = stop_times.merge(
        trips[["trip_id", "shape_id"]], on="trip_id", how="left"
    )

    stop_times_w_shapes = stop_times_w_shape_id.merge(
        shapes,
        how="left",
        left_on=["shape_id", "stop_id"],
        right_on=["shape_id", "shape_model_node_id"],
    )
    stop_times_w_shapes = stop_times_w_shapes.drop(columns=["shape_model_node_id"])
    return stop_times_w_shapes


def _get_applied_projects_from_tables(feed: Feed) -> list[str]:
    """Return a list of applied projects from the feed tables.

    Note: This may or may not return a full accurate account of all the applied projects.
    For better project accounting, please leverage the scenario object.
    """
    applied_projects = set()
    for table_name in feed.table_names:
        table = feed.get_table(table_name)
        if "projects" in table.columns:
            exploded_projects = table.projects.str.split(",").explode().dropna()
            exploded_projects = exploded_projects[exploded_projects != ""]
            applied_projects.update(exploded_projects)
    return list(applied_projects)
