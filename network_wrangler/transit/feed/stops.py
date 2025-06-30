"""Filters and queries of a gtfs stops table and stop_ids."""

from __future__ import annotations

from typing import Union

import pyarrow as pa
from pandera.typing import DataFrame

from ...logger import WranglerLogger
from ...models.gtfs.tables import WranglerStopsTable, WranglerStopTimesTable
from ...utils.models import validate_call_pyd
from .feed import PickupDropoffAvailability


@validate_call_pyd
def stop_id_pattern_for_trip(
    stop_times: DataFrame[WranglerStopTimesTable],
    trip_id: str,
    pickup_dropoff: PickupDropoffAvailability = "either",
) -> list[str]:
    """Returns a stop pattern for a given trip_id given by a list of stop_ids.

    Args:
        stop_times: WranglerStopTimesTable
        trip_id: trip_id to get stop pattern for
        pickup_dropoff: str indicating logic for selecting stops based on piackup and dropoff
            availability at stop. Defaults to "either".
            "either": either pickup_type or dropoff_type > 0
            "both": both pickup_type or dropoff_type > 0
            "pickup_only": only pickup > 0
            "dropoff_only": only dropoff > 0
    """
    from .stop_times import stop_times_for_pickup_dropoff_trip_id  # noqa: PLC0415

    trip_stops = stop_times_for_pickup_dropoff_trip_id(
        stop_times, trip_id, pickup_dropoff=pickup_dropoff
    )
    return trip_stops.stop_id.to_list()


def stops_for_stop_times(
    stops: DataFrame[WranglerStopsTable], stop_times: DataFrame[WranglerStopTimesTable]
) -> DataFrame[WranglerStopsTable]:
    """Filter stops dataframe to only have stops associated with stop_times records."""
    _sel_stops_ge_min = stop_times.stop_id.unique().tolist()
    filtered_stops = stops[stops.stop_id.isin(_sel_stops_ge_min)]
    WranglerLogger.debug(
        f"Filtered stops to {len(filtered_stops)}/{len(stops)} \
                         records that referenced one of {len(stop_times)} stop_times."
    )
    return filtered_stops


def stops_for_trip_id(
    stops: DataFrame[WranglerStopsTable],
    stop_times: DataFrame[WranglerStopTimesTable],
    trip_id: str,
    pickup_dropoff: PickupDropoffAvailability = "any",
) -> DataFrame[WranglerStopsTable]:
    """Returns stops.txt which are used for a given trip_id."""
    stop_ids = stop_id_pattern_for_trip(stop_times, trip_id, pickup_dropoff=pickup_dropoff)
    return stops.loc[stops.stop_id.isin(stop_ids)]


def node_is_stop(
    stops: DataFrame[WranglerStopsTable],
    stop_times: DataFrame[WranglerStopTimesTable],
    node_id: Union[int, list[int]],
    trip_id: str,
    pickup_dropoff: PickupDropoffAvailability = "either",
) -> Union[bool, list[bool]]:
    """Returns boolean indicating if a (or list of) node(s)) is (are) stops for a given trip_id.

    Args:
        stops: WranglerStopsTable
        stop_times: WranglerStopTimesTable
        node_id: node ID for roadway
        trip_id: trip_id to get stop pattern for
        pickup_dropoff: str indicating logic for selecting stops based on piackup and dropoff
            availability at stop. Defaults to "either".
            "either": either pickup_type or dropoff_type > 0
            "both": both pickup_type or dropoff_type > 0
            "pickup_only": only pickup > 0
            "dropoff_only": only dropoff > 0
    """
    trip_stop_nodes = stops_for_trip_id(stops, stop_times, trip_id, pickup_dropoff=pickup_dropoff)[
        "stop_id"
    ]
    if isinstance(node_id, list):
        return [n in trip_stop_nodes.values for n in node_id]
    return node_id in trip_stop_nodes.values
