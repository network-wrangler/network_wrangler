"""Classes and functions for selecting transit trips from a transit network.

Usage:

Create a TransitSelection object by providing a TransitNetwork object and a selection dictionary:

    ```python
    selection_dict = {
        "links": {...},
        "nodes": {...},
        "route_properties": {...},
        "trip_properties": {...},
        "timespans": {...},
    }
    transit_selection = TransitSelection(transit_network, selection_dict)
    ```

Access the selected trip ids or dataframe as follows:

    ```python
    selected_trips = transit_selection.selected_trips
    selected_trips_df = transit_selection.selected_trips_df
    ```

Note: The selection dictionary should conform to the SelectTransitTrips model defined in
the models.projects.transit_selection module.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from pandera.typing import DataFrame

from ..errors import (
    ProjectCardError,
    TransitSelectionEmptyError,
    TransitSelectionNetworkConsistencyError,
)
from ..logger import WranglerLogger
from ..models._base.types import TimeString
from ..models.projects import (
    SelectRouteProperties,
    SelectTransitLinks,
    SelectTransitNodes,
    SelectTransitTrips,
    SelectTripProperties,
)
from ..params import SMALL_RECS
from ..utils.time import filter_df_to_overlapping_timespans
from ..utils.utils import dict_to_hexkey

if TYPE_CHECKING:
    from ..models.gtfs.tables import (
        RoutesTable,
        WranglerFrequenciesTable,
        WranglerShapesTable,
        WranglerTripsTable,
    )
    from .feed.feed import Feed
    from .network import TransitNetwork


class TransitSelection:
    """Object to perform and store information about a selection from a project card "facility".

    Attributes:
        selection_dict: dict: Dictionary of selection criteria
        selected_trips: list: List of selected trips
        selected_trips_df: pd.DataFrame: DataFrame of selected trips
        sel_key: str: Hash of selection_dict
        net: TransitNetwork: Network to select from
    """

    def __init__(
        self,
        net: TransitNetwork,
        selection_dict: dict | SelectTransitTrips,
    ):
        """Constructor for TransitSelection object.

        Args:
            net (TransitNetwork): Transit network object to select from.
            selection_dict: Selection dictionary conforming to SelectTransitTrips
        """
        self.net = net
        self.selection_dict = selection_dict

        # Initialize
        self._selected_trips_df = None
        self.sel_key = dict_to_hexkey(selection_dict)
        self._stored_feed_version = self.net.feed.modification_version

        WranglerLogger.debug(f"...created TransitSelection object: {selection_dict}")

    def __nonzero__(self):
        """Return True if there are selected trips."""
        return len(self.selected_trips_df) > 0

    @property
    def selection_dict(self):
        """Getter for selection_dict."""
        return self._selection_dict

    @selection_dict.setter
    def selection_dict(self, value: dict | SelectTransitTrips):
        self._selection_dict = self.validate_selection_dict(value)

    def validate_selection_dict(self, selection_dict: dict | SelectTransitTrips) -> dict:
        """Check that selection dictionary has valid and used properties consistent with network.

        Checks that selection_dict is a valid TransitSelectionDict:
            - query vars exist in respective Feed tables
        Args:
            selection_dict (dict): selection dictionary

        Raises:
            TransitSelectionNetworkConsistencyError: If not consistent with transit network
            ValidationError: if format not consistent with SelectTransitTrips
        """
        if not isinstance(selection_dict, SelectTransitTrips):
            selection_dict = SelectTransitTrips(**selection_dict)
        selection_dict = selection_dict.model_dump(exclude_none=True, by_alias=True)
        WranglerLogger.debug(f"SELECT DICT - before Validation: \n{selection_dict}")
        _trip_selection_fields = list((selection_dict.get("trip_properties", {}) or {}).keys())
        _missing_trip_fields = set(_trip_selection_fields) - set(self.net.feed.trips.columns)

        if _missing_trip_fields:
            msg = f"Fields in trip selection dictionary but not trips.txt: {_missing_trip_fields}"
            raise TransitSelectionNetworkConsistencyError(msg)

        _route_selection_fields = list((selection_dict.get("route_properties", {}) or {}).keys())
        _missing_route_fields = set(_route_selection_fields) - set(self.net.feed.routes.columns)

        if _missing_route_fields:
            msg = (
                f"Fields in route selection dictionary but not routes.txt: {_missing_route_fields}"
            )
            raise TransitSelectionNetworkConsistencyError(msg)
        return selection_dict

    @property
    def selected_trips(self) -> list:
        """List of selected trip_ids."""
        if self.selected_trips_df is None:
            return []
        return self.selected_trips_df.trip_id.tolist()

    @property
    def selected_trips_df(self) -> DataFrame[WranglerTripsTable]:
        """Lazily evaluates selection for trips or returns stored value in self._selected_trips_df.

        Will re-evaluate if the current feed modification version is different than the stored
        one from the last selection.

        Returns:
            DataFrame[WranglerTripsTable] of selected trips
        """
        if (
            self._selected_trips_df is not None
        ) and self._stored_feed_version == self.net.feed.modification_version:
            return self._selected_trips_df

        self._selected_trips_df = self._select_trips()
        self._stored_feed_version = self.net.feed.modification_version
        return self._selected_trips_df

    @property
    def selected_frequencies_df(self) -> DataFrame[WranglerFrequenciesTable]:
        """DataFrame of selected frequencies."""
        sel_freq_df = self.net.feed.frequencies.loc[
            self.net.feed.frequencies.trip_id.isin(self.selected_trips_df.trip_id)
        ]
        # if timespans are selected, filter to those that overlap
        if self.selection_dict.get("timespans"):
            sel_freq_df = filter_df_to_overlapping_timespans(
                sel_freq_df, self.selection_dict.get("timespans")
            )
        return sel_freq_df

    @property
    def selected_shapes_df(self) -> DataFrame[WranglerShapesTable]:
        """DataFrame of selected shapes.

        Can visualize the selected shapes quickly using the following code:

        ```python
        all_routes = net.feed.shapes.plot(color="gray")
        selection.selected_shapes_df.plot(ax=all_routes, color="red")
        ```

        """
        return self.net.feed.shapes.loc[
            self.net.feed.shapes.shape_id.isin(self.selected_trips_df.shape_id)
        ]

    def _select_trips(self) -> DataFrame[WranglerTripsTable]:
        """Selects transit trips based on selection dictionary.

        Returns:
            DataFrame[WranglerTripsTable]: trips_df DataFrame of selected trips
        """
        return _filter_trips_by_selection_dict(
            self.net.feed,
            self.selection_dict,
        )


def _filter_trips_by_selection_dict(
    feed: Feed,
    sel: SelectTransitTrips,
) -> DataFrame[WranglerTripsTable]:
    trips_df = feed.trips
    _routes_df = feed.routes
    _shapes_df = feed.shapes
    _freq_df = feed.frequencies

    _tot_trips = len(trips_df)

    if sel.get("links"):
        trips_df = _filter_trips_by_links(
            trips_df,
            _shapes_df,
            sel["links"],
        )
        WranglerLogger.debug(f"# Trips after links filter: {len(trips_df)}")
    if sel.get("nodes"):
        trips_df = _filter_trips_by_nodes(trips_df, _shapes_df, sel["nodes"])
        WranglerLogger.debug(f"# Trips after node filter: {len(trips_df)}")
    if sel.get("route_properties"):
        trips_df = _filter_trips_by_route(trips_df, _routes_df, sel["route_properties"])
        WranglerLogger.debug(f"# Trips after route property filter: {len(trips_df)}")
    if sel.get("trip_properties"):
        trips_df = _filter_trips_by_trip(trips_df, sel["trip_properties"])
        WranglerLogger.debug(f"# Trips after trip property filter: {len(trips_df)}")
    if sel.get("timespans"):
        trips_df = _filter_trips_by_timespans(trips_df, _freq_df, sel["timespans"])
        WranglerLogger.debug(f"# Trips after timespans filter: {len(trips_df)}")

    _num_sel_trips = len(trips_df)
    WranglerLogger.debug(f"Selected {_num_sel_trips}/{_tot_trips} trips.")

    if not _num_sel_trips:
        msg = f"No transit trips found with selection: \n{sel}"
        WranglerLogger.error(msg)
        raise TransitSelectionEmptyError(msg)
    return trips_df


def _filter_trips_by_links(
    trips_df: DataFrame[WranglerTripsTable],
    shapes_df: DataFrame[WranglerShapesTable],  # noqa: ARG001
    select_links: SelectTransitLinks | None,
) -> DataFrame[WranglerTripsTable]:
    if select_links is None:
        return trips_df
    msg = "Filtering transit by links not implemented yet."
    raise NotImplementedError(msg)


def _filter_trips_by_nodes(
    trips_df: DataFrame[WranglerTripsTable],
    shapes_df: DataFrame[WranglerShapesTable],
    select_nodes: SelectTransitNodes,
) -> DataFrame[WranglerTripsTable]:
    """Selects transit trips that use any one of a list of nodes in shapes.txt.

    If require = all, the returned trip_ids must traverse all of the nodes
    Else, filter any shapes that use any one of the nodes in node_ids

    Args:
        trips_df: List of trips to filter.
        shapes_df: DataFrame of shapes
        select_nodes: Selection dictionary for nodes
    Returns:
        Copy of filtered trips_df
    """
    _tot_trips = len(trips_df)
    # WranglerLogger.debug(f"Filtering {_tot_trips} trips by nodes.")

    require = select_nodes.get("require", "any")
    model_node_ids = select_nodes.get("model_node_id", [])
    if select_nodes.get("stop_id_GTFS"):
        msg = "GTFS Stop ID transit selection not implemented yet."
        raise NotImplementedError(msg)

    if require == "all":
        shape_ids = (
            shapes_df.groupby("shape_id").filter(
                lambda x: all(i in x["shape_model_node_id"].tolist() for i in model_node_ids)
            )
        ).shape_id.drop_duplicates()
    elif require == "any":
        shape_ids = shapes_df.loc[
            shapes_df["shape_model_node_id"].isin(model_node_ids)
        ].shape_id.drop_duplicates()
    else:
        msg = f"Require must be 'any' or 'all', not {require}"
        raise ProjectCardError(msg)

    trips_df = copy.deepcopy(trips_df.loc[trips_df.shape_id.isin(shape_ids)])

    _sel_trips = len(trips_df)
    WranglerLogger.debug(f"Selected {_sel_trips}/{_tot_trips} trips.")
    if _sel_trips < SMALL_RECS:
        WranglerLogger.debug(f"Selected Trips:\n{trips_df.trip_id}")

    return trips_df


def _filter_trips_by_trip(
    trips_df: DataFrame[WranglerTripsTable], select_trip_properties: SelectTripProperties
) -> DataFrame[WranglerTripsTable]:
    """Filter trips by trip properties.

    Args:
        trips_df (pd.DataFrame): DataFrame of trips to filter
        select_trip_properties (SelectTripProperties): Trip properties to filter by

    Returns:
        pd.DataFrame: Filtered trips
    """
    _missing = set(select_trip_properties.keys()) - set(trips_df.columns)
    if _missing:
        msg = f"Trip selection properties missing from trips.txt: {_missing}"
        raise TransitSelectionNetworkConsistencyError(msg)

    # Select
    _num_unfiltered_trips = len(trips_df)
    # WranglerLogger.debug(f"Filtering {_num_unfiltered_trips} trips by trip properties.")

    trips_df = trips_df.dict_query(select_trip_properties)
    _num_filtered_trips = len(trips_df)

    WranglerLogger.debug(
        f"{_num_filtered_trips}/{_num_unfiltered_trips} trips remain after \
        filtering to trip selection {select_trip_properties}"
    )
    if _num_filtered_trips < SMALL_RECS:
        WranglerLogger.debug(f"Filtered Trips:\n{trips_df.trip_id}")

    return trips_df


def _filter_trips_by_route(
    trips_df: DataFrame[WranglerTripsTable],
    routes_df: DataFrame[RoutesTable],
    select_route_properties: SelectRouteProperties,
) -> DataFrame[WranglerTripsTable]:
    """Filter trips by route properties.

    Args:
        trips_df (pd.DataFrame): DataFrame of trips to filter
        routes_df (pd.DataFrame): DataFrame of routes to filter by
        select_route_properties (SelectRouteProperties): Route properties to filter by

    Returns:
        pd.DataFrame: Filtered trips
    """
    _missing = set(select_route_properties.keys()) - set(routes_df.columns)
    if _missing:
        msg = f"Route selection properties missing from routes.txt: {_missing}"
        raise TransitSelectionNetworkConsistencyError(msg)

    # selection
    _unfiltered_trips = len(trips_df)
    # WranglerLogger.debug(f"Filtering {_unfiltered_trips} trips by route properties.")
    _selected_routes_df = routes_df.dict_query(select_route_properties)

    trips_df = trips_df.loc[trips_df.route_id.isin(_selected_routes_df["route_id"])]
    _num_filtered_trips = len(trips_df)

    WranglerLogger.debug(
        f"{_num_filtered_trips}/{_unfiltered_trips} trips remain after \
        filtering to route selection {select_route_properties}"
    )
    if _num_filtered_trips < SMALL_RECS:
        WranglerLogger.debug(f"Filtered Trips:\n{trips_df.trip_id}")

    return trips_df


def _filter_trips_by_timespans(
    trips_df: DataFrame[WranglerTripsTable],
    freq_df: DataFrame[WranglerFrequenciesTable],
    timespans: list[list[TimeString]],
) -> DataFrame[WranglerTripsTable]:
    """Returns trips that have frequencies that overlap with at least one of the timespans."""
    _unfiltered_trips = len(trips_df)
    # WranglerLogger.debug(f"Filtering {_unfiltered_trips} trips by timespans.")
    # Filter freq to trips in selection
    freq_df = freq_df.loc[freq_df.trip_id.isin(trips_df["trip_id"])]

    # Filter freq to time that overlaps selection
    filtered_trip_ids = filter_df_to_overlapping_timespans(freq_df, timespans).trip_id.tolist()

    _filtered_trips = len(filtered_trip_ids)
    if _filtered_trips == 0:
        WranglerLogger.warning(f"No trips found for timespans: {timespans}")
        WranglerLogger.debug(f"Freq to filter:\n{freq_df[['trip_id', 'start_time', 'end_time']]}")

    WranglerLogger.debug(
        f"{_filtered_trips}/{_unfiltered_trips} trips remain after \
        filtering to timespans {timespans}"
    )

    # Filter trips table to those still in freq table
    filtered_trips_df = trips_df.loc[trips_df.trip_id.isin(filtered_trip_ids)]

    if _filtered_trips < SMALL_RECS:
        WranglerLogger.debug(f"{filtered_trips_df.trip_id}")
    return filtered_trips_df
