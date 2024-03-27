import copy
import hashlib

from typing import Optional, List, Dict, Union, Literal, Annotated

import pandas as pd

from pydantic import validate_call, BaseModel, ValidationError

from ..utils import dict_to_hexkey
from ..utils.time import parse_timespans_to_secs
from ..logger import WranglerLogger

from ..projects.models import (
    SelectTransitTrips,
    SelectRouteProperties,
    SelectTripProperties,
    SelectTransitNodes,
    SelectTransitLinks,
)
from ..models._base.time import Timespan


class TransitSelectionError(Exception):
    pass


class TransitSelectionEmptyError(Exception):
    pass


class TransitSelectionNetworkConsistencyError(TransitSelectionError):
    pass


class TransitSelectionFormatError(TransitSelectionError):
    pass


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
        net: "TransitNetwork",
        selection_dict: SelectTransitTrips,
        pyd_dm_selection_dict: BaseModel = SelectTransitTrips,
    ):
        """Constructor for TransitSelection object.

        Args:
            net (TransitNetwork): Transit network object to select from.
            selection_dict: Selection dictionary conforming to SelectTransitTrips
        """
        self.net = net
        self._pyd_dm_selection_dict = pyd_dm_selection_dict

        # Only want to set selection dictionary upon initialization
        selection_dict = self.validate_selection_dict(selection_dict)
        self._selection_dict = selection_dict

        # Initialize
        self._selected_trips_df = None
        self.sel_key = dict_to_hexkey(selection_dict)
        self._stored_feed_hash = copy.deepcopy(self.net.feed.feed_hash)

        WranglerLogger.debug(f"...created TransitSelection object: {selection_dict}")

    def __nonzero__(self):
        """Return True if there are selected trips."""
        if len(self.selected_trips_df) > 0:
            return True
        return False

    @property
    def selection_dict(self):
        """Getter for selection_dict."""
        return self._selection_dict

    def validate_selection_dict(self, selection_dict: SelectTransitTrips) -> None:
        """Check that selection dictionary has valid and used properties consistent with network.

        Checks that selection_dict is a valid TransitSelectionDict:
            - query vars exist in respective Feed tables
        Args:
            selection_dict (dict): selection dictionary

        Raises:
            TransitSelectionFormatError: If not valid
        """
        # WranglerLogger.debug(f"SELECT DICT - before Validation:\n{selection_dict}")
        try:
            selection_dict = self._pyd_dm_selection_dict.model_validate(
                selection_dict
            ).model_dump(by_alias=True, exclude_none=True)
        except ValidationError as e:
            raise TransitSelectionFormatError(e)
        # WranglerLogger.debug(f"SELECT DICT - after Validation:\n{selection_dict}")
        _trip_selection_fields = list(selection_dict.get("trip_properties", {}).keys())
        _missing_trip_fields = set(_trip_selection_fields) - set(
            self.net.feed.trips.columns
        )

        if _missing_trip_fields:
            raise TransitSelectionNetworkConsistencyError(
                f"Fields in trip selection dictionary but not trips.txt: {_missing_trip_fields}"
            )

        _route_selection_fields = list(
            selection_dict.get("route_properties", {}).keys()
        )
        _missing_route_fields = set(_route_selection_fields) - set(
            self.net.feed.routes.columns
        )

        if _missing_route_fields:
            raise TransitSelectionNetworkConsistencyError(
                f"Fields in route selection dictionary but not routes.txt: {_missing_route_fields}"
            )
        return selection_dict

    @property
    def selected_trips(self) -> list:
        """List of selected trip_ids."""
        if self.selected_trips_df is None:
            return []
        return self.selected_trips_df.trip_id.tolist()

    @property
    def selected_trips_df(self) -> pd.DataFrame:
        """Lazily evaluates selection for trips or returns stored value in self._selected_trips_df.

        Will re-evaluate if the current network hash is different than the stored one from the
        last selection.

        Returns:
            pd.DataFrame: DataFrame of selected trips
        """
        if (
            self._selected_trips_df is not None
        ) and self._stored_feed_hash == self.net.feed.feed_hash:
            return self._selected_trips_df

        self._selected_trips_df = self._select_trips()
        self._stored_feed_hash = copy.deepcopy(self.net.feed.feed_hash)
        return self._selected_trips_df

    def _select_trips(self) -> pd.DataFrame:
        """
        Selects transit trips based on selection dictionary.

        Returns:
            pd.DataFrame: trips_df DataFrame of selected trips
        """

        fk_shp2links, _ = self.net.TRANSIT_FOREIGN_KEYS_TO_ROADWAY["shapes"]["links"]

        return _filter_trips_by_selection_dict(
            self.net.feed,
            self.selection_dict,
            fk_shp2links=fk_shp2links,
        )


def _filter_trips_by_selection_dict(
    feed: "Feed",
    sel: SelectTransitTrips,
    fk_shp2links: str = "shape_model_node_id",
):
    trips_df = feed.trips
    _routes_df = feed.routes
    _shapes_df = feed.shapes
    _freq_df = feed.frequencies

    _tot_trips = len(trips_df)

    if "links" in sel:
        trips_df = _filter_trips_by_links(
            trips_df, _shapes_df, sel["links"], fk_shp2links
        )
        WranglerLogger.debug(f"# Trips after links filter:  {len(trips_df)}")
    if "nodes" in sel:
        trips_df = _filter_trips_by_nodes(
            trips_df, _shapes_df, sel["nodes"], fk_shp2links
        )
        WranglerLogger.debug(f"# Trips after node filter:  {len(trips_df)}")
    if "route_properties" in sel:
        trips_df = _filter_trips_by_route(trips_df, _routes_df, sel["route_properties"])
        WranglerLogger.debug(f"# Trips after route property filter:  {len(trips_df)}")
    if "trip_properties" in sel:
        trips_df = _filter_trips_by_trip(trips_df, sel["trip_properties"])
        WranglerLogger.debug(f"# Trips after trip property filter: {len(trips_df)}")
    if "timespan" in sel:
        trips_df = _filter_trips_by_timespan(trips_df, _freq_df, sel["timespan"])
        WranglerLogger.debug(f"# Trips after timespan filter:  {len(trips_df)}")

    _num_sel_trips = len(trips_df)
    WranglerLogger.debug(f"Selected {_num_sel_trips}/{_tot_trips} trips.")

    if not _num_sel_trips:
        WranglerLogger.error(f"No transit trips Found with selection: \n{sel}")
        raise TransitSelectionEmptyError("No transit trips satisfying selection.")
    return trips_df


def _filter_trips_by_links(
    trips_df: pd.DataFrame,
    shapes_df: pd.DataFrame,
    select_links: Union[SelectTransitLinks, None],
    fk_shape_to_roadway_links: str = "shape_model_node_id",
) -> pd.DataFrame:
    if select_links is None:
        return trips_df
    raise NotImplementedError("Filtering transit by links not implemented yet.")


def _filter_trips_by_nodes(
    trips_df: pd.DataFrame,
    shapes_df: pd.DataFrame,
    select_nodes: SelectTransitNodes,
    fk_shape_to_roadway_links: str = "shape_model_node_id",
) -> pd.DataFrame:
    """
    Selects transit trips that use any one of a list of nodes in shapes.txt.

    If require = all, the returned trip_ids must traverse all of the nodes
    Else, filter any shapes that use any one of the nodes in node_ids

    Args:
        trips_df: List of trips to filter.
        shapes_df: DataFrame of shapes
        select_nodes: Selection dictionary for nodes
        fk_shape_to_roadway_links: Foreign key to roadway links in shapes.txt.
            Defaults to "shape_model_node_id".
    Returns:
        Copy of filtered trips_df
    """

    _tot_trips = len(trips_df)
    # WranglerLogger.debug(f"Filtering {_tot_trips} trips by nodes.")

    require = select_nodes.get("require", "any")
    model_node_ids = select_nodes.get("model_node_id", [])
    if select_nodes.get("stop_id"):
        raise NotImplementedError("Stop ID transit selection not implemented yet.")

    if require == "all":
        shape_ids = (
            shapes_df.groupby("shape_id").filter(
                lambda x: all(
                    i in x[fk_shape_to_roadway_links].tolist() for i in model_node_ids
                )
            )
        ).shape_id.drop_duplicates()
    elif require == "any":
        shape_ids = shapes_df.loc[
            shapes_df[fk_shape_to_roadway_links].isin(model_node_ids)
        ].shape_id.drop_duplicates()
    else:
        raise TransitSelectionFormatError("Require must be 'any' or 'all'")

    trips_df = trips_df.loc[trips_df.shape_id.isin(shape_ids)].copy()

    _sel_trips = len(trips_df)
    WranglerLogger.debug(f"Selected {_sel_trips}/{_tot_trips} trips.")
    if _sel_trips < 10:
        WranglerLogger.debug(f"{trips_df.trip_id}")

    return trips_df


def _filter_trips_by_trip(
    trips_df: pd.DataFrame, select_trip_properties: SelectTripProperties
) -> pd.DataFrame:
    """Filter trips by trip properties.

    Args:
        trips_df (pd.DataFrame): DataFrame of trips to filter
        select_trip_properties (SelectTripProperties): Trip properties to filter by

    Returns:
        pd.DataFrame: Filtered trips
    """

    _missing = set(select_trip_properties.keys()) - set(trips_df.columns)
    if _missing:
        raise TransitSelectionFormatError(
            f"Route slection properties missing from trips.txt: {_missing}"
        )

    # Select
    _num_unfiltered_trips = len(trips_df)
    # WranglerLogger.debug(f"Filtering {_num_unfiltered_trips} trips by trip properties.")

    trips_df = trips_df.dict_query(select_trip_properties)
    _num_filtered_trips = len(trips_df)

    WranglerLogger.debug(
        f"{_num_filtered_trips}/{_num_unfiltered_trips} trips remain after \
        filtering to trip selection {select_trip_properties}"
    )
    if _num_filtered_trips < 10:
        WranglerLogger.debug(f"{trips_df.trip_id}")

    return trips_df


def _filter_trips_by_route(
    trips_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    select_route_properties: SelectRouteProperties,
) -> pd.DataFrame:
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
        raise TransitSelectionFormatError(
            f"Route slection properties missing from routes.txt: {_missing}"
        )

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
    if _num_filtered_trips < 10:
        WranglerLogger.debug(f"{trips_df.trip_id}")

    return trips_df


def _filter_trips_by_timespan(
    trips_df: pd.DataFrame, freq_df: pd.DataFrame, timespans: List[Timespan]
) -> pd.DataFrame:
    _unfiltered_trips = len(trips_df)
    # WranglerLogger.debug(f"Filtering {_unfiltered_trips} trips by timespan.")
    # Filter freq to trips in selection
    freq_df = freq_df.loc[freq_df.trip_id.isin(trips_df["trip_id"])]

    # Filter freq to time that overlaps selection
    selected_freq_df = pd.DataFrame()
    for timespan in timespans:
        selected_freq_df = selected_freq_df.append(
            freq_df[
                (freq_df["end_time"] >= timespan.start_time)
                & (freq_df["start_time"] <= timespan.end_time)
            ]
        )

    # Filter trips table to those still in freq table
    trips_df = trips_df.loc[trips_df.trip_id.isin(selected_freq_df["trip_id"])]
    _filtered_trips = len(trips_df)

    WranglerLogger.debug(
        f"{_filtered_trips}/{_unfiltered_trips} trips remain after \
        filtering to timeframe {timespan}"
    )

    if _filtered_trips < 10:
        WranglerLogger.debug(f"{trips_df.trip_id}")

    return trips_df