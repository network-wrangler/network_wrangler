"""TransitNetwork class for representing a transit network.

Transit Networks are represented as a Wrangler-flavored GTFS Feed and optionally mapped to
a RoadwayNetwork object. The TransitNetwork object is the primary object for managing transit
networks in Wrangler.

Usage:

    ```python
    import network_wrangler as wr

    t = wr.load_transit(stpaul_gtfs)
    t.road_net = wr.load_roadway(stpaul_roadway)
    t = t.apply(project_card)
    write_transit(t, "output_dir")
    ```
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, ClassVar, Optional, Union

import geopandas as gpd
import networkx as nx
from projectcard import ProjectCard, SubProject

from ..configs import DefaultConfig, WranglerConfig
from ..errors import (
    ProjectCardError,
    TransitRoadwayConsistencyError,
    TransitSelectionEmptyError,
    TransitValidationError,
)
from ..logger import WranglerLogger
from ..utils.geo import to_points_gdf
from ..utils.utils import dict_to_hexkey
from .feed.feed import Feed, _get_applied_projects_from_tables
from .geo import (
    shapes_to_shape_links_gdf,
    shapes_to_trip_shapes_gdf,
    stop_times_to_stop_time_links_gdf,
    stop_times_to_stop_time_points_gdf,
)
from .projects import (
    apply_calculated_transit,
    apply_transit_property_change,
    apply_transit_route_addition,
    apply_transit_routing_change,
    apply_transit_service_deletion,
)
from .selection import TransitSelection
from .validate import transit_road_net_consistency

if TYPE_CHECKING:
    from ..roadway.network import RoadwayNetwork


class TransitNetwork:
    """Representation of a Transit Network.

    Typical usage example:
    ``` py
    import network_wrangler as wr

    tc = wr.load_transit(stpaul_gtfs)
    ```

    Attributes:
        feed: gtfs feed object with interlinked tables.
        road_net (RoadwayNetwork): Associated roadway network object.
        graph (nx.MultiDiGraph): Graph for associated roadway network object.
        config (WranglerConfig): Configuration object for the transit network.
        feed_path (str): Where the feed was read in from.
        validated_frequencies (bool): The frequencies have been validated.
        validated_road_network_consistency (): The network has been validated against
            the road network.
    """

    TIME_COLS: ClassVar = ["arrival_time", "departure_time", "start_time", "end_time"]

    def __init__(self, feed: Feed, config: WranglerConfig = DefaultConfig) -> None:
        """Constructor for TransitNetwork.

        Args:
            feed: Feed object representing the transit network gtfs tables
            config: WranglerConfig object. Defaults to DefaultConfig.
        """
        WranglerLogger.debug("Creating new TransitNetwork.")

        self._road_net: Optional[RoadwayNetwork] = None
        self.feed: Feed = feed
        self.graph: nx.MultiDiGraph = None
        self.config: WranglerConfig = config
        # initialize
        self._consistent_with_road_net = False

        # cached selections
        self._selections: dict[str, TransitSelection] = {}

    @property
    def feed_path(self):
        """Pass through property from Feed."""
        return self.feed.feed_path

    @property
    def applied_projects(self) -> list[str]:
        """List of projects applied to the network.

        Note: This may or may not return a full accurate account of all the applied projects.
        For better project accounting, please leverage the scenario object.
        """
        return _get_applied_projects_from_tables(self.feed)

    @property
    def feed(self):
        """Feed associated with the transit network."""
        return self._feed

    @feed.setter
    def feed(self, feed: Feed):
        if not isinstance(feed, Feed):
            msg = f"TransitNetwork's feed value must be a valid Feed instance. \
                             This is a {type(feed)}."
            WranglerLogger.error(msg)
            raise TransitValidationError(msg)
        if self._road_net is None or transit_road_net_consistency(feed, self._road_net):
            self._feed = feed
            self._stored_feed_version = feed.modification_version
        else:
            msg = "Can't assign Feed inconsistent with set Roadway Network."
            WranglerLogger.error(msg)
            raise TransitRoadwayConsistencyError(msg)

    @property
    def road_net(self) -> Union[None, RoadwayNetwork]:
        """Roadway network associated with the transit network."""
        return self._road_net

    @road_net.setter
    def road_net(self, road_net_in: RoadwayNetwork):
        if road_net_in is None or road_net_in.__class__.__name__ != "RoadwayNetwork":
            msg = f"TransitNetwork's road_net: value must be a valid RoadwayNetwork instance. \
                             This is a {type(road_net_in)}."
            WranglerLogger.error(msg)
            raise TransitValidationError(msg)
        if transit_road_net_consistency(self.feed, road_net_in):
            self._road_net = road_net_in
            self._stored_road_net_version = road_net_in.modification_version
            self._consistent_with_road_net = True
        else:
            msg = "Can't assign inconsistent RoadwayNetwork - Roadway Network not \
                   set, but can be referenced separately."
            WranglerLogger.error(msg)
            raise TransitRoadwayConsistencyError(msg)

    @property
    def feed_hash(self):
        """Return the hash of the feed."""
        return self.feed.hash

    @property
    def consistent_with_road_net(self) -> bool:
        """Indicate if road_net is consistent with transit network.

        Will return True if road_net is None, but provide a warning.

        Checks the modification version of when consistency was last evaluated. If transit network
        or roadway network has changed, will re-evaluate consistency and return the updated value.

        Returns:
            Boolean indicating if road_net is consistent with transit network.
        """
        if self.road_net is None:
            WranglerLogger.warning("Roadway Network not set, cannot accurately check consistency.")
            return True
        updated_road = self.road_net.modification_version != self._stored_road_net_version
        updated_feed = self.feed.modification_version != self._stored_feed_version

        if updated_road or updated_feed:
            self._consistent_with_road_net = transit_road_net_consistency(self.feed, self.road_net)
            self._stored_road_net_version = self.road_net.modification_version
            self._stored_feed_version = self.feed.modification_version
        return self._consistent_with_road_net

    def __deepcopy__(self, memo):
        """Returns copied TransitNetwork instance with deep copy of Feed but not roadway net."""
        COPY_REF_NOT_VALUE = ["_road_net"]
        # Create a new, empty instance
        copied_net = self.__class__.__new__(self.__class__)
        # Return the new TransitNetwork instance
        attribute_dict = vars(self)

        # Copy the attributes to the new instance
        for attr_name, attr_value in attribute_dict.items():
            # WranglerLogger.debug(f"Copying {attr_name}")
            if attr_name in COPY_REF_NOT_VALUE:
                # If the attribute is in the COPY_REF_NOT_VALUE list, assign the reference
                setattr(copied_net, attr_name, attr_value)
            else:
                # WranglerLogger.debug(f"making deep copy: {attr_name}")
                # For other attributes, perform a deep copy
                setattr(copied_net, attr_name, copy.deepcopy(attr_value, memo))

        return copied_net

    def deepcopy(self):
        """Returns copied TransitNetwork instance with deep copy of Feed but not roadway net."""
        return copy.deepcopy(self)

    @property
    def stops_gdf(self) -> gpd.GeoDataFrame:
        """Return stops as a GeoDataFrame using set roadway geometry."""
        ref_nodes = self.road_net.nodes_df if self.road_net is not None else None
        return to_points_gdf(self.feed.stops, ref_nodes_df=ref_nodes)

    @property
    def shapes_gdf(self) -> gpd.GeoDataFrame:
        """Return aggregated shapes as a GeoDataFrame using set roadway geometry."""
        ref_nodes = self.road_net.nodes_df if self.road_net is not None else None
        return shapes_to_trip_shapes_gdf(self.feed.shapes, ref_nodes_df=ref_nodes)

    @property
    def shape_links_gdf(self) -> gpd.GeoDataFrame:
        """Return shape-links as a GeoDataFrame using set roadway geometry."""
        ref_nodes = self.road_net.nodes_df if self.road_net is not None else None
        return shapes_to_shape_links_gdf(self.feed.shapes, ref_nodes_df=ref_nodes)

    @property
    def stop_time_links_gdf(self) -> gpd.GeoDataFrame:
        """Return stop-time-links as a GeoDataFrame using set roadway geometry."""
        ref_nodes = self.road_net.nodes_df if self.road_net is not None else None
        return stop_times_to_stop_time_links_gdf(
            self.feed.stop_times, self.feed.stops, ref_nodes_df=ref_nodes
        )

    @property
    def stop_times_points_gdf(self) -> gpd.GeoDataFrame:
        """Return stop-time-points as a GeoDataFrame using set roadway geometry."""
        ref_nodes = self.road_net.nodes_df if self.road_net is not None else None

        return stop_times_to_stop_time_points_gdf(
            self.feed.stop_times, self.feed.stops, ref_nodes_df=ref_nodes
        )

    def get_selection(
        self,
        selection_dict: dict,
        overwrite: bool = False,
    ) -> TransitSelection:
        """Return selection if it already exists, otherwise performs selection.

        Will raise an error if no trips found.

        Args:
            selection_dict (dict): _description_
            overwrite: if True, will overwrite any previously cached searches. Defaults to False.

        Returns:
            Selection: Selection object
        """
        key = dict_to_hexkey(selection_dict)

        if (key not in self._selections) or overwrite:
            WranglerLogger.debug(f"Performing selection from key: {key}")
            self._selections[key] = TransitSelection(self, selection_dict)
        else:
            WranglerLogger.debug(f"Using cached selection from key: {key}")

        if not self._selections[key]:
            msg = f"No links or nodes found for selection dict: \n {selection_dict}"
            WranglerLogger.error(msg)
            raise TransitSelectionEmptyError(msg)
        return self._selections[key]

    def apply(self, project_card: Union[ProjectCard, dict], **kwargs) -> TransitNetwork:
        """Wrapper method to apply a roadway project, returning a new TransitNetwork instance.

        Args:
            project_card: either a dictionary of the project card object or ProjectCard instance
            **kwargs: keyword arguments to pass to project application
        """
        if not (isinstance(project_card, (ProjectCard, SubProject))):
            project_card = ProjectCard(project_card)

        if not project_card.valid:
            msg = f"Project card {project_card.project} not valid."
            WranglerLogger.error(msg)
            raise ProjectCardError(msg)

        if project_card._sub_projects:
            for sp in project_card._sub_projects:
                WranglerLogger.debug(f"- applying subproject: {sp.change_type}")
                self._apply_change(sp, **kwargs)
            return self
        return self._apply_change(project_card, **kwargs)

    def _apply_change(
        self,
        change: Union[ProjectCard, SubProject],
        reference_road_net: Optional[RoadwayNetwork] = None,
    ) -> TransitNetwork:
        """Apply a single change: a single-project project or a sub-project."""
        if not isinstance(change, SubProject):
            WranglerLogger.info(f"Applying Project to Transit Network: {change.project}")

        if change.change_type == "transit_property_change":
            return apply_transit_property_change(
                self,
                self.get_selection(change.transit_property_change["service"]),
                change.transit_property_change["property_changes"],
                project_name=change.project,
            )

        if change.change_type == "transit_routing_change":
            return apply_transit_routing_change(
                self,
                self.get_selection(change.transit_routing_change["service"]),
                change.transit_routing_change["routing"],
                reference_road_net=reference_road_net,
                project_name=change.project,
            )

        if change.change_type == "pycode":
            return apply_calculated_transit(self, change.pycode)

        if change.change_type == "transit_route_addition":
            return apply_transit_route_addition(
                self,
                change.transit_route_addition,
                reference_road_net=reference_road_net,
            )
        if change.change_type == "transit_service_deletion":
            return apply_transit_service_deletion(
                self,
                self.get_selection(change.transit_service_deletion["service"]),
                clean_shapes=change.transit_service_deletion.get("clean_shapes"),
                clean_routes=change.transit_service_deletion.get("clean_routes"),
            )
        msg = f"Not a currently valid transit project: {change}."
        WranglerLogger.error(msg)
        raise NotImplementedError(msg)
