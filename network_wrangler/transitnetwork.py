#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import os
import re
from typing import Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import partridge as ptg
from partridge.config import default_config

from .logger import WranglerLogger
from .utils import parse_time_spans
from .roadwaynetwork import RoadwayNetwork


class TransitNetwork(object):
    """
    Representation of a Transit Network.

    .. highlight:: python

    Typical usage example:
    ::
        import network_wrangler as wr
        stpaul = r'/home/jovyan/work/example/stpaul'
        tc=wr.TransitNetwork.read(path=stpaul)

    Attributes:
        feed (DotDict): Partridge feed mapping dataframes.
        config (nx.DiGraph): Partridge config
        road_net (RoadwayNetwork): Associated roadway network object.
        graph (nx.MultiDiGraph): Graph for associated roadway network object.
        feed_path (str): Where the feed was read in from.
        validated_frequencies (bool): The frequencies have been validated.
        validated_road_network_consistency (): The network has been validated against the road network.
        SHAPES_FOREIGN_KEY (str): foreign key between shapes dataframe and roadway network nodes
        STOPS_FOREIGN_KEY (str): foreign  key between stops dataframe and roadway network nodes
        ID_SCALAR (int): scalar value added to create new IDs when necessary.
        REQUIRED_FILES (list[str]): list of files that the transit network requires.

    .. todo::
      investigate consolidating scalars this with RoadwayNetwork
      consolidate thes foreign key constants into one if possible
    """

    # PK = primary key, FK = foreign key
    SHAPES_FOREIGN_KEY = "shape_model_node_id"
    STOPS_FOREIGN_KEY = "model_node_id"

    ##TODO consolidate these two ^^^ constants if possible

    ID_SCALAR = 100000000

    ##TODO investigate consolidating this with RoadwayNetwork

    REQUIRED_FILES = [
        "agency.txt",
        "frequencies.txt",
        "routes.txt",
        "shapes.txt",
        "stop_times.txt",
        "stops.txt",
        "trips.txt",
    ]

    def __init__(self, feed: DotDict = None, config: nx.DiGraph = None):
        """
        Constructor

        .. todo:: Make graph a reference to associated RoadwayNetwork's graph, not its own thing.
        """
        self.feed: DotDict = feed
        self.config: nx.DiGraph = config
        self.road_net: RoadwayNetwork = None
        self.graph: nx.MultiDiGraph = None
        self.feed_path = None

        self.validated_frequencies = False
        self.validated_road_network_consistency = False

        if not self.validate_frequencies():
            raise ValueError(
                "Transit lines with non-positive frequencies exist in the network"
            )

    @staticmethod
    def empty() -> TransitNetwork:
        """
        Create an empty transit network instance using the default config.

        .. todo:: fill out this method
        """
        ##TODO

        msg = "TransitNetwork.empty is not implemented."
        WranglerLogger.error(msg)
        raise NotImplemented(msg)

    @staticmethod
    def read(feed_path: str) -> TransitNetwork:
        """
        Read GTFS feed from folder and TransitNetwork object

        Args:
            feed_path: where to read transit network files from

        Returns: a TransitNetwork object.
        """
        config = default_config()
        feed = ptg.load_feed(feed_path, config=config)
        WranglerLogger.info("Read in transit feed from: {}".format(feed_path))

        updated_config = TransitNetwork.validate_feed(feed, config)

        # Read in each feed so we can write over them
        editable_feed = DotDict()
        for node in updated_config.nodes.keys():
            # Load (initiate Partridge's lazy load)
            editable_feed[node.replace(".txt", "")] = feed.get(node)

        transit_network = TransitNetwork(feed=editable_feed, config=updated_config)
        transit_network.feed_path = feed_path
        return transit_network

    @staticmethod
    def validate_feed(feed: DotDict, config: nx.DiGraph) -> bool:
        """
        Since Partridge lazily loads the df, load each file to make sure it
        actually works.

        Partridge uses a DiGraph from the networkx library to represent the
        relationships between GTFS files. Each file is a 'node', and the
        relationship between files are 'edges'.

        Args:
            feed: partridge feed
            config: partridge config
        """
        updated_config = copy.deepcopy(config)
        files_not_found = []
        for node in config.nodes.keys():

            n = feed.get(node)
            WranglerLogger.debug("...{}:\n{}".format(node, n[:10]))
            if n.shape[0] == 0:
                WranglerLogger.info(
                    "Removing {} from transit network config because file not found".format(
                        node
                    )
                )
                updated_config.remove_node(node)
                if node in TransitNetwork.REQUIRED_FILES:
                    files_not_found.append(node)

        if files_not_found:
            msg = "Required files not found or valid: {}".format(
                ",".join(files_not_found)
            )
            WranglerLogger.error(msg)
            raise AttributeError(msg)
            return False

        TransitNetwork.validate_network_keys(feed)

        return updated_config

    def validate_frequencies(self) -> bool:
        """
        Validates that there are no transit trips in the feed with zero frequencies.

        Changes state of self.validated_frequencies boolean based on outcome.

        Returns:
            boolean indicating if valid or not.
        """

        _valid = True
        zero_freq = self.feed.frequencies[self.feed.frequencies.headway_secs <= 0]

        if len(zero_freq.index) > 0:
            _valid = False
            msg = "Transit lines {} have non-positive frequencies".format(
                zero_freq.trip_id.to_list()
            )
            WranglerLogger.error(msg)

        self.validated_frequencies = True

        return _valid

    def validate_road_network_consistencies(self) -> bool:
        """
        Validates transit network against the road network for both stops
        and shapes.

        Returns:
            boolean indicating if valid or not.
        """
        if self.road_net is None:
            raise ValueError(
                "RoadwayNetwork not set yet, see TransitNetwork.set_roadnet()"
            )

        valid = True

        valid_stops = self.validate_transit_stops()
        valid_shapes = self.validate_transit_shapes()

        self.validated_road_network_consistency = True

        if not valid_stops or not valid_shapes:
            valid = False
            raise ValueError("Transit network is not consistent with road network.")

        return valid

    def validate_transit_stops(self) -> bool:
        """
        Validates that all transit stops are part of the roadway network.

        Returns:
            Boolean indicating if valid or not.
        """

        if self.road_net is None:
            raise ValueError(
                "RoadwayNetwork not set yet, see TransitNetwork.set_roadnet()"
            )

        stops = self.feed.stops
        nodes = self.road_net.nodes_df

        valid = True

        stop_ids = [int(s) for s in stops[TransitNetwork.STOPS_FOREIGN_KEY].to_list()]
        node_ids = [int(n) for n in nodes[RoadwayNetwork.NODE_FOREIGN_KEY].to_list()]

        if not set(stop_ids).issubset(node_ids):
            valid = False
            missing_stops = list(set(stop_ids) - set(node_ids))
            msg = "Not all transit stops are part of the roadyway network. "
            msg += "Missing stops ({}) from the roadway nodes are {}.".format(
                TransitNetwork.STOPS_FOREIGN_KEY, missing_stops
            )
            WranglerLogger.error(msg)

        return valid

    def validate_transit_shapes(self) -> bool:
        """
        Validates that all transit shapes are part of the roadway network.

        Returns:
            Boolean indicating if valid or not.
        """

        if self.road_net is None:
            raise ValueError(
                "RoadwayNetwork not set yet, see TransitNetwork.set_roadnet()"
            )

        shapes_df = self.feed.shapes
        nodes_df = self.road_net.nodes_df
        links_df = self.road_net.links_df

        valid = True

        # check if all the node ids exist in the network
        shape_ids = [
            int(s) for s in shapes_df[TransitNetwork.SHAPES_FOREIGN_KEY].to_list()
        ]
        node_ids = [int(n) for n in nodes_df[RoadwayNetwork.NODE_FOREIGN_KEY].to_list()]

        if not set(shape_ids).issubset(node_ids):
            valid = False
            missing_shapes = list(set(shape_ids) - set(node_ids))
            msg = "Not all transit shapes are part of the roadyway network. "
            msg += "Missing shapes ({}) from the roadway network are {}.".format(
                TransitNetwork.SHAPES_FOREIGN_KEY, missing_shapes
            )
            WranglerLogger.error(msg)
            return valid

        # check if all the links in transit shapes exist in the network
        # and transit is allowed
        shapes_df = shapes_df.astype({TransitNetwork.SHAPES_FOREIGN_KEY: int})
        unique_shape_ids = shapes_df.shape_id.unique().tolist()

        for id in unique_shape_ids:
            subset_shapes_df = shapes_df[shapes_df["shape_id"] == id]
            subset_shapes_df = subset_shapes_df.sort_values(by=["shape_pt_sequence"])
            subset_shapes_df = subset_shapes_df.add_suffix("_1").join(
                subset_shapes_df.shift(-1).add_suffix("_2")
            )
            subset_shapes_df = subset_shapes_df.dropna()

            merged_df = subset_shapes_df.merge(
                links_df,
                how="left",
                left_on=[
                    TransitNetwork.SHAPES_FOREIGN_KEY + "_1",
                    TransitNetwork.SHAPES_FOREIGN_KEY + "_2",
                ],
                right_on=["A", "B"],
                indicator=True,
            )

            missing_links_df = merged_df.query('_merge == "left_only"')

            # there are shape links which does not exist in the roadway network
            if len(missing_links_df.index) > 0:
                valid = False
                msg = "There are links for shape id {} which are missing in the roadway network.".format(
                    id
                )
                WranglerLogger.error(msg)

            transit_not_allowed_df = merged_df.query(
                '_merge == "both" & drive_access == 0 & bus_only == 0 & rail_only == 0'
            )

            # there are shape links where transit is not allowed
            if len(transit_not_allowed_df.index) > 0:
                valid = False
                msg = "There are links for shape id {} which does not allow transit in the roadway network.".format(
                    id
                )
                WranglerLogger.error(msg)

        return valid

    @staticmethod
    def route_ids_in_routestxt(feed: DotDict) -> Bool:
        """
        Wherever route_id occurs, make sure it is in routes.txt

        Args:
            feed: partridge feed object

        Returns:
            Boolean indicating if feed is okay.
        """
        route_ids_routestxt = set(feed.routes.route_id.tolist())
        route_ids_referenced = set(feed.trips.route_id.tolist())

        missing_routes = route_ids_referenced - route_ids_routestxt

        if missing_routes:
            WranglerLogger.warning(
                "The following route_ids are referenced but missing from routes.txt: {}".format(
                    list(missing_routes)
                )
            )
            return False
        return True

    @staticmethod
    def trip_ids_in_tripstxt(feed: DotDict) -> Bool:
        """
        Wherever trip_id occurs, make sure it is in trips.txt

        Args:
            feed: partridge feed object

        Returns:
            Boolean indicating if feed is okay.
        """
        trip_ids_tripstxt = set(feed.trips.trip_id.tolist())
        trip_ids_referenced = set(
            feed.stop_times.trip_id.tolist() + feed.frequencies.trip_id.tolist()
        )

        missing_trips = trip_ids_referenced - trip_ids_tripstxt

        if missing_trips:
            WranglerLogger.warning(
                "The following trip_ids are referenced but missing from trips.txt: {}".format(
                    list(missing_trips)
                )
            )
            return False
        return True

    @staticmethod
    def shape_ids_in_shapestxt(feed: DotDict) -> Bool:
        """
        Wherever shape_id occurs, make sure it is in shapes.txt

        Args:
            feed: partridge feed object

        Returns:
            Boolean indicating if feed is okay.
        """

        shape_ids_shapestxt = set(feed.shapes.shape_id.tolist())
        shape_ids_referenced = set(feed.trips.shape_id.tolist())

        missing_shapes = shape_ids_referenced - shape_ids_shapestxt

        if missing_shapes:
            WranglerLogger.warning(
                "The following shape_ids from trips.txt are missing from shapes.txt: {}".format(
                    list(missing_shapes)
                )
            )
            return False
        return True

    @staticmethod
    def stop_ids_in_stopstxt(feed: DotDict) -> Bool:
        """
        Wherever stop_id occurs, make sure it is in stops.txt

        Args:
            feed: partridge feed object

        Returns:
            Boolean indicating if feed is okay.
        """
        stop_ids_stopstxt = set(feed.stops.stop_id.tolist())
        stop_ids_referenced = []

        # STOP_TIMES
        stop_ids_referenced.extend(feed.stop_times.stop_id.dropna().tolist())
        stop_ids_referenced.extend(feed.stops.parent_station.dropna().tolist())

        # TRANSFERS
        if feed.get("transfers.txt").shape[0] > 0:
            stop_ids_referenced.extend(feed.transfers.from_stop_id.dropna().tolist())
            stop_ids_referenced.extend(feed.transfers.to_stop_id.dropna().tolist())

        # PATHWAYS
        if feed.get("pathways.txt").shape[0] > 0:
            stop_ids_referenced.extend(feed.pathways.from_stop_id.dropna().tolist())
            stop_ids_referenced.extend(feed.pathways.to_stop_id.dropna().tolist())

        stop_ids_referenced = set(stop_ids_referenced)

        missing_stops = stop_ids_referenced - stop_ids_stopstxt

        if missing_stops:
            WranglerLogger.warning(
                "The following stop_ids from are referenced but missing from stops.txt: {}".format(
                    list(missing_stops)
                )
            )
            return False
        return True

    @staticmethod
    def validate_network_keys(feed: DotDict) -> Bool:
        """
        Validates foreign keys are present in all connecting feed files.

        Args:
            feed: partridge feed object

        Returns:
            Boolean indicating if feed is okay.
        """
        result = True
        result = result and TransitNetwork.route_ids_in_routestxt(feed)
        result = result and TransitNetwork.trip_ids_in_tripstxt(feed)
        result = result and TransitNetwork.shape_ids_in_shapestxt(feed)
        result = result and TransitNetwork.stop_ids_in_stopstxt(feed)
        return result

    def set_roadnet(
        self,
        road_net: RoadwayNetwork,
        graph_shapes: bool = False,
        graph_stops: bool = False,
        validate_consistency: bool = True,
    ) -> None:
        self.road_net: RoadwayNetwork = road_net
        self.graph: nx.MultiDiGraph = RoadwayNetwork.ox_graph(
            road_net.nodes_df, road_net.links_df
        )
        if graph_shapes:
            self._graph_shapes()
        if graph_stops:
            self._graph_stops()

        if validate_consistency:
            self.validate_road_network_consistencies()

    def _graph_shapes(self) -> None:
        """

        .. todo:: Fill out this method.
        """
        existing_shapes = self.feed.shapes
        msg = "_graph_shapes() not implemented yet."
        WranglerLogger.error(msg)
        raise NotImplemented(msg)
        # graphed_shapes = pd.DataFrame()

        # for shape_id in shapes:
        # TODO traverse point by point, mapping shortest path on graph,
        # then append to a list
        # return total list of all link ids
        # rebuild rows in shapes dataframe and add to graphed_shapes
        # make graphed_shapes a GeoDataFrame

        # self.feed.shapes = graphed_shapes

    def _graph_stops(self) -> None:
        """
        .. todo:: Fill out this method.
        """
        existing_stops = self.feed.stops
        msg = "_graph_stops() not implemented yet."
        WranglerLogger.error(msg)
        raise NotImplemented(msg)
        # graphed_stops = pd.DataFrame()

        # for stop_id in stops:
        # TODO

        # self.feed.stops = graphed_stops

    def write(self, path: str = ".", filename: str = None) -> None:
        """
        Writes a network in the transit network standard

        Args:
            path: the path were the output will be saved
            filename: the name prefix of the transit files that will be generated
        """
        WranglerLogger.info("Writing transit to directory: {}".format(path))
        for node in self.config.nodes.keys():

            df = self.feed.get(node.replace(".txt", ""))
            if not df.empty:
                if filename:
                    outpath = os.path.join(path, filename + "_" + node)
                else:
                    outpath = os.path.join(path, node)
                WranglerLogger.debug("Writing file: {}".format(outpath))

                df.to_csv(outpath, index=False)

    @staticmethod
    def transit_net_to_gdf(transit: Union(TransitNetwork, pd.DataFrame)):
        """
        Returns a geodataframe given a TransitNetwork or a valid Shapes DataFrame.

        Args:
            transit: either a TransitNetwork or a Shapes GeoDataFrame

        .. todo:: Make more sophisticated.
        """
        from partridge import geo

        if type(transit) is pd.DataFrame:
            shapes = transit
        else:
            shapes = transit.feed.shapes

        transit_gdf = geo.build_shapes(shapes)
        return transit_gdf

    def apply(self, project_card_dictionary: dict):
        """
        Wrapper method to apply a project to a transit network.

        Args:
            project_card_dictionary: dict
                a dictionary of the project card object

        """
        WranglerLogger.info(
            "Applying Project to Transit Network: {}".format(
                project_card_dictionary["project"]
            )
        )

        def _apply_individual_change(project_dictionary: dict):
            if (
                project_dictionary["category"].lower()
                == "transit service property change"
            ):
                self.apply_transit_feature_change(
                    self.select_transit_features(project_dictionary["facility"]),
                    project_dictionary["properties"],
                )
            elif project_dictionary["category"].lower() == "parallel managed lanes":
                # Grab the list of nodes in the facility from road_net
                # It should be cached because managed lane projects are
                # processed by RoadwayNetwork first via
                # Scenario.apply_all_projects
                try:
                    managed_lane_nodes = self.road_net.selections(
                        self.road_net.build_selection_key(
                            project_dictionary["facility"]
                        )
                    )["route"]
                except ValueError:
                    WranglerLogger.error(
                        "RoadwayNetwork not set yet, see TransitNetwork.set_roadnet()"
                    )

                # Reroute any transit using these nodes
                self.apply_transit_managed_lane(
                    self.select_transit_features_by_nodes(managed_lane_nodes),
                    managed_lane_nodes,
                )
            elif project_dictionary["category"].lower() == "add new route":
                self.add_new_transit_feature(
                    project_dictionary["routes"]
                )
            elif project_dictionary["category"].lower() == "delete transit service":
                self.delete_transit_service(
                    self.select_transit_features(project_dictionary["facility"])
                )
            elif project_dictionary["category"].lower() == "add transit":
                self.apply_python_calculation(project_dictionary["pycode"])
            elif project_dictionary["category"].lower() == "roadway deletion":
                WranglerLogger.warning(
                    "Roadway Deletion not yet implemented in Transit; ignoring"
                )
            else:
                msg = "{} not implemented yet in TransitNetwork; can't apply.".format(
                    project_dictionary["category"]
                )
                WranglerLogger.error(msg)
                raise (msg)

        if project_card_dictionary.get("changes"):
            for project_dictionary in project_card_dictionary["changes"]:
                _apply_individual_change(project_dictionary)
        else:
            _apply_individual_change(project_card_dictionary)

    def apply_python_calculation(
        self, pycode: str, in_place: bool = True
    ) -> Union(None, TransitNetwork):
        """
        Changes roadway network object by executing pycode.

        Args:
            pycode: python code which changes values in the roadway network object
            in_place: update self or return a new roadway network object
        """
        exec(pycode)

    def select_transit_features(self, selection: dict) -> pd.Series:
        """
        combines multiple selections

        Args:
            selection : selection dictionary

        Returns: trip identifiers : list of GTFS trip IDs in the selection
        """
        trip_ids = pd.Series()

        if selection.get("route"):
            for route_dictionary in selection["route"]:
                trip_ids = trip_ids.append(
                    self._select_transit_features(route_dictionary)
                )
        else:
            trip_ids = self._select_transit_features(selection)

        return trip_ids

    def _select_transit_features(self, selection: dict) -> pd.Series:
        """
        Selects transit features that satisfy selection criteria

        Args:
            selection : selection dictionary

        Returns: trip identifiers : list of GTFS trip IDs in the selection
        """
        trips = self.feed.trips
        routes = self.feed.routes
        freq = self.feed.frequencies

        # Turn selection's values into lists if they are not already
        for key in selection.keys():
            if type(selection[key]) not in [list, tuple]:
                selection[key] = [selection[key]]

        # Based on the key in selection, filter trips
        if "trip_id" in selection:
            trips = trips[trips.trip_id.isin(selection["trip_id"])]

        elif "route_id" in selection:
            trips = trips[trips.route_id.isin(selection["route_id"])]

        elif "route_short_name" in selection:
            routes = routes[routes.route_short_name.isin(selection["route_short_name"])]
            trips = trips[trips.route_id.isin(routes["route_id"])]

        elif "route_long_name" in selection:
            matches = []
            for sel in selection["route_long_name"]:
                for route_long_name in routes["route_long_name"]:
                    x = re.search(sel, route_long_name)
                    if x is not None:
                        matches.append(route_long_name)

            routes = routes[routes.route_long_name.isin(matches)]
            trips = trips[trips.route_id.isin(routes["route_id"])]

        else:
            WranglerLogger.error("Selection not supported %s", selection.keys())
            raise ValueError

        # If a time key exists, filter trips using frequency table
        if selection.get("time"):
            selection["time"] = parse_time_spans(selection["time"])
        elif selection.get("time_periods"):
            selection["time"] = []
            for time_period in selection["time_periods"]:
                selection["time"].append(parse_time_spans(
                    [time_period["start_time"], time_period["end_time"]]
                ))
            # Filter freq to trips in selection
            freq = freq[freq.trip_id.isin(trips["trip_id"])]
            freq = freq[freq.start_time.isin([i[0] for i in selection["time"]])]
            freq = freq[freq.end_time.isin([i[1] for i in selection["time"]])]

            # Filter trips table to those still in freq table
            trips = trips[trips.trip_id.isin(freq["trip_id"])]
        elif selection.get("start_time") and selection.get("end_time"):
            selection["time"] = parse_time_spans(
                [selection["start_time"][0], selection["end_time"][0]]
            )
            # Filter freq to trips in selection
            freq = freq[freq.trip_id.isin(trips["trip_id"])]
            freq = freq[freq.start_time == selection["time"][0]]
            freq = freq[freq.end_time == selection["time"][1]]

            # Filter trips table to those still in freq table
            trips = trips[trips.trip_id.isin(freq["trip_id"])]

        # If any other key exists, filter routes or trips accordingly
        for key in selection.keys():
            if key not in [
                "trip_id",
                "route_id",
                "route_short_name",
                "route_long_name",
                "time",
                "start_time",
                "end_time",
                "time_periods",
            ]:
                if key in trips:
                    trips = trips[trips[key].isin(selection[key])]
                elif key in routes:
                    routes = routes[routes[key].isin(selection[key])]
                    trips = trips[trips.route_id.isin(routes["route_id"])]
                else:
                    WranglerLogger.error("Selection not supported %s", key)
                    raise ValueError

        # Check that there is at least one trip in trips table or raise error
        if len(trips) < 1:
            WranglerLogger.error("Selection returned zero trips")
            raise ValueError

        # Return pandas.Series of trip_ids
        return trips["trip_id"]

    def select_transit_features_by_nodes(
        self, node_ids: list, require_all: bool = False
    ) -> pd.Series:
        """
        Selects transit features that use any one of a list of node_ids

        Args:
            node_ids: list (generally coming from nx.shortest_path)
            require_all : bool if True, the returned trip_ids must traverse all of
              the nodes (default = False)

        Returns:
            trip identifiers  list of GTFS trip IDs in the selection
        """
        # If require_all, the returned trip_ids must traverse all of the nodes
        # Else, filter any shapes that use any one of the nodes in node_ids
        if require_all:
            shape_ids = (
                self.feed.shapes.groupby("shape_id").filter(
                    lambda x: all(
                        i in x[TransitNetwork.SHAPES_FOREIGN_KEY].tolist()
                        for i in node_ids
                    )
                )
            ).shape_id.drop_duplicates()
        else:
            shape_ids = self.feed.shapes[
                self.feed.shapes[TransitNetwork.SHAPES_FOREIGN_KEY].isin(node_ids)
            ].shape_id.drop_duplicates()

        # Return pandas.Series of trip_ids
        return self.feed.trips[self.feed.trips.shape_id.isin(shape_ids)].trip_id

    def check_network_connectivity(self, shapes_foreign_key : pd.Series) -> pd.Series:
        """
        check if new shapes contain any links that are not in the roadway network
        """
        shape_links_df = pd.DataFrame(
            {
                "A" : shapes_foreign_key.tolist()[:-1],
                "B" : shapes_foreign_key.tolist()[1:],
            }
        )

        shape_links_df["A"] = shape_links_df["A"].astype(int)
        shape_links_df["B"] = shape_links_df["B"].astype(int)

        shape_links_df = pd.merge(
            shape_links_df,
            self.road_net.links_df[["A", "B", "model_link_id"]],
            how = "left",
            on = ["A", "B"]
        )

        missing_shape_links_df = shape_links_df[shape_links_df["model_link_id"].isnull()]

        if len(missing_shape_links_df) > 0:
            for index, row in missing_shape_links_df.iterrows():
                WranglerLogger.warning(
                    "Missing connections from node {} to node {} for the new routing, find complete path using default graph".format(int(row.A), int(row.B))
                )

                complete_node_list = TransitNetwork.route_between_nodes(self.graph, row.A, row.B)
                complete_node_list = pd.Series([str(int(i)) for i in complete_node_list])

                WranglerLogger.info(
                    "Routing path from node {} to node {} for missing connections: {}.".format(int(row.A), int(row.B), complete_node_list.tolist())
                )

                nodes = shapes_foreign_key.tolist()
                index_replacement_starts = [i for i,d in enumerate(nodes) if d == str(int(row.A))][0]
                index_replacement_ends = [i for i,d in enumerate(nodes) if d == str(int(row.B))][-1]
                shapes_foreign_key = pd.concat(
                    [
                        shapes_foreign_key.iloc[:index_replacement_starts],
                        complete_node_list,
                        shapes_foreign_key.iloc[index_replacement_ends + 1 :],
                    ],
                    ignore_index=True,
                    sort=False,
                )

        return shapes_foreign_key
    
    @staticmethod
    def route_between_nodes(graph, A, B) -> list:
        """
        find complete path when the new shape has connectivity issue
        """

        node_list = nx.shortest_path(
            graph,
            A,
            B,
            weight = "length"
        )

        return node_list
    
    def apply_transit_feature_change(
        self, trip_ids: pd.Series, properties: list, in_place: bool = True
    ) -> Union(None, TransitNetwork):
        """
        Changes the transit attributes for the selected features based on the
        project card information passed

        Args:
            trip_ids : pd.Series
                all trip_ids to apply change to
            properties : list of dictionaries
                transit properties to change
            in_place : bool
                whether to apply changes in place or return a new network

        Returns:
            None
        """
        for i in properties:
            if i["property"] in ["headway_secs"]:
                self._apply_transit_feature_change_frequencies(trip_ids, i, in_place)

            elif i["property"] in ["routing"]:
                self._apply_transit_feature_change_routing(trip_ids, i, in_place)

    def _apply_transit_feature_change_routing(
        self, trip_ids: pd.Series, properties: dict, in_place: bool = True
    ) -> Union(None, TransitNetwork):
        shapes = self.feed.shapes.copy()
        stop_times = self.feed.stop_times.copy()
        stops = self.feed.stops.copy()

        assert shapes[TransitNetwork.SHAPES_FOREIGN_KEY].dtype == "object"
        assert stops[TransitNetwork.STOPS_FOREIGN_KEY].dtype == "object"

        if shapes[TransitNetwork.SHAPES_FOREIGN_KEY].isnull().any():
            WranglerLogger.error(
                "There are null values in the shapes foreign key column, they will be filled with empty strings"
            )
            shapes[TransitNetwork.SHAPES_FOREIGN_KEY].fillna("", inplace=True)
            
        shapes[TransitNetwork.SHAPES_FOREIGN_KEY] = shapes[TransitNetwork.SHAPES_FOREIGN_KEY].apply(lambda x: x.replace(".0",""))
        
        if stops[TransitNetwork.STOPS_FOREIGN_KEY].isnull().any():
            WranglerLogger.error(
                "There are null values in the stops foreign key column, they will be filled with empty strings"
            )
            stops[TransitNetwork.STOPS_FOREIGN_KEY].fillna("", inplace=True)

        stops[TransitNetwork.STOPS_FOREIGN_KEY] = stops[TransitNetwork.STOPS_FOREIGN_KEY].apply(lambda x: x.replace(".0",""))

        # A negative sign in "set" indicates a traversed node without a stop
        # If any positive numbers, stops have changed
        stops_change = False
        # if any(x > 0 for x in properties["set"]):
        # Simplify "set" and "existing" to only stops
        properties["set_stops"] = [str(i) for i in properties["set"] if i > 0]
        if properties.get("existing") is not None:
            properties["existing_stops"] = [
                str(i) for i in properties["existing"] if i > 0
            ]
        if properties["existing_stops"] == properties["set_stops"]:
            stops_change = False
        else:
            stops_change = True

        # Convert ints to objects
        properties["set_shapes"] = [str(abs(i)) for i in properties["set"]]
        if properties.get("existing") is not None:
            properties["existing_shapes"] = [
                str(abs(i)) for i in properties["existing"]
            ]

        # Replace shapes records
        trips = self.feed.trips  # create pointer rather than a copy
        shape_ids = trips[trips["trip_id"].isin(trip_ids)].shape_id
        agency_raw_name = trips[trips["trip_id"].isin(trip_ids)].agency_raw_name.unique()[0]
        for shape_id in set(shape_ids):
            # Check if `shape_id` is used by trips that are not in
            # parameter `trip_ids`
            trips_using_shape_id = trips.loc[trips["shape_id"] == shape_id, ["trip_id"]]
            if not all(trips_using_shape_id.isin(trip_ids)["trip_id"]):
                # In this case, we need to create a new shape_id so as to leave
                # the trips not part of the query alone
                WranglerLogger.warning(
                    "Trips that were not in your query selection use the "
                    "same `shape_id` as trips that are in your query. Only "
                    "the trips' shape in your query will be changed."
                )
                old_shape_id = shape_id
                shape_id = str(int(shape_id) + TransitNetwork.ID_SCALAR)
                if shape_id in shapes["shape_id"].tolist():
                    WranglerLogger.error("Cannot create a unique new shape_id.")
                dup_shape = shapes[shapes.shape_id == old_shape_id].copy()
                dup_shape["shape_id"] = shape_id
                shapes = pd.concat([shapes, dup_shape], ignore_index=True)

            # Pop the rows that match shape_id
            this_shape = shapes[shapes.shape_id == shape_id]

            # Make sure they are ordered by shape_pt_sequence
            this_shape = this_shape.sort_values(by=["shape_pt_sequence"])

            # Build a pd.DataFrame of new shape records
            new_shape_rows = pd.DataFrame(
                {
                    "shape_id": shape_id,
                    "shape_pt_lat": None,  # FIXME Populate from self.road_net?
                    "shape_pt_lon": None,  # FIXME
                    "shape_osm_node_id": None,  # FIXME
                    "shape_pt_sequence": None,
                    TransitNetwork.SHAPES_FOREIGN_KEY: properties["set_shapes"],
                    "agency_raw_name":agency_raw_name,
                }
            )

            check_new_shape_nodes = self.check_network_connectivity(new_shape_rows[TransitNetwork.SHAPES_FOREIGN_KEY])

            if len(check_new_shape_nodes) != len(new_shape_rows):
                new_shape_rows = pd.DataFrame(
                    {
                        "shape_id": shape_id,
                        "shape_pt_lat": None,  # FIXME Populate from self.road_net?
                        "shape_pt_lon": None,  # FIXME
                        "shape_osm_node_id": None,  # FIXME
                        "shape_pt_sequence": None,
                        TransitNetwork.SHAPES_FOREIGN_KEY: check_new_shape_nodes,
                        "agency_raw_name":agency_raw_name,
                    }
                )
                properties["set_shapes"] = check_new_shape_nodes.tolist()

            # If "existing" is specified, replace only that segment
            # Else, replace the whole thing
            if properties.get("existing") is not None:
                # Match list
                nodes = this_shape[TransitNetwork.SHAPES_FOREIGN_KEY].tolist()
                index_replacement_starts = [i for i,d in enumerate(nodes) if d == properties["existing_shapes"][0]][0]
                index_replacement_ends = [i for i,d in enumerate(nodes) if d == properties["existing_shapes"][-1]][-1]
                this_shape = pd.concat(
                    [
                        this_shape.iloc[:index_replacement_starts],
                        new_shape_rows,
                        this_shape.iloc[index_replacement_ends + 1 :],
                    ],
                    ignore_index=True,
                    sort=False,
                )
            else:
                this_shape = new_shape_rows

            # Renumber shape_pt_sequence
            this_shape["shape_pt_sequence"] = np.arange(len(this_shape))

            # Add rows back into shapes
            shapes = pd.concat(
                [shapes[shapes.shape_id != shape_id], this_shape],
                ignore_index=True,
                sort=False,
            )

        # Replace stop_times and stops records (if required)
        if stops_change:
            # If node IDs in properties["set_stops"] are not already
            # in stops.txt, create a new stop_id for them in stops
            existing_fk_ids = set(stops[TransitNetwork.STOPS_FOREIGN_KEY].tolist())
            nodes_df = self.road_net.nodes_df.loc[
                :, [TransitNetwork.STOPS_FOREIGN_KEY, "X", "Y"]
            ]
            for trip_id in trip_ids:
                for fk_i in properties["set_stops"]:
                    if fk_i in existing_fk_ids:
                       existing_agency_raw_name = stops[stops[TransitNetwork.STOPS_FOREIGN_KEY]==fk_i]['agency_raw_name'].to_list()
                       existing_trip_ids = stops[stops[TransitNetwork.STOPS_FOREIGN_KEY]==fk_i]['trip_id'].to_list()
                       existing_stop_id = stops[stops[TransitNetwork.STOPS_FOREIGN_KEY]==fk_i]['stop_id'].iloc[0]
                       if ((agency_raw_name not in existing_agency_raw_name)
                        | (trip_id not in existing_trip_ids)
                       ):
                            stops.loc[
                            len(stops.index) + 1,
                            [
                                "stop_id",
                                "stop_lat",
                                "stop_lon",
                                TransitNetwork.STOPS_FOREIGN_KEY,
                                "trip_id",
                                "agency_raw_name"
                            ],
                            ] = [
                                existing_stop_id,
                                nodes_df.loc[nodes_df[TransitNetwork.STOPS_FOREIGN_KEY] == int(fk_i), "Y"].values[0],
                                nodes_df.loc[nodes_df[TransitNetwork.STOPS_FOREIGN_KEY] == int(fk_i), "X"].values[0],
                                fk_i,
                                trip_id,
                                agency_raw_name
                            ]

                    elif fk_i not in existing_fk_ids:
                        WranglerLogger.info(
                            "Creating a new stop in stops.txt for node ID: {}".format(fk_i)
                        )
                        # Add new row to stops
                        new_stop_id = str(int(fk_i) + TransitNetwork.ID_SCALAR)
                        if new_stop_id in stops["stop_id"].tolist():
                            WranglerLogger.error("Cannot create a unique new stop_id.")
                        
                        stops.loc[
                            len(stops.index) + 1,
                            [
                                "stop_id",
                                "stop_lat",
                                "stop_lon",
                                TransitNetwork.STOPS_FOREIGN_KEY,
                                "trip_id",
                                "agency_raw_name"
                            ],
                        ] = [
                            new_stop_id,
                            nodes_df.loc[nodes_df[TransitNetwork.STOPS_FOREIGN_KEY] == int(fk_i), "Y"].values[0],
                            nodes_df.loc[nodes_df[TransitNetwork.STOPS_FOREIGN_KEY] == int(fk_i), "X"].values[0],
                            fk_i,
                            trip_id,
                            agency_raw_name
                        ]

            # Loop through all the trip_ids
            for trip_id in trip_ids:
                # Pop the rows that match trip_id
                this_stoptime = stop_times[stop_times.trip_id == trip_id]

                # Merge on node IDs using stop_id (one node ID per stop_id)
                this_stoptime = this_stoptime.merge(
                    stops[["stop_id", "trip_id", TransitNetwork.STOPS_FOREIGN_KEY]],
                    how="left",
                    on=["stop_id", "trip_id"],
                )

                # Make sure the stop_times are ordered by stop_sequence
                this_stoptime = this_stoptime.sort_values(by=["stop_sequence"])

                # Build a pd.DataFrame of new shape records from properties
                new_stoptime_rows = pd.DataFrame(
                    {
                        "trip_id": trip_id,
                        "arrival_time": None,
                        "departure_time": None,
                        "pickup_type": None,
                        "drop_off_type": None,
                        "stop_distance": None,
                        "timepoint": None,
                        "stop_is_skipped": None,
                        TransitNetwork.STOPS_FOREIGN_KEY: properties["set_stops"],
                        "agency_raw_name":agency_raw_name,
                    }
                )

                # Merge on stop_id using node IDs (many stop_id per node ID)
                new_stoptime_rows = (
                    new_stoptime_rows.merge(
                        stops[["stop_id", TransitNetwork.STOPS_FOREIGN_KEY]],
                        how="left",
                        on=TransitNetwork.STOPS_FOREIGN_KEY,
                    )
                    .groupby([TransitNetwork.STOPS_FOREIGN_KEY])
                    .head(1)
                )  # pick first

                # If "existing" is specified, replace only that segment
                # Else, replace the whole thing
                if properties.get("existing") is not None:
                    index_replacement_starts = None
                    index_replacement_ends = None
                    # Match list (remember stops are passed in with node IDs)
                    nodes = this_stoptime[TransitNetwork.STOPS_FOREIGN_KEY].tolist()
                    if len(properties["existing_stops"]) == 0:
                        for n in this_shape[TransitNetwork.SHAPES_FOREIGN_KEY]:
                            if n in this_stoptime[TransitNetwork.STOPS_FOREIGN_KEY].values:
                                index_replacement_starts = nodes.index(n)+1
                                index_replacement_ends = nodes.index(n)
                            if n == properties["existing_shapes"][0]:
                                break
                    else:
                        # index_replacement_starts = nodes.index(
                        #     properties["existing_stops"][0]
                        # )
                        # index_replacement_ends = nodes.index(
                        #     properties["existing_stops"][-1]
                        # )
                        indices = [nodes.index(n) for n in properties["existing_stops"]]
                        index_replacement_starts = min(indices)
                        index_replacement_ends = max(indices)

                    this_stoptime = pd.concat(
                        [
                            this_stoptime.iloc[:index_replacement_starts],
                            new_stoptime_rows,
                            this_stoptime.iloc[index_replacement_ends + 1 :],
                        ],
                        ignore_index=True,
                        sort=False,
                    )
                else:
                    this_stoptime = new_stoptime_rows

                # Remove node ID
                del this_stoptime[TransitNetwork.STOPS_FOREIGN_KEY]

                # Renumber stop_sequence
                this_stoptime["stop_sequence"] = np.arange(len(this_stoptime))

                # Add rows back into stoptime
                stop_times = pd.concat(
                    [stop_times[stop_times.trip_id != trip_id], this_stoptime],
                    ignore_index=True,
                    sort=False,
                )

        # Replace self if in_place, else return
        if in_place:
            self.feed.shapes = shapes
            self.feed.stops = stops
            self.feed.stop_times = stop_times
        else:
            updated_network = copy.deepcopy(self)
            updated_network.feed.shapes = shapes
            updated_network.feed.stops = stops
            updated_network.feed.stop_times = stop_times
            return updated_network

    def _apply_transit_feature_change_frequencies(
        self, trip_ids: pd.Series, properties: dict, in_place: bool = True
    ) -> Union(None, TransitNetwork):
        freq = self.feed.frequencies.copy()

        # Grab only those records matching trip_ids (aka selection)
        freq = freq[freq.trip_id.isin(trip_ids)]

        # Check all `existing` properties if given
        if properties.get("existing") is not None:
            if not all(freq.headway_secs == properties["existing"]):
                WranglerLogger.error(
                    "Existing does not match for at least "
                    "1 trip in:\n {}".format(trip_ids.to_string())
                )
                raise ValueError

        # Calculate build value
        if properties.get("set") is not None:
            build_value = properties["set"]
        else:
            build_value = [i + properties["change"] for i in freq.headway_secs]

        # Update self or return a new object
        q = self.feed.frequencies.trip_id.isin(freq["trip_id"])
        if in_place:
            self.feed.frequencies.loc[q, properties["property"]] = build_value
        else:
            updated_network = copy.deepcopy(self)
            updated_network.loc[q, properties["property"]] = build_value
            return updated_network

    def apply_transit_managed_lane(
        self, trip_ids: pd.Series, node_ids: list, in_place: bool = True
    ) -> Union(None, TransitNetwork):
        # Traversed nodes without a stop should be negative integers
        all_stops = self.feed.stops[TransitNetwork.STOPS_FOREIGN_KEY].tolist()
        node_ids = [int(x) if str(x) in all_stops else int(x) * -1 for x in node_ids]

        self._apply_transit_feature_change_routing(
            trip_ids=trip_ids,
            properties={
                "existing": node_ids,
                "set": RoadwayNetwork.get_managed_lane_node_ids(node_ids),
            },
            in_place=in_place,
        )

    def add_new_transit_feature(
        self, routes: dict, in_place: bool = True
    ) -> Union(None, TransitNetwork):
        """
        Add new transit services based on the project card information passed
        Args:
            routes: dict
                new routes
            in_place : bool
                whether to apply changes in place or return a new network
        Returns:
            None or updated transit network
        """

        routes_df = self.feed.routes.copy()
        shapes_df = self.feed.shapes.copy()
        trips_df = self.feed.trips.copy()
        stop_times_df = self.feed.stop_times.copy()
        stops_df = self.feed.stops.copy()
        frequencies_df = self.feed.frequencies.copy()

        # links = self.road_net.links_df.copy()
        nodes = self.road_net.nodes_df.copy()

        # in case any stops missing model_node_id
        stops_missing_model_node_id = (
            stops_df[
                (stops_df[TransitNetwork.STOPS_FOREIGN_KEY].isna())
                | (stops_df[TransitNetwork.STOPS_FOREIGN_KEY]=="")
            ].copy()
        )
        stops_with_model_node_id = (
            stops_df[
                ~(
                    (stops_df[TransitNetwork.STOPS_FOREIGN_KEY].isna())
                    | (stops_df[TransitNetwork.STOPS_FOREIGN_KEY]=="")
                )
            ].copy()
        )

        stops_missing_model_node_id = pd.merge(
            stops_missing_model_node_id.drop(TransitNetwork.STOPS_FOREIGN_KEY, axis = 1),
            nodes[['shst_node_id', TransitNetwork.STOPS_FOREIGN_KEY]],
            how = 'left',
            on = 'shst_node_id'
        )

        stops_final_df = pd.concat([stops_with_model_node_id, stops_missing_model_node_id])
        assert len(stops_final_df) == len(stops_df) 

        stop_id_xref_dict = (
            stops_final_df
            .set_index(TransitNetwork.STOPS_FOREIGN_KEY)["stop_id"]
            .to_dict()
        )
        stop_id_xref_dict = {int(float(key)): int(float(value)) for key, value in stop_id_xref_dict.items()}
        model_node_coord_dict = (
            nodes
            .set_index(TransitNetwork.STOPS_FOREIGN_KEY)[['X', 'Y']]
            .apply(tuple, axis=1)
            .to_dict()
        )
        model_node_coord_dict = {int(float(key)): value for key, value in model_node_coord_dict.items()}
        
        stop_id_max = max(stop_id_xref_dict.values())
        shape_id_max = pd.to_numeric(shapes_df['shape_id'].str.extract(r'(\d+)')[0], errors='coerce').max()

        # define column data type
        route_col_dtypes = {
            "route_id": "object",
            "route_short_name": "object",
            "route_long_name": "object",
            "route_type": "int64",
            "agency_raw_name": "object",
            "agency_id": "object"
        }

        shape_col_dtypes = {
            "shape_id": "object",
            "shape_model_node_id": "object",
            "shape_pt_sequence": "int64",
            "agency_raw_name": "object"
        }

        trip_col_dtypes = {
            "route_id": "object",
            "direction_id": "object",
            "trip_id": "object",
            "shape_id": "object",
            "agency_raw_name": "object"
        }

        freq_col_dtypes = {
            "trip_id": "object",
            "headway_secs": "int64",
            "start_time": "float64",
            "end_time": "float64",
            "agency_raw_name": "object"
        }

        stop_col_dtypes = {
            "stop_id" : "object",
            "stop_lat" : "float64",
            "stop_lon" : "float64",
            "model_node_id" : "object",
            'trip_id': "object",
            "agency_raw_name": "object"
        }

        stop_time_col_dtypes = {
            "trip_id": "object",
            "stop_sequence": "int64",
            "arrival_time": "float64",
            "departure_time": "float64",
            "pickup_type": "float64",
            "drop_off_type": "float64",
            "stop_id": "object",
            "agency_raw_name": "object"
        }

        for route in routes:
            # add route
            agency_id = route["agency_id"]
            add_routes_dict = {
                "route_id": route["route_id"],
                "route_short_name": route["route_short_name"],
                "route_long_name": route["route_long_name"],
                "route_type": route["route_type"],
                "agency_raw_name": route["agency_raw_name"],
                "agency_id": f'{agency_id}'
            }
            add_routes_df = pd.DataFrame([add_routes_dict]).astype(route_col_dtypes)
            routes_df = pd.concat([routes_df, add_routes_df], ignore_index=True, sort=False)

            trip_index = 1
            for trip in route["trips"]:
                # add shape
                shape_id = f"{shape_id_max+1}"
                shape_model_node_id_list = [list(item.keys())[0] if isinstance(item, dict) else item for item in trip["routing"]]
                add_shapes_dict = {
                    "shape_id": shape_id,
                    "shape_model_node_id": shape_model_node_id_list,
                    "shape_pt_sequence": list(range(1,len(shape_model_node_id_list)+1)),
                    "agency_raw_name": route["agency_raw_name"]
                }
                
                add_shapes_df = pd.DataFrame(add_shapes_dict).astype(shape_col_dtypes)
                shapes_df = pd.concat([shapes_df, add_shapes_df], ignore_index=True, sort=False)

                
                for i in trip["headway_sec"]:
                    # add trip
                    trip_id = f"trip{trip_index}_shp{shape_id}"
                    add_trips_dict = {
                        "route_id": route["route_id"],
                        "direction_id": trip["direction_id"],
                        "trip_id": trip_id,
                        "shape_id": shape_id,
                        "agency_raw_name": route["agency_raw_name"]
                    }
                    add_trips_df = pd.DataFrame([add_trips_dict]).astype(trip_col_dtypes)
                    trips_df = pd.concat([trips_df, add_trips_df], ignore_index=True, sort=False)

                    # add frequency
                    headway_secs = list(i.values())[0]
                    time_range = list(i.keys())[0]
                    time_range = [time.strip().strip("'") for time in time_range.strip("()").split(',')]
                    start_time = parse_time_spans(time_range)[0]
                    end_time = parse_time_spans(time_range)[1]

                    add_freqs_dict = {
                        "trip_id": trip_id,
                        "headway_secs": headway_secs,
                        "start_time": start_time,
                        "end_time": end_time,
                        "agency_raw_name": route["agency_raw_name"]
                    }
                    add_freqs_df = pd.DataFrame([add_freqs_dict]).astype(freq_col_dtypes)
                    frequencies_df = pd.concat([frequencies_df, add_freqs_df], ignore_index=True, sort=False)

                    # add stop and stop_times
                    stop_model_node_id_list = []
                    pickup_type = []
                    drop_off_type = []

                    for i in trip['routing']:
                        if (isinstance(i, dict) and 
                           list(i.values())[0] is not None and 
                           list(i.values())[0].get('stop')
                        ):
                            stop_model_node_id_list.append(list(i.keys())[0])
                            drop_off_type.append(0 if list(i.values())[0].get('alight', True) else 1)
                            pickup_type.append(0 if list(i.values())[0].get('board', True) else 1) 

                    # used to build stop_time
                    stop_id_list = [] 

                    # used to add new stops if they are not in the stops.txt
                    new_stop_id_list = []
                    model_node_id_list = []
                    stop_lat_list = []
                    stop_lon_list = []

                    for s in stop_model_node_id_list:
                        s = int(float(s))
                        if s in stop_id_xref_dict.keys():
                            existing_agency_raw_name = (
                                stops_final_df[
                                    stops_final_df[TransitNetwork.STOPS_FOREIGN_KEY]
                                    .astype(float)
                                    .astype(int) == s
                                ]['agency_raw_name'].to_list()
                            )
                            existing_trip_ids = (
                                stops_final_df[
                                    stops_final_df[TransitNetwork.STOPS_FOREIGN_KEY]
                                    .astype(float)
                                    .astype(int) == s
                                ]['trip_id'].to_list()
                            )
                            existing_stop_id = (
                                stops_final_df[
                                    stops_final_df[TransitNetwork.STOPS_FOREIGN_KEY]
                                    .astype(float)
                                    .astype(int) == s
                                ]['stop_id'].iloc[0]
                            )
                            if ((route["agency_raw_name"] not in existing_agency_raw_name)
                                | (trip_id not in existing_trip_ids)
                            ):
                                new_stop_id = existing_stop_id
                                stop_id_list.append(new_stop_id)
                                # add new stop to stops.txt
                                new_stop_id_list.append(new_stop_id)
                                model_node_id_list.append(s)
                                stop_lat_list.append(model_node_coord_dict[s][1])
                                stop_lon_list.append(model_node_coord_dict[s][0])
                                stop_id_xref_dict.update({s: new_stop_id})
                            else:
                                stop_id_list.append(stop_id_xref_dict[s])
                        else:
                            new_stop_id = stop_id_max + 1
                            stop_id_list.append(new_stop_id)
                            # add new stop to stops.txt
                            new_stop_id_list.append(new_stop_id)
                            model_node_id_list.append(s)
                            stop_lat_list.append(model_node_coord_dict[s][1])
                            stop_lon_list.append(model_node_coord_dict[s][0])
                            stop_id_xref_dict.update({s: new_stop_id})
                            stop_id_max += 1
                    
                    # add stops
                    add_stops_df = pd.DataFrame({
                        "stop_id" : new_stop_id_list,
                        "stop_lat" : stop_lat_list,
                        "stop_lon" : stop_lon_list,
                        "model_node_id" : model_node_id_list,
                        'trip_id': trip_id,
                        "agency_raw_name": route["agency_raw_name"]
                    }).astype(stop_col_dtypes)
                    stops_final_df = pd.concat([stops_final_df, add_stops_df], ignore_index=True, sort=False)

                    # add stop_times
                    # TODO: time_to_next_node_sec
                    stop_sequence = list(range(1, len(stop_id_list) + 1))
                    add_stop_times_df = pd.DataFrame({
                        "trip_id": trip_id,
                        "stop_sequence": stop_sequence,
                        "arrival_time": 0,
                        "departure_time": 0,
                        "pickup_type": pickup_type,
                        "drop_off_type": drop_off_type,
                        "stop_id": stop_id_list,
                        "agency_raw_name": route["agency_raw_name"]
                    }).astype(stop_time_col_dtypes)
                    stop_times_df = pd.concat([stop_times_df, add_stop_times_df], ignore_index=True, sort=False)

                    trip_index += 1
                shape_id_max += 1

        # Replace self if in_place, else return
        if in_place:
            self.feed.routes = routes_df
            self.feed.shapes = shapes_df
            self.feed.trips = trips_df
            self.feed.stop_times = stop_times_df
            self.feed.stops = stops_final_df
            self.feed.frequencies = frequencies_df
        else:
            updated_network = copy.deepcopy(self)
            updated_network.feed.routes = routes_df
            updated_network.feed.shapes = shapes_df
            updated_network.feed.trips = trips_df
            updated_network.feed.stop_times = stop_times_df
            updated_network.feed.stops = stops_final_df
            updated_network.feed.frequencies = frequencies_df
            return updated_network

    def delete_transit_service(
        self, trip_ids: pd.Series, in_place: bool = True
    ) -> Union(None, TransitNetwork):
        """
        delete transit service 
        Args:
            trip_ids: pd.Series
                trip ids that need to be deleted from the transit network
            in_place : bool
                whether to apply changes in place or return a new network
        Returns:
            None or updated transit network
        """

        trips_df = self.feed.trips.copy()
        stop_times_df = self.feed.stop_times.copy()
        stops_df = self.feed.stops.copy()
        frequencies_df = self.feed.frequencies.copy()

        delete_trip_list = trip_ids.tolist()
        trips_df = trips_df[~trips_df.trip_id.isin(delete_trip_list)]
        stop_times_df = stop_times_df[~stop_times_df.trip_id.isin(delete_trip_list)]
        stops_df = stops_df[~stops_df.trip_id.isin(delete_trip_list)]
        frequencies_df = frequencies_df[~frequencies_df.trip_id.isin(delete_trip_list)]

        # Replace self if in_place, else return
        if in_place:
            self.feed.trips = trips_df
            self.feed.stop_times = stop_times_df
            self.feed.stops = stops_df
            self.feed.frequencies = frequencies_df
        else:
            updated_network = copy.deepcopy(self)
            updated_network.feed.trips = trips_df
            updated_network.feed.stop_times = stop_times_df
            updated_network.feed.stops = stops_df
            updated_network.feed.frequencies = frequencies_df
            return updated_network

class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    Source: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
