#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import os
import re
from typing import Tuple, Union

import networkx as nx
import pandas as pd
import partridge as ptg
from partridge.config import default_config
from partridge.gtfs import Feed

from .Logger import WranglerLogger
from .Utils import parse_time_spans
from .RoadwayNetwork import RoadwayNetwork


class TransitNetwork(object):
    """
    Representation of a Transit Network.

    Usage:
      import network_wrangler as wr
      stpaul = r'/home/jovyan/work/example/stpaul'
      tc=wr.TransitNetwork.read(path=stpaul)

    """
    # PK = primary key, FK = foreign key
    PK_NODE_ID = "shape_model_node_id"

    def __init__(self, feed: Feed = None, config: nx.DiGraph = None):
        """
        Constructor
        """
        self.feed: Feed = feed
        self.config: nx.DiGraph = config
        self.road_net: RoadwayNetwork = None
        self.graph: nx.MultiDiGraph = None

    @staticmethod
    def validate_feed(feed: Feed, config: nx.DiGraph) -> bool:
        """
        Since Partridge lazily loads the df, load each file to make sure it
        actually works.

        Partridge uses a DiGraph from the networkx library to represent the
        relationships between GTFS files. Each file is a 'node', and the
        relationship between files are 'edges'.
        """
        try:
            for node in config.nodes.keys():
                feed.get(node)
            return True
        except AttributeError:
            return False

    @staticmethod
    def read(feed_path: str, fast: bool = False) -> Tuple[nx.DiGraph, Feed]:
        """
        Read GTFS feed from folder and return a config and Partridge Feed object
        """
        config = default_config()
        feed = ptg.load_feed(feed_path, config=config)

        TransitNetwork.validate_feed(feed, config)

        transit_network = TransitNetwork(feed=feed, config=config)

        return transit_network

    def set_graph(self, road_net: RoadwayNetwork,
                  graph_shapes: bool = True, graph_stops: bool = True
                  ) -> None:
        self.road_net: RoadwayNetwork = road_net
        self.graph: nx.MultiDiGraph = RoadwayNetwork.ox_graph(
            road_net.nodes_df, road_net.links_df
        )
        if graph_shapes:
            self._graph_shapes()
        if graph_stops:
            self._graph_stops()

    def _graph_shapes(self) -> None:
        existing_shapes = self.feed.shapes
        graphed_shapes = pd.DataFrame()

        # for shape_id in shapes:
        # TODO traverse point by point, mapping shortest path on graph,
        # then append to a list
        # return total list of all link ids
        # rebuild rows in shapes dataframe and add to graphed_shapes
        # TODO Make graphed_shapes a GeoDataFrame

        self.feed.shapes = graphed_shapes

    def _graph_stops(self) -> None:
        existing_stops = self.feed.stops
        graphed_stops = pd.DataFrame()

        # for stop_id in stops:
        # TODO

        self.feed.stops = graphed_stops

    def write(self, path: str = ".", filename: str = None) -> None:
        """
        Writes a network in the transit network standard

        Parameters
        ------------
        path: the path were the output will be saved
        filename: the name prefix of the transit files that will be generated
        """
        for node in self.config.nodes.keys():
            df = self.feed.get(node)
            if not df.empty:
                if filename:
                    outpath = os.path.join(path, filename + "_" + node)
                else:
                    outpath = os.path.join(path, node)

                df.to_csv(outpath, index=False)

    def apply(self, project_card_dictionary: dict):
        """
        Wrapper method to apply a project to a transit network.

        args
        ------
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
                WranglerLogger.warning(
                    "Parallel Managed Lanes not implemented yet in Transit"
                )
                # TODO
            else:
                raise (BaseException)

        if project_card_dictionary.get("changes"):
            for project_dictionary in project_card_dictionary["changes"]:
                _apply_individual_change(project_dictionary)
        else:
            _apply_individual_change(project_card_dictionary)

    def select_transit_features(self, selection: dict) -> pd.Series:
        """
        Selects transit features that satisfy selection criteria

        Parameters
        ------------
        selection : dictionary

        Returns
        -------
        trip identifiers : list
           list of GTFS trip IDs in the selection
        """
        trips = self.feed.trips
        routes = self.feed.routes
        freq = self.feed.frequencies

        # Turn selection's values into lists if they are not already
        for key in selection.keys():
            if type(selection[key]) != list:
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
        if selection.get("time") is not None:
            selection["time"] = parse_time_spans(selection["time"])

            # Filter freq to trips in selection
            freq = freq[freq.trip_id.isin(trips["trip_id"])]
            freq = freq[freq.start_time == selection["time"][0]]
            freq = freq[freq.end_time == selection["time"][1]]

            # Filter trips table to those still in freq table
            trips = trips[trips.trip_id.isin(freq["trip_id"])]

        # If any other key exists, filter routes or trips accordingly
        for key in selection.keys():
            if key not in [
                "trip_id", "route_id", "route_short_name", "route_long_name",
                "time"
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

    def apply_transit_feature_change(
        self, trip_ids: pd.Series, properties: dict
    ) -> None:
        """
        Changes the transit attributes for the selected features based on the
        project card information passed

        Parameters
        ------------
        trip_ids : pd.Series
            all trip_ids to apply change to
        properties : list of dictionarys
            transit properties to change

        Returns
        -------
        None
        """
        for i in properties:
            if i["property"] in ["headway_secs"]:
                self.apply_transit_feature_change_frequencies(trip_ids, i)

            elif i['property'] in ['shapes']:
                self.apply_transit_feature_change_shapes(trip_ids, i)

            elif i['property'] in ['stops']:
                self.apply_transit_feature_change_stops(trip_ids, i)

    def apply_transit_feature_change_shapes(
        self, trip_ids: pd.Series, properties: dict, in_place: bool = True
    ) -> Union(None, TransitNetwork):
        shapes = self.feed.shapes
        trips = self.feed.trips

        # Grab only those records matching trip_ids (aka selection)
        shape_ids = trips[trips.trip_id.isin(trip_ids)].shape_id

        # Convert the node IDs in properties to link IDs with lineStrings
        links = self.road_net.links_df
        link_ids = links[links[FK_ROAD_NODE_ID_IN_LINKS_DF].isin(properties['set'])]

        # If 'existing' is specified, replace only that segment
        # TODO

        # Replace
        for shape_id in shape_ids:
            # Pop the rows that match shape_id
            this_shape = shapes[shapes.shape_id == shape_id]
            shapes[shapes.shape_id != shape_id]

            # Match sure they are ordered by shape_pt_sequence
            this_shape = this_shape.sort_values(by=['shape_pt_sequence'])

            # Grab list of link IDs and simplify to unique list
            all_existing_link_ids = this_shape[PK_LINK_ID].unique()

            # Create a new DataFrame
            this_shape = pd.DataFrame({
                PK_LINK_ID: all_existing_link_ids
            })

            # Join on linestring for each link
            this_shape.merge(
                self.road_net.links_df[[PK_ROAD_NET_LINK_ID, 'geometry']],
                how='left',
                left_on=PK_LINK_ID, right_on=PK_ROAD_NET_LINK_ID,
            )

            # Convert GeoDataFrame into DataFrame by exploding each point w/ ID
            # TODO

            # Add shape_id to all rows and point sequence
            # TODO

            # Add rows back into shapes
            shapes = pd.concat([shapes, this_shape])

        # Replace self if in_place, else return
        if in_place:
            self.feed.shapes = shapes
        else:
            return shapes

    def apply_transit_feature_change_stops() -> Union(None, TransitNetwork):
        # TODO

    def apply_transit_feature_change_frequencies(
        self, trip_ids: pd.Series, properties: dict, in_place: bool = True
    ) -> Union(None, TransitNetwork):
        freq = self.feed.frequencies

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
