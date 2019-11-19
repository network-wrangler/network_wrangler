#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import copy

import yaml
import pandas as pd
import geojson
import geopandas as gpd
import json
import networkx as nx
import numpy as np

from pandas.core.frame import DataFrame
from geopandas.geodataframe import GeoDataFrame

from jsonschema import validate
from jsonschema.exceptions import ValidationError
from jsonschema.exceptions import SchemaError

import osmnx as ox

from shapely.geometry import Point, LineString

from .Logger import WranglerLogger
from .Utils import point_df_to_geojson, link_df_to_json, parse_time_spans
from .Utils import offset_lat_lon, haversine_distance
from .ProjectCard import ProjectCard


class RoadwayNetwork(object):
    """
    Representation of a Roadway Network.
    """

    CRS = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    EPSG = 4326

    NODE_FOREIGN_KEY = "osm_node_id"
    LINK_FOREIGN_KEY = ("u","v")

    SEARCH_BREADTH = 5
    MAX_SEARCH_BREADTH = 10
    SP_WEIGHT_FACTOR = 100
    MANAGED_LANES_NODE_ID_SCALAR = 500000
    MANAGED_LANES_LINK_ID_SCALAR = 1000000

    SELECTION_REQUIRES = ["link"]

    UNIQUE_MODEL_LINK_IDENTIFIERS = ["model_link_id", "ShStReferenceId"]
    UNIQUE_NODE_IDENTIFIERS = ["model_node_id"]

    MANAGED_LANES_REQUIRED_ATTRIBUTES = ["A", "B", "model_link_id", "locationReferences"]

    KEEP_SAME_ATTRIBUTES_ML_AND_GP = ["distance", "bike_access", "drive_access",
        "transit_access", "walk_access", "maxspeed", "name", "oneway",
        "ref", "highway", "length"
    ]

    MANAGED_LANES_SCALAR = 500000

    def __init__(self, nodes: GeoDataFrame, links: GeoDataFrame, shapes: GeoDataFrame):
        """
        Constructor
        """

        if not RoadwayNetwork.validate_object_types(nodes, links, shapes):
            sys.exit("RoadwayNetwork: Invalid constructor data type")

        self.nodes_df = nodes
        self.links_df = links
        self.shapes_df = shapes

        # Add non-required fields if they aren't there.
        # for field, default_value in RoadwayNetwork.OPTIONAL_FIELDS:
        #    if field not in self.links_df.columns:
        #        self.links_df[field] = default_value

        self.selections = {}

    @staticmethod
    def read(
        link_file: str, node_file: str, shape_file: str, fast: bool = True
    ) -> RoadwayNetwork:
        ##TODO turn off fast=True as default
        """
        Reads a network from the roadway network standard
        Validates that it conforms to the schema

        args:
        link_file: full path to the link file
        node_file: full path to the node file
        shape_file: full path to the shape file
        fast: boolean that will skip validation to speed up read time
        """
        if not fast:
            if not (
                RoadwayNetwork.validate_node_schema(node_file)
                and RoadwayNetwork.validate_link_schema(link_file)
                and RoadwayNetwork.validate_shape_schema(shape_file)
            ):

                sys.exit("RoadwayNetwork: Data doesn't conform to schema")

        with open(link_file) as f:
            link_json = json.load(f)

        link_properties = pd.DataFrame(link_json)
        link_geometries = [
            LineString(
                [
                    g["locationReferences"][0]["point"],
                    g["locationReferences"][1]["point"],
                ]
            )
            for g in link_json
        ]
        links_df = gpd.GeoDataFrame(link_properties, geometry=link_geometries)
        links_df.crs = RoadwayNetwork.CRS

        shapes_df = gpd.read_file(shape_file)
        shapes_df.crs = RoadwayNetwork.CRS

        # geopandas uses fiona OGR drivers, which doesn't let you have
        # a list as a property type. Therefore, must read in node_properties
        # separately in a vanilla dataframe and then convert to geopandas

        with open(node_file) as f:
            node_geojson = json.load(f)

        node_properties = pd.DataFrame(
            [g["properties"] for g in node_geojson["features"]]
        )
        node_geometries = [
            Point(g["geometry"]["coordinates"]) for g in node_geojson["features"]
        ]

        nodes_df = gpd.GeoDataFrame(node_properties, geometry=node_geometries)

        nodes_df.gdf_name = "network_nodes"

        # set a copy of the  foreign key to be the index so that the
        # variable itself remains queryiable
        nodes_df[RoadwayNetwork.NODE_FOREIGN_KEY+"_idx"]=nodes_df[RoadwayNetwork.NODE_FOREIGN_KEY]
        nodes_df.set_index(RoadwayNetwork.NODE_FOREIGN_KEY+"_idx", inplace=True)

        nodes_df.crs = RoadwayNetwork.CRS
        nodes_df["x"] = nodes_df["geometry"].apply(lambda g: g.x)
        nodes_df["y"] = nodes_df["geometry"].apply(lambda g: g.y)
        # todo: flatten json

        WranglerLogger.info("Read %s links from %s" % (links_df.size, link_file))
        WranglerLogger.info("Read %s nodes from %s" % (nodes_df.size, node_file))
        WranglerLogger.info("Read %s shapes from %s" % (shapes_df.size, shape_file))

        roadway_network = RoadwayNetwork(
            nodes=nodes_df, links=links_df, shapes=shapes_df
        )

        return roadway_network

    def write(self, path: str = ".", filename: str = None) -> None:
        """
        Writes a network in the roadway network standard

        args:
        path: the path were the output will be saved
        filename: the name prefix of the roadway files that will be generated
        """

        if not os.path.exists(path):
            WranglerLogger.debug("\nPath [%s] doesn't exist; creating." % path)
            os.mkdir(path)

        if filename:
            links_file = os.path.join(path, filename + "_" + "link.json")
            nodes_file = os.path.join(path, filename + "_" + "node.geojson")
            shapes_file = os.path.join(path, filename + "_" + "shape.geojson")
        else:
            links_file = os.path.join(path, "link.json")
            nodes_file = os.path.join(path, "node.geojson")
            shapes_file = os.path.join(path, "shape.geojson")

        link_property_columns = self.links_df.columns.values.tolist()
        link_property_columns.remove("geometry")
        links_json = link_df_to_json(self.links_df, link_property_columns)
        with open(links_file, "w") as f:
            json.dump(links_json, f)

        # geopandas wont let you write to geojson because
        # it uses fiona, which doesn't accept a list as one of the properties
        # so need to convert the df to geojson manually first
        property_columns = self.nodes_df.columns.values.tolist()
        property_columns.remove("geometry")

        nodes_geojson = point_df_to_geojson(self.nodes_df, property_columns)

        with open(nodes_file, "w") as f:
            json.dump(nodes_geojson, f)

        self.shapes_df.to_file(shapes_file, driver="GeoJSON")

    @staticmethod
    def validate_object_types(
        nodes: GeoDataFrame, links: GeoDataFrame, shapes: GeoDataFrame
    ):
        """
        Determines if the roadway network is being built with the right object types.
        Returns: boolean

        Does not validate schemas.
        """

        errors = ""

        if not isinstance(nodes, GeoDataFrame):
            error_message = "Incompatible nodes type:{}. Must provide a GeoDataFrame.  ".format(
                type(nodes)
            )
            WranglerLogger.error(error_message)
            errors.append(error_message)
        if not isinstance(links, GeoDataFrame):
            error_message = "Incompatible links type:{}. Must provide a GeoDataFrame.  ".format(
                type(links)
            )
            WranglerLogger.error(error_message)
            errors.append(error_message)
        if not isinstance(shapes, GeoDataFrame):
            error_message = "Incompatible shapes type:{}. Must provide a GeoDataFrame.  ".format(
                type(shapes)
            )
            WranglerLogger.error(error_message)
            errors.append(error_message)

        if errors:
            return False
        return True

    @staticmethod
    def validate_node_schema(
        node_file, schema_location: str = "roadway_network_node.json"
    ):
        """
        Validate roadway network data node schema and output a boolean
        """
        if not os.path.exists(schema_location):
            base_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "schemas"
            )
            schema_location = os.path.join(base_path, schema_location)

        with open(schema_location) as schema_json_file:
            schema = json.load(schema_json_file)

        with open(node_file) as node_json_file:
            json_data = json.load(node_json_file)

        try:
            validate(json_data, schema)
            return True

        except ValidationError as exc:
            WranglerLogger.error("Failed Node schema validation: Validation Error")
            WranglerLogger.error("Node File Loc:{}".format(node_file))
            WranglerLogger.error("Node Schema Loc:{}".format(schema_location))
            WranglerLogger.error(exc.message)

        except SchemaError as exc:
            WranglerLogger.error("Failed Node schema validation: Schema Error")
            WranglerLogger.error("Node Schema Loc:{}".format(schema_location))
            WranglerLogger.error(exc.message)

        return False

    @staticmethod
    def validate_link_schema(
        link_file, schema_location: str = "roadway_network_link.json"
    ):
        """
        Validate roadway network data link schema and output a boolean
        """

        if not os.path.exists(schema_location):
            base_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "schemas"
            )
            schema_location = os.path.join(base_path, schema_location)

        with open(schema_location) as schema_json_file:
            schema = json.load(schema_json_file)

        with open(link_file) as link_json_file:
            json_data = json.load(link_json_file)

        try:
            validate(json_data, schema)
            return True

        except ValidationError as exc:
            WranglerLogger.error("Failed Link schema validation: Validation Error")
            WranglerLogger.error("Link File Loc:{}".format(link_file))
            WranglerLogger.error("Path:{}".format(exc.path))
            WranglerLogger.error(exc.message)

        except SchemaError as exc:
            WranglerLogger.error("Failed Link schema validation: Schema Error")
            WranglerLogger.error("Link Schema Loc: {}".format(schema_location))
            WranglerLogger.error(exc.message)

        return False

    @staticmethod
    def validate_shape_schema(
        shape_file, schema_location: str = "roadway_network_shape.json"
    ):
        """
        Validate roadway network data shape schema and output a boolean
        """

        if not os.path.exists(schema_location):
            base_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "schemas"
            )
            schema_location = os.path.join(base_path, schema_location)

        with open(schema_location) as schema_json_file:
            schema = json.load(schema_json_file)

        with open(shape_file) as shape_json_file:
            json_data = json.load(shape_json_file)

        try:
            validate(json_data, schema)
            return True

        except ValidationError as exc:
            WranglerLogger.error("Failed Shape schema validation: Validation Error")
            WranglerLogger.error("Shape File Loc:{}".format(shape_file))
            WranglerLogger.error("Path:{}".format(exc.path))
            WranglerLogger.error(exc.message)

        except SchemaError as exc:
            WranglerLogger.error("Failed Shape schema validation: Schema Error")
            WranglerLogger.error("Shape Schema Loc: {}".format(schema_location))
            WranglerLogger.error(exc.message)

        return False

    def validate_selection(self, selection: dict) -> Bool:
        """
        Evaluate whetther the selection dictionary contains the
        minimum required values.

        Parameters
        -----------
        selection : dict
            selection dictionary to be evaluated

        Returns
        -------
        boolean value as to whether the selection dictonary is valid.

        """
        if not set(RoadwayNetwork.SELECTION_REQUIRES).issubset(selection):
            err_msg = "Project Card Selection requires: {}".format(
                ",".join(RoadwayNetwork.SELECTION_REQUIRES)
            )
            err_msg += ", but selection only contains: {}".format(",".join(selection))
            WranglerLogger.error(err_msg)
            raise KeyError(err_msg)

        err = []
        for l in selection["link"]:
            for k, v in l.items():
                if k not in self.links_df.columns:
                    err.append(
                        "{} specified in link selection but not an attribute in network\n".format(
                            k
                        )
                    )
        selection_keys = [k for l in selection["link"] for k, v in l.items()]
        unique_link_id = bool(set(RoadwayNetwork.UNIQUE_MODEL_LINK_IDENTIFIERS).intersection(set(selection_keys)))

        if not unique_link_id:
            for k, v in selection["A"].items():
                if k not in self.nodes_df.columns and k != RoadwayNetwork.NODE_FOREIGN_KEY:
                    err.append(
                        "{} specified in A node selection but not an attribute in network\n".format(
                            k
                        )
                    )
            for k, v in selection["B"].items():
                if k not in self.nodes_df.columns and k != RoadwayNetwork.NODE_FOREIGN_KEY:
                    err.append(
                        "{} specified in B node selection but not an attribute in network\n".format(
                            k
                        )
                    )
        if err:
            WranglerLogger.error(
                "ERROR: Selection variables in project card not found in network"
            )
            WranglerLogger.error("\n".join(err))
            WranglerLogger.error(
                "--existing node columns:{}".format(" ".join(self.nodes_df.columns))
            )
            WranglerLogger.error(
                "--existing link columns:{}".format(" ".join(self.links_df.columns))
            )
            raise ValueError()
            return False
        else:
            return True

    def orig_dest_nodes_foreign_key(
        self, selection: dict, node_foreign_key: str = ""
    ) -> tuple:
        """
        Returns the foreign key id (whatever is used in the u and v
        variables in the links file) for the AB nodes as a tuple.

        Parameters
        -----------
        selection : dict
            selection dictionary with A and B keys
        node_foreign_key: str
            variable name for whatever is used by the u and v variable
            in the links_df file.  If nothing is specified, assume whatever
            default is (usually osm_node_id)
        """

        if not node_foreign_key:
            node_foreign_key = RoadwayNetwork.NODE_FOREIGN_KEY
        if len(selection["A"]) > 1:
            raise ("Selection A node dictionary should be of length 1")
        if len(selection["B"]) > 1:
            raise ("Selection B node dictionary should be of length 1")

        A_node_key, A_id = next(iter(selection["A"].items()))
        B_node_key, B_id = next(iter(selection["B"].items()))

        if A_node_key != node_foreign_key:
            A_id = self.nodes_df[self.nodes_df[A_node_key] == A_id][
                node_foreign_key
            ].values[0]
        if B_node_key != node_foreign_key:
            B_id = self.nodes_df[self.nodes_df[B_node_key] == B_id][
                node_foreign_key
            ].values[0]

        return (A_id, B_id)

    @staticmethod
    def get_managed_lane_node_ids(nodes_list):
        return [x + RoadwayNetwork.MANAGED_LANES_SCALAR for x in nodes_list]

    @staticmethod
    def ox_graph(nodes_df, links_df):
        """
        create an osmnx-flavored network graph

        osmnx doesn't like values that are arrays, so remove the variables
        that have arrays.  osmnx also requires that certain variables
        be filled in, so do that too.

        Parameters
        ----------
        nodes_df : GeoDataFrame
        link_df : GeoDataFrame

        Returns
        -------
        networkx multidigraph
        """
        WranglerLogger.debug("starting ox_graph()")
        try:
            graph_nodes = nodes_df.drop(
                ["inboundReferenceId", "outboundReferenceId"], axis=1
            )
        except:
            graph_nodes = nodes_df

        graph_nodes.gdf_name = "network_nodes"
        WranglerLogger.debug("GRAPH NODES: {}".format(graph_nodes.columns))
        graph_nodes['id'] = graph_nodes['osm_node_id']

        graph_links = links_df.copy()
        graph_links['id'] = graph_links['osm_link_id']
        graph_links['key'] = str(graph_links['osm_link_id'])+"_"+str(graph_links['model_link_id'])

        WranglerLogger.debug("starting ox.gdfs_to_graph()")

        G = ox.gdfs_to_graph(graph_nodes, graph_links)

        WranglerLogger.debug("finished ox.gdfs_to_graph()")
        return G

    @staticmethod
    def selection_has_unique_link_id(selection_dict):
        selection_keys = [k for l in selection_dict["link"] for k, v in l.items()]
        return bool(set(RoadwayNetwork.UNIQUE_MODEL_LINK_IDENTIFIERS).intersection(set(selection_keys)))


    def build_selection_key(self, selection_dict):
        sel_query = ProjectCard.build_link_selection_query(
            selection=selection_dict,
            unique_model_link_identifiers=RoadwayNetwork.UNIQUE_MODEL_LINK_IDENTIFIERS
        )

        if RoadwayNetwork.selection_has_unique_link_id(selection_dict):
            return (sel_query)

        A_id, B_id = self.orig_dest_nodes_foreign_key(selection_dict)
        return (sel_query, A_id, B_id)


    def select_roadway_features(
        self, selection: dict, search_mode="drive"
    ) -> GeoDataFrame:
        """
        Selects roadway features that satisfy selection criteria

        Example usage:
            net.select_roadway_features(
              selection = [ {
                #   a match condition for the from node using osm,
                #   shared streets, or model node number
                'from': {'osm_model_link_id': '1234'},
                #   a match for the to-node..
                'to': {'shstid': '4321'},
                #   a regex or match for facility condition
                #   could be # of lanes, facility type, etc.
                'facility': {'name':'Main St'},
                }, ... ])

        Parameters
        ------------
        selection : dictionary
            With keys for:
             A - from node
             B - to node
             link - which includes at least a variable for `name`

        Returns
        -------
        shortest path node route : list
           list of foreign IDs of nodes in the selection route
        """
        WranglerLogger.debug("validating selection")
        self.validate_selection(selection)

        # create a unique key for the selection so that we can cache it
        sel_key = self.build_selection_key(selection)
        WranglerLogger.debug("Selection Key: {}".format(sel_key))

        # if this selection has been queried before, just return the
        # previously selected links
        if sel_key in self.selections:
            if self.selections[sel_key]["selection_found"]:
                return self.selections[sel_key]["selected_links"].index.tolist()
            else:
                msg = "Selection previously queried but no selection found"
                WranglerLogger.error(msg)
                raise Exception(msg)
        self.selections[sel_key] = {}
        self.selections[sel_key]["selection_found"] = False

        unique_model_link_identifer_in_selection = RoadwayNetwork.selection_has_unique_link_id(selection)
        if not unique_model_link_identifer_in_selection:
            A_id, B_id = self.orig_dest_nodes_foreign_key(selection)
        # identify candidate links which match the initial query
        # assign them as iteration = 0
        # subsequent iterations that didn't match the query will be
        # assigned a heigher weight in the shortest path
        WranglerLogger.debug("Building selection query")
        # build a selection query based on the selection dictionary
        modes_to_network_variables = {
            "drive": "drive_access",
            "transit": "transit_access",
            "walk": "walk_access",
            "bike": "bike_access",
        }

        sel_query = ProjectCard.build_link_selection_query(
            selection=selection,
            unique_model_link_identifiers=RoadwayNetwork.UNIQUE_MODEL_LINK_IDENTIFIERS,
            mode=modes_to_network_variables[search_mode]
        )
        WranglerLogger.debug("Selecting features:\n{}".format(sel_query))

        self.selections[sel_key]["candidate_links"] = self.links_df.query(
            sel_query, engine="python"
        )
        WranglerLogger.debug("Completed query")
        candidate_links = self.selections[sel_key][
            "candidate_links"
        ]  # b/c too long to keep that way

        candidate_links["i"] = 0

        if len(candidate_links.index) == 0 and unique_model_link_identifer_in_selection:
            msg = "No links found based on unique link identifiers.\nSelection Failed."
            WranglerLogger.error(msg)
            raise Exception(msg)

        if len(candidate_links.index) == 0:
            WranglerLogger.info("No candidate links in initial search.\nRetrying query using 'ref' instead of 'name'")
            # if the query doesn't come back with something from 'name'
            # try it again with 'ref' instead
            selection_has_name_key = any("name" in d for d in selection["link"])

            if not selection_has_name_key:
                msg = "Not able to complete search using 'ref' instead of 'name' because 'name' not in search."
                WranglerLogger.error(msg)
                raise Exception(msg)

            if not "ref" in self.links_df.columns:
                msg = "Not able to complete search using 'ref' because 'ref' not in network."
                WranglerLogger.error(msg)
                raise Exception(msg)

            WranglerLogger.debug("Trying selection query replacing 'name' with 'ref'")
            sel_query = sel_query.replace("name", "ref")

            self.selections[sel_key]["candidate_links"] = self.links_df.query(
                sel_query, engine="python"
            )
            candidate_links = self.selections[sel_key][
                "candidate_links"
            ]

            candidate_links["i"] = 0

            if len(candidate_links.index) == 0:
                msg = "No candidate links in search using either 'name' or 'ref' in query.\nSelection Failed."
                WranglerLogger.error(msg)
                raise Exception(msg)

        def _add_breadth(candidate_links, nodes, links, i):
            """
            Add outbound and inbound reference IDs to candidate links
            from existing nodes

            Parameters
            -----------
            candidate_links : GeoDataFrame
                df with the links from the previous iteration that we
                want to add on to

            nodes : GeoDataFrame
                df of all nodes in the full network

            links : GeoDataFrame
                df of all links in the full network

            i : int
                iteration of adding breadth

            Returns
            -------
            candidate_links : GeoDataFrame
                updated df with one more degree of added breadth

            node_list_foreign_keys : list
                list of foreign key ids for nodes in the updated candidate links
                to test if the A and B nodes are in there.
            """
            WranglerLogger.debug("-Adding Breadth-")
            node_list_foreign_keys = list(
                set(list(candidate_links["u"]) + list(candidate_links["v"]))
            )
            candidate_nodes = nodes.loc[node_list_foreign_keys]
            WranglerLogger.debug("Candidate Nodes: {}".format(len(candidate_nodes)))
            links_shstRefId_to_add = list(
                set(
                    sum(candidate_nodes["outboundReferenceId"].tolist(), [])
                    + sum(candidate_nodes["inboundReferenceId"].tolist(), [])
                )
                - set(candidate_links["shstRefId"].tolist())
                - set([""])
            )
            ##TODO make unique ID for links in the settings
            #print("Link IDs to add: {}".format(links_shstRefId_to_add))
            # print("Links: ", links_id_to_add)
            links_to_add = links[links.shstRefId.isin(links_shstRefId_to_add)]
            #print("Adding Links:",links_to_add)
            WranglerLogger.debug("Adding {} links.".format(links_to_add.shape[0]))
            links[links.model_link_id.isin(links_shstRefId_to_add)]["i"] = i
            candidate_links = candidate_links.append(links_to_add)
            node_list_foreign_keys = list(
                set(list(candidate_links["u"]) + list(candidate_links["v"]))
            )

            return candidate_links, node_list_foreign_keys

        def _shortest_path():
            WranglerLogger.debug("_shortest_path(): calculating shortest path from graph")
            candidate_links["weight"] = 1 + (
                candidate_links["i"] * RoadwayNetwork.SP_WEIGHT_FACTOR
            )
            candidate_nodes = self.nodes_df.loc[
                list(candidate_links["u"]) + list(candidate_links["v"])
            ]
            WranglerLogger.debug("creating network graph")
            G = RoadwayNetwork.ox_graph(candidate_nodes, candidate_links)

            try:
                WranglerLogger.debug("calculating NX shortest path")
                sp_route = nx.shortest_path(G, A_id, B_id, weight="weight")
                self.selections[sel_key]["candidate_links"] = candidate_links
                sp_links = candidate_links[
                    candidate_links["u"].isin(sp_route)
                    & candidate_links["v"].isin(sp_route)
                ]
                self.selections[sel_key] = {
                    "route": sp_route,
                    "links": sp_links,
                    "graph": G,
                }
                return True
            except:
                return False

        if not unique_model_link_identifer_in_selection:
            # find the node ids for the candidate links
            WranglerLogger.debug("Not a unique ID selection, conduct search")
            node_list_foreign_keys = list(candidate_links["u"]) + list(candidate_links["v"])
            WranglerLogger.debug("Foreign key list: {}".format(node_list_foreign_keys))
            i = 0

            max_i = RoadwayNetwork.SEARCH_BREADTH

            while (
                A_id not in node_list_foreign_keys
                and B_id not in node_list_foreign_keys
                and i <= max_i
            ):
                WranglerLogger.debug("Adding breadth, no shortest path. i: {}, Max i: {}".format(i, max_i))
                i += 1
                candidate_links, node_list_foreign_keys = _add_breadth(
                    candidate_links, self.nodes_df, self.links_df, i
                )
            WranglerLogger.debug("calculating shortest path from graph")
            sp_found = _shortest_path()
            if not sp_found:
                WranglerLogger.info(
                    "No shortest path found with {}, trying greater breadth until SP found".format(
                        i
                    )
                )
            while not sp_found and i <= RoadwayNetwork.MAX_SEARCH_BREADTH:
                WranglerLogger.debug(
                    "Adding breadth, with shortest path iteration. i: {} Max i: {}".format(i, max_i)
                )
                i += 1
                candidate_links, node_list_foreign_keys = _add_breadth(
                    candidate_links, self.nodes_df, self.links_df, i
                )
                sp_found = _shortest_path()

            if sp_found:
                # reselect from the links in the shortest path, the ones with
                # the desired values....ignoring name.
                if len(selection["link"]) > 1:
                    resel_query = ProjectCard.build_link_selection_query(
                        selection=selection,
                        unique_model_link_identifiers=RoadwayNetwork.UNIQUE_MODEL_LINK_IDENTIFIERS,
                        mode=modes_to_network_variables[search_mode],
                        ignore=["name"],
                    )
                    WranglerLogger.info("Reselecting features:\n{}".format(resel_query))
                    self.selections[sel_key]["selected_links"] = self.selections[sel_key][
                        "links"
                    ].query(resel_query, engine="python")
                else:
                    self.selections[sel_key]["selected_links"] = self.selections[sel_key][
                        "links"
                    ]

                self.selections[sel_key]["selection_found"] = True
                # Return pandas.Series of links_ids
                return self.selections[sel_key]["selected_links"].index.tolist()
            else:
                WranglerLogger.error("Couldn't find path from {} to {}".format(A_id, B_id))
                raise ValueError
        else:
            # unique identifier exists and no need to go through big search
            self.selections[sel_key]["selected_links"] = self.selections[sel_key][
                "candidate_links"
            ]
            self.selections[sel_key]["selection_found"] = True

            return self.selections[sel_key]["selected_links"].index.tolist()

    def validate_properties(
        self,
        properties: dict,
        ignore_existing: bool = False,
        require_existing_for_change: bool = False,
    ) -> bool:
        """
        If there are change or existing commands, make sure that that
        property exists in the network.

        Parameters
        -----------
        properties : dict
            properties dictionary to be evaluated
        ignore_existing: bool
            If True, will only warn about properties that specify an "existing"
            value.  If False, will fail.
        require_existing_for_change: bool
            If True, will fail if there isn't a specified value in the
            project card for existing when a change is specified.
        Returns
        -------
        boolean value as to whether the properties dictonary is valid.
        """

        validation_error_message = []

        for p in properties:
            if p["property"] not in self.links_df.columns:
                if p.get("change"):
                    validation_error_message.append(
                        '"Change" is specified for attribute {}, but doesn\'t exist in base network\n'.format(
                            p["property"]
                        )
                    )

                if p.get("existing") and not ignore_existing:
                    validation_error_message.append(
                        '"Existing" is specified for attribute {}, but doesn\'t exist in base network\n'.format(
                            p["property"]
                        )
                    )
                elif p.get("existing"):
                    WranglerLogger.warning(
                        '"Existing" is specified for attribute {}, but doesn\'t exist in base network\n'.format(
                            p["property"]
                        )
                    )

            if p.get("change") and not p.get("existing"):
                if require_existing_for_change:
                    validation_error_message.append(
                        '"Change" is specified for attribute {}, but there isn\'t a value for existing.\nTo proceed, run with the setting require_existing_for_change=False'.format(
                            p["property"]
                        )
                    )
                else:
                    WranglerLogger.warning(
                        '"Change" is specified for attribute {}, but there isn\'t a value for existing.\n'.format(
                            p["property"]
                        )
                    )

        if validation_error_message:
            WranglerLogger.error(" ".join(validation_error_message))
            raise ValueError()

    def apply(self, project_card_dictionary: dict):
        """
        Wrapper method to apply a project to a roadway network.

        args
        ------
        project_card_dictionary: dict
          a dictionary of the project card object

        """

        WranglerLogger.info(
            "Applying Project to Roadway Network: {}".format(
                project_card_dictionary["project"]
            )
        )

        def _apply_individual_change(project_dictionary: dict):

            if project_dictionary["category"].lower() == "roadway property change":
                self.apply_roadway_feature_change(
                    self.select_roadway_features(project_dictionary["facility"]),
                    project_dictionary["properties"],
                )
            elif project_dictionary["category"].lower() == "parallel managed lanes":
                self.apply_managed_lane_feature_change(
                    self.select_roadway_features(project_dictionary["facility"]),
                    project_dictionary["properties"],
                )
            elif project_dictionary["category"].lower() == "add new roadway":
                self.add_new_roadway_feature_change(
                    project_dictionary.get("links"), project_dictionary.get("nodes")
                )
            elif project_dictionary["category"].lower() == "roadway deletion":
                self.delete_roadway_feature_change(
                    project_dictionary.get("links"), project_dictionary.get("nodes")
                )
            else:
                raise (BaseException)

        if project_card_dictionary.get("changes"):
            for project_dictionary in project_card_dictionary["changes"]:
                _apply_individual_change(project_dictionary)
        else:
            _apply_individual_change(project_card_dictionary)

    def apply_roadway_feature_change(
        self, link_idx: list, properties: dict, in_place: bool = True
    ) -> Union(None, RoadwayNetwork):
        """
        Changes the roadway attributes for the selected features based on the
        project card information passed

        args:
        link_idx : list
            lndices of all links to apply change to
        properties : list of dictionarys
            roadway properties to change
        in_place: boolean
            update self or return a new roadway network object
        """

        # check if there are change or existing commands that that property
        #   exists in the network
        # if there is a set command, add that property to network
        self.validate_properties(properties)

        for i, p in enumerate(properties):
            attribute = p["property"]

            # if project card specifies an existing value in the network
            #   check and see if the existing value in the network matches
            if p.get("existing"):
                network_values = self.links_df.loc[link_idx, attribute].tolist()
                if not set(network_values).issubset([p.get("existing")]):
                    WranglerLogger.warning(
                        "Existing value defined for {} in project card does "
                        "not match the value in the roadway network for the "
                        "selected links".format(attribute)
                    )

            if in_place:
                if "set" in p.keys():
                    self.links_df.loc[link_idx, attribute] = p["set"]
                else:
                    self.links_df.loc[link_idx, attribute] = (
                        self.links_df.loc[link_idx, attribute] + p["change"]
                    )
            else:
                if i == 0:
                    updated_network = copy.deepcopy(self)

                if "set" in p.keys():
                    updated_network.links_df.loc[link_idx, attribute] = p["set"]
                else:
                    updated_network.links_df.loc[link_idx, attribute] = (
                        updated_network.links_df.loc[link_idx, attribute] + p["change"]
                    )

                if i == len(properties) - 1:
                    return updated_network

    def apply_managed_lane_feature_change(
        self, link_idx: list, properties: dict, in_place: bool = True
    ) -> Union(None, RoadwayNetwork):
        """
        Apply the managed lane feature changes to the roadway network

        link_idx : list
            lndices of all links to apply change to
        properties : list of dictionarys
            roadway properties to change
        in_place: boolean
            update self or return a new roadway network object
        """

        # flag ML links
        self.links_df['ML'] = 0
        self.links_df.loc[link_idx, 'ML'] = 1

        for p in properties:
            attribute = p["property"]

            if "group" in p.keys():
                attr_value = {}
                attr_value["default"] = p["set"]
                attr_value["timeofday"] = []
                for g in p["group"]:
                    category = g["category"]
                    for tod in g["timeofday"]:
                        attr_value["timeofday"].append(
                            {
                                "category": category,
                                "time": parse_time_spans(tod["time"]),
                                "value": tod["set"],
                            }
                        )

            elif "timeofday" in p.keys():
                attr_value = {}
                attr_value["default"] = p["set"]
                attr_value["timeofday"] = []
                for tod in p["timeofday"]:
                    attr_value["timeofday"].append(
                        {"time": parse_time_spans(tod["time"]), "value": tod["set"]}
                    )

            elif "set" in p.keys():
                attr_value = p["set"]

            else:
                attr_value = ""

            # TODO: decide on connectors info when they are more specific in project card
            if attribute == "ML_ACCESS" and attr_value == "all":
                attr_value = 1

            if attribute == "ML_EGRESS" and attr_value == "all":
                attr_value = 1

            if in_place:
                self.links_df.loc[link_idx, attribute] = attr_value
            else:
                if i == 0:
                    updated_network = copy.deepcopy(self)

                updated_network.links_df.loc[link_idx, attribute] = attr_value

                if i == len(properties) - 1:
                    return updated_network

    def add_new_roadway_feature_change(self, links: dict, nodes: dict) -> None:
        """
        add the new roadway features defined in the project card

        args:
        links : dict
            list of dictionaries
        nodes : dict
            list of dictionaries
        """

        # TODO:
        # validate links dictonary

        # CHECKS:
        # check if new link model_link_id already exists?
        # check if u and v nodes are already present or not?

        def _add_dict_to_df(df, new_dict):
            df_column_names = df.columns
            new_row_to_add = {}

            # add the fields from project card that are in the network
            for property in df_column_names:
                if property in new_dict.keys():
                    if df[property].dtype == np.float64:
                        value = pd.to_numeric(new_dict[property], downcast="float")
                    elif df[property].dtype == np.int64:
                        value = pd.to_numeric(new_dict[property], downcast="integer")
                    else:
                        value = str(new_dict[property])
                else:
                    value = ""

                new_row_to_add[property] = value

            # add the fields from project card that are NOT in the network
            for key, value in new_dict.items():
                if key not in df_column_names:
                    new_row_to_add[key] = new_dict[key]

            out_df = df.append(new_row_to_add, ignore_index=True)
            return out_df

        if nodes is not None:
            for node in nodes:
                self.nodes_df = _add_dict_to_df(self.nodes_df, node)

        if links is not None:
            for link in links:
                self.links_df = _add_dict_to_df(self.links_df, link)

    def delete_roadway_feature_change(
        self,
        links: dict,
        nodes: dict,
        ignore_missing=True
    ) -> None:
        """
        delete the roadway features defined in the project card

        args:
        links : dict
            list of dictionaries
        nodes : dict
            list of dictionaries
        ignore_missing: bool
            If True, will only warn about links/nodes that are missing from
            network but specified to "delete" in project card
            If False, will fail.
        """

        missing_error_message = []

        if links is not None:
            for key, val in links.items():
                missing_links = [v for v in val if v not in self.links_df[key].tolist()]
                if missing_links:
                    message = "Links attribute {} with values as {} does not exist in the network\n".format(
                        key, missing_links)
                    if ignore_missing:
                        WranglerLogger.warning(message)
                    else:
                        missing_error_message.append(message)

                self.links_df = self.links_df[~self.links_df[key].isin(val)]

        if nodes is not None:
            for key, val in nodes.items():
                missing_nodes = [v for v in val if v not in self.nodes_df[key].tolist()]
                if missing_nodes:
                    message = "Nodes attribute {} with values as {} does not exist in the network\n".format(
                        key, missing_links)
                    if ignore_missing:
                        WranglerLogger.warning(message)
                    else:
                        missing_error_message.append(message)

                self.nodes_df = self.nodes_df[~self.nodes_df[key].isin(val)]

        if missing_error_message:
            WranglerLogger.error(" ".join(missing_error_message))
            raise ValueError()

    def get_property_by_time_period_and_group(self, property, time_period = None, category=None):
        '''
        Return a series for the properties with a specific group or time period.

        args
        ------
        property: str
          the variable that you want from network
        time_period: list(str)
          the time period that you are querying for
          i.e. ['16:00', '19:00']
        category: str or list(str)(Optional)
          the group category
          i.e. "sov"

          or

          list of group categories in order of search, i.e.
          ["hov3","hov2"]

        returns
        --------
        pandas series
        '''


        def _get_property(v, time_spans = None, category = None):

            if category and not time_spans:
                WranglerLogger.error("\nShouldn't have a category group without time spans")
                raise ValueError("Shouldn't have a category group without time spans")

            if not category:
                category = ["default"]
            elif isinstance(category, str):
                category = [category]
            search_cats =  [c.lower() for c in category]

            # simple case
            if type(v) in (int, float):
                return v

            #print("VARIABLE:",v)

            # if no time or group specified, but it is a complex link situation
            if not time_spans:
                if v.get("default"):
                    return v["default"]
                else:
                    WranglerLogger.error("\nVariable {} is more complex in network than query".format(v))
                    raise ValueError("Variable {} is more complex in network than query".format(v))

            if v.get("timeofday"):
                categories = []
                for tg in v["timeofday"]:
                    if (tg["time"][0]>= time_spans[0]) and (tg["time"][1]<= time_spans[1]):
                        if tg.get("category"):
                            categories+=(tg["category"])
                            for c in search_cats:
                                print("CAT:", c, tg["category"])
                                if c in tg["category"]:
                                    print("RETURNING:",time_spans,category, tg["value"])
                                    return tg["value"]
                        else:
                            print("RETURNING:",time_spans,category,tg["value"])
                            return tg["value"]

                WranglerLogger.info("\nCouldn't find time period for {}, returning default".format(str(time_spans)))
                if v.get("default"):
                    print("RETURNING:",time_spans, v["default"])
                    return v["default"]
                else:
                    WranglerLogger.error("\nCan't find default; must specify a category in {}".format(str(categories)))
                    raise ValueError("Can't find default, must specify a category in: {}".format(str(categories)))


        time_spans = parse_time_spans(time_period)

        return self.links_df[property].apply(_get_property,time_spans = time_spans, category=category)


    def create_dummy_connector_links(gp_df: GeoDataFrame, ml_df: GeoDataFrame):
        """
        create dummy connector links between the general purpose and managed lanes

        args:
        gp_df : GeoDataFrame
            dataframe of general purpose links (where managed lane also exists)
        ml_df : GeoDataFrame
            dataframe of corresponding managed lane links
        """

        gp_ml_links_df = pd.concat([gp_df, gp_df.add_prefix('ML_')], axis=1, join='inner')

        access_df = pd.DataFrame(columns=gp_df.columns.values.tolist())
        egress_df = pd.DataFrame(columns=gp_df.columns.values.tolist())

        def _get_connector_references(ref_1: list, ref_2: list, type: str):
            if type == "access":
                out_location_reference = [
                    {'sequence': 1, 'point': ref_1[0]["point"]},
                    {'sequence': 2, 'point': ref_2[0]["point"]}
                ]

            if type == "egress":
                out_location_reference = [
                    {'sequence': 1, 'point': ref_2[1]["point"]},
                    {'sequence': 2, 'point': ref_1[1]["point"]}
                ]
            return(out_location_reference)

        for index, row in gp_ml_links_df.iterrows():
            access_row = {}
            access_row["A"] = row["A"]
            access_row["B"] = row["ML_A"]
            access_row["lanes"] = 1
            access_row["model_link_id"] = row["model_link_id"] + row["ML_model_link_id"] + 1
            access_row["access"] = row["ML_access"]
            access_row["drive_access"] = row["drive_access"]
            access_row["locationReferences"] = _get_connector_references(
                row["locationReferences"], row["ML_locationReferences"], "access"
            )
            access_row["distance"] = haversine_distance(
                access_row["locationReferences"][0]["point"],
                access_row["locationReferences"][1]["point"],
            )
            access_row["highway"] = "ml_access"
            access_row["oneway"] = row["oneway"]
            access_row["name"] = row["name"]
            access_row["ref"] = row["ref"]
            access_df = access_df.append(access_row, ignore_index=True)

            egress_row = {}
            egress_row["A"] = row["ML_B"]
            egress_row["B"] = row["B"]
            egress_row["lanes"] = 1
            egress_row["model_link_id"] = row["model_link_id"] + row["ML_model_link_id"] + 2
            egress_row["access"] = row["ML_access"]
            egress_row["drive_access"] = row["drive_access"]
            egress_row["locationReferences"] = _get_connector_references(
                row["locationReferences"], row["ML_locationReferences"], "egress"
            )
            egress_row["distance"] = haversine_distance(
                egress_row["locationReferences"][0]["point"],
                egress_row["locationReferences"][1]["point"],
            )
            egress_row["highway"] = "ml_egress"
            egress_row["oneway"] = row["oneway"]
            egress_row["name"] = row["name"]
            egress_row["ref"] = row["ref"]
            egress_df = egress_df.append(egress_row, ignore_index=True)

        return(access_df, egress_df)


    def create_managed_lane_network(self, in_place = False) -> RoadwayNetwork:
        """
        Create a roadway network with managed lanes links separated out

        args
        ------
        in_place: boolean
            update self or return a new roadway network object

        returns
        --------
        Roadway Network
        """

        link_attributes = self.links_df.columns.values.tolist()
        ml_attributes = [i for i in link_attributes if i.startswith('ML_')]

        non_ml_links_df = self.links_df[self.links_df["ML"]==0]
        non_ml_links_df = non_ml_links_df.drop(ml_attributes, axis = 1)

        ml_links_df = self.links_df[self.links_df["ML"]==1]
        gp_links_df = ml_links_df.drop(ml_attributes, axis = 1)

        for attr in link_attributes:
            if attr in ml_attributes and attr not in ["ML_ACCESS", "ML_EGRESS"]:
                gp_attr = attr.split("_")[1]
                ml_links_df[gp_attr] = ml_links_df[attr]

            if attr not in RoadwayNetwork.KEEP_SAME_ATTRIBUTES_ML_AND_GP and attr not in RoadwayNetwork.MANAGED_LANES_REQUIRED_ATTRIBUTES:
                ml_links_df[attr] = ""

        ml_links_df =  ml_links_df.drop(ml_attributes, axis = 1)

        def _update_location_reference(location_reference: list):
            out_location_reference = copy.deepcopy(location_reference)
            out_location_reference[0]["point"] = offset_lat_lon(out_location_reference[0]["point"])
            out_location_reference[1]["point"] = offset_lat_lon(out_location_reference[1]["point"])
            return(out_location_reference)

        def _get_line_string(location_reference: list):
            return(
                LineString(
                    [
                        location_reference[0]["point"],
                        location_reference[1]["point"]
                    ]
                )
            )

        ml_links_df["A"] = ml_links_df["A"] + RoadwayNetwork.MANAGED_LANES_NODE_ID_SCALAR
        ml_links_df["B"] = ml_links_df["B"] + RoadwayNetwork.MANAGED_LANES_NODE_ID_SCALAR
        ml_links_df["model_link_id"] = ml_links_df["model_link_id"] + RoadwayNetwork.MANAGED_LANES_LINK_ID_SCALAR
        ml_links_df["locationReferences"] = ml_links_df["locationReferences"].apply(
            lambda x : _update_location_reference(x)
        )

        access_links_df, egress_links_df = RoadwayNetwork.create_dummy_connector_links(
            gp_links_df, ml_links_df
        )

        new_links_df = gp_links_df.append(ml_links_df)
        new_links_df = new_links_df.append(access_links_df)
        new_links_df = new_links_df.append(egress_links_df)
        new_links_df = new_links_df.append(non_ml_links_df)
        new_links_df = new_links_df.drop('ML', axis = 1)

        new_links_df["geometry"] = new_links_df["locationReferences"].apply(
            lambda x : _get_line_string(x)
        )

        #only the ml_links_df has the new nodes added
        added_a_nodes = ml_links_df["A"]
        added_b_nodes = ml_links_df["B"]

        new_nodes_df = self.nodes_df

        for a_node in added_a_nodes:
            new_nodes_df = new_nodes_df.append(
                {"model_node_id": a_node,
                 "geometry": Point(new_links_df[new_links_df["A"]==a_node].iloc[0]["locationReferences"][0]["point"]),
                 "drive_node": 1
                },
                ignore_index=True
            )

        for b_node in added_b_nodes:
            if b_node not in new_nodes_df["travelModelId"].tolist():
                new_nodes_df = new_nodes_df.append(
                    {"model_node_id": b_node,
                     "geometry": Point(new_links_df[new_links_df["B"]==b_node].iloc[0]["locationReferences"][1]["point"]),
                     "drive_node": 1
                    },
                    ignore_index=True
                )

        new_nodes_df["x"] = new_nodes_df["geometry"].apply(lambda g: g.x)
        new_nodes_df["y"] = new_nodes_df["geometry"].apply(lambda g: g.y)

        new_shapes_df = self.shapes_df

        # managed lanes, access and egress connectors are new geometry
        for index, row in ml_links_df.iterrows():
            new_shapes_df = new_shapes_df.append({"geometry": row["geometry"]}, ignore_index=True)
        for index, row in access_links_df.iterrows():
            new_shapes_df = new_shapes_df.append({"geometry": row["geometry"]}, ignore_index=True)
        for index, row in egress_links_df.iterrows():
            new_shapes_df = new_shapes_df.append({"geometry": row["geometry"]}, ignore_index=True)

        if in_place:
            self.links_df = new_links_df
            self.nodes_df = new_nodes_df
            self.shapes_df = new_shapes_df
        else:
            out_network = copy.deepcopy(self)
            out_network.links_df = new_links_df
            out_network.nodes_df = new_nodes_df
            out_network.shapes_df = new_shapes_df
            return out_network
