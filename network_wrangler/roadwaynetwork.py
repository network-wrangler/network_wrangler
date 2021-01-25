#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import copy
import numbers
from random import randint
from typing import Union

import folium
import pandas as pd
import geopandas as gpd
import json
import networkx as nx
import numpy as np
import osmnx as ox

from geopandas.geodataframe import GeoDataFrame

from pandas.core.frame import DataFrame

from jsonschema import validate
from jsonschema.exceptions import ValidationError
from jsonschema.exceptions import SchemaError

from shapely.geometry import Point, LineString

from .logger import WranglerLogger
from .projectcard import ProjectCard
from .utils import point_df_to_geojson, link_df_to_json, parse_time_spans
from .utils import offset_location_reference, haversine_distance, create_unique_shape_id
from .utils import create_location_reference_from_nodes, create_line_string
from .utils import update_df

# Coordinate reference system
CRS = 4326  # AKA EPSG:4326, WGS 1984

# Foreign key variables in data model
NODE_FOREIGN_KEY = "model_node_id"
LINK_FOREIGN_KEY = ["A", "B"]
SHAPE_FOREIGN_KEY = "id"  # formerly UNIQUE_SHAPE_ID

# List of variables that are unique such that they can be directly queried and you should get a single item returned
UNIQUE_LINK_IDS = ["model_link_id"]
UNIQUE_NODE_IDS = ["model_node_id"]

# Minimum requirements for selection query
SELECTION_REQUIRES = ["link"]

# Scalar values added to primary keys for nodes and links for corresponding managed lanes
# MTC
MANAGED_LANES_NODE_ID_SCALAR = 4500000
MANAGED_LANES_LINK_ID_SCALAR = 10000000

# Required attributes to specify for managed lanes
MANAGED_LANES_REQUIRED_ATTRIBUTES = [
    "A",
    "B",
    "model_link_id",
    "locationReferences",
]

# Attributes to copy (keep same) from general purpose lanes to corresponding managed lanes
KEEP_SAME_ATTRIBUTES_ML_AND_GP = [
    "distance",
    "bike_access",
    "drive_access",
    "transit_access",
    "walk_access",
    "maxspeed",
    "name",
    "oneway",
    "ref",
    "roadway",
    "length",
    "segment_id",
    "ft",
    "assignable",
]

# Mapping of modes to variables in the network
MODES_TO_NETWORK_LINK_VARIABLES = {
    "drive": ["drive_access"],
    "bus": ["bus_only", "drive_access"],
    "rail": ["rail_only"],
    "transit": ["bus_only", "rail_only", "drive_access"],
    "walk": ["walk_access"],
    "bike": ["bike_access"],
}

MODES_TO_NETWORK_NODE_VARIABLES = {
    "drive": ["drive_access"],
    "rail": ["rail_only", "drive_access"],
    "bus": ["bus_only", "drive_access"],
    "transit": ["bus_only", "rail_only", "drive_access"],
    "walk": ["walk_node"],
    "bike": ["bike_node"],
}

# Primary keys used in graph creation
UNIQUE_LINK_KEY = "model_link_id"
UNIQUE_NODE_KEY = "model_node_id"

# Shortest Path settings for finding routes
SEARCH_BREADTH = 5
MAX_SEARCH_BREADTH = 10
SP_WEIGHT_FACTOR = 100


class NoPathFound(Exception):
    """Raised when can't find path."""

    pass


class RoadwayNetwork(object):
    """
    Representation of a Roadway Network.

    .. highlight:: python

    Typical usage example:
    ::

        net = RoadwayNetwork.read(
            link_filename=MY_LINK_FILE,
            node_filename=MY_NODE_FILE,
            shape_filename=MY_SHAPE_FILE,
            shape_foreign_key ='shape_id',
        )
        my_selection = {
            "link": [{"name": ["I 35E"]}],
            "A": {"osm_node_id": "961117623"},  # start searching for segments at A
            "B": {"osm_node_id": "2564047368"},
        }
        net.select_roadway_features(my_selection)

        my_change = [
            {
                'property': 'lanes',
                'existing': 1,
                'set': 2,
             },
             {
                'property': 'drive_access',
                'set': 0,
              },
        ]

        my_net.apply_roadway_feature_change(
            my_net.select_roadway_features(my_selection),
            my_change
        )

        ml_net = net.create_managed_lane_network(in_place=False)
        ml_net.is_network_connected(mode="drive"))
        _, disconnected_nodes = ml_net.assess_connectivity(mode="walk", ignore_end_nodes=True)
        ml_net.write(filename=my_out_prefix, path=my_dir)

    Attributes:
        nodes_df (GeoDataFrame): node data
        links_df (GeoDataFrame): link data, including start and end
            nodes and associated shape
        shapes_df (GeoDataFrame): detailed shape data
        crs (int): coordinate reference system, ESPG number
        node_foreign_key (str):  variable linking the node table to the link table
        link_foreign_key (list): list of variable linking the link table to the node foreign key
        shape_foreign_key (str): variable linking the links table and shape table
        unique_link_ids (list): list of variables unique to each link
        unique_node_ids (list): list of variables unique to each node
        modes_to_network_link_variables (dict): Mapping of modes to link variables in the network
        modes_to_network_nodes_variables (dict): Mapping of modes to node variables in the network
        managed_lanes_node_id_scalar (int): Scalar values added to primary keys for nodes for
            corresponding managed lanes.
        managed_lanes_link_id_scalar (int): Scalar values added to primary keys for links for
            corresponding managed lanes.
        managed_lanes_required_attributes (list): attributes that must be specified in managed
            lane projects.
        keep_same_attributes_ml_and_gp (list): attributes to copy to managed lanes from parallel
            general purpose lanes.

        selections (dict): dictionary storing selections in case they are made repeatedly
    """

    def __init__(
        self,
        nodes: GeoDataFrame,
        links: GeoDataFrame,
        shapes: GeoDataFrame = None,
        node_foreign_key: str = None,
        link_foreign_key: str = None,
        shape_foreign_key: str = None,
        unique_link_key: str = None,
        unique_node_key: str = None,
        unique_link_ids: list = None,
        unique_node_ids: list = None,
        crs: int = None,
        **kwargs,
    ):
        """
        Constructor
        """
        inputs_valid = [isinstance(x,GeoDataFrame) for x in (nodes, links, shapes)]
        if False in inputs_valid:
            raise(TypeError("Input nodes ({}), links ({})or shapes ({}) not of required type GeoDataFrame".format(inputs_valid)))

        self.nodes_df = nodes
        self.links_df = links
        self.shapes_df = shapes

        self.node_foreign_key = NODE_FOREIGN_KEY if node_foreign_key is None else node_foreign_key
        self.link_foreign_key = LINK_FOREIGN_KEY if link_foreign_key is None else link_foreign_key
        self.shape_foreign_key = SHAPE_FOREIGN_KEY if shape_foreign_key is None else shape_foreign_key

        self.unique_link_key = UNIQUE_LINK_KEY if unique_link_key is None else unique_link_key
        self.unique_node_key = UNIQUE_NODE_KEY if unique_node_key is None else unique_node_key

        self.unique_link_ids = UNIQUE_LINK_IDS if unique_link_ids is None else unique_link_ids
        self.unique_node_ids = UNIQUE_NODE_IDS if unique_node_ids is None else unique_node_ids

        self.crs = CRS if crs is None else crs

        self.__dict__.update(kwargs)

        # Add non-required fields if they aren't there.
        # for field, default_value in RoadwayNetwork.OPTIONAL_FIELDS:
        #    if field not in self.links_df.columns:
        #        self.links_df[field] = default_value
        if not self.validate_uniqueness():
            raise ValueError("IDs in network not unique")
        self.selections = {}

    @staticmethod
    def read(
        link_filename: str,
        node_filename: str,
        shape_filename: str,
        fast: bool = True,
        crs: int = CRS,
        node_foreign_key: str = NODE_FOREIGN_KEY,
        link_foreign_key: list = LINK_FOREIGN_KEY,
        shape_foreign_key: str = SHAPE_FOREIGN_KEY,
        unique_link_key: str = UNIQUE_LINK_KEY,
        unique_node_key: str = UNIQUE_NODE_KEY,
        unique_link_ids: list = UNIQUE_LINK_IDS,
        unique_node_ids: list = UNIQUE_NODE_IDS,
        modes_to_network_link_variables: dict = MODES_TO_NETWORK_LINK_VARIABLES,
        modes_to_network_nodes_variables: dict = MODES_TO_NETWORK_NODE_VARIABLES,
        managed_lanes_link_id_scalar: int = MANAGED_LANES_LINK_ID_SCALAR,
        managed_lanes_node_id_scalar: int = MANAGED_LANES_NODE_ID_SCALAR,
        managed_lanes_required_attributes: list = MANAGED_LANES_REQUIRED_ATTRIBUTES,
        keep_same_attributes_ml_and_gp: list = KEEP_SAME_ATTRIBUTES_ML_AND_GP,
    ) -> RoadwayNetwork:
        """
        Reads a network from the roadway network standard
        Validates that it conforms to the schema

        args:
            node_filename: full path to the node file
            link_filename: full path to the link file
            shape_filename: full path to the shape file
            fast: boolean that will skip validation to speed up read time
            crs: coordinate reference system, ESPG number
            node_foreign_key: variable linking the node table to the link table.
            link_foreign_key:
            shape_foreign_key:
            unique_link_ids:
            unique_node_ids:
            modes_to_network_link_variables:
            modes_to_network_nodes_variables:
            managed_lanes_node_id_scalar:
            managed_lanes_link_id_scalar:
            managed_lanes_required_attributes:
            keep_same_attributes_ml_and_gp:

        Returns: a RoadwayNetwork instance

        .. todo:: Turn off fast=True as default
        """

        WranglerLogger.info("Reading RoadwayNetwork")

        nodes_df,links_df,shapes_df = RoadwayNetwork.load_transform_network(
            node_filename,
            link_filename,
            shape_filename,
            crs = crs,
            node_foreign_key = node_foreign_key,
            validate_schema = not fast,
        )

        roadway_network = RoadwayNetwork(
            nodes=nodes_df,
            links=links_df,
            shapes=shapes_df,
            crs=crs,
            node_foreign_key=node_foreign_key,
            link_foreign_key=link_foreign_key,
            shape_foreign_key=shape_foreign_key,
            unique_link_ids=unique_link_ids,
            unique_node_ids=unique_node_ids,
            unique_link_key=unique_link_key,
            unique_node_key=unique_node_key,
            modes_to_network_link_variables=modes_to_network_link_variables,
            modes_to_network_nodes_variables=modes_to_network_nodes_variables,
            link_filename = link_filename,
            node_filename = node_filename,
            shape_filename = shape_filename,
        )

        return roadway_network

    @staticmethod
    def load_transform_network(
        node_filename: str,
        link_filename: str,
        shape_filename: str,
        crs: int = CRS,
        node_foreign_key: str = NODE_FOREIGN_KEY,
        validate_schema: bool = True,
        **kwargs,
    ) -> tuple:
        """
        Reads roadway network files from disk and transforms them into GeoDataFrames.

        args:
            node_filename: file name for nodes.
            link_filename: file name for links.
            shape_filename: file name for shapes.
            crs: coordinate reference system. Defaults to value in CRS.
            node_foreign_key: variable linking the node table to the link table. Defaults
                to NODE_FOREIGN_KEY.
            validate_schema: boolean indicating if network should be validated to schema.

        returns: tuple of GeodataFrames nodes_df, links_df, shapes_df
        """
        WranglerLogger.debug(
            "Reading RoadwayNetwork from following files:\n   -{}\n   -{}\n   -{}.".format(
                link_filename, node_filename, shape_filename
            )
        )

        if validate_schema:
            if not (
                RoadwayNetwork.validate_node_schema(node_filename)
                and RoadwayNetwork.validate_link_schema(link_filename)
                and RoadwayNetwork.validate_shape_schema(shape_filename)
            ):

                raise ValueError("RoadwayNetwork: Data doesn't conform to schema")

        with open(link_filename) as f:
            link_json = json.load(f)

        link_properties = pd.DataFrame(link_json)
        link_geometries = [
            create_line_string(g["locationReferences"]) for g in link_json
        ]
        links_df = gpd.GeoDataFrame(link_properties, geometry=link_geometries)

        links_df.crs = crs
        # coerce types for booleans which might not have a 1 and are therefore read in as intersection
        bool_columns = [
            "rail_only",
            "bus_only",
            "drive_access",
            "bike_access",
            "walk_access",
            "truck_access",
        ]
        for bc in list(set(bool_columns) & set(links_df.columns)):
            links_df[bc] = links_df[bc].astype(bool)

        shapes_df = gpd.read_file(shape_filename)
        shapes_df.dropna(subset=["geometry", "id"], inplace=True)
        shapes_df.crs = crs

        # geopandas uses fiona OGR drivers, which doesn't let you have
        # a list as a property type. Therefore, must read in node_properties
        # separately in a vanilla dataframe and then convert to geopandas

        with open(node_filename) as f:
            node_geojson = json.load(f)

        node_properties_df = pd.DataFrame(
            [g["properties"] for g in node_geojson["features"]]
        )

        if node_foreign_key not in node_properties_df.columns:
            raise ValueError("Specified `node_foreign_key`: {} not found in {}. Available properties: {}".format(
                node_foreign_key,
                node_filename,
                node_properties_df.columns
            ))

        node_geometries = [
            Point(g["geometry"]["coordinates"]) for g in node_geojson["features"]
        ]

        nodes_df = gpd.GeoDataFrame(node_properties_df, geometry=node_geometries)

        nodes_df.gdf_name = "network_nodes"

        # set a copy of the  foreign key to be the index so that the
        # variable itself remains queryiable
        nodes_df[node_foreign_key + "_idx"] = nodes_df[node_foreign_key]
        nodes_df.set_index(node_foreign_key + "_idx", inplace=True)

        nodes_df.crs = crs
        nodes_df["X"] = nodes_df["geometry"].apply(lambda g: g.x)
        nodes_df["Y"] = nodes_df["geometry"].apply(lambda g: g.y)

        WranglerLogger.info("Read %s links from %s" % (len(links_df), link_filename))
        WranglerLogger.info("Read %s nodes from %s" % (len(nodes_df), node_filename))
        WranglerLogger.info("Read %s shapes from %s" % (len(shapes_df), shape_filename))

        return nodes_df, links_df, shapes_df


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
        """
        links_json = link_df_to_json(self.links_df, link_property_columns)
        with open(links_file, "w") as f:
            json.dump(links_json, f)
        """
        links_json = self.links_df[link_property_columns].to_json(orient = "records")

        with open(links_file, 'w') as f:
            f.write(links_json)

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
    def roadway_net_to_gdf(roadway_net: RoadwayNetwork) -> gpd.GeoDataFrame:
        """
        Turn the roadway network into a GeoDataFrame
        args:
            roadway_net: the roadway network to export

        returns: shapes dataframe

        .. todo:: Make this much more sophisticated, for example attach link info to shapes
        """
        return roadway_net.shapes_df

    def validate_uniqueness(self) -> bool:
        """
        Confirms that the unique identifiers are met.
        """
        valid = True
        for c in self.unique_link_ids:
            if c not in self.links_df.columns:
                valid = False
                msg = "Network doesn't contain unique link identifier: {}".format(c)
                WranglerLogger.error(msg)
            if not self.links_df[c].is_unique:
                valid = False
                msg = "Unique identifier {} is not unique in network links".format(c)
                WranglerLogger.error(msg)
        for c in self.link_foreign_key:
            if c not in self.links_df.columns:
                valid = False
                msg = "Network doesn't contain link foreign key identifier: {}".format(
                    c
                )
                WranglerLogger.error(msg)
        link_foreign_key = self.links_df[self.link_foreign_key].apply(
            lambda x: "-".join(x.map(str)), axis=1
        )
        if not link_foreign_key.is_unique:
            valid = False
            msg = "Foreign key: {} is not unique in network links".format(
                self.link_foreign_key
            )
            WranglerLogger.error(msg)
        for c in self.unique_node_ids:
            if c not in self.nodes_df.columns:
                valid = False
                msg = "Network doesn't contain unique node identifier: {}".format(c)
                WranglerLogger.error(msg)
            if not self.nodes_df[c].is_unique:
                valid = False
                msg = "Unique identifier {} is not unique in network nodes".format(c)
                WranglerLogger.error(msg)
        if self.node_foreign_key not in self.nodes_df.columns:
            valid = False
            msg = "Network doesn't contain node foreign key identifier: {}".format(
                self.node_foreign_key
            )
            WranglerLogger.error(msg)
        elif not self.nodes_df[self.node_foreign_key].is_unique:
            valid = False
            msg = "Foreign key: {} is not unique in network nodes".format(
                self.node_foreign_key
            )
            WranglerLogger.error(msg)
        if self.shape_foreign_key not in self.shapes_df.columns:
            valid = False
            msg = "Network doesn't contain unique shape id: {}".format(
                self.shape_foreign_key
            )
            WranglerLogger.error(msg)
        elif not self.shapes_df[self.shape_foreign_key].is_unique:
            valid = False
            msg = "Unique key: {} is not unique in network shapes".format(
                self.shape_foreign_key
            )
            WranglerLogger.error(msg)
        return valid

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
            WranglerLogger.error("Invalid Node Schema")
            WranglerLogger.error("Node Schema Loc:{}".format(schema_location))
            WranglerLogger.error(json.dumps(exc.message, indent=2))

        return False

    @staticmethod
    def validate_link_schema(
        link_filename, schema_location: str = "roadway_network_link.json"
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

        with open(link_filename) as link_json_file:
            json_data = json.load(link_json_file)

        try:
            validate(json_data, schema)
            return True

        except ValidationError as exc:
            WranglerLogger.error("Failed Link schema validation: Validation Error")
            WranglerLogger.error("Link File Loc:{}".format(link_filename))
            WranglerLogger.error("Path:{}".format(exc.path))
            WranglerLogger.error(exc.message)

        except SchemaError as exc:
            WranglerLogger.error("Invalid Link Schema")
            WranglerLogger.error("Link Schema Loc: {}".format(schema_location))
            WranglerLogger.error(json.dumps(exc.message, indent=2))

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
            WranglerLogger.error("Invalid Shape Schema")
            WranglerLogger.error("Shape Schema Loc: {}".format(schema_location))
            WranglerLogger.error(json.dumps(exc.message, indent=2))

        return False

    def validate_selection(
        self, selection: dict, selection_requires: list = SELECTION_REQUIRES
    ) -> bool:
        """
        Evaluate whetther the selection dictionary contains the
        minimum required values.

        Args:
            selection: selection dictionary to be evaluated

        Returns: boolean value as to whether the selection dictonary is valid.
        """
        if not set(selection_requires).issubset(selection):
            err_msg = "Project Card Selection requires: {}".format(
                ",".join(selection_requires)
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
        unique_link_id = bool(
            set(self.unique_link_ids).intersection(set(selection_keys))
        )

        if not unique_link_id:
            for k, v in selection["A"].items():
                if k not in self.nodes_df.columns and k != self.node_foreign_key:
                    err.append(
                        "{} specified in A node selection but not an attribute in network\n".format(
                            k
                        )
                    )
            for k, v in selection["B"].items():
                if k not in self.nodes_df.columns and k != self.node_foreign_key:
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
        else:
            return True

    def orig_dest_nodes_foreign_key(
        self, selection: dict, node_foreign_key: str = ""
    ) -> tuple:
        """
        Returns the foreign key id (whatever is used in the u and v
        variables in the links file) for the AB nodes as a tuple.

        Args:
            selection : selection dictionary with A and B keys
            node_foreign_key: variable name for whatever is used by the u and v variable
            in the links_df file.  If nothing is specified, assume whatever
            default is (usually osm_node_id)

        Returns: tuple of (A_id, B_id)
        """

        if not node_foreign_key:
            node_foreign_key = self.node_foreign_key
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
    def get_managed_lane_node_ids(
        nodes_list: list, scalar: int = MANAGED_LANES_NODE_ID_SCALAR
    ):
        """
        Transform a list of node IDS by a scalar.
        ..todo #237 what if node ids are not a number?

        Args:
            nodes_list: list of integers
            scalar: value to add to node IDs

        Returns: list of integers
        """
        return [x + scalar for x in nodes_list]

    @staticmethod
    def ox_graph(
        nodes_df: GeoDataFrame,
        links_df: GeoDataFrame,
        node_foreign_key: str = NODE_FOREIGN_KEY,
        link_foreign_key: list = LINK_FOREIGN_KEY,
        unique_link_key: str = UNIQUE_LINK_KEY,
    ):
        """
        create an osmnx-flavored network graph

        osmnx doesn't like values that are arrays, so remove the variables
        that have arrays.  osmnx also requires that certain variables
        be filled in, so do that too.

        Args:
            nodes_df : GeoDataFrame of nodes
            link_df : GeoDataFrame of links
            node_foreign_key: field referenced in `link_foreign_key`
            link_foreign_key: list of attributes that define the link start and end nodes to the node foreign key
            unique_link_key: primary key for links

        Returns: a networkx multidigraph
        """
        WranglerLogger.debug("starting ox_graph()")

        graph_nodes = nodes_df.copy().drop(
            ["inboundReferenceIds", "outboundReferenceIds"], axis=1
        )

        graph_nodes.gdf_name = "network_nodes"
        WranglerLogger.debug("GRAPH NODES: {}".format(graph_nodes.columns))
        graph_nodes["id"] = graph_nodes[node_foreign_key]

        graph_nodes["x"] = graph_nodes["X"]
        graph_nodes["y"] = graph_nodes["Y"]

        graph_links = links_df.copy().drop(
            ["osm_link_id", "locationReferences"], axis=1
        )

        # have to change this over into u,v b/c this is what osm-nx is expecting
        graph_links["u"] = graph_links[link_foreign_key[0]]
        graph_links["v"] = graph_links[link_foreign_key[1]]
        graph_links["id"] = graph_links[unique_link_key]
        graph_links["key"] = graph_links[unique_link_key]

        WranglerLogger.debug("starting ox.gdfs_to_graph()")
        try:
            G = ox.graph_from_gdfs(graph_nodes, graph_links)
        except AttributeError:
            WranglerLogger.debug(
                "Please upgrade your OSMNX package. For now, using depricated osmnx.gdfs_to_graph because osmnx.graph_from_gdfs not found"
            )
            G = ox.gdfs_to_graph(graph_nodes, graph_links)

        WranglerLogger.debug("finished ox.gdfs_to_graph()")
        return G

    def selection_has_unique_link_id(
        self,
        selection_dict: dict,
    ) -> bool:
        """
        Args:
            selection_dictionary: Dictionary representation of selection
                of roadway features, containing a "link" key.

        Returns: A boolean indicating if the selection dictionary contains
            a unique identifier for links.

        """
        selection_keys = [k for l in selection_dict["link"] for k, v in l.items()]
        return bool(
            set(self.unique_link_ids) & set(selection_keys)
        )

    def build_selection_key(self, selection_dict: dict) -> tuple:
        """
        Selections are stored by a key combining the query and the A and B ids.
        This method combines the two for you based on the selection dictionary.

        Args:
            selection_dictonary: Selection Dictionary

        Returns: Tuple serving as the selection key.

        """
        sel_query = ProjectCard.build_link_selection_query(
            selection=selection_dict,
            unique_link_ids=self.unique_link_ids,
        )

        if self.selection_has_unique_link_id(selection_dict):
            return sel_query

        A_id, B_id = self.orig_dest_nodes_foreign_key(selection_dict)
        return (sel_query, A_id, B_id)

    @staticmethod
    def _get_fk_nodes(
        _links: gpd.GeoDataFrame, link_foreign_key: list = LINK_FOREIGN_KEY
    ):
        """Find the nodes for the candidate links.
        """
        _n = list(set([i for fk in link_foreign_key for i in list(_links[fk])]))
        # WranglerLogger.debug("Node foreign key list: {}".format(_n))
        return _n

    def shortest_path(
        self,
        graph_links_df: gpd.GeoDataFrame,
        O_id,
        D_id,
        nodes_df: gpd.GeoDataFrame = None,
        weight_column: str = "i",
        weight_factor: float = SP_WEIGHT_FACTOR,
    ) -> tuple:
        """

        Args:
            graph_links_df:
            O_id: foreign key for start node
            D_id: foreign key for end node
            nodes_df: optional nodes df, otherwise will use network instance
            weight_column: column to use as a weight, defaults to "i"
            weight_factor: any additional weighting to multiply the weight column by, defaults to SP_WEIGHT_FACTOR

        Returns: tuple with length of four
        - Boolean if shortest path found
        - nx Directed graph of graph links
        - route of shortest path nodes as List
        - links in shortest path selected from links_df
        """
        WranglerLogger.debug(
            "Calculating shortest path from {} to {} using {} as weight with a factor of {}".format(
                O_id, D_id, weight_column, weight_factor
            )
        )

        # Prep Graph Links
        if weight_column not in graph_links_df.columns:
            WranglerLogger.warning(
                "{} not in graph_links_df so adding and initializing to 1.".format(
                    weight_column
                )
            )
            graph_links_df[weight_column] = 1

        graph_links_df.loc[:, "weight"] = 1 + (
            graph_links_df[weight_column] * weight_factor
        )

        # Select Graph Nodes
        node_list_foreign_keys = RoadwayNetwork._get_fk_nodes(
            graph_links_df, link_foreign_key=self.link_foreign_key
        )

        if O_id not in node_list_foreign_keys:
            msg = "O_id: {} not in Graph for finding shortest Path".format(O_id)
            WranglerLogger.error(msg)
            raise ValueError(msg)
        if D_id not in node_list_foreign_keys:
            msg = "D_id: {} not in Graph for finding shortest Path".format(D_id)
            WranglerLogger.error(msg)
            raise ValueError(msg)

        if not nodes_df:
            nodes_df = self.nodes_df
        graph_nodes_df = nodes_df.loc[node_list_foreign_keys]

        # Create Graph
        WranglerLogger.debug("Creating network graph")
        G = RoadwayNetwork.ox_graph(
            graph_nodes_df,
            graph_links_df,
            node_foreign_key=self.node_foreign_key,
            link_foreign_key=self.link_foreign_key,
            unique_link_key=self.unique_link_key,
        )

        try:
            sp_route = nx.shortest_path(G, O_id, D_id, weight="weight")
            WranglerLogger.debug("Shortest path successfully routed")
        except nx.NetworkXNoPath:
            WranglerLogger.debug("No SP from {} to {} Found.".format(O_id, D_id))
            return False, G, graph_links_df, None, None

        sp_links = graph_links_df[
            graph_links_df["A"].isin(sp_route) & graph_links_df["B"].isin(sp_route)
        ]

        return True, G, graph_links_df, sp_route, sp_links

    def path_search(
        self,
        candidate_links_df: gpd.GeoDataFrame,
        O_id,
        D_id,
        weight_column: str = "i",
        weight_factor: float = 1.0,
        search_breadth: int = SEARCH_BREADTH,
        max_search_breadth: int = MAX_SEARCH_BREADTH,
    ):
        """

        Args:
            candidate_links: selection of links geodataframe with links likely to be part of path
            O_id: origin node foreigh key ID
            D_id: destination node foreigh key ID
            weight_column: column to use for weight of shortest path. Defaults to "i" (iteration)
            weight_factor: optional weight to multiply the weight column by when finding the shortest path
            search_breadth:

        Returns

        """

        def _add_breadth(
            _candidate_links_df: gpd.GeoDataFrame,
            _nodes_df: gpd.GeoDataFrame,
            _links_df: gpd.GeoDataFrame,
            i: int = None,
        ):
            """
            Add outbound and inbound reference IDs to candidate links
            from existing nodes

            Args:
                _candidate_links_df : df with the links from the previous iteration
                _nodes_df : df of all nodes in the full network
                _links_df : df of all links in the full network
                i : iteration of adding breadth

            Returns:
                candidate_links : GeoDataFrame
                    updated df with one more degree of added breadth

                node_list_foreign_keys : list of foreign key ids for nodes in the updated candidate links
                    to test if the A and B nodes are in there.
            """
            WranglerLogger.debug("-Adding Breadth-")

            if not i:
                WranglerLogger.warning("i not specified in _add_breadth, using 1")
                i = 1

            _candidate_nodes_df = _nodes_df.loc[
                RoadwayNetwork._get_fk_nodes(
                    _candidate_links_df, link_foreign_key=self.link_foreign_key
                )
            ]
            WranglerLogger.debug("Candidate Nodes: {}".format(len(_candidate_nodes_df)))

            # Identify links to add based on outbound and inbound links from nodes
            _links_shstRefId_to_add = list(
                set(
                    sum(_candidate_nodes_df["outboundReferenceIds"].tolist(), [])
                    + sum(_candidate_nodes_df["inboundReferenceIds"].tolist(), [])
                )
                - set(_candidate_links_df["shstReferenceId"].tolist())
                - set([""])
            )
            _links_to_add_df = _links_df[
                _links_df.shstReferenceId.isin(_links_shstRefId_to_add)
            ]

            WranglerLogger.debug("Adding {} links.".format(_links_to_add_df.shape[0]))

            # Add information about what iteration the link was added in
            _links_df[_links_df.model_link_id.isin(_links_shstRefId_to_add)]["i"] = i

            # Append links and update node list
            _candidate_links_df = _candidate_links_df.append(_links_to_add_df)
            _node_list_foreign_keys = RoadwayNetwork._get_fk_nodes(
                _candidate_links_df, link_foreign_key=self.link_foreign_key
            )

            return _candidate_links_df, _node_list_foreign_keys

        # -----------------------------------
        # Set search breadth to zero + set max
        # -----------------------------------
        i = 0
        max_i = search_breadth
        # -----------------------------------
        # Add links to the graph until
        #   (i) the A and B nodes are in the
        #       foreign key list
        #          - OR -
        #   (ii) reach maximum search breadth
        # -----------------------------------
        node_list_foreign_keys = RoadwayNetwork._get_fk_nodes(
            candidate_links_df, link_foreign_key=self.link_foreign_key
        )

        WranglerLogger.debug("Initial set of nodes: {}".format(node_list_foreign_keys))
        while (
            O_id not in node_list_foreign_keys or D_id not in node_list_foreign_keys
        ) and i <= max_i:
            WranglerLogger.debug(
                "Adding breadth - i: {}, Max i: {}] - {} and {} not found in node list.".format(
                    i, max_i, O_id, D_id
                )
            )
            i += 1
            candidate_links_df, node_list_foreign_keys = _add_breadth(
                candidate_links_df, self.nodes_df, self.links_df, i=i,
            )
        # -----------------------------------
        #  Once have A and B in graph,
        #  Try calculating shortest path
        # -----------------------------------
        WranglerLogger.debug("calculating shortest path from graph")
        (
            sp_found,
            graph,
            candidate_links_df,
            shortest_path_route,
            shortest_path_links,
        ) = self.shortest_path(candidate_links_df, O_id, D_id)
        if sp_found:
            return graph, candidate_links_df, shortest_path_route, shortest_path_links

        if not sp_found:
            WranglerLogger.debug(
                "No shortest path found with breadth of {}, trying greater breadth until SP found or max breadth {} reached.".format(
                    i, max_i
                )
            )
        while not sp_found and i <= max_search_breadth:
            WranglerLogger.debug(
                "Adding breadth, with shortest path iteration. i: {} Max i: {}".format(
                    i, max_i
                )
            )
            i += 1
            candidate_links_df, node_list_foreign_keys = _add_breadth(
                candidate_links_df, self.nodes_df, self.links_df, i=i
            )
            (
                sp_found,
                graph,
                candidate_links_df,
                route,
                shortest_path_links,
            ) = self.shortest_path(candidate_links_df, O_id, D_id)

        if sp_found:
            return graph, candidate_links_df, route, shortest_path_links

        if not sp_found:
            msg = "Couldn't find path from {} to {} after adding {} links in breadth".format(
                O_id, D_id, i
            )
            WranglerLogger.error(msg)
            raise NoPathFound(msg)

    def select_roadway_features(
        self,
        selection: dict,
        search_mode="drive",
        force_search=False,
        sp_weight_factor = None,
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

        Args:
            selection : dictionary with keys for:
                 A - from node
                 B - to node
                 link - which includes at least a variable for `name`
            search_mode: mode which you are searching for; defaults to "drive"
            force_search: boolean directing method to perform search even if one
                with same selection dict is stored from a previous search.
            sp_weight_factor: multiple used to discourage shortest paths which
                meander from original search returned from name or ref query.
                If not set here, will default to value of sp_weight_factor in
                RoadwayNetwork instance. If not set there, will defaul to SP_WEIGHT_FACTOR.

        Returns: a list of link IDs in selection
        """
        WranglerLogger.debug("validating selection")
        self.validate_selection(selection)

        if not sp_weight_factor:
            sp_weight_factor = self.__dict__.get("sp_weight_factor")
        if not sp_weight_factor:
            sp_weight_factor = SP_WEIGHT_FACTOR

        # create a unique key for the selection so that we can cache it
        sel_key = self.build_selection_key(selection)
        WranglerLogger.debug("Selection Key: {}".format(sel_key))

        # if this selection has been queried before, just return the
        # previously selected links
        if sel_key in self.selections and not force_search:
            if self.selections[sel_key]["selection_found"]:
                return self.selections[sel_key]["selected_links"].index.tolist()
            else:
                msg = "Selection previously queried but no selection found"
                WranglerLogger.error(msg)
                raise Exception(msg)
        self.selections[sel_key] = {}
        self.selections[sel_key]["selection_found"] = False

        unique_link_identifer_in_selection = self.selection_has_unique_link_id(selection)

        if not unique_link_identifer_in_selection:
            A_id, B_id = self.orig_dest_nodes_foreign_key(selection)
        # identify candidate links which match the initial query
        # assign them as iteration = 0
        # subsequent iterations that didn't match the query will be
        # assigned a heigher weight in the shortest path
        WranglerLogger.debug("Building selection query")
        # build a selection query based on the selection dictionary

        sel_query = ProjectCard.build_link_selection_query(
            selection=selection,
            unique_link_ids=self.unique_link_ids,
            mode=self.modes_to_network_link_variables[search_mode],
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

        if len(candidate_links.index) == 0 and unique_link_identifer_in_selection:
            msg = "No links found based on unique link identifiers.\nSelection Failed."
            WranglerLogger.error(msg)
            raise Exception(msg)

        if len(candidate_links.index) == 0:
            WranglerLogger.debug(
                "No candidate links in initial search.\nRetrying query using 'ref' instead of 'name'"
            )
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
            candidate_links = self.selections[sel_key]["candidate_links"]

            candidate_links["i"] = 0

            if len(candidate_links.index) == 0:
                msg = "No candidate links in search using either 'name' or 'ref' in query.\nSelection Failed."
                WranglerLogger.error(msg)
                raise Exception(msg)

        if unique_link_identifer_in_selection:
            # unique identifier exists and no need to go through big search
            self.selections[sel_key]["selected_links"] = self.selections[sel_key][
                "candidate_links"
            ]
            self.selections[sel_key]["selection_found"] = True

            return self.selections[sel_key]["selected_links"].index.tolist()

        else:
            WranglerLogger.debug("Not a unique ID selection, conduct search.")
            (
                self.selections[sel_key]["graph"],
                self.selections[sel_key]["candidate_links"],
                self.selections[sel_key]["route"],
                self.selections[sel_key]["links"],
            ) = self.path_search(
                self.selections[sel_key]["candidate_links"],
                A_id,
                B_id,
                weight_factor=sp_weight_factor,
            )

            if len(selection["link"]) == 1:
                self.selections[sel_key]["selected_links"] = self.selections[sel_key][
                    "links"
                ]

            # Conduct a "selection on the selection" if have additional requirements to satisfy
            else:
                resel_query = ProjectCard.build_link_selection_query(
                    selection=selection,
                    unique_link_ids=self.unique_link_ids,
                    mode=self.modes_to_network_link_variables[search_mode],
                    ignore=["name"],
                )
                WranglerLogger.debug("Reselecting features:\n{}".format(resel_query))
                self.selections[sel_key]["selected_links"] = self.selections[sel_key][
                    "links"
                ].query(resel_query, engine="python")

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

        Args:
            properties : properties dictionary to be evaluated
            ignore_existing: If True, will only warn about properties
                that specify an "existing" value.  If False, will fail.
            require_existing_for_change: If True, will fail if there isn't
                a specified value in theproject card for existing when a
                change is specified.

        Returns: boolean value as to whether the properties dictonary is valid.
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

        Args:
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
            elif project_dictionary["category"].lower() == "calculated roadway":
                self.apply_python_calculation(project_dictionary["pycode"])
            else:
                raise (BaseException)

        if project_card_dictionary.get("changes"):
            for project_dictionary in project_card_dictionary["changes"]:
                _apply_individual_change(project_dictionary)
        else:
            _apply_individual_change(project_card_dictionary)

    def apply_python_calculation(
        self, pycode: str, in_place: bool = True
    ) -> Union(None, RoadwayNetwork):
        """
        Changes roadway network object by executing pycode.

        Args:
            pycode: python code which changes values in the roadway network object
            in_place: update self or return a new roadway network object
        """
        exec(pycode)

    def apply_roadway_feature_change(
        self, link_idx: list, properties: dict, in_place: bool = True
    ) -> Union(None, RoadwayNetwork):
        """
        Changes the roadway attributes for the selected features based on the
        project card information passed

        Args:
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

        Args:
            link_idx : list of lndices of all links to apply change to
            properties : list of dictionarys roadway properties to change
            in_place: boolean to indicate whether to update self or return
                a new roadway network object

        .. todo:: decide on connectors info when they are more specific in project card
        """

        # add ML flag
        if "managed" in self.links_df.columns:
            self.links_df.loc[link_idx, "managed"] = 1
        else:
            self.links_df["managed"] = 0
            self.links_df.loc[link_idx, "managed"] = 1

        p = 1

        for p in properties:
            attribute = p["property"]
            attr_value = ""

            for idx in link_idx:
                if "group" in p.keys():
                    attr_value = {}

                    if "set" in p.keys():
                        attr_value["default"] = p["set"]
                    elif "change" in p.keys():
                        attr_value["default"] = (
                            self.links_df.at[idx, attribute] + p["change"]
                        )

                    attr_value["timeofday"] = []

                    for g in p["group"]:
                        category = g["category"]
                        for tod in g["timeofday"]:
                            if "set" in tod.keys():
                                attr_value["timeofday"].append(
                                    {
                                        "category": category,
                                        "time": parse_time_spans(tod["time"]),
                                        "value": tod["set"],
                                    }
                                )
                            elif "change" in tod.keys():
                                attr_value["timeofday"].append(
                                    {
                                        "category": category,
                                        "time": parse_time_spans(tod["time"]),
                                        "value": self.links_df.at[idx, attribute]
                                        + tod["change"],
                                    }
                                )

                elif "timeofday" in p.keys():
                    attr_value = {}

                    if "set" in p.keys():
                        attr_value["default"] = p["set"]
                    elif "change" in p.keys():
                        attr_value["default"] = (
                            self.links_df.at[idx, attribute] + p["change"]
                        )

                    attr_value["timeofday"] = []

                    for tod in p["timeofday"]:
                        if "set" in tod.keys():
                            attr_value["timeofday"].append(
                                {
                                    "time": parse_time_spans(tod["time"]),
                                    "value": tod["set"],
                                }
                            )
                        elif "change" in tod.keys():
                            attr_value["timeofday"].append(
                                {
                                    "time": parse_time_spans(tod["time"]),
                                    "value": self.links_df.at[idx, attribute]
                                    + tod["change"],
                                }
                            )
                elif "set" in p.keys():
                    attr_value = p["set"]

                elif "change" in p.keys():
                    attr_value = self.links_df.at[idx, attribute] + p["change"]

                # TODO: decide on connectors info when they are more specific in project card
                if attribute == "ML_ACCESS" and attr_value == "all":
                    attr_value = 1

                if attribute == "ML_EGRESS" and attr_value == "all":
                    attr_value = 1

                if in_place:
                    if attribute in self.links_df.columns and not isinstance(
                        attr_value, numbers.Number
                    ):
                        # if the attribute already exists
                        # and the attr value we are trying to set is not numeric
                        # then change the attribute type to object
                        self.links_df[attribute] = self.links_df[attribute].astype(
                            object
                        )

                    if attribute not in self.links_df.columns:
                        # if it is a new attribute then initialize with NaN values
                        self.links_df[attribute] = "NaN"

                    self.links_df.at[idx, attribute] = attr_value

                else:
                    if i == 1:
                        updated_network = copy.deepcopy(self)

                    if attribute in self.links_df.columns and not isinstance(
                        attr_value, numbers.Number
                    ):
                        # if the attribute already exists
                        # and the attr value we are trying to set is not numeric
                        # then change the attribute type to object
                        updated_network.links_df[attribute] = updated_network.links_df[
                            attribute
                        ].astype(object)

                    if attribute not in updated_network.links_df.columns:
                        # if it is a new attribute then initialize with NaN values
                        updated_network.links_df[attribute] = "NaN"

                    updated_network.links_df.at[idx, attribute] = attr_value

                    if p == len(properties):
                        return updated_network
                    else:
                        p = p + 1

    def add_new_roadway_feature_change(self, links: dict, nodes: dict) -> None:
        """
        add the new roadway features defined in the project card.
        new shapes are also added for the new roadway links.

        args:
            links : list of dictionaries
            nodes : list of dictionaries

        returns: None

        .. todo:: validate links and nodes dictionary
        """

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
                if node.get(self.node_foreign_key) is None:
                    msg = "New link to add doesn't contain link foreign key identifier: {}".format(
                        self.node_foreign_key
                    )
                    WranglerLogger.error(msg)
                    raise ValueError(msg)

                node_query = (
                    self.unique_node_key + " == " + str(node[self.node_foreign_key])
                )
                if not self.nodes_df.query(node_query, engine="python").empty:
                    msg = "Node with id = {} already exist in the network".format(
                        node[self.node_foreign_key]
                    )
                    WranglerLogger.error(msg)
                    raise ValueError(msg)

            for node in nodes:
                self.nodes_df = _add_dict_to_df(self.nodes_df, node)

        if links is not None:
            for link in links:
                for key in self.link_foreign_key:
                    if link.get(key) is None:
                        msg = "New link to add doesn't contain link foreign key identifier: {}".format(
                            key
                        )
                        WranglerLogger.error(msg)
                        raise ValueError(msg)

                ab_query = "A == " + str(link["A"]) + " and B == " + str(link["B"])

                if not self.links_df.query(ab_query, engine="python").empty:
                    msg = "Link with A = {} and B = {} already exist in the network".format(
                        link["A"], link["B"]
                    )
                    WranglerLogger.error(msg)
                    raise ValueError(msg)

                if self.nodes_df[
                    self.nodes_df[self.unique_node_key] == link["A"]
                ].empty:
                    msg = "New link to add has A node = {} but the node does not exist in the network".format(
                        link["A"]
                    )
                    WranglerLogger.error(msg)
                    raise ValueError(msg)

                if self.nodes_df[
                    self.nodes_df[self.unique_node_key] == link["B"]
                ].empty:
                    msg = "New link to add has B node = {} but the node does not exist in the network".format(
                        link["B"]
                    )
                    WranglerLogger.error(msg)
                    raise ValueError(msg)

            for link in links:
                link["new_link"] = 1
                self.links_df = _add_dict_to_df(self.links_df, link)

            # add location reference and geometry for new links
            self.links_df["locationReferences"] = self.links_df.apply(
                lambda x: create_location_reference_from_nodes(
                    self.nodes_df[
                        self.nodes_df[self.node_foreign_key] == x["A"]
                    ].squeeze(),
                    self.nodes_df[
                        self.nodes_df[self.node_foreign_key] == x["B"]
                    ].squeeze(),
                )
                if x["new_link"] == 1
                else x["locationReferences"],
                axis=1,
            )
            self.links_df["geometry"] = self.links_df.apply(
                lambda x: create_line_string(x["locationReferences"])
                if x["new_link"] == 1
                else x["geometry"],
                axis=1,
            )

            self.links_df[self.shape_foreign_key] = self.links_df.apply(
                lambda x: create_unique_shape_id(x["geometry"])
                if x["new_link"] == 1
                else x[self.shape_foreign_key],
                axis=1,
            )

            # add new shapes
            added_links = self.links_df[self.links_df["new_link"] == 1]

            added_shapes_df = pd.DataFrame({"geometry": added_links["geometry"]})
            added_shapes_df[self.shape_foreign_key] = added_shapes_df["geometry"].apply(
                lambda x: create_unique_shape_id(x)
            )
            self.shapes_df = self.shapes_df.append(added_shapes_df)

            self.links_df.drop(["new_link"], axis=1, inplace=True)

    def delete_roadway_feature_change(
        self, links: dict, nodes: dict, ignore_missing=True
    ) -> None:
        """
        delete the roadway features defined in the project card.
        valid links and nodes defined in the project gets deleted
        and shapes corresponding to the deleted links are also deleted.

        Args:
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
            shapes_to_delete = []
            for key, val in links.items():
                missing_links = [v for v in val if v not in self.links_df[key].tolist()]
                if missing_links:
                    message = "Links attribute {} with values as {} does not exist in the network\n".format(
                        key, missing_links
                    )
                    if ignore_missing:
                        WranglerLogger.warning(message)
                    else:
                        missing_error_message.append(message)

                deleted_links = self.links_df[self.links_df[key].isin(val)]
                shapes_to_delete.extend(deleted_links[self.shape_foreign_key].tolist())

                self.links_df.drop(
                    self.links_df.index[self.links_df[key].isin(val)], inplace=True
                )

            self.shapes_df.drop(
                self.shapes_df.index[
                    self.shapes_df[self.shape_foreign_key].isin(shapes_to_delete)
                ],
                inplace=True,
            )

        if nodes is not None:
            for key, val in nodes.items():
                missing_nodes = [v for v in val if v not in self.nodes_df[key].tolist()]
                if missing_nodes:
                    message = "Nodes attribute {} with values as {} does not exist in the network\n".format(
                        key, missing_links
                    )
                    if ignore_missing:
                        WranglerLogger.warning(message)
                    else:
                        missing_error_message.append(message)

                self.nodes_df = self.nodes_df[~self.nodes_df[key].isin(val)]

        if missing_error_message:
            WranglerLogger.error(" ".join(missing_error_message))
            raise ValueError()

    def get_property_by_time_period_and_group(
        self, prop, time_period=None, category=None, default_return=None
    ):
        """
        Return a series for the properties with a specific group or time period.

        args
        ------
        prop: str
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
        default_return: what to return if variable or time period not found. Default is None.

        returns
        --------
        pandas series
        """

        if prop not in list(self.links_df.columns):
            WranglerLogger.warning("Property {} not in links to split, returning as default value: {}".format(prop, default_value))
            return pd.Series([default_return]*len(self.link_df))

        def _get_property(
            v,
            time_spans=None,
            category=None,
            return_partial_match: bool = False,
            partial_match_minutes: int = 60,
        ):
            """

            .. todo:: return the time period with the largest overlap

            """

            if category and not time_spans:
                WranglerLogger.error(
                    "\nShouldn't have a category group without time spans"
                )
                raise ValueError("Shouldn't have a category group without time spans")

            # simple case
            if type(v) in (int, float, str):
                return v

            if not category:
                category = ["default"]
            elif isinstance(category, str):
                category = [category]
            search_cats = [c.lower() for c in category]

            # if no time or group specified, but it is a complex link situation
            if not time_spans:
                if "default" in v.keys():
                    return v["default"]
                else:
                    WranglerLogger.debug("variable: ".format(v))
                    msg = "Variable {} is more complex in network than query".format(v)
                    WranglerLogger.error(msg)
                    raise ValueError(msg)

            if v.get("timeofday"):
                categories = []
                for tg in v["timeofday"]:
                    if (time_spans[0] >= tg["time"][0]) and (
                        time_spans[1] <= tg["time"][1]
                    ):
                        if tg.get("category"):
                            categories += tg["category"]
                            for c in search_cats:
                                print("CAT:", c, tg["category"])
                                if c in tg["category"]:
                                    # print("Var:", v)
                                    # print(
                                    #    "RETURNING:", time_spans, category, tg["value"]
                                    # )
                                    return tg["value"]
                        else:
                            # print("Var:", v)
                            # print("RETURNING:", time_spans, category, tg["value"])
                            return tg["value"]

                    # if there isn't a fully matched time period, see if there is an overlapping one
                    # right now just return the first overlapping ones
                    # TODO return the time period with the largest overlap

                    if (
                        (time_spans[0] >= tg["time"][0])
                        and (time_spans[0] <= tg["time"][1])
                    ) or (
                        (time_spans[1] >= tg["time"][0])
                        and (time_spans[1] <= tg["time"][1])
                    ):
                        overlap_minutes = max(
                            0,
                            min(tg["time"][1], time_spans[1])
                            - max(time_spans[0], tg["time"][0]),
                        )
                        # print("OLM",overlap_minutes)
                        if not return_partial_match and overlap_minutes > 0:
                            WranglerLogger.debug(
                                "Couldn't find time period consistent with {}, but found a partial match: {}. Consider allowing partial matches using 'return_partial_match' keyword or updating query.".format(
                                    time_spans, tg["time"]
                                )
                            )
                        elif (
                            overlap_minutes < partial_match_minutes
                            and overlap_minutes > 0
                        ):
                            WranglerLogger.debug(
                                "Time period: {} overlapped less than the minimum number of minutes ({}<{}) to be considered a match with time period in network: {}.".format(
                                    time_spans,
                                    overlap_minutes,
                                    partial_match_minutes,
                                    tg["time"],
                                )
                            )
                        elif overlap_minutes > 0:
                            WranglerLogger.debug(
                                "Returning a partial time period match. Time period: {} overlapped the minimum number of minutes ({}>={}) to be considered a match with time period in network: {}.".format(
                                    time_spans,
                                    overlap_minutes,
                                    partial_match_minutes,
                                    tg["time"],
                                )
                            )
                            if tg.get("category"):
                                categories += tg["category"]
                                for c in search_cats:
                                    print("CAT:", c, tg["category"])
                                    if c in tg["category"]:
                                        # print("Var:", v)
                                        # print(
                                        #    "RETURNING:",
                                        #    time_spans,
                                        #    category,
                                        #    tg["value"],
                                        # )
                                        return tg["value"]
                            else:
                                # print("Var:", v)
                                # print("RETURNING:", time_spans, category, tg["value"])
                                return tg["value"]

                """
                WranglerLogger.debug(
                    "\nCouldn't find time period for {}, returning default".format(
                        str(time_spans)
                    )
                )
                """
                if "default" in v.keys():
                    # print("Var:", v)
                    # print("RETURNING:", time_spans, v["default"])
                    return v["default"]
                else:
                    # print("Var:", v)
                    WranglerLogger.error(
                        "\nCan't find default; must specify a category in {}".format(
                            str(categories)
                        )
                    )
                    raise ValueError(
                        "Can't find default, must specify a category in: {}".format(
                            str(categories)
                        )
                    )

        time_spans = parse_time_spans(time_period)

        return self.links_df[prop].apply(
            _get_property, time_spans=time_spans, category=category
        )

    def update_distance(
        self,
        links_df: GeoDataFrame = None,
        use_shapes: bool = False,
        units: str = "miles",
        network_variable: str = "distance",
        overwrite: bool = True,
        inplace = True
    ):
        """
        Calculate link distance in specified units to network variable using either straight line
        distance or (if specified) shape distance if available.

        Args:
            links_df: Links GeoDataFrame.  Useful if want to update a portion of network links
                (i.e. only centroid connectors). If not provided, will use entire self.links_df.
            use_shapes: if True, will add length information from self.shapes_df rather than crow-fly.
                If no corresponding shape found in self.shapes_df, will default to crow-fly.
            units: units to use. Defaults to the standard unit of miles. Available units: "meters", "miles".
            network_variable: variable to store link distance in. Defaults to "distance".
            overwrite: Defaults to True and will overwrite all existing calculated distances.
                False will only update NaNs.
            inplace: updates self.links_df

        Returns:
            links_df with updated distance

        """
        if units not in ["miles","meters"]:
            raise NotImplementedError

        if links_df is None:
            links_df = self.links_df.copy()

        msg = "Update distance in {} to variable: {}".format(units, network_variable)
        if overwrite: msg + "\n  - overwriting existing calculated values if found."
        if use_shapes: msg + "\n  - using shapes_df length if found."
        WranglerLogger.debug(msg)

        """
        Start actual process
        """

        temp_links_gdf = links_df.copy()
        temp_links_gdf.crs = "EPSG:4326"
        temp_links_gdf = temp_links_gdf.to_crs(epsg=26915) #in meters

        conversion_from_meters = {"miles": 1/1609.34, "meters": 1}
        temp_links_gdf[network_variable] = temp_links_gdf.geometry.length * conversion_from_meters[units]

        if use_shapes:
            _needed_shapes_gdf = self.shapes_df.loc[
                self.shapes_df[self.shape_foreign_key] in links_df[self.shape_foreign_key]
            ].copy()

            _needed_shapes_gdf = _needed_shapes_gdf.to_crs(epsg=26915)
            _needed_shapes_gdf[network_variable] = _needed_shapes_gdf.geometry.length * conversion_from_meters[units]

            temp_links_gdf = update_df(
                temp_links_gdf,
                _needed_shapes_gdf,
                merge_key = self.shape_foreign_key,
                update_fields = [network_variable],
                method = "update if found",
            )

        if overwrite:
            links_df[network_variable] = temp_links_gdf[network_variable]
        else:
            links_df = update_df(
                links_df,
                temp_links_gdf,
                merge_key = self.unique_link_key,
                update_fields = [network_variable],
                method = "update nan",
            )

        if inplace:
            self.links_df = links_df
        else:
            return links_df

    def create_dummy_connector_links(
        gp_df: GeoDataFrame,
        ml_df: GeoDataFrame,
        access_lanes: int = 1,
        egress_lanes: int = 1,
        access_roadway: str = "ml_access",
        egress_roadway: str = "ml_access",
        access_name_prefix: str = "Access Dummy ",
        egress_name_prefix: str = "Egress Dummy ",
    ):
        """
        create dummy connector links between the general purpose and managed lanes

        args:
            gp_df : GeoDataFrame
                dataframe of general purpose links (where managed lane also exists)
            ml_df : GeoDataFrame
                dataframe of corresponding managed lane links,
            access_lanes: int
                number of lanes in access dummy link
            egress_lanes: int
                number of lanes in egress dummy link
            access_roadway: str
                roaday type for access dummy link
            egress_roadway: str
                roadway type for egress dummy link
            access_name_prefix: str
                prefix for access dummy link name
            egress_name_prefix: str
                prefix for egress dummy link name
        """

        gp_ml_links_df = pd.concat(
            [gp_df, ml_df.add_prefix("ML_")], axis=1, join="inner"
        )

        access_df = gp_df.iloc[0:0, :].copy()
        egress_df = gp_df.iloc[0:0, :].copy()

        def _get_connector_references(ref_1: list, ref_2: list, type: str):
            if type == "access":
                out_location_reference = [
                    {"sequence": 1, "point": ref_1[0]["point"]},
                    {"sequence": 2, "point": ref_2[0]["point"]},
                ]

            if type == "egress":
                out_location_reference = [
                    {"sequence": 1, "point": ref_2[1]["point"]},
                    {"sequence": 2, "point": ref_1[1]["point"]},
                ]
            return out_location_reference

        for index, row in gp_ml_links_df.iterrows():
            _access_row = {
                "A": row["A"],
                "B": row["ML_A"],
                "access": row["ML_access"],
                "drive_access": row["drive_access"],
                "name": access_name_prefix + row["name"],
                "lanes": access_lanes,
                "roadway": access_roadway,
                "model_link_id": (row["model_link_id"] + row["ML_model_link_id"] + 1),
                "locationReferences": _get_connector_references(
                    row["locationReferences"], row["ML_locationReferences"], "access"
                ),
            }

            _access_row["distance"] = haversine_distance(
                _access_row["locationReferences"][0]["point"],
                _access_row["locationReferences"][1]["point"],
            )

            # ref is not a *required* attribute, so make conditional:
            if "ref" in gp_ml_links_df.columns:
                _access_row["ref"] = row["ref"]
            else:
                _access_row["ref"] = ""
            access_df = access_df.append(_access_row, ignore_index=True)

            _egress_row = {
                "A": row["ML_B"],
                "B": row["B"],
                "access": row["ML_access"],
                "drive_access": row["drive_access"],
                "name": egress_name_prefix + row["name"],
                "lanes": egress_lanes,
                "roadway": egress_roadway,
                "model_link_id": (row["model_link_id"] + row["ML_model_link_id"] + 2),
                "locationReferences": _get_connector_references(
                    row["locationReferences"], row["ML_locationReferences"], "egress"
                ),
            }

            _egress_row["distance"] = haversine_distance(
                _egress_row["locationReferences"][0]["point"],
                _egress_row["locationReferences"][1]["point"],
            )

            # ref is not a *required* attribute, so make conditional:
            if "ref" in gp_ml_links_df.columns:
                _egress_row["ref"] = row["ref"]
            else:
                _egress_row["ref"] = ""
            egress_df = egress_df.append(_egress_row, ignore_index=True)

        return (access_df, egress_df)

    def create_managed_lane_network(
        self,
        keep_same_attributes_ml_and_gp: list = None,
        keep_additional_attributes_ml_and_gp: list = [],
        managed_lanes_required_attributes: list = [],
        managed_lanes_node_id_scalar: int = None,
        managed_lanes_link_id_scalar: int = None,
        in_place: bool = False,
    ) -> RoadwayNetwork:
        """
        Create a roadway network with managed lanes links separated out.
        Add new parallel managed lane links, access/egress links,
        and add shapes corresponding to the new links

        args:
            keep_same_attributes_ml_and_gp: list of attributes to copy from general purpose
                lane to managed lane. If not specified, will look for value in the RoadwayNetwork
                instance.  If not found there, will default to KEEP_SAME_ATTRIBUTES_ML_AND_GP.
            keep_additional_attributes_ml_and_gp: list of additional attributes to add. This is useful
                if you want to leave the default attributes and then ALSO some others.
            managed_lanes_required_attributes: list of attributes that are required to be specified
                in new managed lanes. If not specified, will look for value in the RoadwayNetwork
                instance.  If not found there, will default to MANAGED_LANES_REQUIRED_ATTRIBUTES.
            managed_lanes_node_id_scalar: integer value added to original node IDs to create managed
                lane unique ids. If not specified, will look for value in the RoadwayNetwork
                instance.  If not found there, will default to MANAGED_LANES_NODE_ID_SCALAR.
            managed_lanes_link_id_scalar: integer value added to original link IDs to create managed
                lane unique ids. If not specified, will look for value in the RoadwayNetwork
                instance.  If not found there, will default to MANAGED_LANES_LINK_ID_SCALAR.
            in_place: update self or return a new roadway network object

        returns: A RoadwayNetwork instance

        .. todo:: make this a more rigorous test
        """

        WranglerLogger.info("Creating network with duplicated managed lanes")

        if "ml_access" in self.links_df["roadway"].tolist():
            msg = "managed lane access links already exist in network; shouldn't be running create managed lane network. Returning network as-is."
            WranglerLogger.error(msg)
            if in_place:
                return
            else:
                return copy.deepcopy(self)

        # identify parameters to use
        if not keep_same_attributes_ml_and_gp:
            keep_same_attributes_ml_and_gp = self.__dict__.get("keep_same_attributes_ml_and_gp")
        if not keep_same_attributes_ml_and_gp:
            keep_same_attributes_ml_and_gp = KEEP_SAME_ATTRIBUTES_ML_AND_GP

        if not managed_lanes_required_attributes:
            managed_lanes_required_attributes = self.__dict__.get("managed_lanes_required_attributes")
        if not managed_lanes_required_attributes:
            managed_lanes_required_attributes = MANAGED_LANES_REQUIRED_ATTRIBUTES

        if not managed_lanes_node_id_scalar:
            managed_lanes_node_id_scalar = self.__dict__.get("managed_lanes_node_id_scalar")
        if not managed_lanes_node_id_scalar:
            managed_lanes_node_id_scalar = MANAGED_LANES_NODE_ID_SCALAR

        if not managed_lanes_link_id_scalar:
            managed_lanes_link_id_scalar = self.__dict__.get("managed_lanes_link_id_scalar")
        if not managed_lanes_link_id_scalar:
            managed_lanes_link_id_scalar = MANAGED_LANES_LINK_ID_SCALAR

        keep_same_attributes_ml_and_gp = list(
            set(keep_same_attributes_ml_and_gp + keep_additional_attributes_ml_and_gp)
        )

        link_attributes = self.links_df.columns.values.tolist()

        ml_attributes = [i for i in link_attributes if i.startswith("ML_")]

        # non_ml_links are links in the network where there is no managed lane.
        # gp_links are the gp lanes and ml_links are ml lanes respectively for the ML roadways.

        non_ml_links_df = self.links_df[self.links_df["managed"] == 0]
        non_ml_links_df = non_ml_links_df.drop(ml_attributes, axis=1)

        ml_links_df = self.links_df[self.links_df["managed"] == 1]
        gp_links_df = ml_links_df.drop(ml_attributes, axis=1)

        for attr in link_attributes:
            if attr == "name":
                ml_links_df["name"] = "Managed Lane " + gp_links_df["name"]
            elif attr in ml_attributes and attr not in ["ML_ACCESS", "ML_EGRESS"]:
                gp_attr = attr.split("_", 1)[1]
                ml_links_df.loc[:, gp_attr] = ml_links_df[attr]

            if (
                attr not in keep_same_attributes_ml_and_gp
                and attr not in managed_lanes_required_attributes
            ):
                ml_links_df[attr] = ""

        ml_links_df = ml_links_df.drop(ml_attributes, axis=1)

        ml_links_df["managed"] = 1
        gp_links_df["managed"] = 0

        ml_links_df["A"] = ml_links_df["A"] + managed_lanes_node_id_scalar
        ml_links_df["B"] = ml_links_df["B"] + managed_lanes_node_id_scalar
        ml_links_df[self.unique_link_key] = (
            ml_links_df[self.unique_link_key] + managed_lanes_link_id_scalar
        )
        ml_links_df["locationReferences"] = ml_links_df["locationReferences"].apply(
            # lambda x: _update_location_reference(x)
            lambda x: offset_location_reference(x)
        )
        ml_links_df["geometry"] = ml_links_df["locationReferences"].apply(
            lambda x: create_line_string(x)
        )
        ml_links_df[self.shape_foreign_key] = ml_links_df["geometry"].apply(
            lambda x: create_unique_shape_id(x)
        )

        access_links_df, egress_links_df = RoadwayNetwork.create_dummy_connector_links(
            gp_links_df, ml_links_df
        )
        access_links_df["geometry"] = access_links_df["locationReferences"].apply(
            lambda x: create_line_string(x)
        )
        egress_links_df["geometry"] = egress_links_df["locationReferences"].apply(
            lambda x: create_line_string(x)
        )
        access_links_df[self.shape_foreign_key] = access_links_df["geometry"].apply(
            lambda x: create_unique_shape_id(x)
        )
        egress_links_df[self.shape_foreign_key] = egress_links_df["geometry"].apply(
            lambda x: create_unique_shape_id(x)
        )

        out_links_df = gp_links_df.append(ml_links_df)
        out_links_df = out_links_df.append(access_links_df)
        out_links_df = out_links_df.append(egress_links_df)
        out_links_df = out_links_df.append(non_ml_links_df)

        # drop the duplicate links, if Any
        # could happen when a new egress/access link gets created which already exist
        out_links_df = out_links_df.drop_duplicates(
            subset=["A", "B"],
            keep="last"
        )

        # only the ml_links_df could potenitally have the new added nodes
        added_a_nodes = ml_links_df["A"]
        added_b_nodes = ml_links_df["B"]

        out_nodes_df = self.nodes_df

        # add node if it is not already present
        for a_node in added_a_nodes:
            out_nodes_df = out_nodes_df.append(
                {
                    "model_node_id": a_node,
                    "geometry": Point(
                        out_links_df[out_links_df["A"] == a_node].iloc[0][
                            "locationReferences"
                        ][0]["point"]
                    ),
                    "drive_access": 1,
                },
                ignore_index=True,
            )

        for b_node in added_b_nodes:
            if b_node not in out_nodes_df["model_node_id"].tolist():
                out_nodes_df = out_nodes_df.append(
                    {
                        "model_node_id": b_node,
                        "geometry": Point(
                            out_links_df[out_links_df["B"] == b_node].iloc[0][
                                "locationReferences"
                            ][1]["point"]
                        ),
                        "drive_access": 1,
                    },
                    ignore_index=True,
                )

        out_nodes_df["X"] = out_nodes_df["geometry"].apply(lambda g: g.x)
        out_nodes_df["Y"] = out_nodes_df["geometry"].apply(lambda g: g.y)

        out_shapes_df = self.shapes_df

        # managed lanes, access and egress connectors are new geometry
        new_shapes_df = pd.DataFrame(
            {
                "geometry": ml_links_df["geometry"]
                .append(access_links_df["geometry"])
                .append(egress_links_df["geometry"])
            }
        )
        new_shapes_df[self.shape_foreign_key] = new_shapes_df["geometry"].apply(
            lambda x: create_unique_shape_id(x)
        )

        out_shapes_df = out_shapes_df.append(new_shapes_df)
        out_shapes_df = out_shapes_df.drop_duplicates(
            subset=self.shape_foreign_key,
            keep="first"
        )

        out_links_df = out_links_df.reset_index()
        out_nodes_df = out_nodes_df.reset_index()
        out_shapes_df = out_shapes_df.reset_index()

        if in_place:
            self.links_df = out_links_df
            self.nodes_df = out_nodes_df
            self.shapes_df = out_shapes_df
        else:
            out_network = copy.deepcopy(self)
            out_network.links_df = out_links_df
            out_network.nodes_df = out_nodes_df
            out_network.shapes_df = out_shapes_df
            return out_network

    @staticmethod
    def get_modal_links_nodes(
        links_df: DataFrame,
        nodes_df: DataFrame,
        modes: list[str] = None,
        modes_to_network_link_variables: dict = MODES_TO_NETWORK_LINK_VARIABLES,
    ) -> tuple(DataFrame, DataFrame):
        """Returns nodes and link dataframes for specific mode.

        Args:
            links_df: DataFrame of standard network links
            nodes_df: DataFrame of standard network nodes
            modes: list of the modes of the network to be kept, must be in `drive`,`transit`,`rail`,`bus`,
                `walk`, `bike`. For example, if bike and walk are selected, both bike and walk links will be kept.
            modes_to_network_link_variables: dictionary mapping the mode selections to the network variables
                that must bool to true to select that mode. Defaults to MODES_TO_NETWORK_LINK_VARIABLES

        Returns: tuple of DataFrames for links, nodes filtered by mode

        .. todo:: Right now we don't filter the nodes because transit-only
        links with walk access are not marked as having walk access
        Issue discussed in https://github.com/wsp-sag/network_wrangler/issues/145
        modal_nodes_df = nodes_df[nodes_df[mode_node_variable] == 1]
        """
        for mode in modes:
            if mode not in modes_to_network_link_variables.keys():
                msg = "mode value should be one of {}, got {}".format(
                    list(modes_to_network_link_variables.keys()), mode,
                )
                WranglerLogger.error(msg)
                raise ValueError(msg)

        mode_link_variables = list(
            set(
                [
                    mode
                    for mode in modes
                    for mode in modes_to_network_link_variables[mode]
                ]
            )
        )
        mode_node_variables = list(
            set(
                [
                    mode
                    for mode in modes
                    for mode in modes_to_network_link_variables[mode]
                ]
            )
        )

        if not set(mode_link_variables).issubset(set(links_df.columns)):
            msg = "{} not in provided links_df list of columns. Available columns are: \n {}".format(
                set(mode_link_variables) - set(links_df.columns), links_df.columns
            )
            WranglerLogger.error(msg)

        if not set(mode_node_variables).issubset(set(nodes_df.columns)):
            msg = "{} not in provided nodes_df list of columns. Available columns are: \n {}".format(
                set(mode_node_variables) - set(nodes_df.columns), nodes_df.columns
            )
            WranglerLogger.error(msg)

        modal_links_df = links_df.loc[links_df[mode_link_variables].any(axis=1)]

        ##TODO right now we don't filter the nodes because transit-only
        # links with walk access are not marked as having walk access
        # Issue discussed in https://github.com/wsp-sag/network_wrangler/issues/145
        # modal_nodes_df = nodes_df[nodes_df[mode_node_variable] == 1]
        modal_nodes_df = nodes_df

        return modal_links_df, modal_nodes_df

    @staticmethod
    def get_modal_graph(
        links_df: DataFrame,
        nodes_df: DataFrame,
        mode: str = None,
        modes_to_network_link_variables: dict = MODES_TO_NETWORK_LINK_VARIABLES,
    ):
        """Determines if the network graph is "strongly" connected
        A graph is strongly connected if each vertex is reachable from every other vertex.

        Args:
            links_df: DataFrame of standard network links
            nodes_df: DataFrame of standard network nodes
            mode: mode of the network, one of `drive`,`transit`,
                `walk`, `bike`
            modes_to_network_link_variables: dictionary mapping the mode selections to the network variables
                that must bool to true to select that mode. Defaults to MODES_TO_NETWORK_LINK_VARIABLES

        Returns: networkx: osmnx: DiGraph  of network
        """
        if mode not in modes_to_network_link_variables.keys():
            msg = "mode value should be one of {}.".format(
                list(modes_to_network_link_variables.keys())
            )
            WranglerLogger.error(msg)
            raise ValueError(msg)

        _links_df, _nodes_df = RoadwayNetwork.get_modal_links_nodes(
            links_df, nodes_df, modes=[mode],
        )
        G = RoadwayNetwork.ox_graph(_nodes_df, _links_df)

        return G

    def is_network_connected(
        self, mode: str = None, links_df: DataFrame = None, nodes_df: DataFrame = None
    ):
        """
        Determines if the network graph is "strongly" connected
        A graph is strongly connected if each vertex is reachable from every other vertex.

        Args:
            mode:  mode of the network, one of `drive`,`transit`,
                `walk`, `bike`
            links_df: DataFrame of standard network links
            nodes_df: DataFrame of standard network nodes

        Returns: boolean

        .. todo:: Consider caching graphs if they take a long time.
        """

        _nodes_df = nodes_df if nodes_df else self.nodes_df
        _links_df = links_df if links_df else self.links_df

        if mode:
            _links_df, _nodes_df = RoadwayNetwork.get_modal_links_nodes(
                _links_df, _nodes_df, modes=[mode],
            )
        else:
            WranglerLogger.info(
                "Assessing connectivity without a mode\
                specified. This may have limited value in interpretation.\
                To add mode specificity, add the keyword `mode =` to calling\
                this method"
            )

        # TODO: consider caching graphs if they start to take forever
        #      and we are calling them more than once.
        G = RoadwayNetwork.ox_graph(_nodes_df, _links_df)
        is_connected = nx.is_strongly_connected(G)

        return is_connected

    @staticmethod
    def add_incident_link_data_to_nodes(
        links_df: DataFrame = None,
        nodes_df: DataFrame = None,
        link_variables: list = [],
        unique_node_key = UNIQUE_NODE_KEY,
    ) -> DataFrame:
        """
        Add data from links going to/from nodes to node.

        Args:
            links_df: if specified, will assess connectivity of this
                links list rather than self.links_df
            nodes_df: if specified, will assess connectivity of this
                nodes list rather than self.nodes_df
            link_variables: list of columns in links dataframe to add to incident nodes

        Returns:
            nodes DataFrame with link data where length is N*number of links going in/out
        """
        WranglerLogger.debug("Adding following link data to nodes: ".format())

        _link_vals_to_nodes = [x for x in link_variables if x in links_df.columns]
        if link_variables not in _link_vals_to_nodes:
            WranglerLogger.warning(
                "Following columns not in links_df and wont be added to nodes: {} ".format(
                    list(set(link_variables) - set(_link_vals_to_nodes))
                )
            )

        _nodes_from_links_A = nodes_df.merge(
            links_df[["A"] + _link_vals_to_nodes],
            how="outer",
            left_on=unique_node_key,
            right_on="A",
        )
        _nodes_from_links_B = nodes_df.merge(
            links_df[["B"] + _link_vals_to_nodes],
            how="outer",
            left_on=unique_node_key,
            right_on="B",
        )
        _nodes_from_links_ab = pd.concat([_nodes_from_links_A, _nodes_from_links_B])

        return _nodes_from_links_ab

    def identify_segment_endpoints(
        self,
        mode: str = "",
        links_df: DataFrame = None,
        nodes_df: DataFrame = None,
        min_connecting_links: int = 10,
        min_distance: float = None,
        max_link_deviation: int = 2,
    ):
        """

        Args:
            mode:  list of modes of the network, one of `drive`,`transit`,
                `walk`, `bike`
            links_df: if specified, will assess connectivity of this
                links list rather than self.links_df
            nodes_df: if specified, will assess connectivity of this
                nodes list rather than self.nodes_df

        """
        SEGMENT_IDENTIFIERS = ["name", "ref"]

        NAME_PER_NODE = 4
        REF_PER_NODE = 2

        _nodes_df = nodes_df if nodes_df else self.nodes_df
        _links_df = links_df if links_df else self.links_df

        if mode:
            _links_df, _nodes_df = RoadwayNetwork.get_modal_links_nodes(
                _links_df, _nodes_df, modes=[mode],
            )
        else:
            WranglerLogger.warning(
                "Assessing connectivity without a mode\
                specified. This may have limited value in interpretation.\
                To add mode specificity, add the keyword `mode =` to calling\
                this method"
            )

        _nodes_df = RoadwayNetwork.add_incident_link_data_to_nodes(
            links_df=_links_df,
            nodes_df=_nodes_df,
            link_variables=SEGMENT_IDENTIFIERS + ["distance"],
        )
        WranglerLogger.debug("Node/Link table elements: {}".format(len(_nodes_df)))

        # Screen out segments that have blank name AND refs
        _nodes_df = _nodes_df.replace(r"^\s*$", np.nan, regex=True).dropna(
            subset=["name", "ref"]
        )

        WranglerLogger.debug(
            "Node/Link table elements after dropping empty name AND ref : {}".format(
                len(_nodes_df)
            )
        )

        # Screen out segments that aren't likely to be long enough
        # Minus 1 in case ref or name is missing on an intermediate link
        _min_ref_in_table = REF_PER_NODE * (min_connecting_links - max_link_deviation)
        _min_name_in_table = NAME_PER_NODE * (min_connecting_links - max_link_deviation)

        _nodes_df["ref_freq"] = _nodes_df["ref"].map(_nodes_df["ref"].value_counts())
        _nodes_df["name_freq"] = _nodes_df["name"].map(_nodes_df["name"].value_counts())

        _nodes_df = _nodes_df.loc[
            (_nodes_df["ref_freq"] >= _min_ref_in_table)
            & (_nodes_df["name_freq"] >= _min_name_in_table)
        ]

        WranglerLogger.debug(
            "Node/Link table has n = {} after screening segments for min length:\n{}".format(
                len(_nodes_df),
                _nodes_df[
                    [
                        self.unique_node_key,
                        "name",
                        "ref",
                        "distance",
                        "ref_freq",
                        "name_freq",
                    ]
                ],
            )
        )
        # ----------------------------------------
        # Find nodes that are likely endpoints
        # ----------------------------------------

        # - Likely have one incident link and one outgoing link
        _max_ref_endpoints = REF_PER_NODE / 2
        _max_name_endpoints = NAME_PER_NODE / 2
        # - Attach frequency  of node/ref
        _nodes_df = _nodes_df.merge(
            _nodes_df.groupby(by=[self.unique_node_key, "ref"])
            .size()
            .rename("ref_N_freq"),
            on=[self.unique_node_key, "ref"],
        )
        # WranglerLogger.debug("_ref_count+_nodes:\n{}".format(_nodes_df[["model_node_id","ref","name","ref_N_freq"]]))
        # - Attach frequency  of node/name
        _nodes_df = _nodes_df.merge(
            _nodes_df.groupby(by=[self.unique_node_key, "name"])
            .size()
            .rename("name_N_freq"),
            on=[self.unique_node_key, "name"],
        )
        # WranglerLogger.debug("_name_count+_nodes:\n{}".format(_nodes_df[["model_node_id","ref","name","name_N_freq"]]))

        WranglerLogger.debug(
            "Possible segment endpoints:\n{}".format(
                _nodes_df[
                    [
                        self.unique_node_key,
                        "name",
                        "ref",
                        "distance",
                        "ref_N_freq",
                        "name_N_freq",
                    ]
                ]
            )
        )
        # - Filter possible endpoint list based on node/name node/ref frequency
        _nodes_df = _nodes_df.loc[
            (_nodes_df["ref_N_freq"] <= _max_ref_endpoints)
            | (_nodes_df["name_N_freq"] <= _max_name_endpoints)
        ]
        WranglerLogger.debug(
            "{} Likely segment endpoints with req_ref<= {} or freq_name<={} \n{}".format(
                len(_nodes_df),
                _max_ref_endpoints,
                _max_name_endpoints,
                _nodes_df[
                    [
                        self.unique_node_key,
                        "name",
                        "ref",
                        "ref_N_freq",
                        "name_N_freq",
                    ]
                ],
            )
        )
        # ----------------------------------------
        # Assign a segment id
        # ----------------------------------------
        _nodes_df["segment_id"], _segments = pd.factorize(
            _nodes_df.name + _nodes_df.ref
        )
        WranglerLogger.debug("{} Segments:\n{}".format(len(_segments), _segments))

        # ----------------------------------------
        # Drop segments without at least two nodes
        # ----------------------------------------

        # https://stackoverflow.com/questions/13446480/python-pandas-remove-entries-based-on-the-number-of-occurrences
        _nodes_df = _nodes_df[
            _nodes_df.groupby(["segment_id", self.unique_node_key])[
                self.unique_node_key
            ].transform(len)
            > 1
        ]

        WranglerLogger.debug(
            "{} Segments with at least nodes:\n{}".format(
                len(_nodes_df),
                _nodes_df[
                    [self.unique_node_key, "name", "ref", "segment_id"]
                ],
            )
        )

        # ----------------------------------------
        # For segments with more than two nodes, find farthest apart pairs
        # ----------------------------------------

        def _max_segment_distance(row):
            _segment_nodes = _nodes_df.loc[_nodes_df["segment_id"] == row["segment_id"]]
            dist = _segment_nodes.geometry.distance(row.geometry)
            return max(dist.dropna())

        _nodes_df["seg_distance"] = _nodes_df.apply(_max_segment_distance, axis=1)
        _nodes_df = _nodes_df.merge(
            _nodes_df.groupby("segment_id")
            .seg_distance.agg(max)
            .rename("max_seg_distance"),
            on="segment_id",
        )

        _nodes_df = _nodes_df.loc[
            (_nodes_df["max_seg_distance"] == _nodes_df["seg_distance"])
            & (_nodes_df["seg_distance"] > 0)
        ].drop_duplicates(subset=[self.unique_node_key, "segment_id"])

        # ----------------------------------------
        # Reassign segment id for final segments
        # ----------------------------------------
        _nodes_df["segment_id"], _segments = pd.factorize(
            _nodes_df.name + _nodes_df.ref
        )

        WranglerLogger.debug(
            "{} Segments:\n{}".format(
                len(_segments),
                _nodes_df[
                    [
                        self.unique_node_key,
                        "name",
                        "ref",
                        "segment_id",
                        "seg_distance",
                    ]
                ],
            )
        )

        return _nodes_df[
            ["segment_id", self.unique_node_key, "geometry", "name", "ref"]
        ]

    def identify_segment(
        self,
        O_id,
        D_id,
        selection_dict: dict = {},
        mode=None,
        nodes_df=None,
        links_df=None,
    ):
        """
        Args:
            endpoints: list of length of two unique keys of nodes making up endpoints of segment
            selection_dict: dictionary of link variables to select candidate links from, otherwise will create a graph of ALL links which will be both a RAM hog and could result in odd shortest paths.
            segment_variables: list of variables to keep
        """
        _nodes_df = nodes_df if nodes_df else self.nodes_df
        _links_df = links_df if links_df else self.links_df

        if mode:
            _links_df, _nodes_df = RoadwayNetwork.get_modal_links_nodes(
                _links_df, _nodes_df, modes=[mode],
            )
        else:
            WranglerLogger.warning(
                "Assessing connectivity without a mode\
                specified. This may have limited value in interpretation.\
                To add mode specificity, add the keyword `mode =` to calling\
                this method"
            )

        if selection_dict:
            _query = " or ".join(
                [f"{k} == {repr(v)}" for k, v in selection_dict.items()]
            )
            _candidate_links = _links_df.query(_query)
            WranglerLogger.debug(
                "Found {} candidate links from {} total links using following query:\n{}".format(
                    len(_candidate_links), len(_links_df), _query
                )
            )
        else:
            _candidate_links = _links_df

            WranglerLogger.warning(
                "Not pre-selecting links using selection_dict can use up a lot of RAM and also result in odd segment paths."
            )

        WranglerLogger.debug(
            "_candidate links for segment: \n{}".format(
                _candidate_links[["u", "v", "A", "B", "name", "ref"]]
            )
        )

        try:
            _sp = False
            (G, candidate_links, sp_route, sp_links) = self.path_search(
                _candidate_links, O_id, D_id
            )
            _sp = True
        except NoPathFound:
            msg = "Route not found from {} to {} using selection candidates {}".format(
                O_id, D_id, selection_dict
            )
            WranglerLogger.warning(msg)
            sp_links = pd.DataFrame()

        return sp_links

    def assess_connectivity(
        self,
        mode: str = "",
        ignore_end_nodes: bool = True,
        links_df: DataFrame = None,
        nodes_df: DataFrame = None,
    ):
        """Returns a network graph and list of disconnected subgraphs
        as described by a list of their member nodes.

        Args:
            mode:  list of modes of the network, one of `drive`,`transit`,
                `walk`, `bike`
            ignore_end_nodes: if True, ignores stray singleton nodes
            links_df: if specified, will assess connectivity of this
                links list rather than self.links_df
            nodes_df: if specified, will assess connectivity of this
                nodes list rather than self.nodes_df

        Returns: Tuple of
            Network Graph (osmnx flavored networkX DiGraph)
            List of disconnected subgraphs described by the list of their
                member nodes (as described by their `model_node_id`)
        """
        _nodes_df = nodes_df if nodes_df else self.nodes_df
        _links_df = links_df if links_df else self.links_df

        if mode:
            _links_df, _nodes_df = RoadwayNetwork.get_modal_links_nodes(
                _links_df, _nodes_df, modes=[mode],
            )
        else:
            WranglerLogger.warning(
                "Assessing connectivity without a mode\
                specified. This may have limited value in interpretation.\
                To add mode specificity, add the keyword `mode =` to calling\
                this method"
            )

        G = RoadwayNetwork.ox_graph(_nodes_df, _links_df)
        # sub_graphs = [s for s in sorted(nx.strongly_connected_component_subgraphs(G), key=len, reverse=True)]
        sub_graphs = [
            s
            for s in sorted(
                (G.subgraph(c) for c in nx.strongly_connected_components(G)),
                key=len,
                reverse=True,
            )
        ]

        sub_graph_nodes = [
            list(s)
            for s in sorted(nx.strongly_connected_components(G), key=len, reverse=True)
        ]

        # sorted on decreasing length, dropping the main sub-graph
        disconnected_sub_graph_nodes = sub_graph_nodes[1:]

        # dropping the sub-graphs with only 1 node
        if ignore_end_nodes:
            disconnected_sub_graph_nodes = [
                list(s) for s in disconnected_sub_graph_nodes if len(s) > 1
            ]

        WranglerLogger.info(
            "{} for disconnected networks for mode = {}:\n{}".format(
                self.node_foreign_key,
                mode,
                "\n".join(list(map(str, disconnected_sub_graph_nodes))),
            )
        )
        return G, disconnected_sub_graph_nodes

    @staticmethod
    def network_connection_plot(G, disconnected_subgraph_nodes: list):
        """Plot a graph to check for network connection.

        Args:
            G: OSMNX flavored networkX graph.
            disconnected_subgraph_nodes: List of disconnected subgraphs described by the list of their
                member nodes (as described by their `model_node_id`).

        returns: fig, ax : tuple
        """

        colors = []
        for i in range(len(disconnected_subgraph_nodes)):
            colors.append("#%06X" % randint(0, 0xFFFFFF))

        fig, ax = ox.plot_graph(
            G,
            figsize=(16, 16),
            show=False,
            close=True,
            edge_color="black",
            edge_alpha=0.1,
            node_color="black",
            node_alpha=0.5,
            node_size=10,
        )
        i = 0
        for nodes in disconnected_subgraph_nodes:
            for n in nodes:
                size = 100
                ax.scatter(G.nodes[n]["X"], G.nodes[n]["Y"], c=colors[i], s=size)
            i = i + 1

        return fig, ax

    def selection_map(
        self,
        selected_link_idx: list,
        A: Optional[Any] = None,
        B: Optional[Any] = None,
        candidate_link_idx: Optional[List] = [],
    ):
        """
        Shows which links are selected for roadway property change or parallel
        managed lanes category of roadway projects.

        Args:
            selected_links_idx: list of selected link indices
            candidate_links_idx: optional list of candidate link indices to also include in map
            A: optional foreign key of starting node of a route selection
            B: optional foreign key of ending node of a route selection
        """
        WranglerLogger.debug(
            "Selected Links: {}\nCandidate Links: {}\n".format(
                selected_link_idx, candidate_link_idx
            )
        )

        graph_link_idx = list(set(selected_link_idx + candidate_link_idx))
        graph_links = self.links_df.loc[graph_link_idx]

        node_list_foreign_keys = list(
            set(
                [
                    i
                    for fk in self.link_foreign_key
                    for i in list(graph_links[fk])
                ]
            )
        )

        graph_nodes = self.nodes_df.loc[node_list_foreign_keys]

        G = RoadwayNetwork.ox_graph(graph_nodes, graph_links)

        # base map plot with whole graph
        m = ox.plot_graph_folium(
            G, edge_color=None, tiles="cartodbpositron", width="300px", height="250px"
        )

        # plot selection
        selected_links = self.links_df.loc[selected_link_idx]

        for _, row in selected_links.iterrows():
            pl = ox.folium._make_folium_polyline(
                edge=row, edge_color="blue", edge_width=5, edge_opacity=0.8
            )
            pl.add_to(m)

        # if have A and B node add them to base map
        def _folium_node(node_row, color="white", icon=""):
            node_marker = folium.Marker(
                location=[node_row["Y"], node_row["X"]],
                icon=folium.Icon(icon=icon, color=color),
            )
            return node_marker

        if A:

            # WranglerLogger.debug("A: {}\n{}".format(A,self.nodes_df[self.nodes_df[RoadwayNetwork.NODE_FOREIGN_KEY] == A]))
            _folium_node(
                self.nodes_df[self.nodes_df[self.node_foreign_key] == A],
                color="green",
                icon="play",
            ).add_to(m)

        if B:
            _folium_node(
                self.nodes_df[self.nodes_df[self.node_foreign_key] == B],
                color="red",
                icon="star",
            ).add_to(m)

        return m

    def deletion_map(self, links: dict, nodes: dict):
        """
        Shows which links and nodes are deleted from the roadway network
        """
        # deleted_links = None
        # deleted_nodes = None

        missing_error_message = []

        if links is not None:
            for key, val in links.items():
                deleted_links = self.links_df[self.links_df[key].isin(val)]

                node_list_foreign_keys = list(
                    set(
                        [
                            i
                            for fk in self.link_foreign_key
                            for i in list(deleted_links[fk])
                        ]
                    )
                )
                candidate_nodes = self.nodes_df.loc[node_list_foreign_keys]
        else:
            deleted_links = None

        if nodes is not None:
            for key, val in nodes.items():
                deleted_nodes = self.nodes_df[self.nodes_df[key].isin(val)]
        else:
            deleted_nodes = None

        G = RoadwayNetwork.ox_graph(candidate_nodes, deleted_links)

        m = ox.plot_graph_folium(G, edge_color="red", tiles="cartodbpositron")

        def _folium_node(node, color="white", icon=""):
            node_circle = folium.Circle(
                location=[node["Y"], node["X"]],
                radius=2,
                fill=True,
                color=color,
                fill_opacity=0.8,
            )
            return node_circle

        if deleted_nodes is not None:
            for _, row in deleted_nodes.iterrows():
                _folium_node(row, color="red").add_to(m)

        return m

    def addition_map(self, links: dict, nodes: dict):
        """
        Shows which links and nodes are added to the roadway network
        """

        if links is not None:
            link_ids = []
            for link in links:
                link_ids.append(link.get(RoadwayNetwork.UNIQUE_LINK_KEY))

            added_links = self.links_df[
                self.links_df[RoadwayNetwork.UNIQUE_LINK_KEY].isin(link_ids)
            ]
            node_list_foreign_keys = list(
                set(
                    [
                        i
                        for fk in self.link_foreign_key
                        for i in list(added_links[fk])
                    ]
                )
            )
            try:
                candidate_nodes = self.nodes_df.loc[node_list_foreign_keys]
            except:
                return None

        if nodes is not None:
            node_ids = []
            for node in nodes:
                node_ids.append(node.get(self.unique_node_key))

            added_nodes = self.nodes_df[
                self.nodes_df[self.unique_node_key].isin(node_ids)
            ]
        else:
            added_nodes = None

        G = RoadwayNetwork.ox_graph(candidate_nodes, added_links)

        m = ox.plot_graph_folium(G, edge_color="green", tiles="cartodbpositron")

        def _folium_node(node, color="white", icon=""):
            node_circle = folium.Circle(
                location=[node["Y"], node["X"]],
                radius=2,
                fill=True,
                color=color,
                fill_opacity=0.8,
            )
            return node_circle

        if added_nodes is not None:
            for _, row in added_nodes.iterrows():
                _folium_node(row, color="green").add_to(m)

        return m
