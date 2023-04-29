import copy

import numpy as np
import pandas as pd

from typing import Union

from .subnet import Subnet
from .graph import shortest_path
from ..logger import WranglerLogger


"""
(str): default column to use as weights in the shortest path calculations.
"""
SP_WEIGHT_COL = "i"

"""
Union(int, float)): default penalty assigned for each
    degree of distance between a link and a link with the searched-for
    name when searching for paths between A and B node
"""
SP_WEIGHT_FACTOR = 100

"""
(int): default for initial number of links from name-based
    selection that are traveresed before trying another shortest
    path when searching for paths between A and B node
"""
SEARCH_BREADTH = 5

"""
(int): default for maximum number of links traversed between
    links that match the searched name when searching for paths
    between A and B node
"""
MAX_SEARCH_BREADTH = 10


class SegmentFormatError(Exception):
    pass


class SegmentSelectionError(Exception):
    pass


class Segment:
    """Segment is a contiguous section of the roadway network defined by a start and end node and
    a facility of one or more names.

    Segments are defined by a selection dictionary and then searched for on the network using
    a shortest path graph search.


    Usage:

    ```
    selection_dict = {
        "links": {"name":['6th','Sixth','sixth']},
        "start": {"osm_node_id": '187899923'},
        "end": {"osm_node_id": '187865924'}
    }

    net = RoadwayNetwork(...)

    segment = Segment(net = net, selection_dict = selection_dict)

    # lazily evaluated dataframe of links in segment (if found) from segment.net
    segment.segment_links_df

    # lazily evaluated list of nodes primary keys that are in segment (if found)
    segment.segment_nodes
    ```

    attr:
        net: Associated RoadwayNetwork object
        selection: segment selection
        start_node_pk: value of the primary key (usually model_node_id) for segment start node
        end_node_pk: value of the primary key (usually model_node_id) for segment end node
        subnet: Subnet object (and associated graph) on which to do shortest path search
        segment_nodes: list of primary keys of nodes within the selected segment. Will be lazily
            evaluated as the result of connected_path_search().
        segment_nodes_df: dataframe selection from net.modes_df for segment_nodes. Lazily evaluated
            based on segment_nodes.
        segment_links: list of primary keys of links which connect together segment_nodes. Lazily
            evaluated based on segment_links_df.
        segment_links_df: dataframe selection from net.links_df for segment_links. Lazily
            evaluated based on segment_links_df.
    """

    def __init__(
        self,
        net: "RoadwayNetwork",
        selection: 'RoadwaySelection',
        sp_weight_col: str = SP_WEIGHT_COL,
        sp_weight_factor: int = SP_WEIGHT_FACTOR,
        max_search_breadth: int = MAX_SEARCH_BREADTH,
    ):
        """Initialize a roadway segment object.

        Args:
            net (RoadwayNetwork): Associated RoadwayNetwork object
            selection (RoadwaySelection): Selection of type `segment_search`.
            sp_weight_col (str, optional): Column to use for weights in shortest path.  Will not
                likely need to be changed. Defaults to SP_WEIGHT_COL which defaults to `i`.
            sp_weight_factor (int, optional): Factor to multiply sp_weight_col by to use for
                weights in shortest path.  Will not likely need to be changed. Defaults to
                SP_WEIGHT_FACTOR which defaults to `100`.
            max_search_breadth (int, optional):Maximum expansions of the subnet network to find
                the shortest path after the initial selection based on `name`. Will not likely
                need to be changed unless network contains a lot of ambiguity. Defaults to
                MAX_SEARCH_BREADTH which defaults to 10.
        """

        self.net = net
        if not selection.type == "segment_search":
            raise SegmentFormatError("Selection object passed to Segment must be of type\
                                      `segment_search`")
        self.selection = selection

        self.start_node_pk = self._start_node_pk()
        self.end_node_pk = self._end_node_pk()

        self._sp_weight_col = sp_weight_col
        self._sp_weight_factor = sp_weight_factor
        self._max_search_breadth = max_search_breadth

        self.subnet = self._generate_subnet(self.selection_dict)

        # segment members are identified by storing nodes along a route
        self._segment_nodes = None

    @property
    def segment_nodes(self) -> list:
        if self._segment_nodes is None:
            self.connected_path_search()
        return self._segment_nodes

    @property
    def segment_nodes_df(self):
        return self.net.nodes_df[self.net.nodes_df.loc(self.segment_nodes)]

    @property
    def segment_links_df(self):
        return self.net.links_df[
            self.net.links_df[self.net.links_df.params.from_node].isin(
                self.segment_nodes
            )
            & self.net.links_df[self.net.links_df.params.to_node].isin(
                self.segment_nodes
            )
        ]

    @property
    def segment_links(self):
        return self.segment_links_df.index.tolist()

    def connected_path_search(
        self,
    ) -> None:
        """
        Finds a path from start_node_pk to send_node_pk based on the weight col value/factor.
        """
        WranglerLogger.debug(f"Initial set of nodes: {self.subnet_nodes}".format())

        # expand network to find at least the origin and destination nodes
        self.subnet.expand_subnet_to_include_nodes(
            [self.start_node_pk, self.end_node_pk]
        )

        # Once have A and B in graph try calculating shortest path and iteratively
        #    expand if not found.
        WranglerLogger.debug("Calculating shortest path from graph")
        while (
            not self._find_subnet_shortest_path()
            and self._i <= self._max_search_breadth
        ):
            self._i += 1
            WranglerLogger.debug(
                f"Adding breadth to find a connected path in subnet \
                i/max_i: {self._i}/{self._max_search_breadth}"
            )
            self._expand_subnet_breadth()

        if not self.found:
            WranglerLogger.debug(
                f"No connected path found from {self.O.pk} and {self.D_pk}\n\
                self.subnet_links_df:\n{self.subnet_links_df}"
            )
            raise SegmentSelectionError(
                f"No connected path found from {self.O.pk} and {self.D_pk}"
            )

    def _start_node_pk(self):
        """Find start node in selection dict and return its primary key."""
        _search_keys = ["A", "O", "start"]
        _node_dict = None
        for k in _search_keys:
            if self.selection.selection_dict.get(k):
                _node_dict = self.selection.selection_dict[k]
        if not _node_dict:
            raise SegmentFormatError("Can't find start node in selection dict.")
        if len(_node_dict) > 1:
            raise SegmentFormatError(
                "Node selection should only have one value. Found {_node_dict}"
            )
        return self._get_node_pk_from_selection_dict_prop(_node_dict)

    def _end_node_pk(self):
        """Find end node in selection dict and return its primary key."""
        _search_keys = ["B", "D", "end"]
        _node_dict = None
        for k in _search_keys:
            if self.selection.selection_dict.get(k):
                _node_dict = self.selection.selection_dict[k]
        if not _node_dict:
            raise SegmentFormatError("Can't find end node in selection dict.")
        if len(_node_dict) > 1:
            raise SegmentFormatError(
                "Node selection should only have one value. Found {_node_dict}"
            )
        return self._get_node_pk_from_selection_dict_prop(_node_dict)

    def _get_node_pk_from_selection_dict_prop(self, prop_dict: dict) -> Union[str, int]:
        """Return the primary key of a node from a selection dictionary property."""
        if len(prop_dict) != 1:
            WranglerLogger.debug(f"prop_dict: {prop_dict}")
            raise SegmentFormatError("Node selection should have only one value .")
        _node_prop, _val = next(iter(prop_dict.items()))

        if _node_prop == self.net.nodes_df.params.primary_key:
            return _val

        _pk_list = self.net.nodes_df[
            self.net.nodes_df[_node_prop] == _val
        ].index.tolist()
        if len(_pk_list) != 1:
            WranglerLogger.error(
                f"Node selection for segment invalid. Found {len(_pk_list)} \
                in nodes_df with {_node_prop} = {_val}. Should only find one!"
            )
        return _pk_list[0]

    def _generate_subnet(self) -> Subnet:
        """Generate a subnet of the roadway network on which to search for connected segment.

        First will search based on "name" in selection_dict but if not found, will search
        using the "ref" field instead.
        """
        _selection_dict = copy.deepcopy(self.selection.selection_dict)
        # First search for initial set of links using "name" field, combined with values from "ref"

        if "ref" in self.selection.selection_dict:
            _selection_dict["name"] += _selection_dict["ref"]
            del _selection_dict["ref"]

        subnet = Subnet(
            selection_dict=_selection_dict,
            sp_weight_col=self._sp_weight_col,
            sp_weight_factor=self._sp_weight_factor,
            max_search_breadth=self._max_search_breadth,
        )

        if subnet.num_links == 0 and "ref" in self.selection_dict:
            del _selection_dict["name"]
            _selection_dict["ref"] = self.selection_dict["ref"]

            WranglerLogger.debug(f"Searching with ref = {_selection_dict['ref']}")
            subnet = Subnet(selection_dict=_selection_dict)
            _selection_dict = copy.deepcopy(self.selection_dict)

        # i is iteration # for an iterative search for connected paths with larger subnet
        if subnet.num_links == 0:
            WranglerLogger.error(
                f"Selection didn't return subnet links: {_selection_dict}"
            )
            raise SegmentSelectionError("No links found with selection.")
        return subnet

    def _find_subnet_shortest_path(
        self,
    ) -> bool:
        """Finds shortest path from start_node_pk to end_node_pk using self.subnet.graph.

        Args:
            sp_weight_col (str, optional): _description_. Defaults to SP_WEIGHT_COL.
            sp_weight_factor (float, optional): _description_. Defaults to SP_WEIGHT_FACTOR.

        Returns:
            _type_: boolean indicating if shortest path was found
        """

        WranglerLogger.debug(
            f"Calculating shortest path from {self.start_node_pk} to {self.end_node_pk}\
                using {self._sp_weight_col} as \
                weight with a factor of {self._sp_weight_factor}"
        )
        self.subnet._sp_weight_col = self._sp_weight_col
        self.subnet._weight_factor = self._sp_weight_factor

        self._segment_route_nodes = shortest_path(
            self.subnet.graph, self.start_node_pk, self.end_node_pk
        )

        if not self._segment_route_nodes:
            WranglerLogger.debug(
                f"No SP from {self.start_node_pk} to {self.end_node_pk} Found."
            )
            return False

        return True


def identify_segment_endpoints(
    net,
    mode: str,
    min_connecting_links: int = 10,
    min_distance: float = None,
    max_link_deviation: int = 2,
):
    """This has not been revisited or refactored and may or may not contain useful code.

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

    _links_df = net.links_df.mode_query(mode)
    _nodes_df = net.nodes_in_links(
        _links_df,
        net.nodes_df,
    )

    _nodes_df = net.add_incident_link_data_to_nodes(
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
                    net.nodes_df.params.primary_key,
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
        _nodes_df.groupby(by=[net.nodes_df.params.primary_key, "ref"])
        .size()
        .rename("ref_N_freq"),
        on=[net.nodes_df.params.primary_key, "ref"],
    )
    # WranglerLogger.debug("_ref_count+_nodes:\n{}".format(_nodes_df[["model_node_id","ref","name","ref_N_freq"]]))
    # - Attach frequency  of node/name
    _nodes_df = _nodes_df.merge(
        _nodes_df.groupby(by=[net.nodes_df.params.primary_key, "name"])
        .size()
        .rename("name_N_freq"),
        on=[net.nodes_df.params.primary_key, "name"],
    )
    # WranglerLogger.debug("_name_count+_nodes:\n{}".format(_nodes_df[["model_node_id","ref","name","name_N_freq"]]))

    WranglerLogger.debug(
        "Possible segment endpoints:\n{}".format(
            _nodes_df[
                [
                    net.nodes_df.params.primary_key,
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
                    net.nodes_df.params.primary_key,
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
    _nodes_df["segment_id"], _segments = pd.factorize(_nodes_df.name + _nodes_df.ref)
    WranglerLogger.debug("{} Segments:\n{}".format(len(_segments), _segments))

    # ----------------------------------------
    # Drop segments without at least two nodes
    # ----------------------------------------

    # https://stackoverflow.com/questions/13446480/python-pandas-remove-entries-based-on-the-number-of-occurrences
    _nodes_df = _nodes_df[
        _nodes_df.groupby(["segment_id", net.nodes_df.params.primary_key])[
            net.nodes_df.params.primary_key
        ].transform(len)
        > 1
    ]

    WranglerLogger.debug(
        "{} Segments with at least nodes:\n{}".format(
            len(_nodes_df),
            _nodes_df[[net.nodes_df.params.primary_key, "name", "ref", "segment_id"]],
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
    ].drop_duplicates(subset=[net.nodes_df.params.primary_key, "segment_id"])

    # ----------------------------------------
    # Reassign segment id for final segments
    # ----------------------------------------
    _nodes_df["segment_id"], _segments = pd.factorize(_nodes_df.name + _nodes_df.ref)

    WranglerLogger.debug(
        "{} Segments:\n{}".format(
            len(_segments),
            _nodes_df[
                [
                    net.nodes_df.params.primary_key,
                    "name",
                    "ref",
                    "segment_id",
                    "seg_distance",
                ]
            ],
        )
    )

    return _nodes_df[
        ["segment_id", net.nodes_df.params.primary_key, "geometry", "name", "ref"]
    ]
