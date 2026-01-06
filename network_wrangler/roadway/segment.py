"""Segment class and related functions for working with segments of a RoadwayNetwork.

A segment is a contiguous length of RoadwayNetwork defined by start/end nodes + link selections.

Segments are defined by a selection dictionary and then searched for on the network using
a shortest path graph search.

Usage:

```
selection_dict = {
    "links": {"name": ["6th", "Sixth", "sixth"]},
    "from": {"osm_node_id": "187899923"},
    "to": {"osm_node_id": "187865924"},
}

segment = Segment(net, selection)
segment.segment_links_df
segment.segment_nodes
```
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from pandera.typing import DataFrame

from ..errors import SegmentFormatError, SegmentSelectionError, SubnetCreationError
from ..logger import WranglerLogger
from ..models.projects.roadway_selection import SelectLinksDict, SelectNodeDict
from ..params import DEFAULT_SEARCH_MODES
from .graph import shortest_path
from .links.filters import filter_links_to_path
from .subnet import Subnet

if TYPE_CHECKING:
    from ..models.roadway.tables import RoadLinksTable, RoadNodesTable
    from .network import RoadwayNetwork
    from .selection import RoadwayLinkSelection


DEFAULT_MAX_SEARCH_BREADTH: int = 10


"""Factor to multiply sp_weight_col by to use for weights in shortest path.
"""
DEFAULT_SUBNET_SP_WEIGHT_FACTOR: int = 100

"""Column to use for weights in shortest path.
"""
SUBNET_SP_WEIGHT_COL: str = "i"


class Segment:
    """A contiguous length of RoadwayNetwork defined by start/end nodes + link selections.

    Segments are defined by a selection dictionary and then searched for on the network using
    a shortest path graph search.

    Usage:

    ```
    selection_dict = {
        "links": {"name":['6th','Sixth','sixth']},
        "from": {"osm_node_id": '187899923'},
        "to": {"osm_node_id": '187865924'}
    }

    net = RoadwayNetwork(...)

    segment = Segment(net = net, selection)

    # lazily evaluated dataframe of links in segment (if found) from segment.net
    segment.segment_links_df

    # lazily evaluated list of nodes primary keys that are in segment (if found)
    segment.segment_nodes
    ```

    attr:
        net: Associated RoadwayNetwork object
        selection: RoadwayLinkSelection
        from_node_id: value of the primary key (usually model_node_id) for segment start node
        to_node_id: value of the primary key (usually model_node_id) for segment end node
        subnet: Subnet object (and associated graph) on which to do shortest path search
        segment_nodes: list of primary keys of nodes within the selected segment. Will be lazily
            evaluated as the result of connected_path_search().
        segment_nodes_df: dataframe selection from net.modes_df for segment_nodes. Lazily evaluated
            based on segment_nodes.
        segment_links: list of primary keys of links which connect together segment_nodes. Lazily
            evaluated based on segment_links_df.
        segment_links_df: dataframe selection from net.links_df for segment_links. Lazily
            evaluated based on segment_links_df.
        max_search_breadth: maximum number of nodes to search for in connected_path_search.
            Defaults to DEFAULT_MAX_SEGMENT_SEARCH_BREADTH which is 10.
    """

    def __init__(
        self,
        net: RoadwayNetwork,
        selection: RoadwayLinkSelection,
        max_search_breadth: int = DEFAULT_MAX_SEARCH_BREADTH,
    ):
        """Initialize a roadway segment object.

        Args:
            net (RoadwayNetwork): Associated RoadwayNetwork object
            selection (RoadwayLinkSelection): Selection of type `segment`.
            max_search_breadth (int, optional): Maximum number of nodes to search for in
                connected_path_search. Defaults to DEFAULT_MAX_SEGMENT_SEARCH_BREADTH.
        """
        self.net = net
        self.max_search_breadth = max_search_breadth
        if selection.selection_method != "segment":
            msg = "Selection object passed to Segment must be of type `segment`"
            raise SegmentFormatError(msg)
        self.selection = selection

        # segment members are identified by storing nodes along a route
        self._segment_nodes: Union[list, None] = None

        # Initialize calculated, read-only attr.
        self._from_node_id: Union[int, None] = None
        self._to_node_id: Union[int, None] = None

        self.subnet = self._generate_subnet(self.segment_sel_dict)

        WranglerLogger.debug(f"Segment created: {self}")

    @property
    def modes(self) -> list[str]:
        """List of modes in the selection."""
        return self.selection.modes if self.selection.modes else DEFAULT_SEARCH_MODES

    @property
    def segment_sel_dict(self) -> dict:
        """Selection dictionary which only has keys related to initial segment link selection."""
        return self.selection.segment_selection_dict

    @property
    def from_node_id(self) -> int:
        """Find start node in selection dict and return its primary key."""
        if self._from_node_id is not None:
            return self._from_node_id
        self._from_node_id = self.get_node_id(self.selection.selection_data.from_)
        return self._from_node_id

    @property
    def to_node_id(self) -> int:
        """Find end node in selection dict and return its primary key."""
        if self._to_node_id is not None:
            return self._to_node_id
        self._to_node_id = self.get_node_id(self.selection.selection_data.to)
        return self._to_node_id

    @property
    def segment_nodes(self) -> list[int]:
        """Primary keys of nodes in segment."""
        if self._segment_nodes is None:
            WranglerLogger.debug("Segment not found yet so conducting connected_path_search.")
            self.connected_path_search()
        if self._segment_nodes is None:
            msg = "No segment nodes found."
            raise SegmentSelectionError(msg)
        return self._segment_nodes

    @property
    def segment_nodes_df(self) -> DataFrame[RoadNodesTable]:
        """Roadway network nodes filtered to nodes in segment."""
        return self.net.nodes_df.loc[self.segment_nodes]

    @property
    def segment_from_node_s(self) -> DataFrame[RoadNodesTable]:
        """Roadway network nodes filtered to segment start node."""
        return self.segment_nodes_df.loc[self.from_node_id]

    @property
    def segment_to_node_s(self) -> DataFrame[RoadNodesTable]:
        """Roadway network nodes filtered to segment end node."""
        return self.segment_nodes_df.loc[self.to_node_id]

    @property
    def segment_links_df(self) -> DataFrame[RoadLinksTable]:
        """Roadway network links filtered to segment links."""
        modal_links_df = self.net.links_df.mode_query(self.modes)
        segment_links_df = filter_links_to_path(modal_links_df, self.segment_nodes)
        return segment_links_df

    @property
    def segment_links(self) -> list[int]:
        """Primary keys of links in segment."""
        return self.segment_links_df.index.tolist()

    def get_node_id(self, node_selection_data: SelectNodeDict) -> int:
        """Get the primary key of a node based on the selection data."""
        node = self.get_node(node_selection_data)
        return node["model_node_id"].values[0]

    def get_node(self, node_selection_data: SelectNodeDict):
        """Get single node based on the selection data."""
        node_selection_dict = {
            k: v
            for k, v in node_selection_data.asdict.items()
            if k in self.selection.node_query_fields
        }
        node_df = self.net.nodes_df.isin_dict(node_selection_dict)
        if len(node_df) != 1:
            msg = f"Node selection not unique. Found {len(node_df)} nodes."
            raise SegmentSelectionError(msg)
        return node_df

    def connected_path_search(
        self,
    ) -> None:
        """Finds a path from from_node_id to to_node_id based on the weight col value/factor."""
        WranglerLogger.debug("Calculating shortest path from graph")
        _found = False
        _found = self._find_subnet_shortest_path()

        while not _found and self.subnet._i <= self.max_search_breadth:
            self.subnet._expand_subnet_breadth()
            _found = self._find_subnet_shortest_path()

        if not _found:
            msg = f"No connected path found from {self.O.pk} and {self.D_pk}"
            WranglerLogger.debug(msg)
            raise SegmentSelectionError(msg)

    def _generate_subnet(self, selection_dict: dict) -> Subnet:
        """Generate a subnet of the roadway network on which to search for connected segment.

        Args:
            selection_dict: selection dictionary to use for generating subnet
        """
        if not selection_dict:
            msg = "No selection provided to generate subnet from."
            raise SegmentFormatError(msg)

        WranglerLogger.debug(f"Creating subnet from dictionary: {selection_dict}")
        subnet = generate_subnet_from_link_selection_dict(
            self.net,
            modes=self.modes,
            link_selection_dict=selection_dict,
        )
        # expand network to find at least the origin and destination nodes
        subnet.expand_to_nodes(
            [self.from_node_id, self.to_node_id], max_search_breadth=self.max_search_breadth
        )
        return subnet

    def _find_subnet_shortest_path(
        self,
    ) -> bool:
        """Finds shortest path from from_node_id to to_node_id using self.subnet.graph.

        Sets self._segment_nodes to resulting path nodes

        Returns:
            bool: True if shortest path was found
        """
        WranglerLogger.debug(
            f"Calculating shortest path from {self.from_node_id} to {self.to_node_id} using\
        {self.subnet._sp_weight_col} as weight with a factor of {self.subnet._sp_weight_factor}"
        )

        self._segment_nodes = shortest_path(self.subnet.graph, self.from_node_id, self.to_node_id)

        if not self._segment_nodes:
            WranglerLogger.debug(f"No SP from {self.from_node_id} to {self.to_node_id} Found.")
            return False

        return True


def _generate_subnet_link_selection_dict_options(
    link_selection_dict: dict,
) -> list[dict]:
    """Generates a list of link selection dictionaries based on a link selection dictionary.

    Args:
        link_selection_dict (SelectLinksDict): Link selection dictionary.

    Returns:
        list[SelectLinksDict]: List of link selection dictionaries.
    """
    options = []
    # Option 1: As-is selection
    _sd = copy.deepcopy(link_selection_dict)
    options.append(_sd)

    # Option 2: Search for name or ref in name field
    if "ref" in link_selection_dict:
        _sd = copy.deepcopy(link_selection_dict)
        _sd["name"] += _sd["ref"]
        del _sd["ref"]
        options.append(_sd)

    # Option 3: Search for name in ref field
    if "name" in link_selection_dict:
        _sd = copy.deepcopy(link_selection_dict)
        _sd["ref"] = link_selection_dict["name"]
        del _sd["name"]
        options.append(_sd)

    return options


def generate_subnet_from_link_selection_dict(
    net,
    link_selection_dict: dict,
    modes: list[str] = DEFAULT_SEARCH_MODES,
    sp_weight_col: str = SUBNET_SP_WEIGHT_COL,
    sp_weight_factor: float = DEFAULT_SUBNET_SP_WEIGHT_FACTOR,
    **kwargs,
) -> Subnet:
    """Generates a Subnet object from a link selection dictionary.

    First will search based on "name" in selection_dict but if not found, will search
        using the "ref" field instead.

    Args:
        net (RoadwayNetwork): RoadwayNetwork object.
        link_selection_dict: dictionary of attributes to search for.
        modes: List of modes to limit subnet to. Defaults to DEFAULT_SEARCH_MODES.
        sp_weight_col: Column to use for weights in shortest path.  Defaults to SUBNET_SP_WEIGHT_COL.
        sp_weight_factor: Factor to multiply sp_weight_col by to use for weights in shortest path.
            Defaults to DEFAULT_SUBNET_SP_WEIGHT_FACTOR.
        kwargs: other kwargs to pass to Subnet initiation

    Returns:
        Subnet: Subnet object.
    """
    link_sd_options = _generate_subnet_link_selection_dict_options(link_selection_dict)
    for sd in link_sd_options:
        WranglerLogger.debug(f"Trying link selection:\n{sd}")
        subnet_links_df = copy.deepcopy(net.links_df.mode_query(modes))
        subnet_links_df = subnet_links_df.dict_query(sd)
        if len(subnet_links_df) > 0:
            break
    if len(subnet_links_df) == 0:
        WranglerLogger.error(f"Selection didn't return subnet links: {link_selection_dict}")
        msg = "No links found with selection."
        raise SubnetCreationError(msg)

    subnet_links_df["i"] = 0
    subnet = Subnet(
        net=net,
        subnet_links_df=subnet_links_df,
        modes=modes,
        sp_weight_col=sp_weight_col,
        sp_weight_factor=sp_weight_factor,
        **kwargs,
    )

    WranglerLogger.debug(
        f"Found subnet from link selection with {len(subnet.subnet_links_df)} links."
    )
    return subnet


def identify_segment_endpoints(
    net,
    mode: str = "drive",
    min_connecting_links: int = 10,
    max_link_deviation: int = 2,
) -> pd.DataFrame:
    """This has not been revisited or refactored and may or may not contain useful code.

    Args:
        net: RoadwayNetwork to find segments for
        mode:  list of modes of the network, one of `drive`,`transit`,
            `walk`, `bike`. Defaults to "drive".
        min_connecting_links: number of links that should be connected with same name or ref
            to be considered a segment (minus max_link_deviation). Defaults to 10.
        max_link_deviation: maximum links that don't have the same name or ref to still be
            considered a segment. Defaults to 2.

    """
    msg = "This function has not been revisited or refactored to work."
    raise NotImplementedError(msg)
    SEGMENT_IDENTIFIERS = ["name", "ref"]

    NAME_PER_NODE = 4
    REF_PER_NODE = 2

    # make a copy so it is a full dataframe rather than a slice.
    _links_df = copy.deepcopy(net.links_df.mode_query(mode))

    _nodes_df = copy.deepcopy(
        net.nodes_in_links(
            _links_df,
        )
    )
    from .network import add_incident_link_data_to_nodes  # noqa: PLC0415

    _nodes_df = add_incident_link_data_to_nodes(
        links_df=_links_df,
        nodes_df=_nodes_df,
        link_variables=[*SEGMENT_IDENTIFIERS, "distance"],
    )

    # WranglerLogger.debug(f"Node/Link table elements: {len(_nodes_df)}"")

    # Screen out segments that have blank name AND refs
    _nodes_df = _nodes_df.replace(r"^\s*$", np.nan, regex=True).dropna(subset=["name", "ref"])

    # WranglerLogger.debug(f"Node/Link recs after dropping empty name AND ref : {len(_nodes_df)}")

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

    _display_cols = [
        net.nodes_df.model_node_id,
        "name",
        "ref",
        "distance",
        "ref_freq",
        "name_freq",
    ]
    msg = f"Node/Link table has n = {len(_nodes_df)} after screening segments for min length: \n\
        {_nodes_df[_display_cols]}"
    WranglerLogger.debug(msg)

    # ----------------------------------------
    # Find nodes that are likely endpoints
    # ----------------------------------------

    # - Likely have one incident link and one outgoing link
    _max_ref_endpoints = REF_PER_NODE / 2
    _max_name_endpoints = NAME_PER_NODE / 2
    # - Attach frequency  of node/ref
    _nodes_df = _nodes_df.merge(
        _nodes_df.groupby(by=[net.nodes_df.model_node_id, "ref"]).size().rename("ref_N_freq"),
        on=[net.nodes_df.model_node_id, "ref"],
    )

    _display_cols = ["model_node_id", "ref", "name", "ref_N_freq"]
    # WranglerLogger.debug(f"_ref_count+_nodes:\n{_nodes_df[_display_cols]})
    # - Attach frequency  of node/name
    _nodes_df = _nodes_df.merge(
        _nodes_df.groupby(by=[net.nodes_df.model_node_id, "name"]).size().rename("name_N_freq"),
        on=[net.nodes_df.model_node_id, "name"],
    )
    _display_cols = ["model_node_id", "ref", "name", "name_N_freq"]
    # WranglerLogger.debug(f"_name_count+_nodes:\n{_nodes_df[_display_cols]}")

    _display_cols = [
        net.nodes_df.model_node_id,
        "name",
        "ref",
        "distance",
        "ref_N_freq",
        "name_N_freq",
    ]
    # WranglerLogger.debug(f"Possible segment endpoints:\n{_nodes_df[_display_cols]}")
    # - Filter possible endpoint list based on node/name node/ref frequency
    _nodes_df = _nodes_df.loc[
        (_nodes_df["ref_N_freq"] <= _max_ref_endpoints)
        | (_nodes_df["name_N_freq"] <= _max_name_endpoints)
    ]
    _gb_cols = [
        net.nodes_df.model_node_id,
        "name",
        "ref",
        "ref_N_freq",
        "name_N_freq",
    ]

    msg = f"{len(_nodes_df)} Likely segment endpoints with req_ref<= {_max_ref_endpoints} or\
            freq_name<={_max_name_endpoints}\n{_nodes_df.groupby(_gb_cols)}"
    # WranglerLogger.debug(msg)
    # ----------------------------------------
    # Assign a segment id
    # ----------------------------------------
    _nodes_df["segment_id"], _segments = pd.factorize(_nodes_df.name + _nodes_df.ref)

    WranglerLogger.debug(f"{len(_segments)} Segments: \n{chr(10).join(_segments.tolist())}")

    # ----------------------------------------
    # Drop segments without at least two nodes
    # ----------------------------------------

    # https://stackoverflow.com/questions/13446480/python-pandas-remove-entries-based-on-the-number-of-occurrences
    _min_nodes = 2
    _nodes_df = _nodes_df[
        _nodes_df.groupby(["segment_id", net.nodes_df.model_node_id])[
            net.nodes_df.model_node_id
        ].transform(len)
        >= _min_nodes
    ]

    msg = f"{len(_nodes_df)} segments with at least {_min_nodes} nodes: \n\
        {_nodes_df.groupby(['segment_id'])}"
    # WranglerLogger.debug(msg)

    # ----------------------------------------
    # For segments with more than two nodes, find farthest apart pairs
    # ----------------------------------------

    def _max_segment_distance(row):
        _segment_nodes = _nodes_df.loc[_nodes_df["segment_id"] == row["segment_id"]]
        dist = _segment_nodes.geometry.distance(row.geometry)
        return max(dist.dropna())

    _nodes_df["seg_distance"] = _nodes_df.apply(_max_segment_distance, axis=1)
    _nodes_df = _nodes_df.merge(
        _nodes_df.groupby("segment_id").seg_distance.agg(max).rename("max_seg_distance"),
        on="segment_id",
    )

    _nodes_df = _nodes_df.loc[
        (_nodes_df["max_seg_distance"] == _nodes_df["seg_distance"])
        & (_nodes_df["seg_distance"] > 0)
    ].drop_duplicates(subset=[net.nodes_df.model_node_id, "segment_id"])

    # ----------------------------------------
    # Reassign segment id for final segments
    # ----------------------------------------
    _nodes_df["segment_id"], _segments = pd.factorize(_nodes_df.name + _nodes_df.ref)

    _display_cols = [
        net.nodes_df.model_node_id,
        "name",
        "ref",
        "segment_id",
        "seg_distance",
    ]

    WranglerLogger.debug(
        f"Start and end of {len(_segments)} Segments: \n{_nodes_df[_display_cols]}"
    )

    _return_cols = [
        "segment_id",
        net.nodes_df.model_node_id,
        "geometry",
        "name",
        "ref",
    ]
    return _nodes_df[_return_cols]
