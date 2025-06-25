"""Roadway selection classes for selecting links and nodes from a roadway network."""

from __future__ import annotations

import copy
import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal, Union

import pandas as pd

from ..errors import SelectionError
from ..logger import WranglerLogger
from ..models.projects import (
    RoadwaySelectionFormatError,
    SelectFacility,
    SelectLinksDict,
    SelectNodesDict,
)
from ..params import DEFAULT_SEARCH_MODES, SMALL_RECS
from ..utils.models import DatamodelDataframeIncompatableError, coerce_extra_fields_to_type_in_df
from .links.filters import filter_links_to_modes
from .segment import Segment

if TYPE_CHECKING:
    from pandera.typing import DataFrame

    from ..models.roadway.tables import RoadLinksTable, RoadNodesTable
    from .network import RoadwayNetwork

NODE_QUERY_FIELDS: list[str] = ["osm_node_id", "model_node_id"]
LINK_QUERY_FIELDS: list[str] = ["osm_link_id", "model_link_id", "name", "ref"]
SEGMENT_QUERY_FIELDS: list[str] = [
    "name",
    "ref",
    "osm_link_id",
    "model_link_id",
    "modes",
]


class RoadwaySelection(ABC):
    """Abstract base class for RoadwaySelection objects to define interface.

    Properties:
        net (RoadwayNetwork): roadway network related to the selection
        raw_selection_dictionary (dict): raw selection dictionary as input to the object used to
            generate `sel_key`.
        sel_key (str): unique selection key based on selection dictionary
        selection_dictionary (dict): dictionary conforming to `SelectFacility` model
            with a `links` key
        selection_data (SelectFacility): selection dictionary as a SelectFacility model
        selection_types (str): one of `explicit_ids`, `segment` or `all`.
            From selection dictionary.
        feature_types (str): returns `links`.  From selection dictionary.
        ignore_missing (bool): True if missing links should be ignored. From selection dictionary.
        modes (list[str]): list of modes that to search for links for. From selection dictionary.
            "any" will return all links without filtering modes.
        explicit_id_sel_dict (dict): explicit link selection criteria. From selection dictionary.
        additional_sel_dict: link selection criteria that is layered on top of a segment, all,
            or explict ID search (i.e. "lanes": [1,2,3], "drive_access": True).  From selection
            dictionary.
        found (bool): True if links were found
        selected_links (list): list of selected link ids
        selected_links_df (DataFrame[RoadwayLinksTable]): lazily-evaluated selected links.
        segment (Segment): segment object if selection type is segment. Defaults to None if not a
            segment selection.
    """

    def __init__(
        self,
        net: RoadwayNetwork,
        selection_data: Union[SelectFacility, dict],
    ):
        """Constructor for RoadwaySelection object.

        Args:
            net (RoadwayNetwork): Roadway network object to select from.
            selection_data: dictionary conforming to
                `SelectFacility` model with a "links" key or SelectFacility instance.
        """
        WranglerLogger.debug(f"Creating selection from selection dictionary: \n {selection_data}")
        self.net = net
        self.selection_dict = selection_data

    @property
    def feature_types(self) -> Literal["links", "nodes"]:
        """Return the feature type of the selection.."""
        if self.selection_data.links:
            return "links"
        if self.selection_data.nodes:
            return "nodes"
        msg = "SelectFacility must have either links or nodes defined."
        raise RoadwaySelectionFormatError(msg)

    @property
    @abstractmethod
    def selection_method(self):
        """Return the selection method of the selection."""
        raise NotImplementedError

    @property
    def raw_selection_dict(self) -> dict:
        """Raw selection dictionary as input into `net.get_selection()`.

        Used for generating `self.sel_key` and then fed into SelectLinksDict data model where
        additional defaults are set.
        """
        return self._raw_selection_dict

    @raw_selection_dict.setter
    def raw_selection_dict(self, selection_dict: dict):
        self._raw_selection_dict = selection_dict
        self._selection_key = _create_selection_key(selection_dict)

    @property
    def sel_key(self):
        """Return the selection key as generated from `self.raw_selection_dict`."""
        return self._sel_key

    @property
    def selection_data(self):
        """Link selection data from SelectLinksDict."""
        return self._selection_data

    @property
    def selection_dict(self) -> dict:
        """Selection dictionary dictating the selection settings."""
        return self._selection_dict

    @selection_dict.setter
    def selection_dict(self, selection_input: Union[SelectFacility, dict]):
        if isinstance(selection_input, SelectLinksDict):
            selection_input = SelectFacility(links=selection_input)
        elif isinstance(selection_input, SelectNodesDict):
            selection_input = SelectFacility(nodes=selection_input)

        if isinstance(selection_input, SelectFacility):
            self.raw_selection_dict = selection_input.asdict
        else:
            self.raw_selection_dict = selection_input

        if not isinstance(selection_input, SelectFacility):
            self._selection_data = self.validate_selection(SelectFacility(**selection_input))
        else:
            self._selection_data = self.validate_selection(selection_input)

        self._selection_dict = self._selection_data.asdict
        self._stored_net_hash = copy.deepcopy(self.net.network_hash)

    @property
    def node_query_fields(self) -> list[str]:
        """Fields that can be used in a node selection."""
        return NODE_QUERY_FIELDS

    @property
    def link_query_fields(self) -> list[str]:
        """Fields that can be used in a link selection."""
        return LINK_QUERY_FIELDS

    @property
    def segment_query_fields(self) -> list[str]:
        """Fields that can be used in a segment selection."""
        return SEGMENT_QUERY_FIELDS

    @abstractmethod
    def __nonzero__(self) -> bool:
        """Return True if links were selected."""
        raise NotImplementedError

    @property
    @abstractmethod
    def found(self) -> bool:
        """Return True if selection was found."""
        raise NotImplementedError

    @abstractmethod
    def validate_selection(self, selection_data: SelectFacility) -> SelectFacility:
        """Validate that selection_dict is compatible with the network."""
        raise NotImplementedError


class RoadwayLinkSelection(RoadwaySelection):
    """Object to perform and store information about selection from a project card `facility` key.

    Properties:
        net (RoadwayNetwork): roadway network related to the selection
        raw_selection_dictionary (dict): raw selection dictionary as input to the object used to
            generate `sel_key`.
        sel_key (str): unique selection key based on selection dictionary
        selection_dictionary (dict): dictionary conforming to `SelectFacility` model
            with a `links` key
        selection_data (SelectFacility): selection dictionary as a SelectFacility model
        selection_types (str): one of `explicit_ids`, `segment` or `all`.
            From selection dictionary.
        feature_types (str): returns `links`.  From selection dictionary.
        ignore_missing (bool): True if missing links should be ignored. From selection dictionary.
        modes (list[str]): list of modes that to search for links for. From selection dictionary.
            "any" will return all links without filtering modes.
        explicit_id_sel_dict (dict): explicit link selection criteria. From selection dictionary.
        additional_sel_dict: link selection criteria that is layered on top of a segment, all,
            or explict ID search (i.e. "lanes": [1,2,3], "drive_access": True).  From selection
            dictionary.
        found (bool): True if links were found
        selected_links (list): list of selected link ids
        selected_links_df (DataFrame[RoadwayLinksTable]): lazily-evaluated selected links.
        segment (Segment): segment object if selection type is segment. Defaults to None if not a
            segment selection.
    """

    SPECIAL_FIELDS: ClassVar[list[str]] = ["all", "modes", "ignore_missing"]
    # Fields handled explicitly in a special way that shouldn't be included in general selections

    def __init__(
        self,
        net: RoadwayNetwork,
        selection_data: Union[SelectFacility, dict],
    ):
        """Constructor for RoadwayLinkSelection object.

        Args:
            net (RoadwayNetwork): Roadway network object to select from.
            selection_data: dictionary conforming to
                `SelectFacility` model with a "links" key or SelectFacility instance.
        """
        super().__init__(net, selection_data)
        self._selected_links_df: Union[None, DataFrame[RoadLinksTable]] = None
        self._segment: Union[None, Segment] = None
        WranglerLogger.debug(f"Created LinkSelection of type: {self.selection_method}")

    def __nonzero__(self) -> bool:
        """Return True if links were selected."""
        return len(self.selected_links) > 0

    @property
    def selection_method(self) -> Literal["all", "query", "segment"]:
        """One of `all`, `explicit_ids`, or `segment`."""
        if self.selection_data.links and self.selection_data.from_ and self.selection_data.to:
            return "segment"
        if self.selection_data.links.all:
            return "all"
        if self.initial_query_fields:
            return "query"
        msg = "Cannot determine link selection method from selection dictionary."
        WranglerLogger.error(msg + f":\n {self.selection_data}")
        raise RoadwaySelectionFormatError(msg)

    @property
    def ignore_missing(self) -> bool:
        """True if missing links should be ignored. From selection dictionary."""
        return self.selection_data.links.ignore_missing

    @property
    def fields(self) -> list[str]:
        """Fields that can be used in a selections."""
        return self.selection_data.links.fields

    @property
    def modes(self) -> list[str]:
        """List of modes to search for links for. From selection dictionary.

        `any` will return all links without filtering modes.
        """
        return self.selection_data.links.modes or DEFAULT_SEARCH_MODES

    @property
    def initial_query_fields(self) -> list[str]:
        """Fields that can be used in a selection on their own."""
        return [k for k in self.link_query_fields if k in self.fields]

    @property
    def initial_query_selection_dict(self):
        """Return a dictionary of fields for an initial query."""
        return {
            k: v
            for k, v in self.selection_data.links.asdict.items()
            if k in self.initial_query_fields
        }

    @property
    def segment_id_fields(self) -> list[str]:
        """Fields used in an intial segment selection."""
        return [k for k in self.segment_query_fields if k in self.fields]

    @property
    def segment_selection_dict(self):
        """Return a dictionary of fields used for segment link selection."""
        return {k: v for k, v in self.selection_data.links.asdict.items() if k in self.fields}

    @property
    def secondary_selection_fields(self):
        """Return a list of fields that are not part of the initial selection fields."""
        if self.selection_method == "all":
            return list(set(self.fields) - set(self.SPECIAL_FIELDS))
        if self.selection_method == "segment":
            return list(set(self.fields) - set(self.SPECIAL_FIELDS) - set(self.segment_id_fields))
        if self.selection_method == "query":
            return list(
                set(self.fields) - set(self.SPECIAL_FIELDS) - set(self.initial_query_fields)
            )
        msg = f"Unknown selection method: {self.selection_method}."
        raise RoadwaySelectionFormatError(msg)

    @property
    def secondary_selection_dict(self):
        """Return a dictionary of fields that are not part of the initial selection fields."""
        return {
            k: v
            for k, v in self.selection_data.links.asdict.items()
            if k in self.secondary_selection_fields
        }

    @property
    def selected_links(self) -> list[int]:
        """List of selected link ids."""
        if self.selected_links_df is None:
            return []
        return self.selected_links_df.index.tolist()

    @property
    def found(self) -> bool:
        """Return True if links were found."""
        return self.selected_links_df is not None

    @property
    def segment(self) -> Union[None, Segment]:
        """Return the segment object if selection type is segment."""
        if self._segment is None and self.selection_method == "segment":
            WranglerLogger.debug("Creating new segment")
            self._segment = Segment(self.net, self)
        return self._segment

    def create_segment(self, max_search_breadth: int):
        """For running segment with custom max search breadth."""
        WranglerLogger.debug(f"Creating new segment with max_search_breadth {max_search_breadth}")
        self._segment = Segment(self.net, self, max_search_breadth=max_search_breadth)

    @property
    def selected_links_df(self) -> DataFrame[RoadLinksTable]:
        """Lazily evaluates selection for links or returns stored value in self._selected_links_df.

        Will re-evaluate if the current network hash is different than the stored one from the
        last selection.
        """
        if self._selected_links_df is None or self._stored_net_hash != self.net.network_hash:
            self._stored_net_hash = copy.deepcopy(self.net.network_hash)
            self._selected_links_df = self._perform_selection()

        return self._selected_links_df

    def validate_selection(self, sel_data: SelectFacility) -> SelectFacility:
        """Validates that selection_dict is compatible with the network."""
        if sel_data.links is None:
            msg = "Link Selection does not contain links field."
            raise RoadwaySelectionFormatError(msg)
        try:
            sel_data.links = coerce_extra_fields_to_type_in_df(
                sel_data.links, SelectLinksDict, self.net.links_df
            )
        except DatamodelDataframeIncompatableError as e:
            msg = f"Invalid link selection fields: {e}"
            WranglerLogger.error(msg)
            raise RoadwaySelectionFormatError(msg) from e
        return sel_data

    def _perform_selection(self):
        # 1. Initial selection based on selection type
        msg = f"Initial link selection type: {self.feature_types}.{self.selection_method}"
        WranglerLogger.debug(msg)

        if self.selection_method == "query":
            _selected_links_df = self.net.links_df.isin_dict(
                self.initial_query_selection_dict, ignore_missing=self.ignore_missing
            )

        elif self.selection_method == "segment":
            _selected_links_df = self.segment.segment_links_df

        elif self.selection_method == "all":
            _selected_links_df = self.net.links_df

        else:
            msg = f"Didn't understand selection type: {self.selection_type}"
            raise RoadwaySelectionFormatError(msg)

        # 2. Mode selection
        _selected_links_df = filter_links_to_modes(
            _selected_links_df, self.selection_data.links.modes
        )

        # 3. Additional attributes within initial selection
        if self.secondary_selection_dict:
            WranglerLogger.debug(f"Initially selected links: {len(_selected_links_df)}")
            if len(_selected_links_df) < SMALL_RECS:
                msg = f"\n{_selected_links_df[_selected_links_df.attrs['display_cols']]}"
                WranglerLogger.debug(msg)
            WranglerLogger.debug(
                f"Selecting from selection based on: {self.secondary_selection_dict}"
            )
            _selected_links_df = _selected_links_df.dict_query(self.secondary_selection_dict)

        if not len(_selected_links_df):
            WranglerLogger.warning("No links found matching criteria.")
        else:
            WranglerLogger.info(f"Final selected links: {len(_selected_links_df)}")
            if len(_selected_links_df) < SMALL_RECS:
                WranglerLogger.debug(f"\n{_selected_links_df}")

        return _selected_links_df


class RoadwayNodeSelection(RoadwaySelection):
    """Object to perform and store information about node selection from a project card "facility".

    Properties:
        net (RoadwayNetwork): roadway network related to the selection
        raw_selection_dictionary (dict): raw selection dictionary as input to the object used to
            generate `sel_key`.
        sel_key (str): unique selection key based on selection dictionary
        selection_dictionary (dict): dictionary conforming to `SelectFacility` model
            with a `nodes` key
        selection_data (SelectFacility): selection dictionary as a SelectFacility model
        selection_types (str): one of `explicit_ids`, `segment` or `all`.
            From selection dictionary.
        feature_types (str): returns `nodes`.  From selection dictionary.
        ignore_missing (bool): True if missing links should be ignored. From selection dictionary.
        explicit_id_sel_dict (dict): explicit link selection criteria. From selection dictionary.
        found (bool): True if nodes were found
        selected_nodes (list[int]): list of selected nodes ids
        selected_nodes_df (DataFrame[RoadwayNodesTable]): lazily-evaluated selected nodes.
    """

    SPECIAL_FIELDS: ClassVar[list[str]] = ["all", "ignore_missing"]
    # Fields handled explicitly in a special way that shouldn't be included in general selections

    def __init__(
        self,
        net: RoadwayNetwork,
        selection_data: Union[dict, SelectFacility],
    ):
        """Constructor for RoadwayNodeSelection object.

        Args:
            net (RoadwayNetwork): Roadway network object to select from.
            selection_data (Union[dict, SelectFacility]): Selection dictionary with "nodes" key
                conforming to SelectFacility format, or SelectFacility instance.
        """
        super().__init__(net, selection_data)
        self._selected_nodes_df: Union[None, DataFrame[RoadNodesTable]] = None

    def __nonzero__(self) -> bool:
        """Return True if nodes were selected."""
        return len(self.selected_nodes) > 0

    @property
    def selection_method(self) -> Literal["all", "query"]:
        """Node selection methode. Either `all` or `query`."""
        if self.selection_data.nodes.all:
            return "all"
        if self.initial_query_fields:
            return "query"
        msg = "Cannot determine node selection method from selection dictionary."
        WranglerLogger.error(msg + f":\n {self.selection_data}")
        raise RoadwaySelectionFormatError(msg)

    @property
    def ignore_missing(self) -> bool:
        """True if missing nodes should be ignored."""
        return self.selection_data.nodes.ignore_missing

    @property
    def selection_type(self):
        """One of `all` or `explicit_ids`."""
        if self.all:
            return "all"
        if self.explicit_id_fields:
            return "explicit_ids"
        msg = "Select Nodes should have either `all` or an explicit id."
        WranglerLogger.debug(
            msg
            + f" {self.explicit_id_fields}. \
            Found neither in nodes selection: \n{self.model_dump(by_alias=True)}"
        )
        raise RoadwaySelectionFormatError(msg)

    @property
    def fields(self) -> list[str]:
        """Fields that can be used in a selections."""
        return self.selection_data.nodes.fields

    @property
    def initial_query_fields(self) -> list[str]:
        """Fields which can be used in a selection on their own."""
        return [k for k in self.node_query_fields if k in self.fields]

    @property
    def secondary_selection_fields(self):
        """Return a list of fields that are not part of the initial selection fields."""
        if self.selection_method == "all":
            return list(set(self.fields) - set(self.SPECIAL_FIELDS))
        if self.selection_method == "query":
            return list(
                set(self.fields) - set(self.SPECIAL_FIELDS) - set(self.initial_query_fields)
            )
        msg = f"Unknown selection method: {self.selection_method}."
        raise RoadwaySelectionFormatError(msg)

    @property
    def initial_query_selection_dict(self):
        """Return a dictionary of fields for an initial query."""
        return {
            k: v
            for k, v in self.selection_data.nodes.asdict.items()
            if k in self.initial_query_fields
        }

    @property
    def secondary_selection_dict(self):
        """Return a dictionary of fields that are not part of the initial selection fields."""
        return {
            k: v
            for k, v in self.selection_data.nodes.asdict.items()
            if k in self.secondary_selection_fields
        }

    @property
    def selected_nodes(self) -> list[int]:
        """List of selected node ids."""
        if self.selected_nodes_df is None:
            return []
        return self.selected_nodes_df.index.tolist()

    @property
    def found(self) -> bool:
        """True if nodes were found."""
        return self.selected_nodes_df is not None

    @property
    def selected_nodes_df(self) -> DataFrame[RoadNodesTable]:
        """Lazily evaluates selection for nodes or returns stored value in self._selected_nodes_df.

        Will re-evaluate if the current network hash is different than the stored one from the
        last selection.
        """
        if self._selected_nodes_df is None or self._stored_net_hash != self.net.network_hash:
            self._stored_net_hash = self.net.network_hash
            self._selected_nodes_df = self._perform_selection()

        return self._selected_nodes_df

    def validate_selection(self, selection_data: SelectFacility) -> SelectFacility:
        """Validate that selection_dict is compatible with the network."""
        if selection_data.nodes is None:
            msg = "Node Selection does not contain nodes field."
            raise RoadwaySelectionFormatError(msg)

        try:
            selection_data.nodes = coerce_extra_fields_to_type_in_df(
                selection_data.nodes, SelectNodesDict, self.net.nodes_df
            )
        except DatamodelDataframeIncompatableError as e:
            msg = f"Invalid node selection fields: {e}"
            WranglerLogger.error(msg)
            raise RoadwaySelectionFormatError(msg) from e

        return selection_data

    def _perform_selection(self):
        # 1. Initial selection based on selection method
        if self.selection_method == "query":
            _selected_nodes_df = self.net.nodes_df.isin_dict(
                self.initial_query_selection_dict, ignore_missing=self.ignore_missing
            )
        elif self.selection_method == "all":
            _selected_nodes_df = self.net.nodes_df
        else:
            msg = f"Didn't understand selection method: {self.selection_method}"
            raise RoadwaySelectionFormatError(msg)

        # 2. Additional attributes within initial selection
        if self.secondary_selection_dict:
            WranglerLogger.debug(f"Initially selected nodes: {len(_selected_nodes_df)}")
            if len(_selected_nodes_df) < SMALL_RECS:
                msg = f"\n{_selected_nodes_df[_selected_nodes_df.attrs['display_cols']]}"
                WranglerLogger.debug(msg)
            WranglerLogger.debug(
                f"Selecting from selection based on: {self.secondary_selection_dict}"
            )
            _selected_nodes_df = _selected_nodes_df.dict_query(self.secondary_selection_dict)

        if not len(_selected_nodes_df):
            WranglerLogger.warning("No nodes found matching criteria.")
        else:
            WranglerLogger.info(f"Final selected nodes: {len(_selected_nodes_df)}")
            if len(_selected_nodes_df) < SMALL_RECS:
                WranglerLogger.debug(f"\n{_selected_nodes_df}")

        return _selected_nodes_df


def _create_selection_key(
    selection_dict: Union[SelectLinksDict, SelectNodesDict, SelectFacility, dict],
) -> str:
    """Selections are stored by a sha1 hash of the bit-encoded string of the selection dictionary.

    Args:
        selection_dict: Selection Dictionary

    Returns: Hex code for hash
    """
    if isinstance(selection_dict, SelectLinksDict):
        selection_dict = SelectFacility(links=selection_dict)
    elif isinstance(selection_dict, SelectNodesDict):
        selection_dict = SelectFacility(nodes=selection_dict)

    if isinstance(selection_dict, SelectFacility):
        selection_dict = selection_dict.model_dump(exclude_none=True, by_alias=True)
    elif not isinstance(selection_dict, dict):
        WranglerLogger.error(
            f"`selection_dict` arg must be a dictionary or SelectFacility model.\
                             Received: {selection_dict} of type {type(selection_dict)}"
        )
        msg = "Selection dictionary must be a dictionary or SelectFacility model."
        raise SelectionError(msg)
    return hashlib.sha1(str(selection_dict).encode()).hexdigest()
