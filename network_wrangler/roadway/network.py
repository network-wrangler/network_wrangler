"""Roadway Network class and functions for Network Wrangler.

Used to represent a roadway network and perform operations on it.

Usage:

```python
from network_wrangler import load_roadway_from_dir, write_roadway

net = load_roadway_from_dir("my_dir")
net.get_selection({"links": [{"name": ["I 35E"]}]})
net.apply("my_project_card.yml")

write_roadway(net, "my_out_prefix", "my_dir", file_format="parquet")
```
"""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import geopandas as gpd
import ijson
import networkx as nx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from projectcard import ProjectCard, SubProject
from pydantic import BaseModel, field_validator, model_validator
from shapely.geometry import LineString, Point
from shapely.ops import split

from ..configs import DefaultConfig, WranglerConfig, load_wrangler_config
from ..errors import (
    LinkAddError,
    NodeAddError,
    NodeNotFoundError,
    NotNodesError,
    ProjectCardError,
    SelectionError,
    ShapeAddError,
)
from ..logger import WranglerLogger
from ..models.projects.roadway_selection import SelectFacility, SelectLinksDict, SelectNodesDict
from ..models.roadway.tables import RoadLinksTable, RoadNodesAttrs, RoadNodesTable, RoadShapesTable
from ..params import DEFAULT_CATEGORY, DEFAULT_TIMESPAN, LAT_LON_CRS
from ..utils.data import concat_with_attr
from ..utils.models import empty_df_from_datamodel, validate_df_to_model
from .graph import net_to_graph
from .links.create import data_to_links_df
from .links.delete import delete_links_by_ids
from .links.edit import edit_link_geometry_from_nodes
from .links.filters import filter_links_to_ids, filter_links_to_node_ids
from .links.links import node_ids_unique_to_link_ids, shape_ids_unique_to_link_ids
from .links.scopes import prop_for_scope
from .links.validate import validate_links_have_nodes
from .model_roadway import ModelRoadwayNetwork
from .nodes.create import data_to_nodes_df
from .nodes.delete import delete_nodes_by_ids
from .nodes.edit import NodeGeometryChangeTable, edit_node_geometry
from .nodes.filters import filter_nodes_to_links
from .nodes.nodes import node_ids_without_links
from .projects import (
    apply_calculated_roadway,
    apply_new_roadway,
    apply_roadway_deletion,
    apply_roadway_property_change,
)
from .selection import (
    RoadwayLinkSelection,
    RoadwayNodeSelection,
    _create_selection_key,
)
from .shapes.create import df_to_shapes_df
from .shapes.delete import delete_shapes_by_ids
from .shapes.edit import edit_shape_geometry_from_nodes
from .shapes.io import read_shapes
from .shapes.shapes import shape_ids_without_links

if TYPE_CHECKING:
    from networkx import MultiDiGraph

    from ..models._base.types import TimespanString
    from ..transit.network import TransitNetwork


Selections = Union[RoadwayLinkSelection, RoadwayNodeSelection]

# Constants
MIN_SPLIT_SEGMENTS = 2


class RoadwayNetwork(BaseModel):
    """Representation of a Roadway Network.

    Typical usage example:

    ```py
    net = load_roadway(
        links_file=MY_LINK_FILE,
        nodes_file=MY_NODE_FILE,
        shapes_file=MY_SHAPE_FILE,
    )
    my_selection = {
        "link": [{"name": ["I 35E"]}],
        "A": {"osm_node_id": "961117623"},  # start searching for segments at A
        "B": {"osm_node_id": "2564047368"},
    }
    net.get_selection(my_selection)

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
        my_net.get_selection(my_selection),
        my_change
    )

        net.model_net
        net.is_network_connected(mode="drive", nodes=self.m_nodes_df, links=self.m_links_df)
        _, disconnected_nodes = net.assess_connectivity(
            mode="walk",
            ignore_end_nodes=True,
            nodes=self.m_nodes_df,
            links=self.m_links_df
        )
        write_roadway(net,filename=my_out_prefix, path=my_dir, for_model = True)
    ```

    Attributes:
        nodes_df (RoadNodesTable): dataframe of of node records.
        links_df (RoadLinksTable): dataframe of link records and associated properties.
        shapes_df (RoadShapesTable): dataframe of detailed shape records  This is lazily
            created iff it is called because shapes files can be expensive to read.
        _selections (dict): dictionary of stored roadway selection objects, mapped by
            `RoadwayLinkSelection.sel_key` or `RoadwayNodeSelection.sel_key` in case they are
                made repeatedly.
        network_hash: dynamic property of the hashed value of links_df and nodes_df. Used for
            quickly identifying if a network has changed since various expensive operations have
            taken place (i.e. generating a ModelRoadwayNetwork or a network graph)
        model_net (ModelRoadwayNetwork): referenced `ModelRoadwayNetwork` object which will be
            lazily created if None or if the `network_hash` has changed.
        config (WranglerConfig): wrangler configuration object
    """

    model_config = {"arbitrary_types_allowed": True}

    nodes_df: pd.DataFrame
    links_df: pd.DataFrame
    _shapes_df: Optional[pd.DataFrame] = None

    _links_file: Optional[Path] = None
    _nodes_file: Optional[Path] = None
    _shapes_file: Optional[Path] = None

    config: WranglerConfig = DefaultConfig

    _model_net: Optional[ModelRoadwayNetwork] = None
    _selections: dict[str, Selections] = {}
    _modal_graphs: dict[str, dict] = defaultdict(lambda: {"graph": None, "hash": None})

    def __str__(self):
        """Return string representation of RoadwayNetwork.

        Returns:
            str: Summary string showing network statistics and dataframe contents.
        """
        my_str = f"RoadwayNetwork(nodes={len(self.nodes_df)}, links={len(self.links_df)})"
        my_str += f"\nnodes_df (type={type(self.nodes_df)}):\n{self.nodes_df}"
        my_str += f"\nlinks_df (type={type(self.links_df)}):\n{self.links_df}"
        return my_str

    @field_validator("config")
    def validate_config(cls, v):
        """Validate config."""
        return load_wrangler_config(v)

    @field_validator("nodes_df", "links_df")
    def coerce_crs(cls, v):
        """Coerce crs of nodes_df and links_df to LAT_LON_CRS."""
        if v.crs != LAT_LON_CRS:
            WranglerLogger.warning(
                f"CRS of links_df ({v.crs}) doesn't match network crs {LAT_LON_CRS}. \
                    Changing to network crs."
            )
            v.to_crs(LAT_LON_CRS)
        return v

    # TODO: This may be overkill if many edits are being made.
    @model_validator(mode="after")
    def validate_referential_integrity(self):
        """Validate that all nodes referenced in links exist in nodes table."""
        WranglerLogger.debug(
            "validate_referential_integrity(): Validating referential integrity between links and nodes"
        )
        try:
            validate_links_have_nodes(self.links_df, self.nodes_df)
        except Exception as e:
            WranglerLogger.error(f"Referential integrity validation failed: {e}")
            raise

        return self

    @property
    def shapes_df(self) -> pd.DataFrame:
        """Load and return RoadShapesTable.

        If not already loaded, will read from shapes_file and return. If shapes_file is None,
        will return an empty dataframe with the right schema. If shapes_df is already set, will
        return that.
        """
        if (self._shapes_df is None or self._shapes_df.empty) and self._shapes_file is not None:
            self._shapes_df = read_shapes(
                self._shapes_file,
                filter_to_shape_ids=self.links_df.shape_id.to_list(),
                config=self.config,
            )
        # if there is NONE, then at least create an empty dataframe with right schema
        elif self._shapes_df is None:
            self._shapes_df = empty_df_from_datamodel(RoadShapesTable)
            self._shapes_df.set_index("shape_id_idx", inplace=True)

        return self._shapes_df

    @shapes_df.setter
    def shapes_df(self, value):
        self._shapes_df = df_to_shapes_df(value, config=self.config)

    @property
    def network_hash(self) -> str:
        """Hash of the links and nodes dataframes."""
        _value = str.encode(self.links_df.df_hash() + "-" + self.nodes_df.df_hash())

        _hash = hashlib.sha256(_value).hexdigest()
        return _hash

    @property
    def model_net(self) -> ModelRoadwayNetwork:
        """Return a ModelRoadwayNetwork object for this network."""
        if self._model_net is None or self._model_net._net_hash != self.network_hash:
            self._model_net = ModelRoadwayNetwork(self)
        return self._model_net

    @property
    def summary(self) -> dict:
        """Quick summary dictionary of number of links, nodes."""
        d = {
            "links": len(self.links_df),
            "nodes": len(self.nodes_df),
        }
        return d

    @property
    def link_shapes_df(self) -> gpd.GeoDataFrame:
        """Add shape geometry to links if available.

        returns: shapes merged to links dataframe
        """
        _links_df = copy.deepcopy(self.links_df)
        link_shapes_df = _links_df.merge(
            self.shapes_df,
            left_on="shape_id",
            right_on="shape_id",
            how="left",
        )
        link_shapes_df["geometry"] = link_shapes_df["geometry_y"].combine_first(
            link_shapes_df["geometry_x"]
        )
        link_shapes_df = link_shapes_df.drop(columns=["geometry_x", "geometry_y"])
        link_shapes_df = link_shapes_df.set_geometry("geometry")
        return link_shapes_df

    def get_property_by_timespan_and_group(
        self,
        link_property: str,
        category: Optional[Union[str, int]] = DEFAULT_CATEGORY,
        timespan: Optional[TimespanString] = DEFAULT_TIMESPAN,
        strict_timespan_match: bool = False,
        min_overlap_minutes: int = 60,
    ) -> Any:
        """Returns a new dataframe with model_link_id and link property by category and timespan.

        Convenience method for backward compatability.

        Args:
            link_property: link property to query
            category: category to query or a list of categories. Defaults to DEFAULT_CATEGORY.
            timespan: timespan to query in the form of ["HH:MM","HH:MM"].
                Defaults to DEFAULT_TIMESPAN.
            strict_timespan_match: If True, will only return links that match the timespan exactly.
                Defaults to False.
            min_overlap_minutes: If strict_timespan_match is False, will return links that overlap
                with the timespan by at least this many minutes. Defaults to 60.
        """
        from .links.scopes import prop_for_scope  # noqa: PLC0415

        return prop_for_scope(
            self.links_df,
            link_property,
            timespan=timespan,
            category=category,
            strict_timespan_match=strict_timespan_match,
            min_overlap_minutes=min_overlap_minutes,
        )

    def get_selection(
        self,
        selection_dict: Union[dict, SelectFacility],
        overwrite: bool = False,
    ) -> Union[RoadwayNodeSelection, RoadwayLinkSelection]:
        """Return selection if it already exists, otherwise performs selection.

        Args:
            selection_dict (dict): SelectFacility dictionary.
            overwrite: if True, will overwrite any previously cached searches. Defaults to False.
        """
        key = _create_selection_key(selection_dict)
        if (key in self._selections) and not overwrite:
            WranglerLogger.debug(f"Using cached selection from key: {key}")
            return self._selections[key]

        if isinstance(selection_dict, SelectFacility):
            selection_data = selection_dict
        elif isinstance(selection_dict, SelectLinksDict):
            selection_data = SelectFacility(links=selection_dict)
        elif isinstance(selection_dict, SelectNodesDict):
            selection_data = SelectFacility(nodes=selection_dict)
        elif isinstance(selection_dict, dict):
            selection_data = SelectFacility(**selection_dict)
        else:
            msg = "selection_dict arg must be a dictionary or SelectFacility model."
            WranglerLogger.error(
                msg + f" Received: {selection_dict} of type {type(selection_dict)}"
            )
            raise SelectionError(msg)

        WranglerLogger.debug(f"Getting selection from key: {key}  selection_data={selection_data}")
        if "links" in selection_data.fields:
            return RoadwayLinkSelection(self, selection_dict)
        if "nodes" in selection_data.fields:
            return RoadwayNodeSelection(self, selection_dict)
        msg = "Selection data should have either 'links' or 'nodes'."
        WranglerLogger.error(msg + f" Received: {selection_dict}")
        raise SelectionError(msg)

    def modal_graph_hash(self, mode) -> str:
        """Hash of the links in order to detect a network change from when graph created."""
        _value = str.encode(self.links_df.df_hash() + "-" + mode)
        _hash = hashlib.sha256(_value).hexdigest()

        return _hash

    def get_modal_graph(self, mode) -> MultiDiGraph:
        """Return a networkx graph of the network for a specific mode.

        Args:
            mode: mode of the network, one of `drive`,`transit`,`walk`, `bike`
        """
        from .graph import net_to_graph  # noqa: PLC0415

        if self._modal_graphs[mode]["hash"] != self.modal_graph_hash(mode):
            self._modal_graphs[mode]["graph"] = net_to_graph(self, mode)

        return self._modal_graphs[mode]["graph"]

    def apply(
        self,
        project_card: Union[ProjectCard, dict],
        transit_net: Optional[TransitNetwork] = None,
        **kwargs,
    ) -> RoadwayNetwork:
        """Wrapper method to apply a roadway project, returning a new RoadwayNetwork instance.

        Args:
            project_card: either a dictionary of the project card object or ProjectCard instance
            transit_net: optional transit network which will be used to if project requires as
                noted in `SECONDARY_TRANSIT_CARD_TYPES`.  If no transit network is provided, will
                skip anything related to transit network.
            **kwargs: keyword arguments to pass to project application
        """
        if not (isinstance(project_card, (ProjectCard, SubProject))):
            project_card = ProjectCard(project_card)

        # project_card.validate()
        if not project_card.valid:
            msg = f"Project card {project_card.project} not valid."
            WranglerLogger.error(msg)
            raise ProjectCardError(msg)

        if project_card._sub_projects:
            for sp in project_card._sub_projects:
                WranglerLogger.debug(f"- applying subproject: {sp.change_type}")
                self._apply_change(sp, transit_net=transit_net, **kwargs)
            return self
        return self._apply_change(project_card, transit_net=transit_net, **kwargs)

    def _apply_change(
        self,
        change: Union[ProjectCard, SubProject],
        transit_net: Optional[TransitNetwork] = None,
    ) -> RoadwayNetwork:
        """Apply a single change: a single-project project or a sub-project."""
        if not isinstance(change, SubProject):
            WranglerLogger.info(f"Applying Project to Roadway Network: {change.project}")

        if change.change_type == "roadway_property_change":
            return apply_roadway_property_change(
                self,
                self.get_selection(change.roadway_property_change["facility"]),
                change.roadway_property_change["property_changes"],
                project_name=change.project,
            )

        if change.change_type == "roadway_addition":
            return apply_new_roadway(
                self,
                change.roadway_addition,
                project_name=change.project,
            )

        if change.change_type == "roadway_deletion":
            return apply_roadway_deletion(
                self,
                change.roadway_deletion,
                transit_net=transit_net,
            )

        if change.change_type == "pycode":
            return apply_calculated_roadway(self, change.pycode)
        WranglerLogger.error(f"Couldn't find project in: \n{change.__dict__}")
        msg = f"Invalid Project Card Category: {change.change_type}"
        raise ProjectCardError(msg)

    def links_with_link_ids(self, link_ids: list[int]) -> pd.DataFrame:
        """Return subset of links_df based on link_ids list."""
        return filter_links_to_ids(self.links_df, link_ids)

    def links_with_nodes(self, node_ids: list[int]) -> pd.DataFrame:
        """Return subset of links_df based on node_ids list."""
        return filter_links_to_node_ids(self.links_df, node_ids)

    def nodes_in_links(self) -> pd.DataFrame:
        """Returns subset of self.nodes_df that are in self.links_df."""
        return filter_nodes_to_links(self.links_df, self.nodes_df)

    def node_coords(self, model_node_id: int) -> tuple:
        """Return coordinates (x, y) of a node based on model_node_id."""
        try:
            node = self.nodes_df[self.nodes_df.model_node_id == model_node_id]
        except ValueError as err:
            msg = f"Node with model_node_id {model_node_id} not found."
            WranglerLogger.error(msg)
            raise NodeNotFoundError(msg) from err
        return node.geometry.x.values[0], node.geometry.y.values[0]

    def add_links(
        self,
        add_links_df: pd.DataFrame,
        in_crs: int = LAT_LON_CRS,
    ):
        """Validate combined links_df with LinksSchema before adding to self.links_df.

        Args:
            add_links_df: Dataframe of additional links to add.
            in_crs: crs of input data. Defaults to LAT_LON_CRS.
        """
        dupe_recs = self.links_df.model_link_id.isin(add_links_df.model_link_id)

        if dupe_recs.any():
            dupe_ids = self.links_df.loc[dupe_recs, "model_link_id"]
            WranglerLogger.error(
                f"Cannot add links with model_link_id already in network: {dupe_ids}"
            )
            msg = "Cannot add links with model_link_id already in network."
            raise LinkAddError(msg)

        if add_links_df.attrs.get("name") != "road_links":
            add_links_df = data_to_links_df(add_links_df, nodes_df=self.nodes_df, in_crs=in_crs)
        self.links_df = validate_df_to_model(
            concat_with_attr([self.links_df, add_links_df], axis=0), RoadLinksTable
        )

    def add_nodes(
        self,
        add_nodes_df: pd.DataFrame,
        in_crs: int = LAT_LON_CRS,
    ):
        """Validate combined nodes_df with NodesSchema before adding to self.nodes_df.

        Args:
            add_nodes_df: Dataframe of additional nodes to add.
            in_crs: crs of input data. Defaults to LAT_LON_CRS.
        """
        dupe_ids = self.nodes_df.model_node_id.isin(add_nodes_df.model_node_id)
        if dupe_ids.any():
            WranglerLogger.error(
                f"Cannot add nodes with model_node_id already in network: {dupe_ids}"
            )
            msg = "Cannot add nodes with model_node_id already in network."
            raise NodeAddError(msg)
        WranglerLogger.debug(f"add_nodes(): self.nodes_df.tail()\n{self.nodes_df.tail()}")
        WranglerLogger.debug(f"add_nodes(): add_nodes_df:\n{add_nodes_df}")

        # this will perform validation to the nodes schema
        self.nodes_df = data_to_nodes_df(
            nodes_df = concat_with_attr([self.nodes_df, add_nodes_df], axis=0),
            in_crs = in_crs
        )
        # Ensure attrs are preserved after validation
        self.nodes_df.attrs.update(RoadNodesAttrs)
        if self.nodes_df.attrs.get("name") != "road_nodes":
            msg = f"Expected nodes_df to have name 'road_nodes', got {self.nodes_df.attrs.get('name')}"
            raise NotNodesError(msg)

    def add_shapes(
        self,
        add_shapes_df: pd.DataFrame,
        in_crs: int = LAT_LON_CRS,
    ):
        """Validate combined shapes_df with RoadShapesTable efore adding to self.shapes_df.

        Args:
            add_shapes_df: Dataframe of additional shapes to add.
            in_crs: crs of input data. Defaults to LAT_LON_CRS.
        """
        if len(self.shapes_df) > 0:
            dupe_ids = self.shapes_df["shape_id"].isin(add_shapes_df["shape_id"])
            if dupe_ids.any():
                msg = "Cannot add shapes with shape_id already in network."
                WranglerLogger.error(msg + f"\nDuplicates: {dupe_ids}")
                raise ShapeAddError(msg)

        if add_shapes_df.attrs.get("name") != "road_shapes":
            add_shapes_df = df_to_shapes_df(add_shapes_df, in_crs=in_crs, config=self.config)

        WranglerLogger.debug(f"add_shapes_df: \n{add_shapes_df}")
        WranglerLogger.debug(f"self.shapes_df: \n{self.shapes_df}")

        self.shapes_df = validate_df_to_model(
            concat_with_attr([self.shapes_df, add_shapes_df], axis=0), RoadShapesTable
        )

    def delete_links(
        self,
        selection_dict: Union[dict, SelectLinksDict],
        clean_nodes: bool = False,
        clean_shapes: bool = False,
        transit_net: Optional[TransitNetwork] = None,
    ):
        """Deletes links based on selection dictionary and optionally associated nodes and shapes.

        Args:
            selection_dict (SelectLinks): Dictionary describing link selections as follows:
                `all`: Optional[bool] = False. If true, will select all.
                `name`: Optional[list[str]]
                `ref`: Optional[list[str]]
                `osm_link_id`:Optional[list[str]]
                `model_link_id`: Optional[list[int]]
                `modes`: Optional[list[str]]. Defaults to "any"
                `ignore_missing`: if true, will not error when defaults to True.
                ...plus any other link property to select on top of these.
            clean_nodes (bool, optional): If True, will clean nodes uniquely associated with
                deleted links. Defaults to False.
            clean_shapes (bool, optional): If True, will clean nodes uniquely associated with
                deleted links. Defaults to False.
            transit_net (TransitNetwork, optional): If provided, will check TransitNetwork and
                warn if deletion breaks transit shapes. Defaults to None.
        """
        if not isinstance(selection_dict, SelectLinksDict):
            selection_dict = SelectLinksDict(**selection_dict)
        selection_dict = selection_dict.model_dump(exclude_none=True, by_alias=True)
        selection = self.get_selection({"links": selection_dict})
        if isinstance(selection, RoadwayNodeSelection):
            msg = "Selection should be for links, but got nodes."
            raise SelectionError(msg)
        if clean_nodes:
            node_ids_to_delete = node_ids_unique_to_link_ids(
                selection.selected_links, selection.selected_links_df, self.nodes_df
            )
            WranglerLogger.debug(
                f"Dropping nodes associated with dropped links: \n{node_ids_to_delete}"
            )
            self.nodes_df = delete_nodes_by_ids(self.nodes_df, del_node_ids=node_ids_to_delete)

        if clean_shapes:
            shape_ids_to_delete = shape_ids_unique_to_link_ids(
                selection.selected_links, selection.selected_links_df, self.shapes_df
            )
            WranglerLogger.debug(
                f"Dropping shapes associated with dropped links: \n{shape_ids_to_delete}"
            )
            self.shapes_df = delete_shapes_by_ids(
                self.shapes_df, del_shape_ids=shape_ids_to_delete
            )

        self.links_df = delete_links_by_ids(
            self.links_df,
            selection.selected_links,
            ignore_missing=selection.ignore_missing,
            transit_net=transit_net,
        )

    def delete_nodes(
        self,
        selection_dict: Union[dict, SelectNodesDict],
        remove_links: bool = False,
    ) -> None:
        """Deletes nodes from roadway network. Wont delete nodes used by links in network.

        Args:
            selection_dict: dictionary of node selection criteria in the form of a SelectNodesDict.
            remove_links: if True, will remove any links that are associated with the nodes.
                If False, will only remove nodes if they are not associated with any links.
                Defaults to False.

        Raises:
            NodeDeletionError: If not ignore_missing and selected nodes to delete aren't in network
        """
        if not isinstance(selection_dict, SelectNodesDict):
            selection_dict = SelectNodesDict(**selection_dict)
        selection_dict = selection_dict.model_dump(exclude_none=True, by_alias=True)
        _selection = self.get_selection({"nodes": selection_dict})
        assert isinstance(_selection, RoadwayNodeSelection)  # for mypy
        selection: RoadwayNodeSelection = _selection
        if remove_links:
            del_node_ids = selection.selected_nodes
            link_ids = self.links_with_nodes(selection.selected_nodes).model_link_id.to_list()
            WranglerLogger.info(f"Removing {len(link_ids)} links associated with nodes.")
            self.delete_links({"model_link_id": link_ids})
        else:
            unused_node_ids = node_ids_without_links(self.nodes_df, self.links_df)
            del_node_ids = list(set(selection.selected_nodes).intersection(unused_node_ids))

        self.nodes_df = delete_nodes_by_ids(
            self.nodes_df, del_node_ids, ignore_missing=selection.ignore_missing
        )

    def split_link(  # noqa: PLR0912, PLR0915
        self,
        A: int,
        B: int,
        new_model_node_id: int,
        fraction: float,
        split_reverse_link: bool = True,
    ) -> None:
        """Splits a link into two links at a specified fraction of its length.

        Args:
            A: The model_node_id of the start node of the link to split.
            B: The model_node_id of the end node of the link to split.
            new_model_node_id: The model_node_id for the new node to be created at the split point.
            fraction: The fraction along the link's length where the split occurs (0 < fraction < 1).
            split_reverse_link: If True, also splits the reverse link if it exists. Defaults to True.

        Raises:
            ValueError: If the link doesn't exist, fraction is invalid, or new_model_node_id already exists.

        TODO: Use SelectionDictionary rather than A,B?
        TODO: Make new_model_node_id an optional argument because we can use
              `network_wrangler.roadway.nodes.create.generate_node_ids()`
        """
        WranglerLogger.debug(f"split_link() self.links_df.head():\n{self.links_df.head()}")
        # Find the link with given A and B nodes
        matching_links = self.links_df[(self.links_df["A"] == A) & (self.links_df["B"] == B)]

        if matching_links.empty:
            msg = f"Link from node {A} to node {B} not found in network."
            raise ValueError(msg)

        # If multiple links exist between A and B, use the first one and warn
        if len(matching_links) > 1:
            WranglerLogger.warning(
                f"Multiple links found from node {A} to {B}. Using first link with model_link_id {matching_links.index[0]}."
            )

        model_link_idx = matching_links.index[0]

        if not 0 < fraction < 1:
            msg = f"Fraction must be between 0 and 1 (exclusive), got {fraction}."
            raise ValueError(msg)

        if new_model_node_id in self.nodes_df.index:
            msg = f"Node with model_node_id {new_model_node_id} already exists in network."
            raise ValueError(msg)

        # Get the original link
        orig_link = self.links_df.loc[model_link_idx].copy()
        WranglerLogger.debug(f"Splitting link:\n{orig_link}")

        # Get the geometry to split (use shape geometry if available)
        if pd.notna(orig_link.get("shape_id")) and orig_link["shape_id"] in self.shapes_df.index:
            geometry = self.shapes_df.loc[orig_link["shape_id"], "geometry"]
        else:
            geometry = orig_link["geometry"]

        # Calculate the split point
        split_point = geometry.interpolate(fraction, normalized=True)

        # Create the new node
        # The attribute osm_node_id is not required so don't set it
        new_node_data = {
            "model_node_id": new_model_node_id,
            "X": split_point.x,
            "Y": split_point.y,
        }

        # Copy additional attributes from node A if they exist
        node_a_data = self.nodes_df.loc[self.nodes_df["model_node_id"] == orig_link["A"]].iloc[0]
        # Copy all columns except geometry, X, Y, and model_node_id
        for col in self.nodes_df.columns:
            if (
                col not in ["model_node_id", "X", "Y", "geometry", "osm_node_id"]
                and col in node_a_data.index
                and pd.notna(node_a_data[col])
            ):
                new_node_data[col] = node_a_data[col]

        # Create DataFrame for new node and add it
        new_node_df = pd.DataFrame([new_node_data])
        self.add_nodes(new_node_df)

        # Split the geometry
        # Create a small circle around the split point to ensure clean split
        split_geom = split_point.buffer(0.00001)  # Small buffer in degrees
        split_result = split(geometry, split_geom)

        if len(split_result.geoms) >= MIN_SPLIT_SEGMENTS:
            geom1 = split_result.geoms[0]
            geom2 = split_result.geoms[-1]
        else:
            # Fallback: manually create two linestrings
            coords = list(geometry.coords)
            split_idx = int(len(coords) * fraction)
            if split_idx == 0:
                split_idx = 1
            elif split_idx >= len(coords):
                split_idx = len(coords) - 1

            # Insert the split point at the right position
            coords_1 = [*coords[:split_idx], (split_point.x, split_point.y)]
            coords_2 = [(split_point.x, split_point.y), *coords[split_idx:]]

            geom1 = LineString(coords_1)
            geom2 = LineString(coords_2)

        # Create two new links
        # Find the next available model_link_ids
        max_link_id = self.links_df["model_link_id"].max()
        new_link1_id = max_link_id + 1
        new_link2_id = max_link_id + 2

        # First link: A to new node
        link1_data = orig_link.to_dict()
        link1_data["model_link_id"] = new_link1_id
        link1_data["B"] = new_model_node_id
        link1_data["geometry"] = geom1
        link1_data["distance"] = (
            orig_link.get("distance", 0) * fraction if "distance" in orig_link else None
        )
        # Clear shape_id as we're creating new geometry
        link1_data["shape_id"] = None

        # Second link: new node to B
        link2_data = orig_link.to_dict()
        link2_data["model_link_id"] = new_link2_id
        link2_data["A"] = new_model_node_id
        link2_data["geometry"] = geom2
        link2_data["distance"] = (
            orig_link.get("distance", 0) * (1 - fraction) if "distance" in orig_link else None
        )
        # Clear shape_id as we're creating new geometry
        link2_data["shape_id"] = None

        # Handle reverse link if requested
        reverse_link_ids = []
        if split_reverse_link:
            # Look for reverse link (B->A)
            reverse_link = self.links_df[
                (self.links_df["A"] == orig_link["B"]) & (self.links_df["B"] == orig_link["A"])
            ]

            if not reverse_link.empty:
                reverse_link_data = reverse_link.iloc[0].copy()

                # Get reverse geometry
                if (
                    pd.notna(reverse_link_data.get("shape_id"))
                    and reverse_link_data["shape_id"] in self.shapes_df.index
                ):
                    reverse_geometry = self.shapes_df.loc[
                        reverse_link_data["shape_id"], "geometry"
                    ]
                else:
                    reverse_geometry = reverse_link_data["geometry"]

                # Split at (1 - fraction) for reverse
                reverse_split_point = reverse_geometry.interpolate(1 - fraction, normalized=True)

                # Split the reverse geometry
                reverse_split_geom = reverse_split_point.buffer(0.00001)
                reverse_split_result = split(reverse_geometry, reverse_split_geom)

                if len(reverse_split_result.geoms) >= MIN_SPLIT_SEGMENTS:
                    reverse_geom1 = reverse_split_result.geoms[0]
                    reverse_geom2 = reverse_split_result.geoms[-1]
                else:
                    # Fallback for reverse
                    coords = list(reverse_geometry.coords)
                    split_idx = int(len(coords) * (1 - fraction))
                    if split_idx == 0:
                        split_idx = 1
                    elif split_idx >= len(coords):
                        split_idx = len(coords) - 1

                    coords_1 = [*coords[:split_idx], (split_point.x, split_point.y)]
                    coords_2 = [(split_point.x, split_point.y), *coords[split_idx:]]

                    reverse_geom1 = LineString(coords_1)
                    reverse_geom2 = LineString(coords_2)

                # Create reverse links
                new_link3_id = max_link_id + 3
                new_link4_id = max_link_id + 4

                # Reverse link 1: original B to new node
                link3_data = reverse_link_data.to_dict()
                link3_data["model_link_id"] = new_link3_id
                link3_data["B"] = new_model_node_id
                link3_data["geometry"] = reverse_geom1
                link3_data["distance"] = (
                    reverse_link_data.get("distance", 0) * (1 - fraction)
                    if "distance" in reverse_link_data
                    else None
                )
                link3_data["shape_id"] = None

                # Reverse link 2: new node to original A
                link4_data = reverse_link_data.to_dict()
                link4_data["model_link_id"] = new_link4_id
                link4_data["A"] = new_model_node_id
                link4_data["geometry"] = reverse_geom2
                link4_data["distance"] = (
                    reverse_link_data.get("distance", 0) * fraction
                    if "distance" in reverse_link_data
                    else None
                )
                link4_data["shape_id"] = None

                reverse_link_ids = [reverse_link_data["model_link_id"]]

        # Delete original links - we need to delete by the model_link_id values
        orig_model_link_id = orig_link["model_link_id"]
        links_to_delete = [orig_model_link_id, *reverse_link_ids]
        self.delete_links(
            {"model_link_id": links_to_delete, "modes": ["any"]},
            clean_nodes=False,
            clean_shapes=False,
        )

        # Add new links
        new_links_data = [link1_data, link2_data]
        if reverse_link_ids:
            new_links_data.extend([link3_data, link4_data])

        new_links_df = pd.DataFrame(new_links_data)
        self.add_links(new_links_df)

        WranglerLogger.info(
            f"Split link {orig_model_link_id} at fraction {fraction} with new node {new_model_node_id}. "
            f"Created links {new_link1_id} and {new_link2_id}."
        )

        if reverse_link_ids:
            WranglerLogger.info(
                f"Also split reverse link {reverse_link_ids[0]} creating links {new_link3_id} and {new_link4_id}."
            )

    def clean_unused_shapes(self):
        """Removes any unused shapes from network that aren't referenced by links_df."""
        from .shapes.shapes import shape_ids_without_links  # noqa: PLC0415

        del_shape_ids = shape_ids_without_links(self.shapes_df, self.links_df)
        self.shapes_df = self.shapes_df.drop(del_shape_ids)

    def clean_unused_nodes(self):
        """Removes any unused nodes from network that aren't referenced by links_df.

        NOTE: does not check if these nodes are used by transit, so use with caution.
        """
        from .nodes.nodes import node_ids_without_links  # noqa: PLC0415

        node_ids = node_ids_without_links(self.nodes_df, self.links_df)
        self.nodes_df = self.nodes_df.drop(node_ids)

    def move_nodes(
        self,
        node_geometry_change_table: pd.DataFrame,
    ):
        """Moves nodes based on updated geometry along with associated links and shape geometry.

        Args:
            node_geometry_change_table: a table with model_node_id, X, Y, and CRS.
        """
        node_geometry_change_table = NodeGeometryChangeTable(node_geometry_change_table)
        node_ids = node_geometry_change_table.model_node_id.to_list()
        WranglerLogger.debug(f"Moving nodes:\n{node_geometry_change_table}")
        self.nodes_df = edit_node_geometry(self.nodes_df, node_geometry_change_table)
        WranglerLogger.debug(f"Completed edit_node_geometry()")
        self.links_df = edit_link_geometry_from_nodes(self.links_df, self.nodes_df, node_ids)
        WranglerLogger.debug(f"Completed edit_link_geometry_from_nodes()")
        self.shapes_df = edit_shape_geometry_from_nodes(
            self.shapes_df, self.links_df, self.nodes_df, node_ids
        )
        WranglerLogger.debug(f"Completed edit_shape_geometry_from_nodes()")

    def has_node(self, model_node_id: int) -> bool:
        """Queries if network has node based on model_node_id.

        Args:
            model_node_id: model_node_id to check for.
        """
        has_node = self.nodes_df[self.nodes_df.model_node_id].isin([model_node_id]).any()

        return has_node

    def has_link(self, ab: tuple) -> bool:
        """Returns true if network has links with AB values.

        Args:
            ab: Tuple of values corresponding with A and B.
        """
        sel_a, sel_b = ab
        has_link = (
            self.links_df[self.links_df[["A", "B"]]].isin_dict({"A": sel_a, "B": sel_b}).any()
        )
        return has_link

    def is_connected(self, mode: str) -> bool:
        """Determines if the network graph is "strongly" connected.

        A graph is strongly connected if each vertex is reachable from every other vertex.

        Args:
            mode:  mode of the network, one of `drive`,`transit`,`walk`, `bike`
        """
        is_connected = nx.is_strongly_connected(self.get_modal_graph(mode))

        return is_connected


def add_incident_link_data_to_nodes(
    links_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    link_variables: Optional[list] = None,
) -> pd.DataFrame:
    """Add data from links going to/from nodes to node.

    Args:
        links_df: Will assess connectivity of this links list
        nodes_df: Will assess connectivity of this nodes list
        link_variables: list of columns in links dataframe to add to incident nodes

    Returns:
        nodes DataFrame with link data where length is N*number of links going in/out
    """
    WranglerLogger.debug("Adding following link data to nodes: ".format())
    link_variables = link_variables or []

    _link_vals_to_nodes = [x for x in link_variables if x in links_df.columns]
    if link_variables not in _link_vals_to_nodes:
        WranglerLogger.warning(
            f"Following columns not in links_df and wont be added to nodes: {list(set(link_variables) - set(_link_vals_to_nodes))} "
        )

    _nodes_from_links_A = nodes_df.merge(
        links_df[[links_df.A, *_link_vals_to_nodes]],
        how="outer",
        left_on=nodes_df.model_node_id,
        right_on=links_df.A,
    )
    _nodes_from_links_B = nodes_df.merge(
        links_df[[links_df.B, *_link_vals_to_nodes]],
        how="outer",
        left_on=nodes_df.model_node_id,
        right_on=links_df.B,
    )
    _nodes_from_links_ab = concat_with_attr([_nodes_from_links_A, _nodes_from_links_B])

    return _nodes_from_links_ab
