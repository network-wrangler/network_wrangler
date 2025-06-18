"""Functions to create a model roadway network from a roadway network."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import geopandas as gpd
import pandas as pd
from pandera.typing import DataFrame

from ..configs import DefaultConfig
from ..errors import ManagedLaneAccessEgressError
from ..logger import WranglerLogger
from ..models._base.types import RoadwayFileTypes
from ..models.roadway.tables import RoadLinksTable, RoadNodesTable, RoadShapesTable
from ..utils.data import concat_with_attr
from .io import write_roadway
from .links.create import copy_links, data_to_links_df
from .links.edit import _initialize_links_as_managed_lanes
from .links.filters import (
    filter_link_properties_managed_lanes,
    filter_links_to_ml_access_points,
    filter_links_to_ml_egress_points,
)
from .links.links import node_ids_in_links
from .nodes.create import _create_nodes_from_link
from .utils import compare_links, compare_networks

if TYPE_CHECKING:
    from .network import RoadwayNetwork

"""
list of attributes to copy from a general purpose lane to managed lane
   so long as a ML_<prop_name> doesn't exist.
"""
COPY_FROM_GP_TO_ML: list[str] = [
    "ref",
    "roadway",
    "access",
    "distance",
    "bike_access",
    "drive_access",
    "walk_access",
    "bus_only",
    "rail_only",
]

"""
List of attributes to copy from a general purpose lane to access and egress dummy links.
"""
COPY_TO_ACCESS_EGRESS: list[str] = [
    "ref",
    "ML_access",
    "ML_drive_access",
    "ML_bus_only",
    "ML_rail_only",
]

"""
List of attributes that must be provided in managed lanes.
"""
MANAGED_LANES_REQUIRED_ATTRIBUTES: list[str] = [
    "A",
    "B",
    "model_link_id",
]


class ModelRoadwayNetwork:
    """Roadway Network Object compatible with travel modeling.

    Compatability includes:
    (1) separation of managed lane facilities and their connection to general purpose lanes
        using dummy links.

    Attr:
        net: associated RoadwayNetwork object
        links_df: dataframe of model-compatible links
        nodes_df: dataframe of model-compatible nodes
        ml_link_id_lookup:  lookup from general purpose link ids to link ids  of their
            managed lane counterparts.
        ml_node_id_lookup: lookup from general purpose node ids to node ids of their
            managed lane counterparts.
        _net_hash: hash of the the input links and nodes in order to detect changes.

    """

    def __init__(
        self,
        net,
        ml_link_id_lookup: Optional[dict[int, int]] = None,
        ml_node_id_lookup: Optional[dict[int, int]] = None,
    ):
        """Constructor for ModelRoadwayNetwork.

        NOTE: in order to be associated with the RoadwayNetwork, this should be called from
        RoadwayNetwork.model_net which will lazily construct it.

        Args:
            net: Associated roadway network.
            ml_link_id_lookup (dict[int, int]): lookup from general purpose link ids to link ids
                of their managed lane counterparts. Defaults to None which will generate a new one
                using the provided method.
            ml_node_id_lookup (dict[int, int]): lookup from general purpose node ids to node ids
                of their managed lane counterparts. Defaults to None which will generate a new one
                using the provided method.
        """
        self.net = net

        if ml_link_id_lookup is None:
            if self.net.config.IDS.ML_LINK_ID_METHOD == "range":
                self.ml_link_id_lookup = _generate_ml_link_id_lookup_from_range(
                    self.net.links_df, self.net.config.IDS.ML_LINK_ID_RANGE
                )
            elif self.net.config.IDS.ML_LINK_ID_METHOD == "scalar":
                self.ml_link_id_lookup = _generate_ml_link_id_lookup_from_scalar(
                    self.net.links_df, self.net.config.IDS.ML_LINK_ID_SCALAR
                )
            else:
                msg = "ml_link_id_method must be 'range' or 'scalar'."
                WranglerLogger.error(msg + f" Got {self.net.config.IDS.ML_LINK_ID_METHOD}")
                raise ValueError(msg)
        else:
            self.ml_link_id_lookup = ml_link_id_lookup

        if ml_node_id_lookup is None:
            if self.net.config.IDS.ML_NODE_ID_METHOD == "range":
                self.ml_node_id_lookup = _generate_ml_node_id_from_range(
                    self.net.nodes_df, self.net.links_df, self.net.config.IDS.ML_NODE_ID_RANGE
                )
            elif self.net.config.IDS.ML_NODE_ID_METHOD == "scalar":
                self.ml_node_id_lookup = _generate_ml_node_id_lookup_from_scalar(
                    self.net.nodes_df, self.net.links_df, self.net.config.IDS.ML_NODE_ID_SCALAR
                )
            else:
                msg = "ml_node_id_method must be 'range' or 'scalar'."
                WranglerLogger.error(msg + f" Got {self.net.config.IDS.ML_NODE_ID_METHOD}")
                raise ValueError(msg)
        else:
            self.ml_node_id_lookup = ml_node_id_lookup

        if len(self.net.links_df.of_type.managed) == 0:
            self.links_df, self.nodes_df = self.net.links_df, self.net.nodes_df
        else:
            self.links_df, self.nodes_df = model_links_nodes_from_net(
                self.net, self.ml_link_id_lookup, self.ml_node_id_lookup
            )
        self._net_hash = copy.deepcopy(net.network_hash)

    @property
    def ml_config(self) -> dict:
        """Convenience method for lanaged lane configuration."""
        return self.net.config.MODEL_ROADWAY

    @property
    def shapes_df(self) -> DataFrame[RoadShapesTable]:
        """Shapes dataframe."""
        return self.net.shapes_df

    @property
    def ml_links_df(self) -> pd.DataFrame:
        """Managed lanes links."""
        return self.links_df.of_type.managed

    @property
    def gp_links_df(self) -> pd.DataFrame:
        """GP lanes on links that have managed lanes next to them."""
        return self.links_df.of_type.parallel_general_purpose

    @property
    def dummy_links_df(self) -> pd.DataFrame:
        """GP lanes on links that have managed lanes next to them."""
        return self.links_df.of_type.dummy

    @property
    def summary(self) -> dict:
        """Quick summary dictionary of number of links, nodes."""
        d = {"links": len(self.links_df), "nodes": len(self.nodes_df)}
        return d

    @property
    def compare_links_df(self) -> pd.DataFrame:
        """Comparison of the original network and the model network."""
        return compare_links([self.net.links_df, self.links_df], names=["Roadway", "ModelRoadway"])

    @property
    def compare_net_df(self) -> pd.DataFrame:
        """Comparison of the original network and the model network."""
        return compare_networks([self.net, self], names=["Roadway", "ModelRoadway"])

    def write(
        self,
        out_dir: Path = Path(),
        convert_complex_link_properties_to_single_field: bool = False,
        prefix: str = "",
        file_format: RoadwayFileTypes = "geojson",
        overwrite: bool = True,
        true_shape: bool = False,
    ) -> None:
        """Writes a network in the roadway network standard.

        Args:
            out_dir: the path were the output will be saved.
            convert_complex_link_properties_to_single_field: if True, will convert complex properties to a
                single column consistent with v0 format.  This format is NOT valid
                with parquet and many other softwares. Defaults to False.
            prefix: the name prefix of the roadway files that will be generated.
            file_format: the format of the output files. Defaults to "geojson".
            overwrite: if True, will overwrite the files if they already exist. Defaults to True.
            true_shape: if True, will write the true shape of the links as found from shapes.
                Defaults to False.
        """
        write_roadway(
            self,
            out_dir=out_dir,
            convert_complex_link_properties_to_single_field=convert_complex_link_properties_to_single_field,
            prefix=prefix,
            file_format=file_format,
            overwrite=overwrite,
            true_shape=true_shape,
        )


def _generate_ml_link_id_lookup_from_range(links_df, link_id_range: tuple[int]):
    """Generate a lookup from general purpose link ids to link ids their managed lane counterparts.

    Will be divisable by LINK_IDS_DIVISIBLE_BY which defaults to 10.
    """
    LINK_IDS_DIVISIBLE_BY = 10
    og_ml_link_ids = links_df.of_type.managed.model_link_id
    link_id_list = [i for i in range(*link_id_range) if i % LINK_IDS_DIVISIBLE_BY == 0]
    avail_ml_link_ids = set(link_id_list) - set(links_df.model_link_id.unique().tolist())
    if len(avail_ml_link_ids) < len(og_ml_link_ids):
        msg = f"{len(avail_ml_link_ids)} of {len(og_ml_link_ids)} new link ids\
                         available for provided range: {link_id_range}."
        raise ValueError(msg)
    new_link_ids = list(avail_ml_link_ids)[: len(og_ml_link_ids)]
    return dict(zip(og_ml_link_ids, new_link_ids))


def _generate_ml_node_id_from_range(nodes_df, links_df, node_id_range: tuple[int]):
    """Generate a lookup for managed lane node ids to their general purpose lane counterparts."""
    og_ml_node_ids = node_ids_in_links(links_df.of_type.managed, nodes_df)
    avail_ml_node_ids = set(range(*node_id_range)) - set(nodes_df.model_node_id.unique().tolist())
    if len(avail_ml_node_ids) < len(og_ml_node_ids):
        msg = f"{len(avail_ml_node_ids)} of {len(og_ml_node_ids)} new nodes ids\
               available for provided range: {node_id_range}."
        raise ValueError(msg)
    new_ml_node_ids = list(avail_ml_node_ids)[: len(og_ml_node_ids)]
    return dict(zip(og_ml_node_ids.tolist(), new_ml_node_ids))


def _generate_ml_link_id_lookup_from_scalar(links_df: DataFrame[RoadLinksTable], scalar: int):
    """Generate a lookup from general purpose link ids to their managed lane counterparts."""
    og_ml_link_ids = links_df.of_type.managed.model_link_id
    link_id_list = [i + scalar for i in og_ml_link_ids]
    if links_df.model_link_id.isin(link_id_list).any():
        msg = f"New link ids generated by scalar {scalar} already exist. Try a different scalar."
        raise ValueError(msg)
    return dict(zip(og_ml_link_ids, link_id_list))


def _generate_ml_node_id_lookup_from_scalar(nodes_df, links_df, scalar: int):
    """Generate a lookup for managed lane node ids to their general purpose lane counterparts."""
    og_ml_node_ids = node_ids_in_links(links_df.of_type.managed, nodes_df)
    node_id_list = og_ml_node_ids + scalar
    if nodes_df.model_node_id.isin(node_id_list).any():
        msg = f"New node ids generated by scalar {scalar} already exist. Try a different scalar."
        raise ValueError(msg)
    return dict(zip(og_ml_node_ids.tolist(), node_id_list.tolist()))


def model_links_nodes_from_net(
    net: RoadwayNetwork, ml_link_id_lookup: dict[int, int], ml_node_id_lookup: dict[int, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a roadway network with managed lanes links separated out.

    Add new parallel managed lane links, access/egress links,
    and add shapes corresponding to the new links

    Args:
        net: RoadwayNetwork instance
        ml_link_id_lookup: lookup table for managed lane link ids to their general purpose lane
            counterparts.
        ml_node_id_lookup: lookup table for managed lane node ids to their general purpose lane
            counterparts.

    returns: tuple of links and nodes dataframes with managed lanes separated out
    """
    WranglerLogger.info("Separating managed lane links from general purpose links")

    copy_cols_gp_ml = list(
        set(COPY_FROM_GP_TO_ML + net.config.MODEL_ROADWAY.ADDITIONAL_COPY_FROM_GP_TO_ML)
    )
    _m_links_df = _separate_ml_links(
        net.links_df,
        ml_link_id_lookup,
        ml_node_id_lookup,
        offset_meters=net.config.MODEL_ROADWAY.ML_OFFSET_METERS,
        copy_from_gp_to_ml=copy_cols_gp_ml,
    )
    _m_nodes_df = _create_ml_nodes_from_links(_m_links_df, ml_node_id_lookup)
    m_nodes_df = concat_with_attr([net.nodes_df, _m_nodes_df])

    copy_ae_fields = list(
        set(COPY_TO_ACCESS_EGRESS + net.config.MODEL_ROADWAY.ADDITIONAL_COPY_TO_ACCESS_EGRESS)
    )
    _access_egress_links_df = _create_dummy_connector_links(
        net.links_df,
        m_nodes_df,
        ml_link_id_lookup,
        ml_node_id_lookup,
        copy_fields=copy_ae_fields,
    )
    m_links_df = concat_with_attr([_m_links_df, _access_egress_links_df])
    return m_links_df, m_nodes_df


def _separate_ml_links(
    links_df: DataFrame[RoadLinksTable],
    link_id_lookup: dict[int, int],
    node_id_lookup: dict[int, int],
    offset_meters: float = DefaultConfig.MODEL_ROADWAY.ML_OFFSET_METERS,
    copy_from_gp_to_ml: list[str] = COPY_FROM_GP_TO_ML,
) -> gpd.GeoDataFrame:
    """Separate managed lane links from general purpose links."""
    no_ml_links_df = copy.deepcopy(links_df.of_type.general_purpose_no_parallel_managed)
    gp_links_df = _create_parallel_gp_lane_links(links_df)
    ml_links_df = _create_separate_managed_lane_links(
        links_df,
        link_id_lookup,
        node_id_lookup,
        offset_meters=offset_meters,
        copy_from_gp_to_ml=copy_from_gp_to_ml,
    )
    WranglerLogger.debug(
        f"Separated ML links: \
        \n  no parallel ML: {len(no_ml_links_df)}\
        \n  parallel GP: {len(gp_links_df)}\
        \n  separate ML: {len(ml_links_df)}"
    )

    m_links_df = concat_with_attr([ml_links_df, gp_links_df, no_ml_links_df], axis=0)

    return m_links_df


def _create_parallel_gp_lane_links(links_df: DataFrame[RoadLinksTable]) -> pd.DataFrame:
    """Create df with parallel general purpose lane links."""
    ml_properties = filter_link_properties_managed_lanes(links_df)
    keep_c = [c for c in links_df.columns if c not in ml_properties]
    gp_links_df = copy.deepcopy(links_df[keep_c].of_type.managed)
    gp_links_df["managed"] = -1
    return gp_links_df


def _create_separate_managed_lane_links(
    links_df: DataFrame[RoadLinksTable],
    link_id_lookup: dict[int, int],
    node_id_lookup: dict[int, int],
    offset_meters: float = DefaultConfig.MODEL_ROADWAY.ML_OFFSET_METERS,
    copy_from_gp_to_ml: list[str] = COPY_FROM_GP_TO_ML,
) -> tuple[gpd.GeoDataFrame]:
    """Creates df with separate links for managed lanes."""
    # make sure there are correct fields even if managed = 1 was set outside of wrangler
    links_df = _initialize_links_as_managed_lanes(
        links_df, links_df.of_type.managed.index.values, geometry_offset_meters=offset_meters
    )

    # columns to keep in order to use to create ML versions of them
    ML_MUST_KEEP_COLS = [
        "managed",
        "ML_access_point",
        "ML_egress_point",
    ]

    ML_POST_COPY_RENAME_COLS = {
        "source_A": "GP_A",
        "source_B": "GP_B",
    }

    ml_props = filter_link_properties_managed_lanes(links_df)
    ml_rename_props = dict(zip(ml_props, strip_ML_from_prop_list(ml_props)))

    ml_links_df = copy_links(
        links_df.of_type.managed,
        link_id_lookup=link_id_lookup,
        node_id_lookup=node_id_lookup,
        offset_meters=offset_meters,
        updated_geometry_col="ML_geometry",
        copy_properties=list(set(copy_from_gp_to_ml + ML_MUST_KEEP_COLS)),
        rename_properties=ml_rename_props,
        name_prefix="Managed Lane of",
    )

    ml_links_df = ml_links_df.rename(columns=ML_POST_COPY_RENAME_COLS)

    return ml_links_df


def _create_dummy_connector_links(
    links_df: DataFrame[RoadLinksTable],
    m_nodes_df: DataFrame[RoadNodesTable],
    ml_link_id_lookup: dict[int, int],
    ml_node_id_lookup: dict[int, int],
    copy_fields: list[str] = COPY_TO_ACCESS_EGRESS,
) -> DataFrame[RoadLinksTable]:
    """Create dummy connector links between the general purpose and managed lanes.

    Args:
        links_df: RoadLinksTable of network links
        m_nodes_df: GeoDataFrame of model nodes
        ml_link_id_lookup: lookup table for managed lane link ids to their general purpose lane
        ml_node_id_lookup: lookup table for managed lane node ids to their general purpose lane
        copy_fields: list of fields to copy from the general purpose links to the dummy links.
            Defaults to COPY_TO_ACCESS_EGRESS.

    returns: GeoDataFrame of access and egress dummy connector links to add to m_links_df
    """
    WranglerLogger.debug("Creating access and egress dummy connector links")
    # 1. Align the managed lane and associated general purpose lanes in the same records
    copy_cols = [c for c in copy_fields if c in links_df.columns]

    keep_for_calcs_cols = [
        "A",
        "B",
        "model_link_id",
        "name",
        "ML_access_point",
        "ML_egress_point",
    ]
    keep_cols = list(set(copy_cols + keep_for_calcs_cols))

    # 2 - Create access and egress link dataframes from aligned records
    # if ML_access_point is specified, only have access at those points. Same for egress.
    access_df = copy.deepcopy(filter_links_to_ml_access_points(links_df)[keep_cols])
    egress_df = copy.deepcopy(filter_links_to_ml_egress_points(links_df)[keep_cols])

    if len(access_df) == 0:
        msg = "No access points to managed lanes found."
        raise ManagedLaneAccessEgressError(msg)
    if len(egress_df) == 0:
        msg = "No egress points to managed lanes found."
        raise ManagedLaneAccessEgressError(msg)

    # access link should go from A_GP to A_ML
    access_df["B"] = access_df["A"].map(ml_node_id_lookup)
    access_df["GP_model_link_id"] = access_df["model_link_id"]
    access_df["model_link_id"] = 1000 + access_df["GP_model_link_id"].map(ml_link_id_lookup)
    access_df["name"] = "Access Dummy " + access_df["name"]
    access_df["roadway"] = "ml_access_point"
    access_df["model_link_id_idx"] = access_df["model_link_id"]
    access_df.set_index("model_link_id_idx", inplace=True)

    # egress link should go from B_ML to B_GP
    egress_df["A"] = egress_df["B"].map(ml_node_id_lookup)
    egress_df["GP_model_link_id"] = egress_df["model_link_id"]
    egress_df["model_link_id"] = 2000 + egress_df["GP_model_link_id"].map(ml_link_id_lookup)
    egress_df["name"] = "Egress Dummy " + egress_df["name"]
    egress_df["roadway"] = "ml_egress_point"
    egress_df["model_link_id_idx"] = egress_df["model_link_id"]
    egress_df.set_index("model_link_id_idx", inplace=True)

    # combine to one dataframe
    access_egress_df = concat_with_attr([access_df, egress_df], axis=0)

    # 3 - Determine property values
    access_egress_df["lanes"] = 1
    access_egress_df = access_egress_df.rename(
        columns=dict(zip(copy_cols, strip_ML_from_prop_list(copy_cols)))
    )

    # 5 - Add geometry
    access_egress_df = data_to_links_df(access_egress_df, nodes_df=m_nodes_df)
    WranglerLogger.debug(f"access_egress_df['geometry']: \n {access_egress_df['geometry']}")

    WranglerLogger.debug(f"Access Egress Links Created: {len(access_egress_df)}")

    return access_egress_df


def _create_ml_nodes_from_links(
    ml_links_df: DataFrame[RoadLinksTable],
    ml_node_id_lookup: dict[int, int],
) -> DataFrame[RoadNodesTable]:
    """Creates managed lane nodes from geometry already generated by links."""
    a_nodes_df = _create_nodes_from_link(ml_links_df.of_type.managed, 0, "A")
    b_nodes_df = _create_nodes_from_link(ml_links_df.of_type.managed, -1, "B")
    ml_nodes_df = a_nodes_df.combine_first(b_nodes_df)

    ml_nodes_df["GP_model_node_id"] = ml_nodes_df["model_node_id"].map(ml_node_id_lookup)
    return ml_nodes_df


def strip_ML_from_prop_list(property_list: list[str]) -> list[str]:
    """Strips 'ML_' from property list but keeps necessary access/egress point cols."""
    keep_same = ["ML_access_point", "ML_egress_point"]
    pl = [p.removeprefix("ML_") if p not in keep_same else p for p in property_list]
    pl = [p.replace("_ML_", "_") if p not in keep_same else p for p in pl]
    return pl
