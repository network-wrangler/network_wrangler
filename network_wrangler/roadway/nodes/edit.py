"""Edits RoadNodesTable properties.

NOTE: Each public method will return a new, whole copy of the RoadNodesTable with associated edits.
Private methods may return mutated originals.
"""

import copy

import geopandas as gpd
from pandera import DataFrameModel, Field
from pandera.typing import DataFrame, Series
from pydantic import ConfigDict

from ...configs import DefaultConfig, WranglerConfig
from ...errors import NodeChangeError
from ...logger import WranglerLogger
from ...models._base.records import RecordModel
from ...models.projects.roadway_changes import RoadPropertyChange
from ...models.roadway.tables import RoadNodesAttrs, RoadNodesTable
from ...params import LAT_LON_CRS
from ...utils.data import update_df_by_col_value, validate_existing_value_in_df
from ...utils.models import validate_call_pyd, validate_df_to_model


class NodeGeometryChangeTable(DataFrameModel):
    """DataFrameModel for setting node geometry given a model_node_id."""

    model_node_id: Series[int]
    X: Series[float] = Field(coerce=True)
    Y: Series[float] = Field(coerce=True)
    in_crs: Series[int] = Field(default=LAT_LON_CRS)

    class Config:
        """Config for NodeGeometryChangeTable."""

        add_missing_columns = True


class NodeGeometryChange(RecordModel):
    """Value for setting node geometry given a model_node_id."""

    model_config = ConfigDict(extra="ignore")
    X: float
    Y: float
    in_crs: int | None = LAT_LON_CRS


@validate_call_pyd
def edit_node_geometry(
    nodes_df: DataFrame[RoadNodesTable],
    node_geometry_change_table: DataFrame[NodeGeometryChangeTable],
) -> DataFrame[RoadNodesTable]:
    """Returns copied nodes table with geometry edited.

    Should be called from network so that accompanying links and shapes are also updated.

    Args:
        nodes_df: RoadNodesTable to edit
        node_geometry_change_table: NodeGeometryChangeTable with geometry changes
    """
    # TODO write wrapper on validate call so don't have to do this
    nodes_df.attrs.update(RoadNodesAttrs)
    WranglerLogger.debug(f"Updating node geometry for {len(node_geometry_change_table)} nodes.")
    WranglerLogger.debug(f"Original nodes_df: \n{nodes_df.head()}")
    # for now, require in_crs is the same for whole column
    if node_geometry_change_table.in_crs.nunique() != 1:
        msg = f"in_crs must be the same for all nodes. Got: {node_geometry_change_table.in_crs}"
        WranglerLogger.error(msg)
        raise NodeChangeError(msg)

    in_crs = node_geometry_change_table.loc[0, "in_crs"]

    # Create a table with all the new node geometry
    geo_s = gpd.points_from_xy(node_geometry_change_table.X, node_geometry_change_table.Y)
    geo_df = gpd.GeoDataFrame(node_geometry_change_table, geometry=geo_s, crs=in_crs)
    geo_df = geo_df.to_crs(LAT_LON_CRS)
    WranglerLogger.debug(f"Updated geometry geo_df: \n{geo_df}")

    # Update the nodes table with the new geometry
    nodes_df = update_df_by_col_value(
        nodes_df, geo_df, "model_node_id", properties=["X", "Y", "geometry"]
    )
    nodes_df = validate_df_to_model(nodes_df, RoadNodesTable)

    WranglerLogger.debug(f"Updated nodes_df: \n{nodes_df.head()}")

    return nodes_df


def edit_node_property(
    nodes_df: DataFrame[RoadNodesTable],
    node_idx: list[int],
    prop_name: str,
    prop_change: dict | RoadPropertyChange,
    project_name: str | None = None,
    config: WranglerConfig = DefaultConfig,
    _geometry_ok: bool = False,
) -> DataFrame[RoadNodesTable]:
    """Return copied nodes table with node property edited.

    Args:
        nodes_df: RoadNodesTable to edit
        node_idx: list of node indices to change
        prop_name: property name to change
        prop_change: dictionary of value from project_card
        project_name: optional name of the project to be applied
        config: WranglerConfig instance.
        _geometry_ok: if False, will not let you change geometry-related fields. Should
            only be changed to True by internal processes that know that geometry is changing
            and will update it in appropriate places in network. Defaults to False.
            GENERALLY DO NOT TURN THIS ON.
    """
    if not isinstance(prop_change, RoadPropertyChange):
        prop_change = RoadPropertyChange(**prop_change)
    prop_dict = prop_change.model_dump(exclude_none=True, by_alias=True)

    # Allow the project card to override the default behavior of raising an error
    existing_value_conflict = prop_change.get(
        "existing_value_conflict", config.EDITS.EXISTING_VALUE_CONFLICT
    )

    # Should not be used to update node geometry fields unless explicity set to OK:
    if prop_name in nodes_df.attrs["geometry_props"] and not _geometry_ok:
        msg = f"Cannot unilaterally change geometry property."
        raise NodeChangeError(msg)

    # check existing if necessary
    if not _check_existing_value_conflict(
        nodes_df, node_idx, prop_name, prop_dict, existing_value_conflict
    ):
        return nodes_df

    nodes_df = copy.deepcopy(nodes_df)

    # if it is a new attribute then initialize with NaN values
    if prop_name not in nodes_df:
        nodes_df[prop_name] = None

    # `set` and `change` just affect the simple property
    if "set" in prop_dict:
        nodes_df.loc[node_idx, prop_name] = prop_dict["set"]
    elif "change" in prop_dict:
        nodes_df.loc[node_idx, prop_name] = nodes_df.loc[prop_name].apply(
            lambda x: x + prop_dict["change"]
        )
    else:
        msg = f"Couldn't find correct node change spec in: {prop_dict}"
        raise NodeChangeError(msg)

    if project_name is not None:
        nodes_df.loc[node_idx, "projects"] += f"{project_name},"

    nodes_df = validate_df_to_model(nodes_df, RoadNodesTable)
    return nodes_df


def _check_existing_value_conflict(
    nodes_df: DataFrame[RoadNodesTable],
    node_idx: list[int],
    prop_name: str,
    prop_dict: dict,
    existing_value_conflict: str,
) -> bool:
    """Check if existing value conflict is OK."""
    if "existing" not in prop_dict:
        return True

    if validate_existing_value_in_df(nodes_df, node_idx, prop_name, prop_dict["existing"]):
        return True

    WranglerLogger.warning(f"Existing {prop_name} != prop_dict['existing'].")
    if existing_value_conflict == "error":
        msg = f"Conflict between specified existing and actual existing values."
        raise NodeChangeError(msg)
    if existing_value_conflict == "skip":
        WranglerLogger.warning(
            f"Skipping change for {prop_name} because of conflict with existing value."
        )
        return False
    WranglerLogger.warning(f"Changing {prop_name} despite conflict with existing value.")
    return True
