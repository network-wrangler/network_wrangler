"""Datamodels for Roadway Network Tables.

This module contains the datamodels used to validate the format and types of
Roadway Network tables.

Includes:

- RoadLinksTable
- RoadNodesTable
- RoadShapesTable
- ExplodedScopedLinkPropertyTable
"""

from __future__ import annotations

import datetime as dt
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
import pandera as pa
from pandas import Int64Dtype as Int64
from pandera import DataFrameModel, Field
from pandera.typing import Series
from pandera.typing.geopandas import GeoSeries

from ...logger import WranglerLogger
from .._base.db import TableForeignKeys, TablePrimaryKeys
from .._base.tables import validate_pyd
from .types import ScopedLinkValueList

RoadLinksAttrs = {
    "name": "road_links",
    "primary_key": "model_link_id",
    "source_file": None,
    "display_cols": ["model_link_id", "osm_link_id", "name"],
    "explicit_ids": ["model_link_id", "osm_link_id"],
    "geometry_props": ["geometry"],
    "idx_col": "model_link_id_idx",
}


class RoadLinksTable(DataFrameModel):
    """Datamodel used to validate if links_df is of correct format and types.

    Attributes:
        model_link_id (int): Unique identifier for the link.
        A (int): `model_node_id` of the link's start node. Foreign key to `road_nodes`.
        B (int): `model_node_id` of the link's end node. Foreign key to `road_nodes`.
        geometry (GeoSeries): **Warning**: this attribute is controlled by wrangler and should not be explicitly user-edited.
            Simple A-->B geometry of the link.
        name (str): Name of the link.
        rail_only (bool): If the link is only for rail. Default is False.
        bus_only (bool): If the link is only for buses. Default is False.
        drive_access (bool): If the link allows driving. Default is True.
        bike_access (bool): If the link allows biking. Default is True.
        walk_access (bool): If the link allows walking. Default is True.
        truck_access (bool): If the link allows trucks. Default is True.
        distance (float): Length of the link.
        roadway (str): Type of roadway per [OSM definitions](https://wiki.openstreetmap.org/wiki/Key:highway#Roads).
            Default is "road".
        projects (str): **Warning**: this attribute is controlled by wrangler and should not be explicitly user-edited.
            Comma-separated list of project names applied to the link. Default is "".
        managed (int): **Warning**: this attribute is controlled by wrangler and should not be explicitly user-edited.
            Indicator for the type of managed lane facility. Values can be:

            - 0 indicating no managed lane on this link.
            - 1 indicates that there is a managed lane on the link (std network) or that the link is a
                managed lane (model network).
            - -1 indicates that there is a parallel managed lane derived from this link (model network).
        shape_id (str): Identifier referencing the primary key of the shapes table. Default is None.
        lanes (int): Default number of lanes on the link. Default is 1.
        sc_lanes (Optional[list[dict]]: List of scoped link values for the number of lanes. Default is None.
            Example: `[{'timespan':['12:00':'15:00'], 'value': 3},{'timespan':['15:00':'19:00'], 'value': 2}]`.

        price (float): Default price to use the link. Default is 0.
        sc_price (Optional[list[dict]]): List of scoped link values for the price. Default is None.
            Example: `[{'timespan':['15:00':'19:00'],'category': 'sov', 'value': 2.5}]`.
        ref (Optional[str]): Reference numbers for link referring to a route or exit number per the
            [OSM definition](https://wiki.openstreetmap.org/wiki/Key:ref). Default is None.
        access (Optional[Any]): User-defined method to note access restrictions for the link. Default is None.
        ML_projects (Optional[str]): **Warning**: this attribute is controlled by wrangler and should not be explicitly user-edited.
            Comma-separated list of project names applied to the managed lane. Default is "".
        ML_lanes (Optional[int]): Default number of lanes on the managed lane. Default is None.
        ML_price (Optional[float]): Default price to use the managed lane. Default is 0.
        ML_access (Optional[Any]): User-defined method to note access restrictions for the managed lane. Default is None.
        ML_access_point (Optional[bool]): If the link is an access point for the managed lane. Default is False.
        ML_egress_point (Optional[bool]): If the link is an egress point for the managed lane. Default is False.
        sc_ML_lanes (Optional[list[dict]]): List of scoped link values for the number of lanes on the managed lane.
            Default is None.
        sc_ML_price (Optional[list[dict]]): List of scoped link values for the price of the managed lane. Default is None.
        sc_ML_access (Optional[list[dict]]): List of scoped link values for the access restrictions of the managed lane.
            Default is None.
        ML_geometry (Optional[GeoSeries]): **Warning**: this attribute is controlled by wrangler and should not be explicitly user-edited.
            Simple A-->B geometry of the managed lane. Default is None.
        ML_shape_id (Optional[str]): Identifier referencing the primary key of the shapes table for the managed lane.
            Default is None.
        osm_link_id (Optional[str]): Identifier referencing the OSM link ID. Default is "".
        GP_A (Optional[int]): **Warning**: this attribute is controlled by wrangler and should not be explicitly user-edited.
            Identifier referencing the primary key of the associated general purpose link start node for
            a managed lane link in a model network. Default is None.
        GP_B (Optional[int]): **Warning**: this attribute is controlled by wrangler and should not be explicitly user-edited.
            Identifier referencing the primary key of the associated general purpose link end node for
            a managed lane link in a model network. Default is None.

    !!! tip "User Defined Properties"

        Additional properites may be defined and are assumed to have the same definition of OpenStreetMap if they
        have overlapping property names.

    ### Properties for parallel managed lanes

    Properties for parallel managed lanes are prefixed with `ML_`. (Almost) any property,
    including an ad-hoc one, can be made to apply to a parallel managed lane by applying
    the prefix `ML_`, e.g. `ML_lanes`

    !!! warning

        The following properties should **not** be assigned an `ML_` prefix by the user
        because they are assigned one within networkwrangler:

        - `name`
        - `A`
        - `B`
        - `model_link_id`

    ### Time- or category-dependent properties

    The following properties can be time-dependent, category-dependent, or both by adding `sc_`.
    The "plain" property without the prefix becomes the default when no scoped property applies.

    | Property | # of Lanes | Price |
    | -----------| ----------------- | ---------------- |
    | Default value | `lanes` | `price` |
    | Time- and/or category-dependent value | `sc_lanes` | `sc_price` |
    | Default value for managed lane | `ML_lanes` | `ML_price` |
    | Time- and/or category-dependent value for managed lane | `sc_ML_lanes` | `sc_ML_price` |


    ??? note "previous format for scoped properties"

        Some previous tooling was developed around a previous method for serializing scoped properties.  In order to retain compatability with this format:

        - `load_roadway_from_dir()`, `read_links()`, and associated functions will "sniff" the network for the old format and apply the converter function `translate_links_df_v0_to_v1()`
        - `write_links()` has an boolean attribute to `convert_complex_properties_to_single_field` which can also be invoked from `write_roadway()` as `convert_complex_link_properties_to_single_field`.

    #### Defining time-dependent properties

    Time-dependent properties are defined as a list of dictionaries with timespans and values.

    - Timespans must be defined as a list of HH:MM or HH:MM:SS using a 24-hour clock: `('06:00':'09:00')`.
    - Timespans must not intersect.

    !!! example  "Time-dependent property"

        $3 peak-period pricing

        ```python
        # default price
        'price' = 0
        'sc_price':
        [
            {
                'time':['06:00':'09:00'],
                'value': 3
            },
            {
                'timespan':['16:00':'19:00'],
                'value': 3,
            }
        ]
        ```

    #### Defining time- and category-dependent properties

    Properties co-dependent on time- and category are defined as a list of dictionaries with value, category and time defined.

    !!! example "time- and category-dependent property"

        A pricing strategy which only applies in peak period for trucks and sovs:

        ```python
        # default price
        "price": 0
        # price scoped by time of day
        "sc_price":
        [
            {
                'timespan':['06:00':'09:00'],
                'category': ('sov','truck'),
                'value': 3
            },
            {
                'timespan':['16:00':'19:00'],
                'category': ('sov','truck'),
                'value': 3,
            }
        ]
        ```

    !!! tip

        There is no limit on other, user-defined properties being listed as time-dependent or time- and category-dependent.

    !!! example "User-defined variable by time of day"

        Define a variable `access` to represent which categories can access the network and vary it by time of day.

        ```python
        #access
        {
            # default value for access
            'access': ('any'),
            # scoped value for access
            'sc_access': [
                {
                    'timespan':['06:00':'09:00'],
                    'value': ('no-trucks')
                },
                {
                    'timespan':['16:00':'19:00'],
                    'value': ('hov2','hov3','trucks')
                }
            ]
        }
        ```
    """

    model_link_id: Series[int] = Field(coerce=True, unique=True)
    model_link_id_idx: Optional[Series[int]] = Field(coerce=True, unique=True)
    A: Series[int] = Field(nullable=False, coerce=True)
    B: Series[int] = Field(nullable=False, coerce=True)
    geometry: GeoSeries = Field(nullable=False)
    name: Series[str] = Field(nullable=False, default="unknown")
    rail_only: Series[bool] = Field(coerce=True, nullable=False, default=False)
    bus_only: Series[bool] = Field(coerce=True, nullable=False, default=False)
    drive_access: Series[bool] = Field(coerce=True, nullable=False, default=True)
    bike_access: Series[bool] = Field(coerce=True, nullable=False, default=True)
    walk_access: Series[bool] = Field(coerce=True, nullable=False, default=True)
    distance: Series[float] = Field(coerce=True, nullable=False)

    roadway: Series[str] = Field(nullable=False, default="road")
    projects: Series[str] = Field(coerce=True, default="")
    managed: Series[int] = Field(coerce=True, nullable=False, default=0)

    shape_id: Series[str] = Field(coerce=True, nullable=True)
    lanes: Series[int] = Field(coerce=True, nullable=False)
    price: Series[float] = Field(coerce=True, nullable=False, default=0)

    # Optional Fields
    ref: Optional[Series[str]] = Field(coerce=True, nullable=True, default=None)
    access: Optional[Series[Any]] = Field(coerce=True, nullable=True, default=None)

    sc_lanes: Optional[Series[object]] = Field(coerce=True, nullable=True, default=None)
    sc_price: Optional[Series[object]] = Field(coerce=True, nullable=True, default=None)

    ML_projects: Series[str] = Field(coerce=True, default="")
    ML_lanes: Optional[Series[Int64]] = Field(coerce=True, nullable=True, default=None)
    ML_price: Optional[Series[float]] = Field(coerce=True, nullable=True, default=0)
    ML_access: Optional[Series[Any]] = Field(coerce=True, nullable=True, default=True)
    ML_access_point: Optional[Series[bool]] = Field(
        coerce=True,
        default=False,
    )
    ML_egress_point: Optional[Series[bool]] = Field(
        coerce=True,
        default=False,
    )
    sc_ML_lanes: Optional[Series[object]] = Field(
        coerce=True,
        nullable=True,
        default=None,
    )
    sc_ML_price: Optional[Series[object]] = Field(
        coerce=True,
        nullable=True,
        default=None,
    )
    sc_ML_access: Optional[Series[object]] = Field(
        coerce=True,
        nullable=True,
        default=None,
    )

    ML_geometry: Optional[GeoSeries] = Field(nullable=True, coerce=True, default=None)
    ML_shape_id: Optional[Series[str]] = Field(nullable=True, coerce=True, default=None)

    truck_access: Optional[Series[bool]] = Field(coerce=True, nullable=True, default=True)
    osm_link_id: Series[str] = Field(coerce=True, nullable=True, default="")
    # todo this should be List[dict] but ranch output something else so had to have it be Any.
    locationReferences: Optional[Series[Any]] = Field(
        coerce=True,
        nullable=True,
        default="",
    )

    GP_A: Optional[Series[Int64]] = Field(coerce=True, nullable=True, default=None)
    GP_B: Optional[Series[Int64]] = Field(coerce=True, nullable=True, default=None)

    class Config:
        """Config for RoadLinksTable."""

        add_missing_columns = True
        coerce = True
        unique: ClassVar[list[str]] = ["A", "B"]

    @pa.check("sc_*", regex=True, element_wise=True)
    def check_scoped_fields(cls, scoped_value: Series) -> Series[bool]:
        """Checks that all fields starting with 'sc_' or 'sc_ML_' are valid ScopedLinkValueList.

        Custom check to validate fields starting with 'sc_' or 'sc_ML_'
        against a ScopedLinkValueItem model, handling both mandatory and optional fields.
        """
        if scoped_value is None or (not isinstance(scoped_value, list) and pd.isna(scoped_value)):
            return True
        return validate_pyd(scoped_value, ScopedLinkValueList)


RoadNodesAttrs = {
    "name": "road_nodes",
    "primary_key": "model_node_id",
    "source_file": None,
    "display_cols": ["model_node_id", "osm_node_id", "X", "Y"],
    "explicit_ids": ["model_node_id", "osm_node_id"],
    "geometry_props": ["X", "Y", "geometry"],
    "idx_col": "model_node_id_idx",
}


class RoadNodesTable(DataFrameModel):
    """Datamodel used to validate if links_df is of correct format and types.

    Must have a record for each node used by the `links` table and by the transit `shapes`, `stop_times`, and `stops` tables.

    Attributes:
        model_node_id (int): Unique identifier for the node.
        osm_node_id (Optional[str]): Reference to open street map node id. Used for querying. Not guaranteed to be unique.
        X (float): Longitude of the node in WGS84. Must be in the range of -180 to 180.
        Y (float): Latitude of the node in WGS84. Must be in the range of -90 to 90.
        geometry (GeoSeries): **Warning**: this attribute is controlled by wrangler and should not be explicitly user-edited.
    """

    model_node_id: Series[int] = Field(coerce=True, unique=True, nullable=False)
    model_node_idx: Optional[Series[int]] = Field(coerce=True, unique=True, nullable=False)
    X: Series[float] = Field(coerce=True, nullable=False)
    Y: Series[float] = Field(coerce=True, nullable=False)
    geometry: GeoSeries

    # optional fields
    osm_node_id: Series[str] = Field(
        coerce=True,
        nullable=True,
        default="",
    )
    projects: Series[str] = Field(coerce=True, default="")
    inboundReferenceIds: Optional[Series[list[str]]] = Field(coerce=True, nullable=True)
    outboundReferenceIds: Optional[Series[list[str]]] = Field(coerce=True, nullable=True)

    class Config:
        """Config for RoadNodesTable."""

        add_missing_columns = True
        coerce = True
        _pk: ClassVar[TablePrimaryKeys] = ["model_node_id"]


RoadShapesAttrs = {
    "name": "road_shapes",
    "primary_key": "shape_id",
    "source_file": None,
    "display_cols": ["shape_id"],
    "explicit_ids": ["shape_id"],
    "geometry_props": ["geometry"],
    "idx_col": "shape_id_idx",
}


class RoadShapesTable(DataFrameModel):
    """Datamodel used to validate if shapes_df is of correct format and types.

    Should have a record for each `shape_id` referenced in `links` table.

    Attributes:
        shape_id (str): Unique identifier for the shape.
        geometry (GeoSeries): **Warning**: this attribute is controlled by wrangler and should not be explicitly user-edited.
            Geometry of the shape.
        ref_shape_id (Optional[str]): Reference to another `shape_id` that it may
            have been created from. Default is None.
    """

    shape_id: Series[str] = Field(unique=True)
    shape_id_idx: Optional[Series[int]] = Field(unique=True)

    geometry: GeoSeries = Field()
    ref_shape_id: Optional[Series] = Field(nullable=True)

    class Config:
        """Config for RoadShapesTable."""

        coerce = True
        _pk: ClassVar[TablePrimaryKeys] = ["shape_id"]


class ExplodedScopedLinkPropertyTable(DataFrameModel):
    """Datamodel used to validate an exploded links_df by scope."""

    model_link_id: Series[int]
    category: Series[Any]
    timespan: Series[list[str]]
    start_time: Series[dt.datetime]
    end_time: Series[dt.datetime]
    scoped: Series[Any] = Field(default=None, nullable=True)

    class Config:
        """Config for ExplodedScopedLinkPropertySchema."""

        name = "ExplodedScopedLinkPropertySchema"
        coerce = True
