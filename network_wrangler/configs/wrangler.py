"""Configuration for parameters for Network Wrangler.

Users can change a handful of parameters which control the way Wrangler runs.  These parameters
can be saved as a wrangler config file which can be read in repeatedly to make sure the same
parameters are used each time.

Usage:
    At runtime, you can specify configurable parameters at the scenario level which will then also
    be assigned and accessible to the roadway and transit networks.

    ```python
    create_scenario(...config = myconfig)
    ```

    Or if you are not using Scenario functionality, you can specify the config when you read in a
    RoadwayNetwork.

    ```python
    load_roadway_from_dir(**roadway, config=myconfig)
    load_transit(**transit, config=myconfig)
    ```

    `my_config` can be a:

    - Path to a config file in yaml/toml/json (recommended),
    - List of paths to config files (in case you want to split up various sub-configurations)
    - Dictionary which is in the same structure of a config file, or
    - A `WranglerConfig()`  instance.

If not provided, Wrangler will use reasonable defaults.

!!! Example "Default Wrangler Configuration Values"

    If not explicitly provided, the following default values are used:

    ```yaml

    IDS:
        TRANSIT_SHAPE_ID_METHOD: scalar
        TRANSIT_SHAPE_ID_SCALAR: 1000000
        ROAD_SHAPE_ID_METHOD: scalar
        ROAD_SHAPE_ID_SCALAR: 1000
        ML_LINK_ID_METHOD: range
        ML_LINK_ID_RANGE: (950000, 999999)
        ML_LINK_ID_SCALAR: 15000
        ML_NODE_ID_METHOD: range
        ML_NODE_ID_RANGE: (950000, 999999)
        ML_NODE_ID_SCALAR: 15000
    EDITS:
        EXISTING_VALUE_CONFLIC: warn
        OVERWRITE_SCOPED: conflicting
    MODEL_ROADWAY:
        ML_OFFSET_METERS: int = -10
        ADDITIONAL_COPY_FROM_GP_TO_ML: []
        ADDITIONAL_COPY_TO_ACCESS_EGRESS: []
    CPU:
        EST_PD_READ_SPEED:
            csv: 0.03
            parquet: 0.005
            geojson: 0.03
            json: 0.15
            txt: 0.04
    ```

Extended usage:
    Load the default configuration:

    ```python
    from network_wrangler.configs import DefaultConfig
    ```

    Access the configuration:

    ```python
    from network_wrangler.configs import DefaultConfig
    DefaultConfig.MODEL_ROADWAY.ML_OFFSET_METERS
    >> -10
    ```

    Modify the default configuration in-line:

    ```python
    from network_wrangler.configs import DefaultConfig

    DefaultConfig.MODEL_ROADWAY.ML_OFFSET_METERS = 20
    ```

    Load a configuration from a file:

    ```python
    from network_wrangler.configs import load_wrangler_config

    config = load_wrangler_config("path/to/config.yaml")
    ```

    Set a configuration value:

    ```python
    config.MODEL_ROADWAY.ML_OFFSET_METERS = 10
    ```

"""

from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass

from .utils import ConfigItem


@dataclass
class EditsConfig(ConfigItem):
    """Configuration for Edits.

    Attributes:
        EXISTING_VALUE_CONFLICT: Only used if 'existing' provided in project card and
            `existing` doesn't match the existing network value. One of `error`, `warn`, or `skip`.
            `error` will raise an error, `warn` will warn the user, and `skip` will skip the change
            for that specific property (note it will still apply any remaining property changes).
            Defaults to `warn`. Can be overridden by setting `existing_value_conflict` in
            a `roadway_property_change` project card.

        OVERWRITE_SCOPED: How to handle conflicts with existing values.
            Should be one of "conflicting", "all", or False.
            "conflicting" will only overwrite values where the scope only partially overlaps with
            the existing value. "all" will overwrite all the scoped values. "error" will error if
            there is any overlap. Default is "conflicting". Can be changed at the project-level
            by setting `overwrite_scoped` in a `roadway_property_change` project card.
    """

    EXISTING_VALUE_CONFLICT: Literal["warn", "error", "skip"] = "warn"
    OVERWRITE_SCOPED: Literal["conflicting", "all", "error"] = "conflicting"


@dataclass
class IdGenerationConfig(ConfigItem):
    """Model Roadway Configuration.

    Attributes:
        TRANSIT_SHAPE_ID_METHOD: method for creating a shape_id for a transit shape.
            Should be "scalar".
        TRANSIT_SHAPE_ID_SCALAR: scalar value to add to general purpose lane to create a
            shape_id for a transit shape.
        ROAD_SHAPE_ID_METHOD: method for creating a shape_id for a roadway shape.
            Should be "scalar".
        ROAD_SHAPE_ID_SCALAR: scalar value to add to general purpose lane to create a
            shape_id for a roadway shape.
        ML_LINK_ID_METHOD: method for creating a model_link_id for an associated
            link for a parallel managed lane.
        ML_LINK_ID_RANGE: range of model_link_ids to use when creating an associated
            link for a parallel managed lane.
        ML_LINK_ID_SCALAR: scalar value to add to general purpose lane to create a
            model_link_id when creating an associated link for a parallel managed lane.
        ML_NODE_ID_METHOD: method for creating a model_node_id for an associated node
            for a parallel managed lane.
        ML_NODE_ID_RANGE: range of model_node_ids to use when creating an associated
            node for a parallel managed lane.
        ML_NODE_ID_SCALAR: scalar value to add to general purpose lane node ides create
            a model_node_id when creating an associated nodes for parallel managed lane.
    """

    TRANSIT_SHAPE_ID_METHOD: Literal["scalar"] = "scalar"
    TRANSIT_SHAPE_ID_SCALAR: int = 1000000
    ROAD_SHAPE_ID_METHOD: Literal["scalar"] = "scalar"
    ROAD_SHAPE_ID_SCALAR: int = 1000
    ML_LINK_ID_METHOD: Literal["range", "scalar"] = "scalar"
    ML_LINK_ID_RANGE: tuple[int, int] = (950000, 999999)
    ML_LINK_ID_SCALAR: int = 3000000
    ML_NODE_ID_METHOD: Literal["range", "scalar"] = "range"
    ML_NODE_ID_RANGE: tuple[int, int] = (950000, 999999)
    ML_NODE_ID_SCALAR: int = 15000


@dataclass
class ModelRoadwayConfig(ConfigItem):
    """Model Roadway Configuration.

    Attributes:
        ML_OFFSET_METERS: Offset in meters for managed lanes.
        ADDITIONAL_COPY_FROM_GP_TO_ML: Additional fields to copy from general purpose to managed
            lanes.
        ADDITIONAL_COPY_TO_ACCESS_EGRESS: Additional fields to copy to access and egress links.
    """

    ML_OFFSET_METERS: int = -10
    ADDITIONAL_COPY_FROM_GP_TO_ML: list[str] = Field(default_factory=list)
    ADDITIONAL_COPY_TO_ACCESS_EGRESS: list[str] = Field(default_factory=list)


@dataclass
class TransitConfig(ConfigItem):
    """Transit Configuration.

    Attributes:
        K_NEAREST_CANDIDATES: Number of nearest candidate nodes to consider in stop matching
            when using name scoring. Used in
            [`match_bus_stops_to_roadway_nodes()`][network_wrangler.utils.transit.match_bus_stops_to_roadway_nodes].
        NAME_MATCH_WEIGHT: Weight for name match score in combined scoring. 0.9 means 90% name
            match, 10% distance. Used in
            [`match_bus_stops_to_roadway_nodes()`][network_wrangler.utils.transit.match_bus_stops_to_roadway_nodes].
        MIN_SUBSTRING_MATCH_LENGTH: Minimum string length required for substring matching.
            Prevents spurious matches with single letters. Used in
            [`assess_stop_name_roadway_compatibility()`][network_wrangler.utils.transit.assess_stop_name_roadway_compatibility].
        SHAPE_DISTANCE_TOLERANCE: Maximum ratio of path distance to shortest distance in
            shape-aware routing. 1.10 means paths up to 110% of shortest distance are considered.
            Used in [`route_shapes_between_stops()`][network_wrangler.utils.transit.route_shapes_between_stops]
            and [`find_shape_aware_shortest_path()`][network_wrangler.utils.transit.find_shape_aware_shortest_path].
        MAX_SHAPE_CANDIDATE_PATHS: Maximum number of candidate paths to evaluate when doing
            shape-aware routing. Used in
            [`find_shape_aware_shortest_path()`][network_wrangler.utils.transit.find_shape_aware_shortest_path].
        NEAREST_K_SHAPES_TO_STOPS: Number of nearest shape points to check for each stop.
        FIRST_LAST_SHAPE_STOP_IDX: For loops, the first stop must match one of the first
            FIRST_LAST_SHAPE_STOP_IDX shapes, and the last stop must match one of the last
            FIRST_LAST_SHAPE_STOP_IDX shapes. Used in
            [`route_shapes_between_stops()`][network_wrangler.utils.transit.route_shapes_between_stops].
        MAX_DISTANCE_STOP_FEET: Maximum distance in feet for a stop to match to a node.
            Used in [`match_bus_stops_to_roadway_nodes()`][network_wrangler.utils.transit.match_bus_stops_to_roadway_nodes].
        MAX_DISTANCE_STOP_METERS: Maximum distance in meters for a stop to match to a node.
            Used in [`match_bus_stops_to_roadway_nodes()`][network_wrangler.utils.transit.match_bus_stops_to_roadway_nodes].
    """

    K_NEAREST_CANDIDATES: int = 20
    NAME_MATCH_WEIGHT: float = 0.9
    MIN_SUBSTRING_MATCH_LENGTH: int = 3
    SHAPE_DISTANCE_TOLERANCE: float = 1.10
    MAX_SHAPE_CANDIDATE_PATHS: int = 20
    NEAREST_K_SHAPES_TO_STOPS: int = 20
    FIRST_LAST_SHAPE_STOP_IDX: int = 10
    MAX_DISTANCE_STOP_FEET: float = 528.0  # 0.10 * 5280 FEET_PER_MILE
    MAX_DISTANCE_STOP_METERS: float = 150.0  # 0.15 * 1000 METERS_PER_KILOMETER


@dataclass
class CpuConfig(ConfigItem):
    """CPU Configuration -  Will not change any outcomes.

    Attributes:
        EST_PD_READ_SPEED: Read sec / MB - WILL DEPEND ON SPECIFIC COMPUTER
    """

    EST_PD_READ_SPEED: dict[str, float] = Field(
        default_factory=lambda: {
            "csv": 0.03,
            "parquet": 0.005,
            "geojson": 0.03,
            "json": 0.15,
            "txt": 0.04,
        }
    )


@dataclass
class WranglerConfig(ConfigItem):
    """Configuration for Network Wrangler.

    Attributes:
        IDS: Parameteters governing how new ids are generated.
        MODEL_ROADWAY: Parameters governing how the model roadway is created.
        TRANSIT: Parameters governing transit network processing.
        CPU: Parameters for accessing CPU information. Will not change any outcomes.
        EDITS: Parameters governing how edits are handled.
    """

    IDS: IdGenerationConfig = IdGenerationConfig()
    MODEL_ROADWAY: ModelRoadwayConfig = ModelRoadwayConfig()
    TRANSIT: TransitConfig = TransitConfig()
    CPU: CpuConfig = CpuConfig()
    EDITS: EditsConfig = EditsConfig()


DefaultConfig = WranglerConfig()
