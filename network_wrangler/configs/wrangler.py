"""Configuration for parameters for Network Wrangler.

Users can change a handful of parameters which control the way Wrangler runs.  These parameters
can be saved as a wrangler config file which can be read in repeatedly to make sure the same
parameters are used each time.

!!! tip "YAML Configuration (Recommended)"

    The easiest way to configure parameters is using a YAML file. This is especially useful for
    users who are less familiar with Python, as you can simply point to a YAML file path rather
    than writing Python code.

    **Start with the template:** Copy `examples/wrangler.config.yml.template` to your project
    directory. The template contains all available parameters with default values and guidance
    on when to change them. Simply uncomment and modify only the parameters you need.

    ```bash
    cp examples/wrangler.config.yml.template wrangler.config.yml
    ```

    Then load it in your code:

    ```python
    from network_wrangler.configs import load_wrangler_config

    config = load_wrangler_config("wrangler.config.yml")
    ```

    See `examples/wrangler.config.yml.template` for the complete template with all parameters
    and documentation, or `examples/stpaul/wrangler.config.yml` for a working example.

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

??? Example "Default Wrangler Configuration Values"

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
        EXISTING_VALUE_CONFLICT: warn
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
    !!! warning "Do Not Modify DefaultConfig in Python"

        **Do not modify `DefaultConfig` directly in Python code.** Instead, always use YAML or TOML
        configuration files. Modifying `DefaultConfig` directly can lead to unexpected behavior
        and makes it difficult to track configuration changes.

    **Recommended approach - Use YAML/TOML configuration files:**

    ```python
    from network_wrangler.configs import load_wrangler_config

    # Load configuration from a YAML or TOML file
    config = load_wrangler_config("path/to/wrangler.config.yml")

    # Use the config when loading networks or creating scenarios
    road_net = load_roadway_from_dir("path/to/roadway", config=config)
    ```

    **Accessing default values (read-only):**

    If you need to read the default configuration values (without modifying them):

    ```python
    from network_wrangler.configs import DefaultConfig

    # Read default value (read-only)
    default_offset = DefaultConfig.MODEL_ROADWAY.ML_OFFSET_METERS
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
        CPU: Parameters for accessing CPU information. Will not change any outcomes.
        EDITS: Parameters governing how edits are handled.
    """

    IDS: IdGenerationConfig = Field(default_factory=IdGenerationConfig)
    MODEL_ROADWAY: ModelRoadwayConfig = Field(default_factory=ModelRoadwayConfig)
    CPU: CpuConfig = Field(default_factory=CpuConfig)
    EDITS: EditsConfig = Field(default_factory=EditsConfig)


DefaultConfig = WranglerConfig()
