"""Scenario configuration for Network Wrangler.

You can build a scenario and write out the output from a scenario configuration file using the code below.  This is very useful when you are running a specific scenario with minor variations over again because you can enter your config file into version control.  In addition to the completed roadway and transit files, the output will provide a record of how the scenario was run.

Usage:
    ```python
        from scenario import build_scenario_from_config
        my_scenario = build_scenario_from_config(my_scenario_config)
    ```

    Where `my_scenario_config` can be a:

    - Path to a scenario config file in yaml/toml/json (recommended),
    - Dictionary which is in the same structure of a scenario config file, or
    - A `ScenarioConfig()`  instance.

!!! tip "Notes on relative paths in scenario configs"

    - Relative paths are recognized by a preceeding ".".
    - Relative paths within `output_scenario` for `roadway`, `transit`, and `project_cards` are interpreted to be relative to `output_scenario.path`.
    - All other relative paths are interpreted to be relative to directory of the scenario config file. (Or if scenario config is provided as a dictionary, relative paths will be interpreted as relative to the current working directory.)

??? example "Example Scenario Config"
    ```yaml
    name: "my_scenario"
    base_scenario:
        roadway:
            dir: "path/to/roadway_network"
            file_format: "geojson"
            read_in_shapes: True
        transit:
            dir: "path/to/transit_network"
            file_format: "txt"
        applied_projects:
            - "project1"
            - "project2"
        conflicts:
            "project3": ["project1", "project2"]
            "project4": ["project1"]
    projects:
        project_card_filepath:
            - "path/to/projectA.yaml"
            - "path/to/projectB.yaml"
        filter_tags:
            - "tag1"
    output_scenario:
        overwrite: True
        roadway:
            out_dir: "path/to/output/roadway"
            prefix: "my_scenario"
            file_format: "geojson"
            true_shape: False
        transit:
            out_dir: "path/to/output/transit"
            prefix: "my_scenario"
            file_format: "txt"
        project_cards:
            out_dir: "path/to/output/project_cards"

    wrangler_config: "path/to/wrangler_config.yaml"
    ```

Extended Usage:
    Load a configuration from a file:

    ```python
    from network_wrangler.configs import load_scenario_config

    my_scenario_config = load_scenario_config("path/to/config.yaml")
    ```

    Access the configuration:

    ```python
    my_scenario_config.base_transit_network.path
    >> path/to/transit_network
    ```
"""

from datetime import datetime
from pathlib import Path

from projectcard.io import _resolve_rel_paths

from ..models._base.types import RoadwayFileTypes, TransitFileTypes
from .utils import ConfigItem
from .wrangler import DefaultConfig

ProjectCardFilepath = Path | str
ProjectCardFilepaths = Path | list[Path] | str | list[str]

DEFAULT_SCENARIO_NAME: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

DEFAULT_BASE_DIR: Path = Path.cwd()
DEFAULT_OUTPUT_DIR: Path = Path("./output")

DEFAULT_PROJECT_IN_PATHS: ProjectCardFilepaths = []
DEFAULT_PROJECT_TAGS: list[str] = []
DEFAULT_PROJECT_WRITE: bool = True
DEFAULT_PROJECT_OUT_DIR: Path = Path("./projects")

DEFAULT_ROADWAY_IN_DIR: Path = Path()
DEFAULT_ROADWAY_IN_FORMAT: RoadwayFileTypes = "geojson"
DEFAULT_ROADWAY_SHAPE_READ: bool = False
DEFAULT_ROADWAY_WRITE: bool = True
DEFAULT_ROADWAY_OUT_DIR: Path = Path("./roadway")
DEFAULT_ROADWAY_OUT_FORMAT: RoadwayFileTypes = "geojson"

DEFAULT_TRANSIT_IN_DIR = Path()
DEFAULT_TRANSIT_IN_FORMAT: TransitFileTypes = "txt"
DEFAULT_TRANSIT_WRITE: bool = True
DEFAULT_TRANSIT_OUT_DIR = Path("./transit")
DEFAULT_TRANSIT_OUT_FORMAT: TransitFileTypes = "txt"


class ProjectsConfig(ConfigItem):
    """Configuration for projects in a scenario.

    Attributes:
        project_card_filepath: where the project card is.  A single path, list of paths,
            a directory, or a glob pattern. Defaults to None.
        filter_tags: List of tags to filter the project cards by.
    """

    def __init__(
        self,
        base_path: Path = DEFAULT_BASE_DIR,
        project_card_filepath: ProjectCardFilepaths = DEFAULT_PROJECT_IN_PATHS,
        filter_tags: list[str] = DEFAULT_PROJECT_TAGS,
    ):
        """Constructor for ProjectsConfig."""
        self.project_card_filepath = _resolve_rel_paths(project_card_filepath, base_path=base_path)
        self.filter_tags = filter_tags


class RoadwayNetworkInputConfig(ConfigItem):
    """Configuration for the road network in a scenario.

    Attributes:
        dir: Path to directory with roadway network files.
        file_format: File format for the roadway network files. Should be one of RoadwayFileTypes.
            Defaults to "geojson".
        read_in_shapes: If True, will read in the shapes of the roadway network. Defaults to False.
        boundary_geocode: Geocode of the boundary. Will use this to filter the roadway network.
        boundary_file: Path to the boundary file. If provided and both boundary_gdf and
            boundary_geocode are not provided, will use this to filter the roadway network.
    """

    def __init__(
        self,
        base_path: Path = DEFAULT_BASE_DIR,
        dir: Path = DEFAULT_ROADWAY_IN_DIR,
        file_format: RoadwayFileTypes = DEFAULT_ROADWAY_IN_FORMAT,
        read_in_shapes: bool = DEFAULT_ROADWAY_SHAPE_READ,
        boundary_geocode: str | None = None,
        boundary_file: Path | None = None,
    ):
        """Constructor for RoadwayNetworkInputConfig."""
        if dir is not None and not Path(dir).is_absolute():
            self.dir = (base_path / Path(dir)).resolve()
        else:
            self.dir = Path(dir)
        self.file_format = file_format
        self.read_in_shapes = read_in_shapes
        self.boundary_geocode = boundary_geocode
        self.boundary_file = boundary_file


class RoadwayNetworkOutputConfig(ConfigItem):
    """Configuration for writing out the resulting roadway network for a scenario.

    Attributes:
        out_dir: Path to write the roadway network files to if you don't want to use the default.
        prefix: Prefix to add to the file name. If not provided will use the scenario name.
        file_format: File format to write the roadway network to. Should be one of
            RoadwayFileTypes. Defaults to "geojson".
        true_shape: If True, will write the true shape of the roadway network. Defaults to False.
        write: If True, will write the roadway network. Defaults to True.
    """

    def __init__(
        self,
        out_dir: Path = DEFAULT_ROADWAY_OUT_DIR,
        base_path: Path = DEFAULT_BASE_DIR,
        convert_complex_link_properties_to_single_field: bool = False,
        prefix: str | None = None,
        file_format: RoadwayFileTypes = DEFAULT_ROADWAY_OUT_FORMAT,
        true_shape: bool = False,
        write: bool = DEFAULT_ROADWAY_WRITE,
    ):
        """Constructor for RoadwayNetworkOutputConfig."""
        if out_dir is not None and not Path(out_dir).is_absolute():
            self.out_dir = (base_path / Path(out_dir)).resolve()
        else:
            self.out_dir = Path(out_dir)

        self.convert_complex_link_properties_to_single_field = (
            convert_complex_link_properties_to_single_field
        )
        self.prefix = prefix
        self.file_format = file_format
        self.true_shape = true_shape
        self.write = write


class TransitNetworkInputConfig(ConfigItem):
    """Configuration for the transit network in a scenario.

    Attributes:
        dir: Path to the transit network files. Defaults to ".".
        file_format: File format for the transit network files. Should be one of TransitFileTypes.
            Defaults to "txt".
    """

    def __init__(
        self,
        base_path: Path = DEFAULT_BASE_DIR,
        dir: Path = DEFAULT_TRANSIT_IN_DIR,
        file_format: TransitFileTypes = DEFAULT_TRANSIT_IN_FORMAT,
    ):
        """Constructor for TransitNetworkInputConfig."""
        if dir is not None and not Path(dir).is_absolute():
            self.feed = (base_path / Path(dir)).resolve()
        else:
            self.feed = Path(dir)
        self.file_format = file_format


class TransitNetworkOutputConfig(ConfigItem):
    """Configuration for the transit network in a scenario.

    Attributes:
        out_dir: Path to write the transit network files to if you don't want to use the default.
        prefix: Prefix to add to the file name. If not provided will use the scenario name.
        file_format: File format to write the transit network to. Should be one of
            TransitFileTypes. Defaults to "txt".
        write: If True, will write the transit network. Defaults to True.
    """

    def __init__(
        self,
        base_path: Path = DEFAULT_BASE_DIR,
        out_dir: Path = DEFAULT_TRANSIT_OUT_DIR,
        prefix: str | None = None,
        file_format: TransitFileTypes = DEFAULT_TRANSIT_OUT_FORMAT,
        write: bool = DEFAULT_TRANSIT_WRITE,
    ):
        """Constructor for TransitNetworkOutputCOnfig."""
        if out_dir is not None and not Path(out_dir).is_absolute():
            self.out_dir = (base_path / Path(out_dir)).resolve()
        else:
            self.out_dir = Path(out_dir)
        self.write = write
        self.prefix = prefix
        self.file_format = file_format


class ProjectCardOutputConfig(ConfigItem):
    """Configuration for outputing project cards in a scenario.

    Attributes:
        out_dir: Path to write the project card files to if you don't want to use the default.
        write: If True, will write the project cards. Defaults to True.
    """

    def __init__(
        self,
        base_path: Path = DEFAULT_BASE_DIR,
        out_dir: Path = DEFAULT_PROJECT_OUT_DIR,
        write: bool = DEFAULT_PROJECT_WRITE,
    ):
        """Constructor for ProjectCardOutputConfig."""
        if out_dir is not None and not Path(out_dir).is_absolute():
            self.out_dir = (base_path / Path(out_dir)).resolve()
        else:
            self.out_dir = Path(out_dir)
        self.write = write


class ScenarioInputConfig(ConfigItem):
    """Configuration for the writing the output of a scenario.

    Attributes:
        roadway: Configuration for writing out the roadway network.
        transit: Configuration for writing out the transit network.
        applied_projects: List of projects to apply to the base scenario.
        conflicts: Dict of projects that conflict with the applied_projects.
    """

    def __init__(
        self,
        base_path: Path = DEFAULT_BASE_DIR,
        roadway: dict | None = None,
        transit: dict | None = None,
        applied_projects: list[str] | None = None,
        conflicts: dict | None = None,
    ):
        """Constructor for ScenarioInputConfig."""
        if roadway is not None:
            self.roadway: RoadwayNetworkInputConfig | None = RoadwayNetworkInputConfig(
                **roadway, base_path=base_path
            )
        else:
            self.roadway = None

        if transit is not None:
            self.transit: TransitNetworkInputConfig | None = TransitNetworkInputConfig(
                **transit, base_path=base_path
            )
        else:
            self.transit = None

        self.applied_projects = applied_projects if applied_projects is not None else []
        self.conflicts = conflicts if conflicts is not None else {}


class ScenarioOutputConfig(ConfigItem):
    """Configuration for the writing the output of a scenario.

    Attributes:
        roadway: Configuration for writing out the roadway network.
        transit: Configuration for writing out the transit network.
        project_cards: Configuration for writing out the project cards.
        overwrite: If True, will overwrite the files if they already exist. Defaults to True
    """

    def __init__(
        self,
        path: Path = DEFAULT_OUTPUT_DIR,
        base_path: Path = DEFAULT_BASE_DIR,
        roadway: dict | None = None,
        transit: dict | None = None,
        project_cards: dict | None = None,
        overwrite: bool = True,
    ):
        """Constructor for ScenarioOutputConfig."""
        if not Path(path).is_absolute():
            self.path = (base_path / Path(path)).resolve()
        else:
            self.path = Path(path)

        roadway = roadway if roadway else RoadwayNetworkOutputConfig().to_dict()
        transit = transit if transit else TransitNetworkOutputConfig().to_dict()
        self.roadway = RoadwayNetworkOutputConfig(**roadway, base_path=self.path)
        self.transit = TransitNetworkOutputConfig(**transit, base_path=self.path)

        if project_cards is not None:
            self.project_cards: ProjectCardOutputConfig | None = ProjectCardOutputConfig(
                **project_cards, base_path=self.path
            )
        else:
            self.project_cards = None

        self.overwrite = overwrite


class ScenarioConfig(ConfigItem):
    """Scenario configuration for Network Wrangler.

    Attributes:
        base_path: base path of the scenario. Defaults to cwd.
        name: Name of the scenario.
        base_scenario: information about the base scenario
        projects: information about the projects to apply on top of the base scenario
        output_scenario: information about how to output the scenario
        wrangler_config: wrangler configuration to use
    """

    def __init__(
        self,
        base_scenario: dict,
        projects: dict,
        output_scenario: dict,
        base_path: Path = DEFAULT_BASE_DIR,
        name: str = DEFAULT_SCENARIO_NAME,
        wrangler_config=DefaultConfig,
    ):
        """Constructor for ScenarioConfig."""
        self.base_path = Path(base_path) if base_path is not None else Path.cwd()
        self.name = name
        self.base_scenario = ScenarioInputConfig(**base_scenario, base_path=base_path)
        self.projects = ProjectsConfig(**projects, base_path=base_path)
        self.output_scenario = ScenarioOutputConfig(**output_scenario, base_path=base_path)
        self.wrangler_config = wrangler_config
