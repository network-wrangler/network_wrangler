"""Scenario objects manage how a collection of projects is applied to the networks.

Scenarios are built from a base scenario and a list of project cards.

A project card is a YAML file (or similar) that describes a change to the network. The project
card can contain multiple changes, each of which is applied to the network in sequence.

## Create a Scenario

Instantiate a scenario by seeding it with a base scenario and optionally some project cards.

```python
from network_wrangler import create_scenario

my_scenario = create_scenario(
    base_scenario=my_base_year_scenario,
    card_search_dir=project_card_directory,
    filter_tags=["baseline2050"],
)
```

A `base_year_scenario` is a dictionary representation of key components of a scenario:

- `road_net`: RoadwayNetwork instance
- `transit_net`: TransitNetwork instance
- `applied_projects`: list of projects that have been applied to the base scenario so that the
    scenario knows if there will be conflicts with future projects or if a future project's
    pre-requisite is satisfied.
- `conflicts`: dictionary of conflicts for project that have been applied to the base scenario so
    that the scenario knows if there will be conflicts with future projects.

```python
my_base_year_scenario = {
    "road_net": load_from_roadway_dir(STPAUL_DIR),
    "transit_net": load_transit(STPAUL_DIR),
    "applied_projects": [],
    "conflicts": {},
}
```

## Add Projects to a Scenario

In addition to adding projects when you create the scenario, project cards can be **added** to a
scenario using the `add_project_cards` method.

```python
from projectcard import read_cards

project_card_dict = read_cards(card_location, filter_tags=["Baseline2030"], recursive=True)
my_scenario.add_project_cards(project_card_dict.values())
```

Where `card_location` can be a single path, list of paths, a directory, or a glob pattern.

## Apply Projects to a Scenario

Projects can be **applied** to a scenario using the `apply_all_projects` method. Before applying
projects, the scenario will check that all pre-requisites are satisfied, that there are no conflicts,
and that the projects are in the planned projects list.

If you want to check the order of projects before applying them, you can use the `queued_projects`
prooperty.

```python
my_scenario.queued_projects
my_scenario.apply_all_projects()
```

You can **review** the resulting scenario, roadway network, and transit networks.

```python
my_scenario.applied_projects
my_scenario.road_net.links_gdf.explore()
my_scenario.transit_net.feed.shapes_gdf.explore()
```

## Write a Scenario to Disk

Scenarios (and their networks) can be **written** to disk using the `write` method which
in addition to writing out roadway and transit networks, will serialize the scenario to
a yaml-like file and can also write out the project cards that have been applied.

```python
my_scenario.write(
    "output_dir",
    "scenario_name_to_use",
    overwrite=True,
    projects_write=True,
    file_format="parquet",
)
```

??? example "Example Serialized Scenario File"

    ```yaml
    applied_projects: &id001
    - project a
    - project b
    base_scenario:
    applied_projects: *id001
    roadway:
        dir: /Users/elizabeth/Documents/urbanlabs/MetCouncil/NetworkWrangler/working/network_wrangler/examples/small
        file_format: geojson
    transit:
        dir: /Users/elizabeth/Documents/urbanlabs/MetCouncil/NetworkWrangler/working/network_wrangler/examples/small
    config:
    CPU:
        EST_PD_READ_SPEED:
        csv: 0.03
        geojson: 0.03
        json: 0.15
        parquet: 0.005
        txt: 0.04
    IDS:
        ML_LINK_ID_METHOD: range
        ML_LINK_ID_RANGE: &id002 !!python/tuple
        - 950000
        - 999999
        ML_LINK_ID_SCALAR: 15000
        ML_NODE_ID_METHOD: range
        ML_NODE_ID_RANGE: *id002
        ML_NODE_ID_SCALAR: 15000
        ROAD_SHAPE_ID_METHOD: scalar
        ROAD_SHAPE_ID_SCALAR: 1000
        TRANSIT_SHAPE_ID_METHOD: scalar
        TRANSIT_SHAPE_ID_SCALAR: 1000000
    MODEL_ROADWAY:
        ADDITIONAL_COPY_FROM_GP_TO_ML: []
        ADDITIONAL_COPY_TO_ACCESS_EGRESS: []
        ML_OFFSET_METERS: -10
    conflicts: {}
    corequisites: {}
    name: first_scenario
    prerequisites: {}
    roadway:
    dir: /Users/elizabeth/Documents/urbanlabs/MetCouncil/NetworkWrangler/working/network_wrangler/tests/out/first_scenario/roadway
    file_format: parquet
    transit:
    dir: /Users/elizabeth/Documents/urbanlabs/MetCouncil/NetworkWrangler/working/network_wrangler/tests/out/first_scenario/transit
    file_format: txt
    ```

## Load a scenario from disk

And if you want to reload scenario that you "wrote", you can use the `load_scenario` function.

```python
from network_wrangler import load_scenario

my_scenario = load_scenario("output_dir/scenario_name_to_use_scenario.yml")
```

"""

from __future__ import annotations

import copy
import pprint
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import yaml
from projectcard import ProjectCard, SubProject, read_cards, write_card

from .configs import (
    DefaultConfig,
    ScenarioConfig,
    WranglerConfig,
    load_scenario_config,
    load_wrangler_config,
)
from .configs.scenario import ScenarioInputConfig, ScenarioOutputConfig
from .errors import (
    ProjectCardError,
    ScenarioConflictError,
    ScenarioCorequisiteError,
    ScenarioPrerequisiteError,
)
from .logger import WranglerLogger
from .roadway.io import load_roadway_from_dir, write_roadway
from .roadway.network import RoadwayNetwork
from .transit.io import load_transit, write_transit
from .transit.network import TransitNetwork
from .utils.io_dict import load_dict
from .utils.io_table import prep_dir
from .utils.utils import topological_sort

if TYPE_CHECKING:
    from .models._base.types import RoadwayFileTypes, TransitFileTypes

"""
List of attributes that are suggested to be in a base scenario dictionary.
"""
BASE_SCENARIO_SUGGESTED_PROPS: list[str] = [
    "road_net",
    "transit_net",
    "applied_projects",
    "conflicts",
]

"""
List of card types that that will be applied to the transit network.
"""
TRANSIT_CARD_TYPES: list[str] = [
    "transit_property_change",
    "transit_routing_change",
    "transit_route_addition",
    "transit_service_deletion",
]

"""
List of card types that that will be applied to the roadway network.
"""
ROADWAY_CARD_TYPES: list[str] = [
    "roadway_property_change",
    "roadway_deletion",
    "roadway_addition",
    "pycode",
]


"""
List of card types that that will be applied to the transit network AFTER being applied to
the roadway network.
"""
SECONDARY_TRANSIT_CARD_TYPES: list[str] = [
    "roadway_deletion",
]


class Scenario:
    """Holds information about a scenario.

    Typical usage example:

    ```python
    my_base_year_scenario = {
        "road_net": load_roadway(
            links_file=STPAUL_LINK_FILE,
            nodes_file=STPAUL_NODE_FILE,
            shapes_file=STPAUL_SHAPE_FILE,
        ),
        "transit_net": load_transit(STPAUL_DIR),
    }

    # create a future baseline scenario from base by searching for all cards in dir w/ baseline tag
    project_card_directory = Path(STPAUL_DIR) / "project_cards"
    my_scenario = create_scenario(
        base_scenario=my_base_year_scenario,
        card_search_dir=project_card_directory,
        filter_tags=["baseline2050"],
    )

    # check project card queue and then apply the projects
    my_scenario.queued_projects
    my_scenario.apply_all_projects()

    # check applied projects, write it out, and create a summary report.
    my_scenario.applied_projects
    my_scenario.write("baseline")
    my_scenario.summary

    # Add some projects to create a build scenario based on a list of files.
    build_card_filenames = [
        "3_multiple_roadway_attribute_change.yml",
        "road.prop_changes.segment.yml",
        "4_simple_managed_lane.yml",
    ]
    my_scenario.add_projects_from_files(build_card_filenames)
    my_scenario.write("build2050")
    my_scenario.summary
    ```

    Attributes:
        base_scenario: dictionary representation of a scenario
        road_net: instance of RoadwayNetwork for the scenario
        transit_net: instance of TransitNetwork for the scenario
        project_cards: Mapping[ProjectCard.name,ProjectCard] Storage of all project cards by name.
        queued_projects: Projects which are "shovel ready" - have had pre-requisits checked and
            done any required re-ordering. Similar to a git staging, project cards aren't
            recognized in this collecton once they are moved to applied.
        applied_projects: list of project names that have been applied
        projects: list of all projects either planned, queued, or applied
        prerequisites:  dictionary storing prerequiste info as `projectA: [prereqs-for-projectA]`
        corequisites:  dictionary storing corequisite info as`projectA: [coreqs-for-projectA]`
        conflicts: dictionary storing conflict info as `projectA: [conflicts-for-projectA]`
        config: WranglerConfig instance.
    """

    def __init__(
        self,
        base_scenario: Union[Scenario, dict],
        project_card_list: Optional[list[ProjectCard]] = None,
        config: Optional[Union[WranglerConfig, dict, Path, list[Path]]] = None,
        name: str = "",
    ):
        """Constructor.

        Args:
            base_scenario: A base scenario object to base this isntance off of, or a dict which
                describes the scenario attributes including applied projects and respective
                conflicts. `{"applied_projects": [],"conflicts":{...}}`
            project_card_list: Optional list of ProjectCard instances to add to planned projects.
                Defaults to None.
            config: WranglerConfig instance or a dictionary of configuration settings or a path to
                one or more configuration files. Configurations that are not explicity set will
                default to the values in the default configuration in
                `/configs/wrangler/default.yml`.
            name: Optional name for the scenario.
        """
        WranglerLogger.info("Creating Scenario")
        self.config = load_wrangler_config(config)

        if project_card_list is None:
            project_card_list = []

        if isinstance(base_scenario, Scenario):
            base_scenario = base_scenario.__dict__

        self.base_scenario: dict = extract_base_scenario_metadata(base_scenario)

        if not set(BASE_SCENARIO_SUGGESTED_PROPS) <= set(base_scenario.keys()):
            WranglerLogger.warning(
                f"Base_scenario doesn't contain {BASE_SCENARIO_SUGGESTED_PROPS}"
            )
        self.name: str = name
        # if the base scenario had roadway or transit networks, use them as the basis.
        self.road_net: Optional[RoadwayNetwork] = copy.deepcopy(
            base_scenario.pop("road_net", None)
        )

        self.transit_net: Optional[TransitNetwork] = copy.deepcopy(
            base_scenario.pop("transit_net", None)
        )
        if self.road_net and self.transit_net:
            self.transit_net.road_net = self.road_net

        # Set configs for networks to be the same as scenario.
        if isinstance(self.road_net, RoadwayNetwork):
            self.road_net.config = self.config
        if isinstance(self.transit_net, TransitNetwork):
            self.transit_net.config = self.config

        self.project_cards: dict[str, ProjectCard] = {}
        self._planned_projects: list[str] = []
        self._queued_projects = None
        self.applied_projects: list[str] = base_scenario.pop("applied_projects", [])

        self.prerequisites: dict[str, list[str]] = base_scenario.pop("prerequisites", {})
        self.corequisites: dict[str, list[str]] = base_scenario.pop("corequisites", {})
        self.conflicts: dict[str, list[str]] = base_scenario.pop("conflicts", {})

        for p in project_card_list:
            self._add_project(p)

    @property
    def projects(self):
        """Returns a list of all projects in the scenario: applied and planned."""
        return self.applied_projects + self._planned_projects

    @property
    def queued_projects(self):
        """Returns a list version of _queued_projects queue.

        Queued projects are thos that have been planned, have all pre-requisites satisfied, and
        have been ordered based on pre-requisites.

        If no queued projects, will dynamically generate from planned projects based on
        pre-requisites and return the queue.
        """
        if not self._queued_projects:
            self._check_projects_requirements_satisfied(self._planned_projects)
            self._queued_projects = self.order_projects(self._planned_projects)
        return list(self._queued_projects)

    def __str__(self):
        """String representation of the Scenario object."""
        s = [f"{key}: {value}" for key, value in self.__dict__.items()]
        return "\n".join(s)

    def _add_dependencies(self, project_name, dependencies: dict) -> None:
        """Add dependencies from a project card to relevant scenario variables.

        Updates existing "prerequisites", "corequisites" and "conflicts".
        Lowercases everything to enable string matching.

        Args:
            project_name: name of project you are adding dependencies for.
            dependencies: Dictionary of depndencies by dependency type and list of associated
                projects.
        """
        project_name = project_name.lower()

        for d, v in dependencies.items():
            _dep = list(map(str.lower, v))
            WranglerLogger.debug(f"Adding {_dep} to {project_name} dependency table.")
            self.__dict__[d].update({project_name: _dep})

    def _add_project(
        self,
        project_card: ProjectCard,
        validate: bool = True,
        filter_tags: Optional[list[str]] = None,
    ) -> None:
        """Adds a single ProjectCard instances to the Scenario.

        Checks that a project of same name is not already in scenario.
        If selected, will validate ProjectCard before adding.
        If provided, will only add ProjectCard if it matches at least one filter_tags.

        Resets scenario queued_projects.

        Args:
            project_card (ProjectCard): ProjectCard instance to add to scenario.
            validate (bool, optional): If True, will validate the projectcard before
                being adding it to the scenario. Defaults to True.
            filter_tags: If used, will only add the project card if
                its tags match one or more of these filter_tags. Defaults to []
                which means no tag-filtering will occur.

        """
        filter_tags = filter_tags or []
        project_name = project_card.project.lower()
        filter_tags = list(map(str.lower, filter_tags))

        if project_name in self.projects:
            msg = f"Names not unique from existing scenario projects: {project_card.project}"
            raise ProjectCardError(msg)

        if filter_tags and set(project_card.tags).isdisjoint(set(filter_tags)):
            WranglerLogger.debug(
                f"Skipping {project_name} - no overlapping tags with {filter_tags}."
            )
            return

        if validate:
            project_card.validate()

        WranglerLogger.info(f"Adding {project_name} to scenario.")
        self.project_cards[project_name] = project_card
        self._planned_projects.append(project_name)
        self._queued_projects = None
        self._add_dependencies(project_name, project_card.dependencies)

    def add_project_cards(
        self,
        project_card_list: list[ProjectCard],
        validate: bool = True,
        filter_tags: Optional[list[str]] = None,
    ) -> None:
        """Adds a list of ProjectCard instances to the Scenario.

        Checks that a project of same name is not already in scenario.
        If selected, will validate ProjectCard before adding.
        If provided, will only add ProjectCard if it matches at least one filter_tags.

        Args:
            project_card_list: List of ProjectCard instances to add to
                scenario.
            validate (bool, optional): If True, will require each ProjectCard is validated before
                being added to scenario. Defaults to True.
            filter_tags: If used, will filter ProjectCard instances
                and only add those whose tags match one or more of these filter_tags.
                Defaults to [] - which means no tag-filtering will occur.
        """
        filter_tags = filter_tags or []
        for p in project_card_list:
            self._add_project(p, validate=validate, filter_tags=filter_tags)

    def _check_projects_requirements_satisfied(self, project_list: list[str]):
        """Checks all requirements are satisified to apply this specific set of projects.

        Including:
        1. has an associaed project card
        2. is in scenario's planned projects
        3. pre-requisites satisfied
        4. co-requisies satisfied by applied or co-applied projects
        5. no conflicing applied or co-applied projects

        Args:
            project_list: list of projects to check requirements for.
        """
        self._check_projects_planned(project_list)
        self._check_projects_have_project_cards(project_list)
        self._check_projects_prerequisites(project_list)
        self._check_projects_corequisites(project_list)
        self._check_projects_conflicts(project_list)

    def _check_projects_planned(self, project_names: list[str]) -> None:
        """Checks that a list of projects are in the scenario's planned projects."""
        _missing_ps = [p for p in project_names if p not in self._planned_projects]
        if _missing_ps:
            msg = f"Projects are not in planned projects: \n {_missing_ps}. \
                Add them by using add_project_cards()."
            WranglerLogger.debug(msg)
            raise ValueError(msg)

    def _check_projects_have_project_cards(self, project_list: list[str]) -> bool:
        """Checks that a list of projects has an associated project card in the scenario."""
        _missing = [p for p in project_list if p not in self.project_cards]
        if _missing:
            WranglerLogger.error(
                f"Projects referenced which are missing project cards: {_missing}"
            )
            return False
        return True

    def _check_projects_prerequisites(self, project_names: list[str]) -> None:
        """Check a list of projects' pre-requisites have been or will be applied to scenario."""
        if set(project_names).isdisjoint(set(self.prerequisites.keys())):
            return
        _prereqs = []
        for p in project_names:
            _prereqs += self.prerequisites.get(p, [])
        _projects_applied = self.applied_projects + project_names
        _missing = list(set(_prereqs) - set(_projects_applied))
        if _missing:
            WranglerLogger.debug(
                f"project_names: {project_names}\nprojects_have_or_will_be_applied: \
                    {_projects_applied}\nmissing: {_missing}"
            )
            msg = f"Missing {len(_missing)} pre-requisites."
            raise ScenarioPrerequisiteError(msg)

    def _check_projects_corequisites(self, project_names: list[str]) -> None:
        """Check a list of projects' co-requisites have been or will be applied to scenario."""
        if set(project_names).isdisjoint(set(self.corequisites.keys())):
            return
        _coreqs = []
        for p in project_names:
            _coreqs += self.corequisites.get(p, [])
        _projects_applied = self.applied_projects + project_names
        _missing = list(set(_coreqs) - set(_projects_applied))
        if _missing:
            WranglerLogger.debug(
                f"project_names: {project_names}\nprojects_have_or_will_be_applied: \
                    {_projects_applied}\nmissing: {_missing}"
            )
            msg = f"Missing {len(_missing)} corequisites."
            raise ScenarioCorequisiteError(msg)

    def _check_projects_conflicts(self, project_names: list[str]) -> None:
        """Checks that list of projects' conflicts have not been or will be applied to scenario."""
        # WranglerLogger.debug("Checking Conflicts...")
        projects_to_check = project_names + self.applied_projects
        # WranglerLogger.debug(f"\nprojects_to_check:{projects_to_check}\nprojects_with_conflicts:{set(self.conflicts.keys())}")
        if set(projects_to_check).isdisjoint(set(self.conflicts.keys())):
            # WranglerLogger.debug("Projects have no conflicts to check")
            return
        _conflicts = []
        for p in project_names:
            _conflicts += self.conflicts.get(p, [])
        _conflict_problems = [p for p in _conflicts if p in projects_to_check]
        if _conflict_problems:
            WranglerLogger.warning(f"Conflict Problems: \n{_conflict_problems}")
            _conf_dict = {
                k: v
                for k, v in self.conflicts.items()
                if k in projects_to_check and not set(v).isdisjoint(set(_conflict_problems))
            }
            WranglerLogger.debug(f"Problematic Conflicts: \n{_conf_dict}")
            msg = f"Found {len(_conflict_problems)} conflicts: {_conflict_problems}"
            raise ScenarioConflictError(msg)

    def order_projects(self, project_list: list[str]) -> deque:
        """Orders a list of projects based on moving up pre-requisites into a deque.

        Args:
            project_list: list of projects to order

        Returns: deque for applying projects.
        """
        project_list = [p.lower() for p in project_list]
        assert self._check_projects_have_project_cards(project_list)

        # build prereq (adjacency) list for topological sort
        adjacency_list: dict[str, list] = defaultdict(list)
        visited_list: dict[str, bool] = defaultdict(bool)

        for project in project_list:
            visited_list[project] = False
            if not self.prerequisites.get(project):
                continue
            for prereq in self.prerequisites[project]:
                # this will always be true, else would have been flagged in missing \
                # prerequsite check, but just in case
                if prereq.lower() in project_list:
                    if adjacency_list.get(prereq.lower()):
                        adjacency_list[prereq.lower()].append(project)
                    else:
                        adjacency_list[prereq.lower()] = [project]

        # sorted_project_names is topological sorted project card names (based on prerequsiite)
        _ordered_projects = topological_sort(
            adjacency_list=adjacency_list, visited_list=visited_list
        )

        if set(_ordered_projects) != set(project_list):
            _missing = list(set(project_list) - set(_ordered_projects))
            msg = f"Project sort resulted in missing projects: {_missing}"
            raise ValueError(msg)

        project_deque = deque(_ordered_projects)

        WranglerLogger.debug(f"Ordered Projects: \n{project_deque}")

        return project_deque

    def apply_all_projects(self):
        """Applies all planned projects in the queue."""
        # Call this to make sure projects are appropriately queued in hidden variable.
        self.queued_projects  # noqa: B018

        # Use hidden variable.
        while self._queued_projects:
            self._apply_project(self._queued_projects.popleft())

        # set this so it will trigger re-queuing any more projects.
        self._queued_projects = None

    def _apply_change(self, change: Union[ProjectCard, SubProject]) -> None:
        """Applies a specific change specified in a project card.

        Change type must be in at least one of:
        - ROADWAY_CARD_TYPES
        - TRANSIT_CARD_TYPES

        Args:
            change: a project card or subproject card
        """
        if change.change_type in ROADWAY_CARD_TYPES:
            if not self.road_net:
                msg = "Missing Roadway Network"
                raise ValueError(msg)
            if change.change_type in SECONDARY_TRANSIT_CARD_TYPES and self.transit_net:
                self.road_net.apply(change, transit_net=self.transit_net)
            else:
                self.road_net.apply(change)
        if change.change_type in TRANSIT_CARD_TYPES:
            if not self.transit_net:
                msg = "Missing Transit Network"
                raise ValueError(msg)
            self.transit_net.apply(change)

        if change.change_type not in ROADWAY_CARD_TYPES + TRANSIT_CARD_TYPES:
            msg = f"Project {change.project}: Don't understand project cat: {change.change_type}"
            raise ProjectCardError(msg)

    def _apply_project(self, project_name: str) -> None:
        """Applies project card to scenario.

        If a list of changes is specified in referenced project card, iterates through each change.

        Args:
            project_name (str): name of project to be applied.
        """
        project_name = project_name.lower()

        WranglerLogger.info(
            f"Applying {project_name} from file:\
                            {self.project_cards[project_name].file}"
        )

        p = self.project_cards[project_name]
        WranglerLogger.debug(f"types: {p.change_types}")
        WranglerLogger.debug(f"type: {p.change_type}")
        if p._sub_projects:
            for sp in p._sub_projects:
                WranglerLogger.debug(f"- applying subproject: {sp.change_type}")
                self._apply_change(sp)

        else:
            self._apply_change(p)

        self._planned_projects.remove(project_name)
        self.applied_projects.append(project_name)

    def apply_projects(self, project_list: list[str]):
        """Applies a specific list of projects from the planned project queue.

        Will order the list of projects based on pre-requisites.

        NOTE: does not check co-requisites b/c that isn't possible when applying a single project.

        Args:
            project_list: List of projects to be applied. All need to be in the planned project
                queue.
        """
        project_list = [p.lower() for p in project_list]

        self._check_projects_requirements_satisfied(project_list)
        ordered_project_queue = self.order_projects(project_list)

        while ordered_project_queue:
            self._apply_project(ordered_project_queue.popleft())

        # Set so that when called again it will retrigger queueing from planned projects.
        self._ordered_projects = None

    def write(
        self,
        path: Path,
        name: str,
        overwrite: bool = True,
        roadway_write: bool = True,
        transit_write: bool = True,
        projects_write: bool = True,
        roadway_convert_complex_link_properties_to_single_field: bool = False,
        roadway_out_dir: Optional[Path] = None,
        roadway_prefix: Optional[str] = None,
        roadway_file_format: RoadwayFileTypes = "parquet",
        roadway_true_shape: bool = False,
        transit_out_dir: Optional[Path] = None,
        transit_prefix: Optional[str] = None,
        transit_file_format: TransitFileTypes = "txt",
        projects_out_dir: Optional[Path] = None,
    ) -> Path:
        """Writes scenario networks and summary to disk and returns path to scenario file.

        Args:
            path: Path to write scenario networks and scenario summary to.
            name: Name to use.
            overwrite: If True, will overwrite the files if they already exist.
            roadway_write: If True, will write out the roadway network.
            transit_write: If True, will write out the transit network.
            projects_write: If True, will write out the project cards.
            roadway_convert_complex_link_properties_to_single_field: If True, will convert complex
                link properties to a single field.
            roadway_out_dir: Path to write the roadway network files to.
            roadway_prefix: Prefix to add to the file name.
            roadway_file_format: File format to write the roadway network to
            roadway_true_shape: If True, will write the true shape of the roadway network
            transit_out_dir: Path to write the transit network files to.
            transit_prefix: Prefix to add to the file name.
            transit_file_format: File format to write the transit network to
            projects_out_dir: Path to write the project cards to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.road_net and roadway_write:
            if roadway_out_dir is None:
                roadway_out_dir = path / "roadway"
            roadway_out_dir.mkdir(parents=True, exist_ok=True)

            write_roadway(
                net=self.road_net,
                out_dir=roadway_out_dir,
                prefix=roadway_prefix or name,
                convert_complex_link_properties_to_single_field=roadway_convert_complex_link_properties_to_single_field,
                file_format=roadway_file_format,
                true_shape=roadway_true_shape,
                overwrite=overwrite,
            )
        if self.transit_net and transit_write:
            if transit_out_dir is None:
                transit_out_dir = path / "transit"
            transit_out_dir.mkdir(parents=True, exist_ok=True)
            write_transit(
                self.transit_net,
                out_dir=transit_out_dir,
                prefix=transit_prefix or name,
                file_format=transit_file_format,
                overwrite=overwrite,
            )
        if projects_write:
            if projects_out_dir is None:
                projects_out_dir = path / "projects"
            write_applied_projects(
                self,
                out_dir=projects_out_dir,
                overwrite=overwrite,
            )

        scenario_data = self.summary
        if transit_write:
            scenario_data["transit"] = {
                "dir": str(transit_out_dir),
                "file_format": transit_file_format,
            }
        if roadway_write:
            scenario_data["roadway"] = {
                "dir": str(roadway_out_dir),
                "file_format": roadway_file_format,
            }
        if projects_write:
            scenario_data["project_cards"] = {"dir": str(projects_out_dir)}
        scenario_file_path = Path(path) / f"{name}_scenario.yml"
        with scenario_file_path.open("w") as f:
            yaml.dump(scenario_data, f, default_flow_style=False, allow_unicode=True)
        return scenario_file_path

    @property
    def summary(self) -> dict:
        """A high level summary of the created scenario and public attributes."""
        skip = ["road_net", "base_scenario", "transit_net", "project_cards", "config"]
        summary_dict = {
            k: v for k, v in self.__dict__.items() if not k.startswith("_") and k not in skip
        }
        summary_dict["config"] = self.config.to_dict()

        """
        # Handle nested dictionary for "base_scenario"
        skip_base = ["project_cards"]
        if "base_scenario" in self.__dict__:
            base_summary_dict = {
                k: v
                for k, v in self.base_scenario.items()
                if not k.startswith("_") and k not in skip_base
            }
            summary_dict["base_scenario"] = base_summary_dict
        """

        return summary_dict


def create_scenario(
    base_scenario: Optional[Union[Scenario, dict]] = None,
    name: str = datetime.now().strftime("%Y%m%d%H%M%S"),
    project_card_list=None,
    project_card_filepath: Optional[Union[list[Path], Path]] = None,
    filter_tags: Optional[list[str]] = None,
    config: Optional[Union[dict, Path, list[Path], WranglerConfig]] = None,
) -> Scenario:
    """Creates scenario from a base scenario and adds project cards.

    Project cards can be added using any/all of the following methods:
    1. List of ProjectCard instances
    2. List of ProjectCard files
    3. Directory and optional glob search to find project card files in

    Checks that a project of same name is not already in scenario.
    If selected, will validate ProjectCard before adding.
    If provided, will only add ProjectCard if it matches at least one filter_tags.

    Args:
        base_scenario: base Scenario scenario instances of dictionary of attributes.
        name: Optional name for the scenario. Defaults to current datetime.
        project_card_list: List of ProjectCard instances to create Scenario from. Defaults
            to [].
        project_card_filepath: where the project card is.  A single path, list of paths,
        a directory, or a glob pattern. Defaults to None.
        filter_tags: If used, will only add the project card if
            its tags match one or more of these filter_tags. Defaults to []
            which means no tag-filtering will occur.
        config: Optional wrangler configuration file or dictionary or instance. Defaults to
            default config.
    """
    base_scenario = base_scenario or {}
    project_card_list = project_card_list or []
    filter_tags = filter_tags or []

    scenario = Scenario(base_scenario, config=config, name=name)

    if project_card_filepath:
        project_card_list += list(
            read_cards(project_card_filepath, filter_tags=filter_tags).values()
        )

    if project_card_list:
        scenario.add_project_cards(project_card_list, filter_tags=filter_tags)

    return scenario


def write_applied_projects(scenario: Scenario, out_dir: Path, overwrite: bool = True) -> None:
    """Summarizes all projects in a scenario to folder.

    Args:
        scenario: Scenario instance to summarize.
        out_dir: Path to write the project cards.
        overwrite: If True, will overwrite the files if they already exist.
    """
    outdir = Path(out_dir)
    prep_dir(out_dir, overwrite=overwrite)

    for p in scenario.applied_projects:
        if p in scenario.project_cards:
            card = scenario.project_cards[p]
        elif p in scenario.base_scenario["project_cards"]:
            card = scenario.base_scenario["project_cards"][p]
        else:
            continue
        filename = Path(card.__dict__.get("file", f"{p}.yml")).name
        outpath = outdir / filename
        write_card(card, outpath)


def load_scenario(
    scenario_data: Union[dict, Path],
    name: str = datetime.now().strftime("%Y%m%d%H%M%S"),
) -> Scenario:
    """Loads a scenario from a file written by Scenario.write() as the base scenario.

    Args:
        scenario_data: Scenario data as a dict or path to scenario data file
        name: Optional name for the scenario. Defaults to current datetime.
    """
    if not isinstance(scenario_data, dict):
        WranglerLogger.debug(f"Loading Scenario from file: {scenario_data}")
        scenario_data = load_dict(scenario_data)
    else:
        WranglerLogger.debug("Loading Scenario from dict.")

    base_scenario_data = {
        "roadway": scenario_data.get("roadway"),
        "transit": scenario_data.get("transit"),
        "applied_projects": scenario_data.get("applied_projects", []),
        "conflicts": scenario_data.get("conflicts", {}),
    }
    base_scenario = _load_base_scenario_from_config(
        base_scenario_data, config=scenario_data["config"]
    )
    my_scenario = create_scenario(
        base_scenario=base_scenario, name=name, config=scenario_data["config"]
    )
    return my_scenario


def create_base_scenario(
    roadway: Optional[dict] = None,
    transit: Optional[dict] = None,
    applied_projects: Optional[list] = None,
    conflicts: Optional[dict] = None,
    config: WranglerConfig = DefaultConfig,
) -> dict:
    """Creates a base scenario dictionary from roadway and transit network files.

    Args:
        roadway: kwargs for load_roadway_from_dir
        transit: kwargs for load_transit from dir
        applied_projects: list of projects that have been applied to the base scenario.
        conflicts: dictionary of conflicts that have been identified in the base scenario.
            Takes the format of `{"projectA": ["projectB", "projectC"]}` showing that projectA,
            which has been applied, conflicts with projectB and projectC and so they shouldn't be
            applied in the future.
        config: WranglerConfig instance.
    """
    applied_projects = applied_projects or []
    conflicts = conflicts or {}
    if roadway:
        road_net = load_roadway_from_dir(**roadway, config=config)
    else:
        road_net = None
        WranglerLogger.info(
            "No roadway directory specified, base scenario will have empty roadway network."
        )

    if transit:
        transit_net = load_transit(**transit, config=config)
        if roadway:
            transit_net.road_net = road_net
    else:
        transit_net = None
        WranglerLogger.info(
            "No transit directory specified, base scenario will have empty transit network."
        )

    base_scenario = {
        "road_net": road_net,
        "transit_net": transit_net,
        "applied_projects": applied_projects,
        "conflicts": conflicts,
    }

    return base_scenario


def _load_base_scenario_from_config(
    base_scenario_data: Union[dict, ScenarioInputConfig], config: WranglerConfig = DefaultConfig
) -> dict:
    """Loads a scenario from a file written by Scenario.write() as the base scenario.

    Args:
        base_scenario_data: Scenario data that conforms to ScenarioInputConfig or is
            ScenarioInputConfig instance.
        config: WranglerConfig instance. Defaults to DefaultConfig.
    """
    if not isinstance(base_scenario_data, ScenarioInputConfig):
        base_scenario_data = ScenarioInputConfig(**base_scenario_data)

    base_scenario = create_base_scenario(**base_scenario_data.to_dict(), config=config)

    return base_scenario


def extract_base_scenario_metadata(base_scenario: dict) -> dict:
    """Extract metadata from base scenario rather than keeping all of big files.

    Useful for summarizing a scenario.
    """
    _skip_copy = ["road_net", "transit_net", "config"]
    out_dict = {k: v for k, v in base_scenario.items() if k not in _skip_copy}
    if isinstance(base_scenario.get("road_net"), RoadwayNetwork):
        nodes_file_path = base_scenario["road_net"].nodes_df.attrs.get("source_file", None)
        if nodes_file_path is not None:
            out_dict["roadway"] = {
                "dir": str(Path(nodes_file_path).parent),
                "file_format": str(nodes_file_path.suffix).lstrip("."),
            }
    if isinstance(base_scenario.get("transit_net"), TransitNetwork):
        feed_path = base_scenario["transit_net"].feed.feed_path
        if feed_path is not None:
            out_dict["transit"] = {"dir": str(feed_path)}
    return out_dict


def build_scenario_from_config(
    scenario_config: Union[Path, list[Path], ScenarioConfig, dict],
) -> Scenario:
    """Builds a scenario from a dictionary configuration.

    Args:
        scenario_config: Path to a configuration file, list of paths, or a dictionary of
            configuration.
    """
    WranglerLogger.info(f"Building Scenario from Configuration: {scenario_config}")
    scenario_config = load_scenario_config(scenario_config)
    WranglerLogger.debug(f"{pprint.pformat(scenario_config)}")

    base_scenario = create_base_scenario(
        **scenario_config.base_scenario.to_dict(), config=scenario_config.wrangler_config
    )

    my_scenario = create_scenario(
        base_scenario=base_scenario,
        config=scenario_config.wrangler_config,
        **scenario_config.projects.to_dict(),
    )

    my_scenario.apply_all_projects()

    write_args = _scenario_output_config_to_scenario_write(scenario_config.output_scenario)
    my_scenario.write(**write_args, name=scenario_config.name)
    return my_scenario


def _scenario_output_config_to_scenario_write(
    scenario_output_config: ScenarioOutputConfig,
) -> dict:
    """Converts a ScenarioOutputConfig to a dictionary for use in write method."""
    _exc = ["roadway", "transit", "project_cards"]
    scenario_write_args = {k: v for k, v in scenario_output_config.items() if k not in _exc}
    roadway_args = {f"roadway_{k}": v for k, v in scenario_output_config.roadway.items()}
    transit_args = {f"transit_{k}": v for k, v in scenario_output_config.transit.items()}

    scenario_write_args.update(roadway_args)
    scenario_write_args.update(transit_args)

    if scenario_output_config.project_cards is not None:
        project_args = {
            f"projects_{k}": v for k, v in scenario_output_config.project_cards.items()
        }
        scenario_write_args.update(project_args)

    return scenario_write_args
