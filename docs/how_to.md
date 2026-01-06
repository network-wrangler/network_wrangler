# How To

## Build a Scenario using API

::: network_wrangler.scenario
    options:
        members: []
        heading_level: 3
    handlers:
      python:
        options:
          show_root_toc_entry: false

!!! tip "additional examples"

    You can see additional scenario creating capabilities in the example jupyter notebook `Scenario Building Example.ipynb`.

## Build a Scenario from a Scenario Configuration File

::: network_wrangler.configs.scenario
    options:
        members: []
        heading_level: 3
    handlers:
      python:
        options:
          show_root_toc_entry: false

## Change Wrangler Configuration

!!! example "Using YAML Configuration File (Recommended)"

    The easiest way to configure Network Wrangler parameters is using a YAML file. This is especially
    useful for users less familiar with Python, as you can simply point to a YAML file path.

    **Step 1:** Create a YAML configuration file (e.g., `wrangler.config.yml`):

    ```yaml
    IDS:
      ML_LINK_ID_METHOD: scalar
      ML_LINK_ID_SCALAR: 3000000
    MODEL_ROADWAY:
      ML_OFFSET_METERS: -10
    EDITS:
      EXISTING_VALUE_CONFLICT: warn
    ```

    **Step 2:** Load and use the configuration:

    ```python
    from network_wrangler.configs import load_wrangler_config
    from network_wrangler import load_roadway_from_dir
    from pathlib import Path

    # Load configuration from YAML file
    config = load_wrangler_config("path/to/wrangler.config.yml")

    # Use the configuration when loading networks
    road_net = load_roadway_from_dir(
        "path/to/roadway",
        config=config
    )
    ```

    **Step 3:** Or use it when building scenarios:

    ```python
    from network_wrangler.scenario import create_scenario, create_base_scenario

    base_scenario = create_base_scenario(
        roadway={"dir": "path/to/roadway"},
        config=config
    )
    my_scenario = create_scenario(
        base_scenario=base_scenario,
        config=config
    )
    ```

    See `examples/stpaul/wrangler.config.yml` for a complete example with all available parameters.

!!! warning "Configuration Best Practices"

    **Always use YAML or TOML configuration files** for setting Wrangler parameters. Do not modify
    `DefaultConfig` or other Python code to change configuration values. Using configuration files:
    
    - Makes it easy to track and version control your settings
    - Allows you to use different configurations for different projects
    - Is more accessible for users less familiar with Python
    - Prevents accidental global state changes

!!! tip "Alternative Configuration Methods"

    For advanced use cases, you can also pass a dictionary directly to configuration functions.
    However, YAML/TOML files are strongly recommended. See the full API documentation below for
    all options.

::: network_wrangler.configs.wrangler
    options:
        heading_level: 3
    handlers:
      python:
        options:
          show_root_toc_entry: false

## Review changes beetween networks

!!! example "Review Added Managed Lanes"

    ```python
    from network_wrangler import load_roadway_from_dir
    from projectcard import read_card
    from pathlib import Path

    EXAMPLE_DIR = Path.cwd().parent / "examples"
    STPAUL = EXAMPLE_DIR / "stpaul"
    STPAUL_ROAD = load_roadway_from_dir(STPAUL)

    card_path = STPAUL / "project_cards" / "road.prop_change.managed_lanes.yml"
    card = read_card(card_path)
    stpaul_build = STPAUL_ROAD.apply(card)

    ml_map = STPAUL_ROAD.links_df[STPAUL_ROAD.links_df.managed > 0].explore(
        color="blue",
        tiles="CartoDB positron",
        name="Managed Lanes",
        style_kwds={"opacity": 0.6, "weight": 20}
    )

    added_managed_lanes = stpaul_build.links_df[(stpaul_build.links_df.managed > 0) & (STPAUL_ROAD.links_df.managed == 0)]

    added_managed_lanes.explore(
        m=ml_map,
        color="red",
        name="Added Managed Lanes",
        style_kwds={"opacity": 0.6, "weight": 20}
    )
    ```

!!! tip "additional examples"
    You can see additional scenario review capabilities in the example jupyter notebook `Visual Checks.ipynb`.

## Review selected facilities

!!! example "Review selected links"

    ```python
    from network_wrangler import load_roadway_from_dir
    from pathlib import Path

    EXAMPLE_DIR = Path.cwd().parent / "examples"
    STPAUL = EXAMPLE_DIR / "stpaul"

    STPAUL_ROAD = load_roadway_from_dir(STPAUL)
    sel_dict = {
      "links": {
          "modes": ["walk"],
          "name": ["Valley Street"],
      },
      "from": {"model_node_id": 174762},
      "to": {"model_node_id": 43041},
    }
    STPAUL_ROAD.get_selection(sel_dict).selected_links_df.explore(
      color="red", style_kwds={"opacity": 0.6, "weight": 20}
    )
    ```

!!! tip "additional examples"

    You can see additional interactive exploration of how selections work and how to review them in the Jupyter notebook `Roadway Network Search.ipynb`.

## Create your own example data from Open Street Map

::: network_wrangler.bin.build_basic_osm_roadnet
    options:
        show_bases: false
        show_root_toc_entry: false
        heading_level: 3
        show_source: false
        members: false

!!! tip "additional examples"

    You can review the process in this script step-wise and interactively create your own networks from OSM with variation in the underlying assumptions in the Jupyter notebook `Create Network from OSM.ipynb`.

## Review separated model network managed lanes

!!! example "Review model network"

    ```python
    m_net = stpaul_build.model_net
    model_net_map = m_net.gp_links_df.explore(
        tiles="CartoDB positron",
        color="blue",
        style_kwds={"opacity": 0.6, "weight": 10}
    )
    m_net.ml_links_df.explore(m=model_net_map, color="red", style_kwds={"opacity": 0.6, "weight": 10})
    m_net.dummy_links_df.explore(m=model_net_map, color="green", style_kwds={"opacity": 0.6, "weight": 10})
    ```

!!! tip "additional examples"

    You can learn more about visualization of networks in the Jupyter notebook `Network Viewer.ipynb`.

{!
  include-markdown("https://raw.githubusercontent.com/network-wrangler/projectcard/refs/heads/main/docs/how-to.md")
!}
