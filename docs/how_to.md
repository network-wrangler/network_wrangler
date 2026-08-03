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
          show_root_heading: false
          show_source: false
          show_submodules: false
          show_classes: false
          show_functions: false

## Change Wrangler Configuration

::: network_wrangler.configs.wrangler
    options:
      members: []
      heading_level: 3
    handlers:
      python:
        options:
          show_root_toc_entry: false
          show_root_toc_entry: false
          show_root_heading: false
          show_source: false
          show_submodules: false
          show_classes: false
          show_functions: false

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

## Create a Network from OSM and GTFS

The [`notebook/Create Network from OSM.ipynb`](https://github.com/network-wrangler/network_wrangler/blob/main/notebook/Create%20Network%20from%20OSM.ipynb) notebook provides an interactive walkthrough of building a network step by step. The [`mtc_wrangler`](https://github.com/BayAreaMetro/mtc_wrangler/) script [`create_baseyear_network/create_mtc_network_from_OSM.py`](https://github.com/BayAreaMetro/mtc_wrangler/blob/main/create_baseyear_network/create_mtc_network_from_OSM.py) shows a complete production pipeline, summarized below.

**Step 1: Download OSM road network**

Use [`osmnx`](https://osmnx.readthedocs.io) to fetch the raw road graph for your geography. Enable caching to avoid repeated downloads during development:

```python
import osmnx
osmnx.settings.use_cache = True
g = osmnx.graph_from_place('San Francisco, California, USA', network_type='all')
# or for a bounding box:
# g = osmnx.graph_from_bbox(bbox, network_type='all')
```

**Step 2: Simplify topology**

Project to a local CRS and consolidate nearby intersections to remove unnecessary intermediate nodes while preserving connectivity:

```python
g = osmnx.projection.project_graph(g, to_crs=local_crs)
g = osmnx.simplification.consolidate_intersections(
    g, tolerance=30, rebuild_graph=True, dead_ends=True, reconnect_edges=True
)
nodes_gdf, links_gdf = osmnx.graph_to_gdfs(g)
```

**Step 3: Create a RoadwayNetwork**

Rename and augment columns to meet the wrangler schema (adding `A`, `B`, `model_link_id`, `drive_access`, `walk_access`, etc.), then load into a [`RoadwayNetwork`](api.md#network_wrangler.roadway.network.RoadwayNetwork) using [`load_roadway_from_dataframes()`](api_roadway.md#network_wrangler.roadway.io.load_roadway_from_dataframes):

```python
import network_wrangler as nw

road_net = nw.load_roadway_from_dataframes(
    links_df=links_gdf,
    nodes_df=nodes_gdf,
    shapes_df=links_gdf,
)
```

Before adding centroids, apply any additional enrichment to the network: setting facility types, managed lane fields, controlled-access highway flags, bridge toll link attributes, and per-link `{mode}_centroid_fit` values (see [`FitForCentroidConnection`](api_roadway.md#network_wrangler.roadway.centroids.FitForCentroidConnection)).

**Step 4: Add zone centroids and connectors**

Add zone centroid nodes and create connector links using [`add_centroid_nodes()`](api_roadway.md#network_wrangler.roadway.centroids.add_centroid_nodes) and [`add_centroid_connectors()`](api_roadway.md#network_wrangler.roadway.centroids.add_centroid_connectors).

```python
from network_wrangler.roadway.centroids import add_centroid_nodes, add_centroid_connectors

# TAZ connectors
add_centroid_nodes(road_net, taz_zones_gdf, zone_id="TAZ_NODE")
add_centroid_connectors(
    road_net, taz_zones_gdf, zone_id="TAZ_NODE", mode="drive",
    local_crs=local_crs,
    zone_buffer_distance=20,    # units of local_crs — search radius beyond zone boundary
    num_centroid_connectors=4,  # max connectors per zone
    max_mode_graph_degrees=4,   # exclude high-degree nodes (e.g. motorway ramps)
)
```

The connector selection algorithm first picks the node with the best fitness and shortest distance, then selects each additional connector to maximize angular separation from already-selected ones, ensuring good spatial distribution around the centroid.

**Step 5: Prepare GTFS transit data**

Load the GTFS feed and filter it to the target service date and geography using [`load_feed_from_path()`](api_transit.md#network_wrangler.transit.feed.io.load_feed_from_path):

```python
from network_wrangler.transit.feed.io import load_feed_from_path

gtfs_feed = load_feed_from_path(
    input_gtfs_path,
    service_ids_filter=service_ids,  # pre-filter to a specific operating day
)
```

Additional cleanup — dropping irrelevant agencies, filtering stops to the study area boundary, removing duplicate consecutive stops — should be applied before the next step.

**Step 6: Create TransitNetwork**

Conflate transit stops to roadway nodes and create access links, converting the GTFS-flavored feed to a wrangler-flavored [`TransitNetwork`](api.md#network_wrangler.transit.network.TransitNetwork):

```python
from network_wrangler.transit.feed.io import create_feed_from_gtfs_model

feed = create_feed_from_gtfs_model(
    gtfs_feed, road_net,
    local_crs=local_crs,
    timeperiods=time_periods,
    frequency_method='median_headway',
    add_stations_and_links=True,
)
transit_net = nw.load_transit(feed)
```

**Write to files**

```python
road_net.write(out_dir=output_dir, prefix="my_network", file_format="geojson")
nw.write_transit(transit_net, output_dir, prefix="my_transit")
```

## Create a Network from Overture

The `notebook/Create Network from Overture.ipynb` notebook explores using [Overture Maps](https://overturemaps.org/) data as an alternative roadway source. Data is downloaded with the [Overture Maps Python CLI](https://docs.overturemaps.org/getting-data/overturemaps-py/):

```bash
pip install overturemaps
overturemaps download --bbox=<west,south,east,north> -f geoparquet --type=segment   -o segments.parquet
overturemaps download --bbox=<west,south,east,north> -f geoparquet --type=connector -o connectors.parquet
```

Overture `segment` features map to wrangler links and `connector` features map to nodes. Most attribute mappings are straightforward (name, roadway class, access modes). Overture uses GERS hex IDs which must be converted to integers for use as `model_link_id`, `A`, and `B`. Speed limits and other scoped values are stored as nested JSON in parquet and require parsing before use.

!!! warning "Overture data not yet suitable for production use"

    The `lanes` attribute is absent from the Overture dataset, making it impossible to determine reliably which links should be one-way vs. two-way or to correctly set link directionality. The notebook is therefore an exploratory reference rather than a complete workflow.

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
