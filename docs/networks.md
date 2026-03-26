# Networks

## Roadway Network Format

RoadwayNetworks must be defined in the following format, which leverages [Open Street Map](http://openstreetmap.org) which uses a tag-based format (e.g. json, xml).  This may be changed in the future to align with the tabular-based [General Modeling Network Specification](https://github.com/zephyr-data-specs/GMNS).

A network is defined by a set of nodes and links which connect them.  Shapes may be optionally specified for each link in a separate file.

!!! tip "file serialization formats"

    While the default serialiation for roadway networks is `json`/`geojson` and for transit data is `csv`, networks can also be stored – more efficiently – in parquet files with a similar structure. Other workable file serializations include shapefiles, csvs, and anything that can be read by pandas or geopandas. This can be noted in most I/O procedures by including the keyword argument `file_format = <format>`.

### Roadway Validation

RoadwayNetworks can be validated using the following tools:

=== "CLI"

    ```bash
    python validate_roadway.py <network_directory> <file_format> [-s] [--output_dir <output_dir>]
    ```
    Where:

    - `network_directory`: The roadway network file directory.
    - `file_format`: The suffices of roadway network file name.
    - `-s`, `--strict`: Validate the roadway network strictly without parsing and filling in data.
    -  `--output_dir`: The output directory for the validation report.

=== "Python API"

    ```python
    from network_wrangler.roadway.validate import validate_roadway_in_dir
    validate_roadway_in_dir(
        directory=<network_directory>,
        file_format=<file_format>,
        strict=<strict_bool>>,
        output_dir=<output_dir>
    )
    ```
    Where:

    - `network_directory`: The roadway network file directory.
    - `file_format`: The suffices of roadway network file name.
    - `strict`: Validate the roadway network strictly without parsing and filling in data.
    -  `output_dir`: The output directory for the validation report.

### Examples

Network Wrangler is packaged with two examples located in the `/examples` directory:

- St Paul, MN
- Small which is a several block exerpt of St Paul and is infinitely easier to trouble-shoot quickly.

### Road Nodes

A  valid `geojson`, `shp`, or `parquet` file.

::: network_wrangler.models.roadway.tables.RoadNodesTable
    options:
        members: []
        heading_level: 3
        show_bases: false
    handlers:
      python:
        options:
          show_root_toc_entry: false

### Road Links

A  valid `geojson`, `shp`, `parquet`, or `json` file.

::: network_wrangler.models.roadway.tables.RoadLinksTable
    options:
        members: []
        heading_level: 3
        show_bases: false
    handlers:
      python:
        options:
          show_root_toc_entry: false

### Road Shapes

A  valid `geojson`, `shp`, or `parquet` file with `LineString` geometry features and the folowing `properties`.

::: network_wrangler.models.roadway.tables.RoadShapesTable
    options:
        members: []
        heading_level: 3
        show_bases: false
    handlers:
        python:
            options:
            show_root_toc_entry: false

## Transit Network Format

::: network_wrangler.models.gtfs.tables
    options:
        show_bases: false
        members: []

Transit Networks must use the the [GTFS](https://www.gtfs.org) Schedule format with the following additional constraints:

1. At this time, only frequency-based schedules are supported.
2. Each `stop_id` must be a node in the RoadwayNetwork.
3. `shapes.txt` is *required* (it is optional in GTFS) and must have the added field `model_node_id` associating a specific location with a node on the `RoadwayNetwork`.

### Stops

::: network_wrangler.models.gtfs.tables.WranglerStopsTable
    options:
        heading_level: 3
        show_bases: false
        members: []
    handlers:
        python:
            options:
                show_root_toc_entry: false

### Routes

::: network_wrangler.models.gtfs.tables.RoutesTable
    options:
        heading_level: 3
        show_bases: false
        members: []
    handlers:
        python:
            options:
                show_root_toc_entry: false

### Trips

::: network_wrangler.models.gtfs.tables.WranglerTripsTable
    options:
        members: []
        heading_level: 3
        show_bases: false
    handlers:
        python:
            options:
                show_root_toc_entry: false

### Stop_times

::: network_wrangler.models.gtfs.tables.WranglerStopTimesTable
    options:
        members: []
        heading_level: 3
        show_bases: false
    handlers:
        python:
            options:
                show_root_toc_entry: false

### Shapes

::: network_wrangler.models.gtfs.tables.WranglerShapesTable
    options:
        members: []
        heading_level: 3
        show_bases: false
    handlers:
        python:
            options:
                show_root_toc_entry: false

### Frequencies

::: network_wrangler.models.gtfs.tables.WranglerFrequenciesTable
    options:
        members: []
        heading_level: 3
        show_bases: false
    handlers:
        python:
            options:
                show_root_toc_entry: false

### Agencies

::: network_wrangler.models.gtfs.tables.AgenciesTable
    options:
        heading_level: 3
        show_bases: false
        members: []
    handlers:
        python:
            options:
                show_root_toc_entry: false

### Transit Validation

TransitNetworks can be validated using the following tools:

=== "CLI"

    ```bash
    python validate_transit.py <network_dir> <file_format> [-s] [--output_dir <output_dir>] [--road_dir <road_dir>] [--road_file_format <road_file_format]
    ```
    Where:

    - `network_dir`: The transit network file directory.
    - `file_format`: The suffices of transit network .
    - `--output_dir`: The output directory for the validation report.
    - `--road_dir`: The directory roadway network. if want to validate the transit network to it.
    - `--road_file_format`: The file format for roadway network. Defaults to 'geojson'.

=== "Python API"

    ```python
    from network_wrangler.transit.validate import validate_transit_in_dir
    validate_transit_in_dir(
        dir=<network_dir>,
        file_format=<network_file_format>,
        road_dir=<road_dir>,
        road_file_format=<road_file_format,
    )
    ```
    Where:

    - `network_dir`: The roadway network file directory.
    - `network_file_format`: The file format of the transit files.
    - `road_dir`: The directory roadway network. if want to validate the transit network to it.
    - `road_file_format`: The file format for roadway network. Defaults to 'geojson'.

## Project Cards

Project Cards, which define changes to the roadway and transit networks must use the [ProjectCard](https://github.com/networkwrangler/projectcard) standard.

## Model Roadway Network Export Format

In order to separately calculate the delay when the networks are assigned in static roadway network assignment the roadway network must be exported with separate managed lane links.

To acheive this, RoadwayNetwork objects have the option to be exported in the ModelRoadwayNetwork format, which separates a link into two: set of managed lanes and a set of general purpose lanes which are connected by a set of dummy connector links.  

### Managed Lane Links

All properties preceded by `ML_` will be copied, without that prefix, to the managed lane links.

The following are controlled by parameters which can be set using WranglerConfig:

Geometry of managed lanes will be defined as a shape offset by the parameter `ML_OFFSET_METERS`.
Properties defined in the parameter `ADDITIONAL_COPY_FROM_GP_TO_ML` are also copied from the parent link.

New `model_node_id` s and `model_link_ids` are generated based either on ranges or using a scalar from the GP link based on: `ML_LINK_ID_METHOD`, `ML_NODE_ID_METHOD`, `ML_LINK_ID_RANGE`, `ML_NODE_ID_RANGE`, `ML_LINK_ID_SCALAR`, `ML_NODE_ID_SCALAR`

`name` is created as "managed lane of `name of GP link`"

Relationship to the general purpose lanes is retained using the fields `GP_A`, `GP_B`, `GP_model_link_id`.

Managed-lane link-ids are generated as multiples of 10.

### Dummy Connector Links

Dummy connector links are generated between the general purpose lane links and managed lane links at points defined by the variable `ML_access_point` and `ML_egress_point`.  If a managed lane is created without explictly setting these values, network wrangler will assume that the managed lanes can be accessed at any node.

The parameter `ADDITIONAL_COPY_TO_ACCESS_EGRESS` defines what additional attributes are copied from the general purpose lane to the access and egress links.

`name` is created as "<access/egress> dummy link"

`model_link_id` is created as follows, noting that `model_link_id` s for managed lanes will be multiples of 10:

- 1 + managed lane's `model_link_id` for access links
- 2 + managed lane's `model_link_id` for access links

::: network_wrangler.configs.wrangler.ModelRoadwayConfig
::: network_wrangler.configs.wrangler.IdGenerationConfig

## Network Management

Several functions assist in managing networks themselves including converting serialization formats and clipping to various geographic bounds.

### Viewing

Because the roadway network is already in GeoDataFrames, they can be easily viewed in Jupyter Notebooks using commands such as:

```python
my_small_net = SMALL_ROAD.links_df).explore(tiles="CartoDB positron")
SMALL_ROAD.nodes_df.explore(m=my_small_net, color="grey")
my_small_net
```

For larger networks, you might want to sample objects if you are just trying to get the general picture in order to save memory.

```python
stpaul_net = STPAUL_ROAD.links_df.sample(300).explore(tiles="CartoDB positron")
STPAUL_ROAD.nodes_df.sample(300).explore(m=stpaul_net, color="grey")
stpaul_net
```

For transit, you have access to GeoDataFrames for both shapes and stops:

```python
STPAUL_TRANSIT.shapes_gdf.explore(m=stpaul_net, color="limegreen")
STPAUL_TRANSIT.stops_gdf.explore(m=stpaul_net, color="limegreen")
stpaul_net
```

### Clipping

There are two options for getting networks that are subsets of the larger one:

1. clipping networks already loaded from files. Useful when you are already manipulating the objects.
2. filtering the network when it is read in.

In both instances, you have the option to filter based on one of three methods:

1. `boundary_file` which is a geojson or shapefile that you want to filter to.
2. `boundary_gdf` passing a geodataframe with a single polygon record that you want to filter to.
3. `boundary_geocode` which queries open streetmap for a jurisdiction matching the provided name, e.g. "St Paul, MN, USA".

Transit additionally has the option to be filtered to a roadway network.

#### Clipping loaded networks

If your network is already loaded from disk into a RoadwayNetwork or TransitNetwork, you can clip it using `clip_roadway()` or `clip_transit()`:

```python
from network_wrangler.roadway import clip_roadway
clipped_road_eco = clip_roadway(STPAUL_ROAD, boundary_file=TEST_DATA / "ecolab.geojson")
clipped_road_eco.links_df.explore(m= color="hotpink")
```

```python
from network_wrangler.transit.clip import clip_transit
clipped_transit_eco = clip_transit(STPAUL_TRANSIT, roadway_net=clipped_road_eco)
```

#### Filtering on being read from disk

To filter roadway networks on being read-in, you can use the same parameters (`boundary_gdf`, `boundary_geocode` or `boundary_file`, available in `load_roadway_from_dir`, and all associated methods.  

```python
downtown_net = load_roadway_from_dir(STPAUL_DIR, boundary_geocode="Downtown, St Paul, MN, USA")
```

This feature is not yet implemented in transit.

### Converting Serialization

To convert file serialization without reading it into objects, you can use the method `convert_roadway_file_serialization()`:

```python
from roadway.io import convert_roadway_file_serialization

convert_roadway_file_serialization(
    my_dir, # the path to the input directory.
    "geojson", # the file format of the input files. Defaults to "geojson".
    my_new_dir, # the path were the output will be saved.
    "parquet", # the format of the output files. Defaults to "parquet".
    "new", # the name prefix of the roadway files that will be generated. Defaults to "".
    overwrite = True, # if True, will overwrite the files if they already exist. Defaults to True.
)
```

!!! Note "clip to"

    Note you can also pass one of `boundary_geocode`, `boundary_gdf` or `boundary_file` to clip while you are converting the file serialization.
