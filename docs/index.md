---
hide:
 - navigation
---

Network Wrangler is a Python library for managing travel model network scenarios.

## System Requirements

Network Wrangler should be operating system agonistic and has been tested on Ubuntu and Mac OS.

Network Wrangler does require Python 3.10+.  If you have a different version of Python installed (e.g. from ArcGIS), `conda` or a similar virtual environment manager can care of installing it for you in the installation instructions below.

!!! tip "installing conda"

    In order to assist in installation, its helpful to have [miniconda](https://docs.conda.io/en/latest/miniconda.html) or another virtual environment manager installed to make sure the network wrangler dependencies don't interfer with any other package requirements you have. If you don't have any of these already, we recommend starting with Miniconda as it has the smallest footprint. `conda` is the environment manager that is contained within both the Anaconda and mini-conda applications.

??? warning "installing conda or Anaconda (the GUI version) for the first time on computer with Cube or ArcGIS?"

    For ArcGIS / Cube users - Recommend Install conda by leaving boxes unchecked for advanced options – system path and register anaconda. On some systems, checking these boxes will break Cube and ArcGis systems.

## Installation

### Create and Activate Virtual Environment

Create and/or activate the virtual environment where you want to install Network Wrangler.

!!! example "Option 1. Create a new conda environment for wrangler using conda"

    ```bash
    conda config --add channels conda-forge
    conda create python=3.11 -n wrangler #if you don't already have a virtual environment
    conda activate wrangler
    ```

!!! example "Option 2. Use pre-packaged conda-environment with all dependencies"

    Network wrangler comes packaged with a conda environment that already has all the required dependencies.  This can be helpful if you are having trouble with different versions of requirements since `conda` can be better at sorting through them than `pip`.

    ```bash
    conda config --add channels conda-forge
    conda env create -f environments/conda/environment.yml
    conda activate wrangler
    ```

### Step 2. Consider which Dependencies you want

The **core requirements** for network wrangler are specified in `pyproject.toml` and **will be automatically checked and installed when you install network wrangler**.

Additional, optional libraries are specified in separate requirements files to reduce the bloat of the minimum installation.

| **File**                   | **pip Option Code** | **Purpose**          |
|----------------------------|---------------------|----------------------|
| `requirements.viz.txt`     | `viz` | Requirements for running visualizations.  |
| `requirements.docs.txt`    | `docs` | Requirements for building documentation.  |
| `requirements.tests.txt`   | `tests` | Requirements for running tests.           |

If you want to view your networks or use jupyter notebooks, will likely want to install at least the visualization dependencies, which you can always do later as follows:

```bash
conda activate wrangler
pip install -r requirements.viz.txt
```

!!! warning "install additional dependencies using pip"

    You don't have to separately install these dependencies. You can also install them when you install network wrangler itself using command like follows using the respective pip option code or a list of them:

    ```shell
    conda activate wrangler
    pip install network-wrangler[viz]
    ```

!!! warning "tricky dependencies"

    `rtree`, `geopandas` and `osmnx` can have some tricky co-dependencies.  If don't already have an up-to-date installation of them, we've had the best success installing them using [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) (as opposed to pip).

    ```bash
    conda install rtree geopandas osmnx
    ```

### Step 3. Install Wrangler

=== "Latest Release"

    ```bash
    pip install network-wrangler
    ```

=== "From GitHub"

    Only necessary if you want to test features before they have released as official versions.

    ```bash
    pip install git+https://github.com/wsp-sag/network_wrangler.git@develop#egg=network_wrangler
    ```

    !!! tip

        If you wanted to install from a specific tag/version number or branch, replace `@develop` with `@<branchname>`  or `@tag`

### Common Installation Issues

??? question "libstdc++ is deprecated"

    > clang: warning: libstdc++ is deprecated; move to libc++ with a minimum deployment target of OS X 10.9 [-Wdeprecated]`

    If you are using MacOS, you might need to update your [xcode command line tools and headers](https://developer.apple.com/downloads/)

??? question "libspatialindex_c or Missing GEOS module"

    > "OSError: Could not find libspatialindex_c library file

    or

    > Shapely, a pre-requisite, doesn't install propertly because it is missing GEOS module

    Try installing them using conda.

    ```bash
    conda uninstall geopandas rtree shapely
    conda install rtree shapely geopandas
    ```

??? question "Conda is unable to install a library or to update to a specific library version"

    Add libraries from conda-forge

    ```bash
    conda install -c conda-forge *library*
    ```

??? question "User does not have permission to install in directories"

    Try [running Anaconda as an administrator](https://techcommunity.microsoft.com/t5/windows-11/how-to-run-app-as-administrator-by-default-in-windows-11/td-p/3033704), even if you are an admin on your machine.

## Quickstart

To get a feel for the API and using project cards, please refer to the "Wrangler Quickstart" jupyter notebook.

To start the notebook, open a command line in the network_wrangler top-level directory and type:

`jupyter notebook`

## Usage

```python

import network_wrangler

##todo this is just an example for now

network_wrangler.setup_logging()

## Network Manipulation
my_network = network_wrangler.read_roadway_network(...) # returns
my_network.apply_project_card(...) # returns
my_network.write_roadway_network(...) # returns

## Scenario Building
my_scenario = network_wrangler.create_scenario(
        base_scenario=my_base_scenario,
        card_search_dir=project_card_directory,
        tags = ["baseline-2050"]
        )
my_scenario.apply_all_projects()
my_scenario.write("my_project/baseline", "baseline-2050")
my_scenario.summarize(outfile="scenario_summary_baseline.txt")

my_scenario.add_projects_from_files(list_of_build_project_card_files)
my_scenario.queued_projects
my_scenario.apply_all_projects()
my_scenario.write("my_project/build", "baseline")
```

## Attribution

NetworkWrangler was developed using resources from the [Metropolitan Transportation Commission](bayareametro.gov), [Metropolitan Council MN](https://metrocouncil.org/), and in-kind time from [UrbanLabs LLC](https://urbanlabs.io) and [WSP](www.wsp.com).  It is currently maintained using in-kind time...so please be patient.

This project is built upon the ideas and concepts implemented in the [network wrangler project](https://github.com/sfcta/networkwrangler) by the [San Francisco County Transportation Authority](http://github.com/sfcta) and expanded upon by the [Metropolitan Transportation Commission](https://github.com/BayAreaMetro/NetworkWrangler).

While Network Wrangler as written here is based on these concepts, the code is distinct and builds upon other packages such as `geopandas` and `pydantic` which hadn't been implemented when networkwrangler 1.0 was developed.

## Contributing

Contributions are welcome. Please review [contributing guidelines and instructions](development.md).

## Companion Software

[ProjectCard](https://network-wrnagler.github.io/projectcard): Initially part of NetworkWrangler, the functionality for reading, writing and validating ProjectCard objects was pulled out into a separate project so that it could be used by other entities without necessitating NetworkWrangler.

## Having an issue?

🪲 NetworkWrangler may contain bugs.

🤔 Also, since it has primarily been used by its developers, the documentation may contain some omissions or not be entirely clear.

But we'd love to make it better! Please report bugs or incorrect/unclear/missing documentation with a [GitHub Issue](https://github.com/wsp-sag/network-wrangler/issues) -  or [fix them yourself with a pull request](development.md)!

## License

[Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)

## Release History

{!
    include-markdown "../CHANGELOG.md"
    heading-offset=1
!}
