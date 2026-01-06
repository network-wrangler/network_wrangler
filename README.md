Network Wrangler is a Python library for managing travel model network scenarios.

## System Requirements

Network Wrangler should be operating system agonistic and has been tested on Ubuntu and Mac OS.

Network Wrangler does require Python 3.7+.  If you have a different version of Python installed (e.g. from ArcGIS), `conda` or a similar virtual environment manager can care of installing it for you in the installation instructions below.

## Installation

Requirements for basic network_wranglerare functionality as well as enhanced *development/testing*, *visualization* and *documentation* functionalities are stored in `requirements*.txt` and `pyproject.toml` but are automatically installed when using `pip`.

Ready to install network wrangler?

### Latest Official Version

```bash
pip install network-wrangler
```

### From GitHub

```bash
pip install git+https://github.com/network-wrangler/network_wrangler.git@main#egg=network_wrangler
```

If you wanted to install from a specific tag/version number or branch, replace `@main` with `@<branchname>`  or `@tag`

### From Clone

If you are going to be working on Network Wrangler locally, you might want to clone it to your local machine and install it from the clone.  The -e will install it in [editable mode](https://pip.pypa.io/en/stable/reference/pip_install/?highlight=editable#editable-installs).

If you have [GitHub desktop](https://desktop.github.com/) installed, you can either do this by using the GitHub user interface by clicking on the green button "clone or download" in the [main network wrangler repository page](https://github.com/network-wrangler/network_wrangler).

Otherwise, you can use the command prompt to navigate to the directory that you would like to store your network wrangler clone and then using a [git command](https://git-scm.com/downloads) to clone it.

```bash
cd path to where you want to put wrangler
git clone https://github.com/network-wrangler/network_wrangler
```

Then you should be able to install Network Wrangler in "develop" mode.

```bash
cd network_wrangler
pip install -e .
```

## Quickstart

To get a feel for the API and using project cards, please refer to the "Wrangler Quickstart" jupyter notebook.

To start the notebook, open a command line in the network_wrangler top-level directory and type:

`jupyter notebook`

## Documentation

Documentation can be built from the `/docs` folder using the command: `make html`

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

This project is built upon the ideas and concepts implemented in the [network wrangler project](https://github.com/sfcta/networkwrangler) by the [San Francisco County Transportation Authority](http://github.com/sfcta) and expanded upon by the [Metropolitan Transportation Commission](https://github.com/BayAreaMetro/NetworkWrangler).

While Network Wrangler as written here is based on these concepts, the code is distinct and builds upon other packages such as `geopandas` and `pydantic` which hadn't been implemented when networkwrangler 1.0 was developed.

## Contributing

Pull requests are welcome. Please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License

[Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
