# Contributing to Network Wrangler

## Setup

### Recommended Tools

- [GitHub desktop](https://desktop.github.com/) to manage access to the main repository.
- [Git](https://git-scm.com/downloads) to conduct required version control.
- [MiniConda](https://docs.anaconda.com/miniconda/miniconda-install/) to manage your Python environments.
- [VSCode](https://code.visualstudio.com/download) to edit and test code.
- Some type of terminal application (note, this comes with Mac/Ubuntu).

### Setup Virtual Environment

Create and/or activate the virtual environment where you want to install Network Wrangler.

!!! example "Creating and activating a virtual environment using conda"
    ```bash
    conda config --add channels conda-forge
    conda create python=3.11 -n wrangler-dev #if you don't already have a virtual environment
    conda activate wrangler-dev
    ```

### Clone

To effectively work on Network Wrangler locally, install it from a clone by either:

1. Use the GitHub user interface by clicking on the green button "clone or download" in the [main network wrangler repository page](https://github.com/wsp-sag/network_wrangler).
2. Use the command prompt in a terminal to navigate to the directory that you would like to store your network wrangler clone and then using a [git command](https://git-scm.com/downloads) to clone it.

!!! example "Clone network wrangler"
    ```bash
    cd path to where you want to put wrangler
    git clone https://github.com/wsp-sag/network_wrangler
    ```

### Install in Develop Mode

Install Network Wrangler in ["develop" mode](https://pip.pypa.io/en/stable/reference/pip_install/?highlight=editable#editable-installs) using the `-e` flag so that changes to your code will be reflected when you are using and testing network wrangler:

!!! example "Install Network Wrangler from Clone"
    ```bash
    cd network_wrangler
    pip install -e .
    ```

!!! example "Install development dependencies"
    ```bash
    pip install -r requirements.tests.txt
    pip install -r requirements.docs.txt
    ```

### IDE Settings

### VSCode

Select conda env as Python interpreter:

    - `cmd-shift-P`: Python: Select Interpreter

If you are using VS Code, here are some recommended extensions and settings to leverage the IDE capabilities:

| Extension | Purpose |
| --------- | ------- |
| Microsoft Python | Pytest integration, code-completion |
| Astral Ruff | Linting and formatting |
| Microsoft Jupyter | Edit and run python notebooks |
| Microsoft Data Wrangler | Review and edit data in pandas dataframes |
| David Anson markdownlint | Lint markdown |
| Github Pull Requests | Manage github issues and PRs |
| Dvir Yitzchaki parquet-viewer | Review parquet data as json |
| Random Fractals Inc. Geo Data Viewer | Render geojson data |

Leveraging these extensions to their full potential may take some configuration. Here are some examples. YMMV.

!!! example "`settings.json` for VS Code"

    ```json
    {
        "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
        "source.fixAll": "explicit",
        "source.organizeImports": "explicit"
        }
    },
        "python.testing.pytestArgs": [
            "tests"
        ],
        "python.testing.unittestEnabled": false,
        "python.testing.pytestEnabled": true,
        "python.testing.cwd": "",
        "python.testing.autoTestDiscoverOnSaveEnabled": true,
        "python.defaultInterpreterPath": "/usr/bin/env python",
        "python.testing.pytestPath": "/opt/miniconda3/envs/wrangler-dev/bin/pytest",
    }

    ```

!!! example "Code > Settings > Settings"

    For tests to run in conda environment, add path to it. To find it, you can run `conda info --envs`

    `@id:python.condaPath`: `opt/miniconda3/envs/wrangler-dev`

## Development Workflow

1. Create [an issue](https://github.com/wsp-sag/network_wrangler/issues) for any features/bugs that you are working on.
2. [Create a branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository) to work on a new issue (or checkout an existing one where the issue is being worked on).  
3. Develop comprehensive tests in the `/tests` folder.
4. Modify code including inline documentation such that it passes *all*  tests (not just your new ones)
5. Lint code using `pre-commit run --all-files`
6. Fill out information in the [pull request template](https://github.com/wsp-sag/network_wrangler/blob/master/.github/pull_request_template.md)
7. Submit all pull requests to the `develop` branch.
8. Core developer will review your pull request and suggest changes.
9. After requested changes are complete, core developer will sign off on pull-request merge.

!!! tip
    Keep pull requests small and focused. One issue is best.

!!! tip
    Don't forget to update any associated [documentation](#documentation) as well!

## Diagram Maintenance

Architecture and data flow diagrams live as mermaid blocks in `docs/design.md`. When your PR changes module relationships, class hierarchies, or data flow, update the relevant diagram in the same PR. Current diagrams cover: ecosystem data flow, core object model, roadway selection, project application flow, scoped property resolution, and IO pipeline.

## Documentation

Documentation is stored in the `/docs` folder and created by [`mkdocs`](https://www.mkdocs.org/) using the [`material-for-mkdocs`](https://squidfunk.github.io/mkdocs-material/) theme.

!!! example "Build and locally serve documentation"
    ```bash
    mkdocs serve
    ```

Documentation is deployed using the [`mike`](https://github.com/jimporter/mike) package and Github Actions configured in `.github/workflows/` for each "ref" (i.e. branch) in the network_wrangler repository.

## Making sure your code works

### Linting and Type Checking

Before even running the tests, its a good idea to lint and check the types of the code using pre-commit:

!!! example Run pre-commit
    ```bash
    pre-commit run --all-files
    ```

Your code **must** pass the pre-commit tests as a part of continuous integration, so you might as well fix anything now if it arises.

### Adding Tests

Tests live in `/tests` organized by module (`test_roadway/`, `test_transit/`, `test_models/`, `test_utils/`).

**Fixtures**: Use session-scoped fixtures in `tests/conftest.py` for expensive setup like loading networks. Module-specific fixtures go in subdirectory `conftest.py` files.

**Markers**:

- `@pytest.mark.skipci` — tests that should not run in CI (e.g., require large data or external services)
- `@pytest.mark.failing` — known failures being investigated

**Naming**: Test files match source files (`test_links.py` for `links.py`). Test functions start with `test_` and describe the behavior being tested.

**Pandas compatibility**: Tests must pass on both pandas 2.x and 3.x. Use `pd.testing.assert_frame_equal()` for DataFrame comparisons. Be aware that string dtype behavior differs between versions.

**Benchmarks**: Performance-critical tests go in `test_benchmarks.py` and are tracked via `pytest-benchmark` across PRs.

### Running Tests

Tests and test data reside in the `/tests` directory:

!!! example Run all tests
    ```bash
    pytest
    ```

Your code **must** pass the these tests as a part of continuous integration, so you might as well fix anything now if it arises.

### Profiling Performance

When you run the tests, their performance is profiled using `pytest-profiling` and profiles for tests are stored in `.prof` directory. If you want to explore what is taking time in a particular test, you can do so using products like [`snakviz`](https://jiffyclub.github.io/snakeviz/)

!!! example "Explore performance of a test"
    ```bash
    snakeviz .prof/<test_name>.prof
    ```

We also benchmark some specific tests (`test_benchmarks.py`) that we want to compare when reviewing pull requests. If you want to review how you are doing on these benchmarks you can save the benchmarks when you run pytestand compare these numbers to another branch.

!!! example "Compare benchmarks between branches"
    ```
    pytest --benchmark-save=branch_1
    git checkout branch_2
    pytest --benchmark-save=branch_2
    pytest-benchmark compare branch_1 branch_2
    ```

## Evaluate Code Maintainability

Using [Radon](https://radon.readthedocs.io/)

> Maintainability Index is a software metric which measures how maintainable (easy to support and change) the source code is. The maintainability index is calculated as a factored formula consisting of SLOC (Source Lines Of Code), Cyclomatic Complexity and Halstead volume.

```bash
radon mi ../network_wrangler -s
```

> Cyclomatic Complexity corresponds to the number of decisions a block of code contains plus 1. This number (also called McCabe number) is equal to the number of linearly independent paths through the code. This number can be used as a guide when testing conditional logic in blocks.

```bash
radon cc ../network_wrangler --average
```

## Continuous Integration

Continuous Integration is managed by Github Actions in `.github/workflows`.  
All tests other than those with the decorator `@pytest.mark.skipci` will be run.

## Project Governance

The project is currently governed by representatives of its two major organizational contributors:

- Metropolitan Council (MN)
- Metropolitan Transportation Commission (California)

## Code of Conduct

Contributors to the Network Wrangler Project are expected to read and follow the [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md) for the project.
