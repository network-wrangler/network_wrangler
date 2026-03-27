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

## Documentation

Documentation is stored in the `/docs` folder and created by [`mkdocs`](https://www.mkdocs.org/) using the [`material-for-mkdocs`](https://squidfunk.github.io/mkdocs-material/) theme.

!!! example "Build and locally serve documentation"
    ```bash
    mkdocs serve
    ```

Documentation is deployed using the [`mike`](https://github.com/jimporter/mike) package and Github Actions configured in `.github/workflows/` for each "ref" (i.e. branch) in the network_wrangler repository.

### References

- [MkDocs User Guide: Configuration](https://www.mkdocs.org/user-guide/configuration/)
- [mkdocstrings-python](https://mkdocstrings.github.io/python/usage/)
- [Material for MkDocs Setup](https://squidfunk.github.io/mkdocs-material/setup/)
- [Admonitions](https://squidfunk.github.io/mkdocs-material/reference/admonitions/#supported-types)

## Making sure your code works

### Linting and Type Checking

Before even running the tests, its a good idea to lint and check the types of the code using pre-commit:

!!! example Run pre-commit
    ```bash
    pre-commit run --all-files
    ```

Your code **must** pass the pre-commit tests as a part of continuous integration, so you might as well fix anything now if it arises.

#### Common Ruff Error Codes

Network Wrangler uses [Ruff](https://docs.astral.sh/ruff/) for linting. Here are some common error codes you may encounter:

**Complexity Issues:**
- [`PLR0912`](https://docs.astral.sh/ruff/rules/too-many-branches/): Too many branches (>12) - Consider refactoring the function or add `# noqa: PLR0912` if complexity is necessary
- [`PLR0915`](https://docs.astral.sh/ruff/rules/too-many-statements/): Too many statements (>50) - Break down into smaller functions or add `# noqa: PLR0915` if justified

**Code Style:**
- [`PLR2004`](https://docs.astral.sh/ruff/rules/magic-value-comparison/): Magic value used in comparison - Define constants instead of hardcoding values
- [`PLC0415`](https://docs.astral.sh/ruff/rules/import-outside-toplevel/): Import should be at top-level - Move imports to file top (use `# noqa: PLC0415` for intentional lazy imports)
- [`SIM102`](https://docs.astral.sh/ruff/rules/collapsible-if/): Use a single if statement - Combine nested if statements with `and`
- [`SIM108`](https://docs.astral.sh/ruff/rules/if-else-block-instead-of-if-exp/): Use ternary operator - Replace if-else blocks with ternary for simple assignments
- [`RUF005`](https://docs.astral.sh/ruff/rules/collection-literal-concatenation/): Consider unpacking instead of concatenation - Use `[*list1, item]` instead of `list1 + [item]`

**Exception Handling:**
- [`EM101`](https://docs.astral.sh/ruff/rules/raw-string-in-exception/): Exception must not use string literal - Assign message to variable first
- [`EM102`](https://docs.astral.sh/ruff/rules/f-string-in-exception/): Exception must not use f-string literal - Assign f-string to variable first

**Function Arguments:**
- [`ARG001`](https://docs.astral.sh/ruff/rules/unused-function-argument/): Unused function argument - Prefix with `_` if intentionally unused
- [`B007`](https://docs.astral.sh/ruff/rules/unused-loop-control-variable/): Loop control variable not used - Prefix with `_` (e.g., `for _idx, item in ...`)
- [`B023`](https://docs.astral.sh/ruff/rules/function-uses-loop-variable/): Function uses loop variable - Use default arguments to capture values (e.g., `lambda x, var=loop_var: ...`)

**Other Common Issues:**
- [`C403`](https://docs.astral.sh/ruff/rules/unnecessary-list-comprehension-set/): Unnecessary list comprehension - Use set comprehension directly `{item for ...}` instead of `set([item for ...])`
- [`C414`](https://docs.astral.sh/ruff/rules/unnecessary-double-cast-or-process/): Unnecessary `list()` call within `sorted()` - Remove redundant conversion
- [`PTH110`](https://docs.astral.sh/ruff/rules/os-path-exists/): Use `Path.exists()` instead of `os.path.exists()`

For a complete list of rules, see the [Ruff Rules Documentation](https://docs.astral.sh/ruff/rules/).

### Adding Tests

..to come

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
