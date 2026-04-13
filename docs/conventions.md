# Coding Conventions

This page documents coding patterns and conventions specific to network_wrangler. For general setup and workflow, see [Contributing](development.md).

## Python & Style

- **Python 3.10+** with `from __future__ import annotations` for forward references
- **Line length**: 99 characters (enforced by ruff)
- **Docstrings**: Google-style (enforced by ruff `pydocstyle` convention)
- **Type hints**: use throughout; PEP 604 union syntax (`X | None` not `Optional[X]`)
- **Linting**: ruff (replaces flake8, black, isort)
- **Type checking**: mypy (target Python 3.10, skip missing imports)
- **Formatting**: `ruff format` with docstring code formatting enabled

## Imports

- Top-level imports preferred
- **Lazy imports** allowed where needed to avoid circular dependencies (`PLC0415` is suppressed in ruff)
- Common circular-dep patterns: `roadway.network` ↔ `roadway.model_roadway`, `transit.network` ↔ `roadway.network`

## Pandas Patterns

- **String dtype**: `pandas.options.future.infer_string` is set to `False` in `__init__.py` for pandera compatibility. Be aware when working with string columns — they use `object` dtype, not `StringDtype`.
- **CI matrix**: code must work on both pandas 2.x and 3.x (Python 3.10 + pandas 3 is excluded)
- **Prefer vectorized operations** over `.apply()` or row iteration
- **Use `.loc[]` for assignment** to avoid `SettingWithCopyWarning`
- **Categorical columns**: use for low-cardinality columns (e.g., mode, facility type) when performance matters
- **Avoid chained indexing**: use `.loc[rows, cols]` not `df[col][rows]`

## Validation Strategy

Pandera validation and foreign key checks are expensive. Where and when to validate is a deliberate design choice driven by performance:

### Pandera Schema Validation

All DataFrame schemas are Pandera `DataFrameModel` subclasses in `network_wrangler/models/`:

- `models/roadway/tables.py`: `RoadLinksTable`, `RoadNodesTable`, `RoadShapesTable`
- `models/gtfs/tables.py`: GTFS feed table schemas (`WranglerStopsTable`, `WranglerStopTimesTable`, etc.)
- `models/projects/`: schemas for project card change types

**When to validate:**

- **Reading from disk**: full schema validation on load (this is where bad data enters the system)
- **Writing to disk**: full schema validation before write (guarantee output correctness)
- **After project application**: validate only the columns/tables that the project actually touched — not the entire network
- **Between internal functions**: do **not** run full pandera validation. Trust that data was validated at the boundary. If a specific internal function needs a precondition, use a lightweight assertion or check on just the relevant column, not a full `DataFrameModel.validate()` call.

**Why**: Full pandera validation on a Bay Area-scale network (~100k links) can take seconds per call. Validating inside tight loops or between every internal function call can turn a sub-second edit into a multi-minute operation.

**Patterns:**

- Annotate DataFrames as `DataFrame[SchemaModel]` in function signatures for documentation, but don't rely on pandera's implicit validation in internal code paths
- Schema coercion is enabled — pandera will attempt to cast types before failing
- See `docs/data_models.md` for full schema reference

### Foreign Key Consistency

Foreign key relationships (e.g., link A_node/B_node referencing node model_node_id, or GTFS stop_times.trip_id referencing trips.trip_id) follow a similar boundary pattern:

- **Guaranteed at IO boundaries**: foreign keys are validated when loading from disk and before writing to disk
- **Guaranteed before and after each project application**: a project may temporarily break FK relationships during intermediate steps (e.g., adding links before adding their nodes), but FKs must be consistent when the project handler returns
- **Not guaranteed within internal editing functions**: intermediate steps of a project handler may have dangling references. This is expected and acceptable — do not add FK checks inside handler helper functions.

**Why**: FK validation requires joins across tables. Doing this after every small edit (add a link, delete a node, update a shape reference) multiplies the cost of project application.

## GeoDataFrame / GeoPandas

- Roadway links and nodes are stored as `GeoDataFrame` with geometry columns
- Shapes are lazily loaded (`RoadwayNetwork.shapes_df` property)
- CRS handling: networks store their CRS; transformations should preserve or explicitly convert
- Use `gpd.GeoDataFrame` type hints, not plain `pd.DataFrame`, when geometry is required

## Pydantic Models

- `RoadwayNetwork` is a Pydantic `BaseModel` — use `model_config` for Pydantic settings, not class-level `Config`
- `WranglerConfig` is Pydantic-based — override via YAML/TOML/JSON files
- `ProjectCard` comes from the `projectcard` package — never reimplement card parsing

## Scoped Properties

A key domain concept. Link properties can be **scoped** by `category` (e.g., `sov`, `hov2`) and `timespan` (e.g., `["6:00", "9:00"]`):

- Scoped values stored as lists of `ScopedLinkValueItem` on link records
- Resolved at query time via `roadway/links/scopes.py::prop_for_scope()`
- `ModelRoadwayNetwork` explodes scoped properties into flat columns: `{prop}_{timeperiod}_{category}` (e.g., `lanes_AM_sov`)
- See `docs/design.md` for the resolution flowchart

## Error Handling

- Use custom exceptions from `network_wrangler/errors.py`
- Error messages should be actionable — reference the specific project card, file, or field
- Pandera validation errors should be wrapped with context (which table, which column)
- See `USERS.md` for UX principles around error messages

## Testing

- **Framework**: pytest
- **Markers**: `@pytest.mark.skipci` (skip in CI), `@pytest.mark.failing` (known failures)
- **Fixtures**: session-scoped in `tests/conftest.py` for expensive setup (loading networks); module-level `conftest.py` in subdirectories
- **Benchmarks**: `test_benchmarks.py` tracked via `pytest-benchmark`, compared across PRs
- **Profiling**: `pytest-profiling` stores profiles in `.prof/`; explore with `snakeviz`
- **Coverage**: only measured on Python 3.13 + pandas 3 in CI

## Commits

Use conventional commit format:

- `fix:` — bug fix
- `feat:` — new feature
- `chore:` — maintenance, dependencies, CI
- `docs:` — documentation only
- `style:` — formatting, linting (no logic change)
- `perf:` — performance improvement
- `refactor:` — code change that neither fixes a bug nor adds a feature
- `test:` — adding or updating tests

## Diagram Maintenance

When modifying code that changes module relationships, class hierarchies, or data flow, update the corresponding mermaid diagram in `docs/design.md` in the same PR. Current diagrams:

- Ecosystem data flow
- Core object model (class diagram)
- Roadway selection & search (class diagram)
- Project application flow (sequence diagram)
- Scoped property resolution (flowchart)
- IO pipeline (flowchart)
