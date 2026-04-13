# Users & UX Context

This file helps contributors and AI coding assistants understand who uses network_wrangler and in what context. Use this when making design decisions about APIs, error messages, defaults, and feature direction.

## Personas

### 1. Transportation Planners (primary end users)

- Non-programmers or light scripters who run scenarios via config files and project cards
- Interact mostly through YAML project cards and CLI tools (`network_wrangler/bin/`)
- Need clear, actionable error messages and forgiving input handling
- Tooling context: typically use a travel demand model GUI (e.g., Cube) with network_wrangler as a preprocessing step

### 2. Model Developers (power users)

- Python-proficient staff at agencies (Met Council, MTC, etc.) building/extending travel demand models
- Call the Python API directly: `load_roadway()`, `create_scenario()`, `scenario.apply_all_projects()`
- Care about performance, type safety, and predictable behavior
- Tooling context: Jupyter notebooks, Python scripts, CI pipelines

### 3. Tool Developers (ecosystem contributors)

- Build on top of network_wrangler (cube_wrangler, met_council_wrangler, regional forks)
- Need stable public APIs, clear extension points, and well-documented Pandera schemas
- Tooling context: developing in parallel repos that import network_wrangler

### 4. CI/Automation (non-interactive)

- Scripts and pipelines that run network_wrangler in batch mode for scenario building
- Need deterministic behavior, machine-readable errors, and meaningful exit codes
- Tooling context: GitHub Actions, regional agency build systems

## UX Principles

When making implementation decisions, consider:

- **Error messages** should be actionable and reference the specific project card, data file, or field that caused the issue. Include what was expected vs. what was found.
- **Defaults** should work for the common case (single-region, standard GTFS, reasonable performance settings).
- **Breaking API changes** affect tool developers and CI — require deprecation warnings and a migration path.
- **CLI interface** (`network_wrangler/bin/`) is the primary touchpoint for transportation planners — keep it simple, with good `--help` text.
- **Performance** matters most for model developers working at regional scale (Bay Area ~100k links, Twin Cities ~50k links).
- **Validation errors** from Pandera should surface clearly, not as cryptic stack traces. Wrap them with context about which table/column failed.
- **File format flexibility**: support geojson, parquet, and CSV, but recommend parquet for performance. Auto-detect format when possible.
