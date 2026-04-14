# AGENTS.md

Context for AI coding assistants working on network_wrangler.

## Quick Start

```bash
pip install -e ".[tests]"
pre-commit install
pytest                           # run all tests
pre-commit run --all-files       # lint + format + type check
mkdocs serve                     # local docs
```

## Key Docs (read these, don't duplicate)

- **Architecture & design**: `docs/design.md`
- **Data models & schemas**: `docs/data_models.md`
- **Coding conventions**: `docs/conventions.md`
- **Setup & workflow**: `CONTRIBUTING.md`
- **Users & UX context**: `USERS.md`
- **Network data model**: `docs/networks.md`
- **Usage guide**: `docs/how_to.md`

## Ecosystem

Four repos under [network-wrangler](https://github.com/network-wrangler) org:

```
Cube Files → cube_wrangler → ProjectCards (.yml)
                                    ↓
                           network_wrangler applies cards
                                    ↓
                 met_council_wrangler transforms network
                                    ↓
                           Cube-ready model network
```

| Repo | Role |
|------|------|
| [`projectcard`](https://github.com/network-wrangler/projectcard) | ProjectCard schema, validation, read/write |
| `network_wrangler` (this repo) | Load networks, apply project cards, write output |
| [`cube_wrangler`](https://github.com/network-wrangler/cube_wrangler) | Diff Cube model files, emit ProjectCards |
| [`met_council_wrangler`](https://github.com/Metropolitan-Council/met_council_wrangler) | Region-specific transformations for Met Council |

**Downstream impact**: changes to scoped-property logic, project handlers, or `ProjectCard`/`SubProject` interfaces affect `cube_wrangler` and `met_council_wrangler`. Check before changing public APIs.

## Conventions & Testing

See `docs/conventions.md` for full details on: coding style, validation strategy, pandas/pandera/geopandas patterns, scoped properties, testing, commits, and diagram maintenance.

## Task-Specific Skills

Behavioral guidance for common task types lives in `skills/` (Anthropic Agent Skill format, usable by Claude Code, Codex, and other skill-aware agents):

- `skills/debugging-root-cause/` — root-cause discipline for bugs, test failures, pandera/FK errors
- `skills/performance-work/` — measure-before-optimize workflow for regional-scale networks
- `skills/feature-design/` — user-persona + ecosystem-impact checks before adding features

Agents without skill auto-invocation can read these files directly when the task matches.
