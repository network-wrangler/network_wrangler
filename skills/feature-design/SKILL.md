---
name: feature-design
description: Use whenever adding new functionality, extending public APIs, adding project card change types, adding config options, changing DataFrame schemas, or making choices that affect how users or downstream repos (projectcard, cube_wrangler, met_council_wrangler) interact with network_wrangler. Use this before writing code for any new feature, even small ones — includes "how should I add X", "where does X belong", "new change type". Consults USERS.md personas and checks ecosystem impact before committing to an approach.
---

# Feature Design: Users First, Ecosystem Aware

Network Wrangler is a library with multiple user personas and a downstream ecosystem. Design decisions ripple outward. Before writing code, think about who's affected and how.

## Required Steps

### 1. Read USERS.md

Which persona(s) is this feature for?

- **Transportation planners** — YAML project cards, CLI, error messages matter most
- **Model developers** — Python API, performance, predictability matter most
- **Tool developers** — stable public API, pandera schemas, extension points
- **CI/automation** — deterministic, machine-readable, good exit codes

A feature that helps one persona can harm another. Surface the tradeoff explicitly.

### 2. Check Ecosystem Impact

Will this change affect downstream repos? Check before committing to an approach:

- **`projectcard`**: changes to project card schema, change types, or validation
- **`cube_wrangler`**: changes to scoped-property logic, `prop_for_scope()`, public `roadway.links` / `transit.feed` APIs, or DataFrame schemas
- **`met_council_wrangler`**: changes to `RoadwayNetwork` / `TransitNetwork` public interface, `WranglerConfig`, or project application behavior

If affected: note it in the PR description, and ideally draft a companion PR in the downstream repo.

### 3. Prefer Minimal Changes

In order of preference:

1. **Extend** existing abstractions with new optional behavior
2. **Compose** — add a new module that uses existing APIs
3. **Parameterize** — add a `WranglerConfig` option with a sensible default
4. **Refactor** existing abstractions (last resort — this is where ecosystem breakage happens)

Red flag: if your design requires changing a Pandera schema, a public function signature, or a `ProjectCard` change type, the blast radius is large. Consider whether the same goal can be met with a narrower change.

### 4. Consider Defaults

Defaults should serve the **common case** (single region, standard GTFS, reasonable performance settings). Uncommon needs should be reachable via config, not the default.

- Does the default behavior match what a transportation planner would expect?
- If the feature introduces a choice, what default minimizes surprise for existing users?
- Can the default change be a `WranglerConfig` option with backwards compatibility?

### 5. Design the Error Path

Before writing the happy path, think about failures:

- What can go wrong? (Bad input, missing data, schema mismatch, FK violation)
- Where is the right place to detect it? (IO boundary? Project application?)
- What's the error message? Include: what was expected, what was found, which record/file/field

See `USERS.md` UX Principles — "Error messages should be actionable and reference the specific project card, data file, or field."

### 6. Choose the Right Extension Point

Network Wrangler has established extension points. Use them instead of creating new patterns:

- **New project card change type**: add to `models/projects/`, add handler in `roadway/projects/` or `transit/projects/`, update `apply()` dispatch
- **New IO format**: extend `roadway/io.py` or `transit/io.py` — format auto-detection is already there
- **New config option**: extend `WranglerConfig` in `configs/wrangler.py`
- **New validation**: add to the appropriate pandera schema in `models/`
- **New selection type**: extend `RoadwaySelection` or `TransitSelection`

If your feature doesn't fit an existing pattern, that's a signal — either you're introducing a genuinely new concept (document it in `docs/design.md`) or you're solving the wrong problem.

### 7. Document the Design

Before implementation is complete:

- Add or update the relevant `docs/` page
- Update the relevant mermaid diagram in `docs/design.md` (see `CONTRIBUTING.md` diagram convention)
- If this is a new project card change type, document in `docs/networks.md` or `docs/how_to.md`
- API changes appear automatically via mkdocstrings — but the design narrative doesn't

## Project-Specific Considerations

- **Scoped properties** are a core abstraction — extensions should use scoping rather than inventing parallel mechanisms
- **Pandera schemas** are the data contract — schema changes are breaking changes for tool developers
- **`ProjectCard` comes from `projectcard`** — never parse cards yourself; changes to card semantics go there first
- **`WranglerConfig` is the right place** for tunable behavior — don't add ad-hoc globals or environment variables
- **Lazy evaluation** is a pattern throughout (`shapes_df`, `model_net`, selections) — new derived objects should usually follow suit

## Anti-Patterns

- Adding a feature "just for Met Council" — region-specific logic belongs in `met_council_wrangler`
- Adding Cube-specific output formats — belongs in `cube_wrangler`
- Hardcoding time periods, vehicle categories, or area types — these are parameters, not constants
- Creating parallel APIs (a new `load_roadway_v2()` alongside `load_roadway()`) — evolve the existing one with deprecation

## When You're Unsure

- Sketch the API surface (function signatures, project card YAML example, config keys) *before* implementing
- Share the sketch with the user for feedback — cheap to revise at this stage
- Check if a similar feature already exists that you could extend
- Ask: "If I were a tool developer building on this, would I be surprised by this design?"
