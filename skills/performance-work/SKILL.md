---
name: performance-work
description: Use whenever performance, speed, memory, or validation overhead comes up in network_wrangler — optimizing code, profiling, benchmarking, "this is slow", "can we speed this up", reducing memory, or evaluating whether pandera validation is too expensive. Use this even when the fix seems obvious. Requires measurement before and after — no "this should be faster" claims. Applies to regional-scale networks (50k-100k links) where intuition about hotspots is often wrong.
---

# Performance Work: Measure, Don't Guess

Performance claims without measurement are hypotheses, not improvements. This project runs on regional-scale networks (Twin Cities ~50k links, Bay Area ~100k links) where assumptions about what's slow are often wrong.

## Required Steps

### 1. Establish a Baseline

Before changing anything:

- Run the relevant benchmark in `tests/test_benchmarks.py` or write one
- Save the baseline: `pytest --benchmark-save=before`
- Note the dataset size (link/node/trip counts) — performance is not linear

If there's no existing benchmark for what you're optimizing, **add one first**. Optimizations without benchmarks regress silently.

### 2. Profile to Find the Actual Hotspot

Assumed hotspots are usually wrong. Use tools:

- **`pytest-profiling`** stores profiles in `.prof/` when tests run
- **`snakeviz .prof/<test_name>.prof`** visualizes the call tree
- **`cProfile`** for targeted profiling of a function

Look for:

- Unexpected time in pandera validation (see validation strategy below)
- `apply()` / row iteration — almost always replaceable with vectorized ops
- Repeated `network_hash` computation (should be cached)
- `GeoDataFrame` operations that materialize geometry unnecessarily
- Pandas operations that trigger `object` dtype fallback

### 3. Respect the Validation Strategy

Pandera validation is **expensive** — full validation of `RoadLinksTable` on a Bay Area network can take seconds. See `docs/conventions.md` for the full strategy. Summary:

- **Validate at IO boundaries and at project boundaries** — not between internal functions
- If profiling shows pandera hotspots inside tight loops or internal helpers, that's usually the bug
- Foreign key validation is similar — guaranteed at IO and project boundaries, not intermediate steps
- If you need a lightweight precondition check, assert on one column, don't call `SchemaModel.validate()`

### 4. Optimize at the Right Level

Order of impact (usually):

1. **Algorithm** — O(n²) → O(n) beats any micro-optimization
2. **Avoiding work** — caching, lazy evaluation, skipping validated data
3. **Data structure** — categorical dtypes, sparse representations, appropriate indexes
4. **Vectorization** — replacing `.apply()` with vectorized ops
5. **Implementation** — faster library functions, avoiding copies

Don't start at step 5.

### 5. Measure Again

After the change:

- Run the same benchmark: `pytest --benchmark-save=after`
- Compare: `pytest-benchmark compare before after`
- Confirm the improvement on realistic data (not just tiny test fixtures)
- Run the full test suite to verify no regressions

### 6. Document the Measurement

In the PR:

- Include before/after numbers with dataset size
- Name the bottleneck you addressed (e.g., "pandera validation called per-link in edit loop")
- Note any tradeoffs (memory, code complexity, validation coverage)

Use `perf:` as the commit prefix.

## Project-Specific Patterns

- **Scoped property access**: `prop_for_scope()` can be called many times per project application. If profiling shows it hot, consider batch resolution rather than per-link calls.
- **`ModelRoadwayNetwork`**: lazily created and cached on `RoadwayNetwork.model_net`. Recomputing it is expensive — don't invalidate the cache unless the underlying network actually changed.
- **Network hash**: `RoadwayNetwork.network_hash` is used for cache invalidation. Recomputing from scratch is O(links+nodes); excessive calls mean something is invalidating too eagerly.
- **GeoDataFrame materialization**: operations that compute geometry on demand (e.g., `link_shapes_df`) can be costly. Cache when reused.
- **GTFS table joins**: `stop_times` is the largest GTFS table. Operations that join it repeatedly are performance-critical; profile carefully.

## Anti-Patterns to Watch For

- "This feels slow, I'll add caching" — measure first, then cache the actual hotspot
- "Let me disable validation here to speed it up" — validation costs are a *symptom*; the real fix is validating at the right boundary (see `docs/conventions.md`)
- "I'll parallelize this" — usually a last resort after algorithm/vectorization wins. Adds complexity and debugging difficulty.
- Optimizing code paths that aren't in the measured hotspot

## When You're Unsure

- Ask: "What's the expected time complexity of this operation? Does the measured time match?"
- Share benchmark numbers with the user rather than describing performance qualitatively
- Escalate tradeoffs (e.g., more memory for less CPU) rather than deciding silently
