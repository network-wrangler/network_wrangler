---
name: debugging-root-cause
description: Use whenever something is broken, failing, or behaving unexpectedly in network_wrangler — bugs, test failures, pandera/FK validation errors, "why is this not working", flaky tests, or surprising output. Use this even for seemingly simple fixes. Finds the actual root cause before proposing changes and prevents band-aid patches, suppressed exceptions, try/except-and-return-default fixes, or tests rewritten to match buggy behavior.
---

# Debugging: Root Cause First

When something is broken, the goal is to **understand why** before fixing what. Band-aid fixes compound over time and create harder bugs later.

## Required Steps

### 1. Reproduce

You cannot fix what you cannot reproduce. Before proposing any fix:

- Write a minimal reproduction (smallest test, smallest input, fewest steps)
- Confirm the bug is deterministic; if flaky, investigate the non-determinism *first* — it may be the actual root cause
- If the bug only appears with a specific pandas version, note that; it may reveal a dtype or behavior assumption
- If the bug is in a test, run it in isolation (`pytest tests/path/to/test.py::test_name -v`) to rule out test pollution

### 2. Form a Hypothesis

Don't start changing code until you can state:

- What the code is *actually* doing
- What it *should* be doing
- Why the gap exists (not just where — why)

"I think adding X will fix it" is not a hypothesis. "The scoped property resolver returns the default because the timespan comparison uses string equality instead of interval overlap" is a hypothesis.

### 3. Narrow the Search Space

- Binary search through recent commits (`git bisect`) when the bug is new
- Isolate components: does it fail at IO, during project application, or at validation?
- For pandera errors, inspect the actual DataFrame state just before validation — don't just read the error message
- For FK consistency failures, check whether the violation is expected mid-project (see `docs/conventions.md` on validation boundaries)

### 4. Fix the Cause, Not the Symptom

**Red flags** that you're patching symptoms, not fixing the cause:

- Catching an exception and returning a default
- Adding an `if` check that bypasses the buggy path
- Suppressing a warning without understanding what it indicated
- Adding `pd.option_context` or pandera `validate=False` to "make tests pass"
- Rewriting the test to match the current (buggy) behavior

If you're tempted to do any of these, **stop** and explain in writing why the original path was wrong. The fix should address that explanation.

### 5. Prove the Fix

- Write (or update) a regression test that would have caught the bug
- Run the full test suite — not just the failing test — to verify no new breakage
- For performance bugs, include a benchmark (see `skills/performance-work/SKILL.md`)

### 6. Document the Root Cause

In the PR description or commit message, explain:

- What the actual root cause was (not what you changed)
- Why the previous code was wrong (design assumption that didn't hold, race condition, dtype mismatch, etc.)
- Why the fix addresses that specific cause

This is what distinguishes a `fix:` commit from a `chore:` commit. Future contributors reading `git blame` need the *why*.

## Project-Specific Gotchas

- **pandas 2 vs 3 string dtypes**: the `__init__.py` sets `future.infer_string = False` for pandera compat. Bugs involving string columns may behave differently if this is bypassed.
- **Scoped properties**: if a property returns a wrong value, check `prop_for_scope()` and the list of `ScopedLinkValueItem` on the link, not just the direct column value.
- **Validation boundaries**: pandera/FK failures in internal functions may indicate you're validating where you shouldn't — or that an earlier step left invalid state. See `docs/conventions.md`.
- **Network hash caching**: `RoadwayNetwork` caches `model_net` keyed on `network_hash`. Stale derived objects after modification often mean the hash didn't invalidate.
- **Ecosystem**: a bug in `cube_wrangler` or `met_council_wrangler` tests may actually be a network_wrangler API change. Check if the downstream repo pinned an old version.

## When You're Stuck

- Read the error message all the way through (including stack trace interior, not just the top line)
- Check `docs/design.md` for architectural assumptions that may not match your mental model
- Ask: "What would have to be true for this to happen?" then verify each assumption
- Escalate to the user with specifics: what you've tried, what you've ruled out, what remains unknown
