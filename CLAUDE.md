# CLAUDE.md

Read `AGENTS.md` for project context, conventions, ecosystem overview, and doc pointers.
Read `USERS.md` for user personas and UX principles.

## Claude-Specific Instructions

- Use `pre-commit run --all-files` to validate before suggesting code is complete
- When modifying public APIs, check downstream impact on `cube_wrangler` and `met_council_wrangler` (see ecosystem in AGENTS.md)
- Prefer reading existing docs (`docs/design.md`, `docs/conventions.md`, `CONTRIBUTING.md`) over asking the user to re-explain project structure
- When project cards are involved, use `from projectcard import ProjectCard, read_cards` — never reimplement card parsing

## Related Repository Paths (Local)

| Repo | Path |
|------|------|
| `projectcard` | `/Users/elizabethsall/Documents/GitHub/projectcard` |
| `cube_wrangler` | `/Users/elizabethsall/Documents/GitHub/cube_wrangler` |
| `met_council_wrangler` | `/Users/elizabethsall/Documents/GitHub/met_council_wrangler` |
