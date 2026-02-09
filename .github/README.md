# GitHub Actions and Workflows

This directory contains the CI/CD configuration for **network_wrangler**: workflows that run on pull requests and pushes, plus reusable composite actions used by those workflows.

---

## Workflows

Workflows live in [`.github/workflows/`](workflows/). They run on different triggers and coordinate linting, testing, docs, benchmarks, and releases.

| Workflow | Trigger | Lint | Format | Tests | Benchmark | Docs | Coverage |
|----------|---------|:----:|:------:|:-----:|:---------:|:----:|:--------:|
| **PR Checks** (`pullrequest.yml`) | `pull_request` (opened, synchronize, reopened) | ✓ | ✓ (fix) | ✓ | ✓ | ✓ | ✓ |
| **CI** (`push.yml`) | `push` to `main` or `develop` | ✓ | ✓ (check) | ✓ | ✓ | ✓ | — |
| **Prepare Release** (`prepare-release.yml`) | Release **created** or manual | — | — | — | — | — | — |
| **Publish Release** (`publish.yml`) | Release **published** or manual | — | — | — | — | ✓ | — |
| **Clean Documentation** (`clean-docs.yml`) | Branch/tag **deleted** or PR **closed** | — | — | — | — | ✓ (delete) | — |

- **Lint**: `ruff check` (PR: auto-fix and commit; Push: check only).  
- **Format**: `ruff format` (PR: apply fixes; Push: check only).  
- **Tests**: pytest on Python 3.10–3.13
- **Docs**: build and deploy to GitHub Pages (PR/Push/Release); **Clean** removes a version when a branch is deleted or PR closed.  
- **Benchmark**: run and compare benchmarks.
- **Coverage**: post coverage comment on PR (when base is `main`/`develop`).

### Workflow details

- **PR Checks**  
    - Lint job can auto-commit formatting/lint fixes to the PR branch (with `[skip ci]`).  
    - Tests run in a matrix (3.10–3.13); only 3.13 produces coverage and benchmark artifacts.  
    - Benchmark and coverage jobs run on Python 3.13 and only when the PR base is `main` or `develop`.  
    - Docs are built per-PR branch; a comment with the docs URL is posted when the PR is opened.

- **Push (main/develop)**  
    - Same test matrix and artifact strategy.
    - Benchmark comparison is only on Python 3.13 and against the previous commit on the branch.  
    - Docs are deployed for the pushed branch name.

- **Releases**  
    - **Prepare**: runs on release *created* (or manual); ensures version matches tag, publishes to TestPyPI, and verifies install.  
    - **Publish**: runs on release *published* as a pre-release or release (or manual); publishes to PyPI and then deploys release docs.

- **Clean docs**  
    - Uses `get-branch-name` to resolve the branch/tag from the event, then deletes that version from the docs site (skips `main` and `develop`).

---

## Reusable Actions

Reusable actions live in [`.github/actions/`](actions/). Workflows call them with `uses: ./.github/actions/<name>`.

| Action | Purpose |
|--------|---------|
| **setup-python-uv** | Sets up the requested Python version, installs [uv](https://github.com/astral-sh/uv), and caches UV packages (keyed by `pyproject.toml`). Used by lint, test, docs, and benchmark jobs. |
| **get-branch-name** | Outputs a normalized branch (or tag) name from the GitHub event (`push`, `pull_request`, `delete`, etc.). Used by docs and clean-docs workflows. |
| **build-docs** | Installs deps with `.[docs]`, runs `mike deploy` for the given branch name, and updates the `latest` alias when the branch is `main`. |
| **compare-benchmarks** | Compares `benchmark.json` either to the previous commit (`push`) or to the base branch (`pr`). Commits `benchmark.json` to the branch and, for PRs, posts a comment with the comparison (and regression warning if applicable). |
| **post-coverage** | Downloads the `coverage-py3.13` artifact, normalizes paths into a `coverage/` directory, and uses `MishaKav/pytest-coverage-comment` to post a coverage comment on the PR. |

---

## Other contents

- **Issue templates** ([`ISSUE_TEMPLATE/`](ISSUE_TEMPLATE/)) – Templates for bugs, features, docs, performance, and chores.  
- **Pull request template** ([`pull_request_template.md`](pull_request_template.md)) – Default body for new pull requests.
