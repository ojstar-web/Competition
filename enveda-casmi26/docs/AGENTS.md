# Repository Guidelines

## Project Structure & Module Organization

`src/` contains the Python pipeline: `baseline.py` is the original spectral-library baseline, phase-specific scripts implement later experiments, and `kaggle_submit.py` builds test-only submissions. `tests/test_*.py` holds focused pytest checks; `tests/verify_run.py` verifies a specific baseline run. Keep official inputs in `data/`, run outputs and receipts in `outputs/`, and external models or reference code in `references/`. Read `README.md` and the relevant `PHASE*.md` or `SUBMISSION.md` before changing a frozen workflow.

## Setup, Run & Test Commands

- `.venv/bin/python -m pip install -r requirements.txt` installs the pinned core dependencies into the project environment.
- `.venv/bin/python -m pytest -q tests` runs the local unit tests.
- `.venv/bin/python src/baseline.py --data data --output outputs/<unique-run-id>` runs the CPU baseline; use a new output directory because the script refuses to overwrite one.

MIST experiments use the separate `.venv_mist_cpu` environment; follow the matching phase document for their exact inputs and commands. There is no repository-wide build step.

## Coding Style & Naming

Use four-space indentation, `snake_case` for Python modules, functions, and variables, and descriptive `test_*.py` names. Keep scientific constants and data contracts explicit. Extend the existing phase entry point instead of adding a parallel runner. No root formatter or linter configuration is present; keep edits consistent with nearby code.

## Testing & Scientific Checks

Add focused pytest cases for changed parsing, ranking, or submission behavior. No coverage threshold is configured. For a run, check its manifest, input and output identities, and CSV schema before reporting success. Preserve the frozen development/sealed split: do not use sealed truth for tuning. A local development score is not a Kaggle score.

## Commits & Pull Requests

This directory has no Git metadata, so an existing commit-message convention cannot be verified. If contributing through Git, use a short imperative subject that names the affected phase or component. In a pull request, describe the changed behavior, inputs, affected run or artifact identities, tests performed, and any change to scientific comparability. Include output samples or screenshots only when they clarify a changed artifact.
