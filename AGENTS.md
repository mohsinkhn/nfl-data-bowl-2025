# Repository Guidelines

## Project Structure & Module Organization
- `src/datasets.py` handles ingest and feature engineering for Next Gen Stats tracking frames; extend functions here to keep loaders reusable.
- `src/models.py` defines the PyTorch attention blocks and spatio-temporal modules used for trajectory prediction; add new architectures alongside clear docstrings.
- `data/` stores raw competition assets. Treat as read-only and avoid committing large new drops—prefer derived tables under `data/processed/`.
- Top-level notebooks (`0_eda.ipynb`, `base_model.ipynb`) capture exploratory analysis and baselines; clone them when starting a new study to preserve history.
- `dashboard_prompt.txt` tracks dashboard requirements; keep it aligned with changes pushed to any visualization or product repo.

## Build, Test, and Development Commands
- Install dependencies with `uv sync --all-extras` (recommended) or `python -m pip install -e .[dev]` inside a Python 3.12 environment.
- Format and lint with `uv run black src tests` and `uv run flake8 src tests`. Replace `uv run` with your launcher if using another tool.
- Run the suite via `uv run pytest` or `uv run pytest --cov` for coverage; this honors `pyproject.toml` settings.

## Coding Style & Naming Conventions
- Code is formatted with `black` (88-char lines); keep imports sorted logically and prefer double quotes.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and upper-case constants for static hyperparameters.
- Name tensors and DataFrames by semantic role (`player_states`, `attn_weights`) and include brief shape comments when dimensions are non-obvious.

## Testing Guidelines
- Place tests under `tests/` with filenames `test_*.py`; structure them to mirror modules in `src/`.
- Reuse fixtures via `tests/conftest.py` and keep sample plays lightweight (e.g., 1-2 drives) to maintain fast CI.
- Aim for ≥80% statement coverage on `src/`. Use `uv run pytest --cov=src --cov-report=term-missing` before submitting a PR.

## Commit & Pull Request Guidelines
- Follow the existing short, imperative commit style (`set up`, `naseline conv`). Keep subject lines ≤72 chars and expand details in the body when necessary.
- Reference related issues (`Refs #12`), note dataset revisions, and record key metrics or charts.
- Pull requests should summarize motivation, list functional changes, attach metric tables or notebook links, and call out data or config migrations.

## Data & Security Notes
- Store credentials and API keys in a local `.env`; load them via `dotenv` or `os.environ` rather than committing secrets.
- When publishing notebooks, clear outputs or replace sensitive cells with stubs before pushing.
