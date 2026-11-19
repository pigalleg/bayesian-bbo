# Project Rules & Guidelines

## Python Environments and Notebook Kernels

- When a notebook needs a kernel or the requested kernel cannot be found, search the repository for an existing `.venv` and use its Python interpreter/kernel when suitable.
- Never create a virtual environment yourself, including with `python -m venv`, `uv venv`, or equivalent commands. If no suitable environment exists, ask the user how they want to proceed.

## Project Map

- Python package: `src/bayesian_bbo/`. Reuse package APIs rather than duplicating optimization logic in notebooks.
- `data.py`: observations, weekly imports (`append_weekly_batch`, `append_observation`), and prediction persistence via `save_predictions`.
- `history.py`: validated weekly prediction and outcome history (`load_weekly_history`).
- `kernels.py`: shared kernel constructors (`make_matern_32_kernel`, `make_rational_quadratic_kernel`, `make_comparison_kernels`).
- `surrogate.py`: GP fitting (`fit_surrogate`); `acquisition.py`: EI/UCB; `optimisation.py`: `BayesianOptimizer`, `OptimisationConfig`, `find_prediction_regions`.
- `validation.py`: cross-validation and explicit-split kernel evaluation (`evaluate_kernel_splits`, `cross_validate_kernels`).
- `visualisation.py`: GP heatmaps/slices, calibration, prediction planes, and optimum progress; `submission.py`: six-decimal submission formatting.
- `settings.py` loads `config.yaml` (there is no separate settings YAML). `CONFIG` includes candidate count and integer-keyed per-function `length_caps`.
- `notebooks/week_1/` preserves historical hardcoded length caps. Later weekly notebooks read shared caps from `CONFIG`; do not migrate week-1 caps implicitly.

## Analysis Context and Reusable Workflow

- Read [the documentation index](doc/README.md), [data flow](doc/data-flow.md), and the relevant weekly workflow (`doc/week-*-workflow.md`) before changing analysis or exports. Verify current files rather than relying on previous chat names or numbers.
- Use the `bayesian-notebook-workflow` skill ([.agents/skills/bayesian-notebook-workflow/SKILL.md](.agents/skills/bayesian-notebook-workflow/SKILL.md)) for GP diagnostics, alpha/xi sweeps, candidate selection, notebook execution, and weekly exports.
- Keep notebooks compact and readable. Put reusable candidate generation/selection in the package; use optimizer APIs instead of notebook-local pools and argmax logic.
- Validate using the existing `.venv`: `.venv/bin/python -m pytest tests/<relevant_test>.py -q`. Distinguish syntax checks, reduced-pool execution, and full million-candidate execution in reports.
