# Python Environments and Notebook Kernels

- When a notebook needs a kernel or the requested kernel cannot be found, search the repository for an existing `.venv` and use its Python interpreter/kernel when suitable.
- Never create a virtual environment yourself, including with `python -m venv`, `uv venv`, or equivalent commands. If no suitable environment exists, ask the user how they want to proceed.

## Project map

- Python package: `src/bayesian_bbo/`. Reuse package APIs rather than duplicating optimization logic in notebooks.
- `data.py`: observations, weekly imports, prediction persistence via `save_predictions`.
- `surrogate.py`: GP fitting; `acquisition.py`: EI/UCB; `optimisation.py`: `BayesianOptimizer` and `OptimisationConfig`.
- `visualisation.py`: GP heatmaps/slices, calibration and optimum progress; `submission.py`: six-decimal submission formatting.
- `settings.py` loads `config.yaml` (there is no separate settings YAML). `CONFIG` includes candidate count and integer-keyed per-function `length_caps`.
- `notebooks/week_2/` preserves historical hardcoded length caps. Week-3 notebooks read shared caps from `CONFIG`; do not migrate week-2 caps implicitly.

## Analysis context and reusable workflow

- Read [the documentation index](../doc/README.md), [data flow](../doc/data-flow.md), and the relevant weekly workflow before changing analysis or exports.
- [Week 3 workflow](../doc/week-3-workflow.md) records candidate rationale, selected alpha/xi values, notebook order and export semantics. Verify current files rather than relying on previous chat names or numbers.
- Use [the bayesian-notebook-workflow skill](skills/bayesian-notebook-workflow/SKILL.md) for GP diagnostics, alpha/xi sweeps, candidate selection, notebook execution and weekly exports.
- Keep notebooks compact and readable. Put reusable candidate generation/selection in the package; use optimizer APIs instead of notebook-local pools and argmax logic.
- Validate using the existing `.venv`: `.venv/bin/python -m pytest tests/<relevant_test>.py -q`. Distinguish syntax checks, reduced-pool execution and full million-candidate execution in reports.
