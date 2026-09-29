# Bayesian black-box optimisation

This is the **starting point** of an evolving capstone project. The task is to maximise eight unknown scalar-valued functions using input/output evaluations only. Their formulas are not available and must not be inferred from the repository. Inputs have dimension-specific coordinates in `[0, 1)`; submitted coordinates require exactly six decimal places.

| Function | Dimensions | Provided initial observations |
| --- | ---: | ---: |
| 1 | 2 | 10 |
| 2 | 2 | 10 |
| 3 | 3 | 15 |
| 4 | 4 | 30 |
| 5 | 4 | 20 |
| 6 | 5 | 20 |
| 7 | 6 | 30 |
| 8 | 8 | 40 |

## Current baseline

The `src/bayesian_bbo` package loads and validates supplied observations, fits a Gaussian Process surrogate, and proposes unused six-decimal candidates from `[0, 1)` using Expected Improvement (default) or Upper Confidence Bound. The default evaluation budget is 20 **new** queries per function. Each new real observation can be recorded separately from the original data; the next proposal refits on all observations. `Observations` retains the complete input/output history and exposes the best measured input/output (maximisation). The candidate search is a random pool, not an assertion of global optimality. No evaluation service, function formulas, results, or conclusions are supplied here.

## Install and explore

Use Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
jupyter notebook
```

See [data/README.md](data/README.md) for the required file layout. Initial files are **not** included in this repository. After placing your provided files, open `notebooks/01_data_exploration.ipynb`, then `02_function_1_baseline.ipynb`. Notebook `03_other_functions.ipynb` provides a configurable starting point for later functions. The notebooks do not execute objective evaluations automatically.

The same workflow is available in Python:

```python
from pathlib import Path
from bayesian_bbo import (
    BayesianOptimizer, OptimisationConfig, append_observation,
    format_submission, load_observations, remaining_budget,
)

root = Path("data")
observations = load_observations(root, 1)
optimizer = BayesianOptimizer(
    observations, OptimisationConfig(budget=remaining_budget(observations, 1), seed=42)
)
if optimizer.remaining:
    point = optimizer.suggest()
    print(format_submission(point))  # Submit this candidate to the real evaluator.
# Once the actual scalar result is known:
# value = ...  # Replace with the real observed output; never invent an evaluation.
# append_observation(root, 1, point, value)
# optimizer.observe(point, value)
# Repeat suggest/evaluate/append/observe until the budget is exhausted.
```

If you restart, `load_observations` includes previously saved evaluations; `remaining_budget` subtracts these from the configurable total (20 by default). Only record confirmed outputs. `run(evaluate)` is available when a genuine callable evaluator exists; it does not automatically persist results.

## Project evolution

`notebooks/` holds exploratory work, `results/` is reserved for measured outputs and convergence artefacts, and `reports/` contains explicitly unfinished templates for the data datasheet, model card, and final summary. Future updates will compare surrogate and acquisition choices, add new observations after submissions, document decisions and failures, and report actual results, plots, limitations, and conclusions when available. Neither the current baseline nor any strategy is claimed to be optimal.
