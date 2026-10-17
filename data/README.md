# Data placement and provenance

Place the supplied arrays in `data/function_1/` through `data/function_9/`, each containing `initial_inputs.npy` and `initial_outputs.npy`. Initial inputs must have shape `(n, d)` for the function's listed dimension; the number of initial observations `n` is flexible. Outputs must have shape `(n,)` or `(n, 1)`. Values must be finite, inputs in `[0, 1)`, and outputs scalar. Do not modify these originals.

`all_observations.npz` in each `data/function_N/` directory is the runtime source of truth. It contains the complete cumulative `inputs`, `outputs`, `expected_outputs`, and `stds` arrays for that function. `load_observations(Path("data"), function_id)` reads this file; before the first weekly import it can fall back to the supplied initial `.npy` files. Initial rows have `NaN` expected outputs and standard deviations until predictions are available. Older aggregates without `stds` load with missing (`NaN`) uncertainties.

Each file also has a `weeks` array giving the week each row was recorded in. Week 0 is the initial data; each weekly batch is labelled with its week number. `load_observations(..., up_to_week=N)` returns only rows with week less than or equal to `N`, so analyses can be restricted to the data available at a given week. Without `up_to_week`, all rows are returned. Files without `weeks` raise an error. To add labels to older files, run `scripts/add_week_labels.py`: it is a dry run by default, and `--apply` writes the change and keeps a backup named `all_observations.pre-week-labels.npz` next to each file.

For each weekly delivery, keep `inputs.json`, `outputs.json`, and `predictions.json` in a numbered folder under `data/weekly_updates/`. `inputs.json` and `outputs.json` are JSON arrays: inputs are nested arrays of coordinates (for example, `[[0.2, 0.3]]`), and outputs are an array of numbers in matching row order. `predictions.json` is a JSON object with three equally sized arrays in the same row order: `expected_outputs` (the previous week's GP posterior mean for each input), `stds` (the posterior standard deviation for each input, finite and nonnegative), and `expected_best_outputs` (the expected best-so-far after evaluating the suggested input: the best output observed when the prediction was made plus expected improvement with ξ = 0). Unlike `expected_outputs`, `expected_best_outputs` accounts for retaining the existing best if the candidate's output is worse. It is computed by `expected_best_output` from the saved posterior mean, standard deviation, and best observed output, without refitting. The first evaluated batch is stored under `week_1`; subsequent batches increment that number. These JSON files are tracking records; import all three together with `append_weekly_batch`. The original NumPy-repr text attachments are archived in `data/weekly_updates/text_exports/` as `inputs.txt` and `outputs.txt`; `scripts/convert_weekly_txt.py` converts them into each week's JSON files, with `--first-week` setting the week of the first list (default 1). By default, row 1 maps to function 1, row 2 to function 2, and so on. For a subset or different order, pass `function_ids` explicitly. Identical re-imports are skipped; conflicting outputs or expectations are rejected. Function-level `.npz` files are ignored by Git, so back them up or manage them with a data versioning tool.

Newly generated `predictions.json` files also contain `acquisition_scores` and `acquisition_settings`, with one entry per input in the same row order. These record the score actually used to select the candidate: EI with the configured `xi` and incumbent `best` (`{"name": "ei", "xi": 0.01, "best": 1.0}`), or UCB with its configured `kappa` (`{"name": "ucb", "kappa": 1.96}`). Week 1 records UCB scores from the Week 0 notebook; Week 2 records EI scores using each function's selected `xi` from the Week 1 notebook. These scores are distinct from the zero-`xi` improvement used for `expected_best_outputs`. `suggest_with_prediction()` returns `(point, mean, std, acquisition_score)`. Pass the scores and settings together to `save_predictions`; older files without either field remain readable. Acquisition metadata stays in the weekly JSON and is not imported into `all_observations.npz`. `inputs.json` remains an array of coordinate arrays.

The expected outputs and standard deviations are the predictions made in the previous week for the current week's inputs, in the same row order. `append_weekly_batch` imports actual inputs, outputs, and predictions together into each function's `all_observations.npz`. The week is taken from the batch folder name (`week_N` labels rows as week N); pass `week=...` to override it. Each prediction field must contain one finite value per input, and standard deviations must be nonnegative. Missing uncertainties for already recorded inputs can be filled by re-importing; conflicting values are rejected. `append_observation(..., week=N, std=value)` records a single observation for week `N`; `week` is required when writing to disk. The `std` argument records a single uncertainty, or leaves it missing when omitted, while preserving previous values. Export NumPy values as ordinary lists and floats with `json.dumps(..., allow_nan=False)`; NumPy `array(...)` expressions are not valid JSON.

For example:

```python
from pathlib import Path
from bayesian_bbo import append_weekly_batch

batch = Path("data/weekly_updates/week_1")
append_weekly_batch(
	Path("data"),
	batch / "inputs.json",
	batch / "outputs.json",
	batch / "predictions.json",
)
```

Record provenance, evaluation dates, any known constraints, and data limitations in `reports/data-datasheet.md` as actual observations arrive.
