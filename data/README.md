# Data placement and provenance

Place the supplied arrays in `data/function_1/` through `data/function_9/`, each containing `initial_inputs.npy` and `initial_outputs.npy`. Initial inputs must have shape `(n, d)` for the function's listed dimension; the number of initial observations `n` is flexible. Outputs must have shape `(n,)` or `(n, 1)`. Values must be finite, inputs in `[0, 1)`, and outputs scalar. Do not modify these originals.

`all_observations.npz` in each `data/function_N/` directory is the runtime source of truth. It contains the complete cumulative `inputs`, `outputs`, `expected_outputs`, and `stds` arrays for that function. `load_observations(Path("data"), function_id)` reads this file; before the first weekly import it can fall back to the supplied initial `.npy` files. Initial rows have `NaN` expected outputs and standard deviations until predictions are available. Older aggregates without `stds` load with missing (`NaN`) uncertainties.

For each weekly delivery, keep `inputs.json`, `outputs.json`, and `expected_outputs.json` in a numbered folder under `data/weekly_updates/`. Each file contains a JSON array: inputs are nested arrays of coordinates (for example, `[[0.2, 0.3]]`), while outputs and expected outputs are arrays of numbers in matching row order. Week two is already stored under `week_2`; use `week_3` for the next batch. These JSON files are tracking records; import all three together with `append_weekly_batch`. By default, row 1 maps to function 1, row 2 to function 2, and so on. For a subset or different order, pass `function_ids` explicitly. Identical re-imports are skipped; conflicting outputs or expectations are rejected. Function-level `.npz` files are ignored by Git, so back them up or manage them with a data versioning tool.

The expected outputs are the predictions made in the previous week for the current week's inputs, in the same row order. `append_weekly_batch` imports actual inputs, outputs, and expectations together into each function's `all_observations.npz`. If a neighboring `stds.json` exists, it imports those posterior standard deviations too; pass `stds_path=...` to use a different path. The file must contain one finite, nonnegative number per input in the same order. Missing uncertainties for already recorded inputs can be filled by re-importing; conflicting values are rejected. `append_observation(..., std=value)` records a single uncertainty, or leaves it missing when omitted, while preserving previous values. Export NumPy values as ordinary lists and floats with `json.dumps(..., allow_nan=False)`; NumPy `array(...)` expressions are not valid JSON.

For example:

```python
from pathlib import Path
from bayesian_bbo import append_weekly_batch

batch = Path("data/weekly_updates/week_2")
append_weekly_batch(
	Path("data"),
	batch / "inputs.json",
	batch / "outputs.json",
	batch / "expected_outputs.json",
	stds_path=batch / "stds.json",
)
```

Record provenance, evaluation dates, any known constraints, and data limitations in `reports/data-datasheet.md` as actual observations arrive.
