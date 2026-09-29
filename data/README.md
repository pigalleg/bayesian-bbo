# Data placement and provenance

Place the supplied arrays in `data/function_1/` through `data/function_8/`, each containing `initial_inputs.npy` and `initial_outputs.npy`. Initial inputs must have shape `(n, d)` for the function's listed dimension and observation count (see the root README); outputs must have shape `(n,)` or `(n, 1)`. Values must be finite, inputs in `[0, 1)`, and outputs scalar. Do not modify these originals.

After a real evaluation, call `append_observation(Path("data"), function_id, point, observed_value)`. This writes/updates `added_observations.npz` in the relevant function directory, holding cumulative `inputs` and `outputs` arrays. Concurrent appends to the same function are serialised with a local lock file (Unix-like systems). `load_observations` combines the original and added arrays in chronological order. Keep a backup of your data externally: datasets and results are ignored by Git by default, and this repository has no provided evaluation data.

Record provenance, evaluation dates, any known constraints, and data limitations in `reports/data-datasheet.md` as actual observations arrive.
