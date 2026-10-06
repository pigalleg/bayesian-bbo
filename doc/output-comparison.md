# Expected and observed outputs

The Week 2 notebook, [`notebooks/week_2/02_output_comparison.ipynb`](../notebooks/week_2/02_output_comparison.ipynb), compares predictions recorded for evaluation inputs with the subsequently observed outputs. It reads cumulative `all_observations.npz` files through `load_observations`; weekly JSON files remain the tracking records used to import those observations.

## Row selection and alignment

For each function with an aggregate file, the notebook selects rows whose `expected_outputs` value is finite. Each selected row retains its function ID, input, expected output, observed output, and posterior standard deviation. The data loader keeps these arrays aligned by row, so each prediction is compared with the actual output for the same input.

Initial observations without predictions have missing expected outputs and are excluded. If no function has a matched prediction, the notebook stops with an error rather than producing an empty comparison.

## Metrics

The signed error is:

`error = observed_output - expected_output`

A positive error means the prediction was too low (underprediction); a negative error means it was too high (overprediction). `absolute_error` is the magnitude of this difference.

The signed z-score scales error by the prediction's standard deviation:

`z_score = error / std`

Positive and negative z-scores retain the error direction; `absolute_z_score` reports its magnitude in standard-deviation units. A z-score is missing when `std` is missing, non-finite, or zero. Consequently, raw errors and z-scores are not interchangeable: raw errors use each function's output units, while z-scores are uncertainty-scaled.

## Summary and plots

The notebook groups rows by function and reports the matched observation count, mean expected and observed output, mean signed error, largest signed error, mean standard deviation, count of valid z-scores, mean z-score, and largest absolute z-score. `max_error` is the maximum signed error, not the maximum absolute error.

The expected-versus-observed bar plots use a separate y-axis per function because output scales may differ. Bars are displayed in matched-row order, not sorted by error or input value.

## Scope and limitations

- The notebook includes only functions with an existing `all_observations.npz` file and only rows with finite expected outputs.
- A missing or zero standard deviation prevents uncertainty-normalized scoring but does not remove the row from raw-error summaries.
- Mean and maximum signed errors can hide large errors in the opposite direction; inspect absolute errors or absolute z-scores when magnitude is the concern.
- The plot creates a fixed grid of eight axes. If matched data includes all nine functions defined by the package, the current loop plots only the first eight function IDs. Increase the grid size or compute it from the number of functions before relying on the plot for all nine.
- The charts and summaries compare predictions with later observations; they do not establish global optimum quality or prove that one optimisation strategy is superior.

The notebook implementation is the source of truth for these calculations. Data storage and weekly import rules are documented in [data placement and provenance](../data/README.md).