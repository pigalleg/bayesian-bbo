# Project documentation

- [Benchmark functions](benchmark-functions.md) summarizes the eight functions described in Mini-lesson 12.8, their input dimensions, and the supplied application examples.
- [Week 1 workflow](week-1-workflow.md) lists the Week 1 notebooks in order, the kernel policy, the Matérn 3/2 decision for Notebook 06, and the caveats for next week.
- [Week 2 workflow](week-2-workflow.md) explains candidate choices, notebook order, selected xi values, GP diagnostics and the week-3 export.
- [Week 3 workflow](week-3-workflow.md) records the week-3 progress/calibration analysis, Week 4 candidate policy and exports.
- [Week 4 workflow](week-4-workflow.md) records the week-4 analysis, alpha decisions, kernel choices, and verified Week 5 proposal.
- [Week 5 workflow](week-5-workflow.md) records measured outcomes, notebook order, kernel/bound diagnostics, the Function 2 local GP experiment, and the verified Week 6 proposal and export safeguards.
- [Week 6 workflow](week-6-workflow.md) records measured Week 6 outcomes, cumulative calibration, and the Weeks 1-6 per-function strategy summary.
- [All-function surrogate diagnostics](../notebooks/week_5/02_all_functions_surrogate_diagnostics.ipynb) compares five kernels across all functions using shuffled folds and pre-week rolling-origin backtests.
- [Length-scale cap sensitivity](../notebooks/week_5/03_length_scale_sensitivity.ipynb) plots maximized log marginal likelihood against the length-scale cap for all eight functions and five kernels using observations through Week 5.
- [Length-scale lower-bound sensitivity](../notebooks/week_5/04_length_scale_lower_bound_sensitivity.ipynb) compares length-scale lower bounds from `0.001` down to `0.00001` with maximum length caps and noise bounds held fixed.
- [Data flow](data-flow.md) describes how data moves from the initial arrays to the observation files, and the meaning of each field in the `.npz` and `.json` files.
- [Function 5 kernel diagnostics](../notebooks/week_4/05_function_5_kernel_diagnostics.ipynb) compares paired CV folds, individual held-out predictions, interval widths, historical next-week predictions, and a training-mean baseline. It is read-only and uses the week-4 cutoff by default.

The lesson describes Functions 1-8. The repository also defines Function 9 as a one-dimensional function; the supplied lesson does not describe it.