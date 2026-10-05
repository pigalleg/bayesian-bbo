# Data datasheet — to be completed during the capstone

Initial observations are provided externally for nine black-box functions. The number of initial observations may vary by function. See `data/README.md` for the expected file layout. This document is **not** a completed datasheet.

As data becomes available, record its source and provenance, collection and submission dates, dimensionality and validation outcomes, any missing/invalid observations, licensing or sharing constraints, and how new evaluations were incorporated.

## Weekly update received 2026-10-05

Received one input/output observation for each function 1–8 in text attachments, now archived as `inputs.json` and `outputs.json` alongside `expected_outputs.json` in `data/weekly_updates/week_2/`. Conversion to JSON preserved all numeric values and row order. The evaluation date and further source provenance were not supplied. The input dimensions matched the configured functions; inputs and outputs were finite, inputs were within `[0, 1)`, and no inputs duplicated existing observations. Their actual and expected values are aligned in each function's `all_observations.npz`. Expected means were computed from initial observations using an RBF kernel with fixed length scale 0.1. No update was received for function 9.
