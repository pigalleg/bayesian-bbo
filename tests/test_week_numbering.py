import json
from pathlib import Path
import runpy
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_week_label_utility_assigns_zero_to_initial_data(tmp_path):
    directory = tmp_path / "function_1"
    directory.mkdir()
    np.save(directory / "initial_inputs.npy", [[0.1, 0.2]])
    np.savez(
        directory / "all_observations.npz",
        inputs=[[0.1, 0.2], [0.3, 0.4]], outputs=[1., 2.],
    )
    batch = tmp_path / "weekly_updates" / "week_1"
    batch.mkdir(parents=True)
    (batch / "inputs.json").write_text("[[0.3, 0.4]]", encoding="utf-8")
    (batch / "outputs.json").write_text("[2.0]", encoding="utf-8")
    label_function = runpy.run_path(str(SCRIPTS / "add_week_labels.py"))["label_function"]

    _, arrays = label_function(tmp_path, 1)

    np.testing.assert_array_equal(arrays["weeks"], [0, 1])
    np.testing.assert_array_equal(arrays["outputs"], [1., 2.])


def test_text_conversion_defaults_to_first_evaluated_week_one(tmp_path, monkeypatch):
    source = tmp_path / "outputs.txt"
    source.write_text(
        json.dumps([1.] * 8) + "\n" + json.dumps([2.] * 8) + "\n", encoding="utf-8",
    )
    destination = tmp_path / "weekly_updates"
    monkeypatch.setattr(
        sys, "argv",
        ["convert_weekly_txt.py", str(source), "--weeks-dir", str(destination)],
    )
    main = runpy.run_path(str(SCRIPTS / "convert_weekly_txt.py"))["main"]

    main()

    assert json.loads((destination / "week_1/outputs.json").read_text()) == [1.] * 8
    assert json.loads((destination / "week_2/outputs.json").read_text()) == [2.] * 8
    assert not (destination / "week_0").exists()
