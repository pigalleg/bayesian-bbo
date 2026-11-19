import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from bayesian_bbo.data import FUNCTION_SPECS, load_observations


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "watch_weekly_updates.py"


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def prepare_week(root):
    for function_id, dimension in FUNCTION_SPECS.items():
        folder = root / f"function_{function_id}"
        folder.mkdir(parents=True)
        np.save(folder / "initial_inputs.npy", [[0.1] * dimension])
        np.save(folder / "initial_outputs.npy", [1.0])
    folder = root / "weekly_updates" / "week_2"
    folder.mkdir(parents=True)
    points = [[0.2] * dimension for dimension in FUNCTION_SPECS.values()]
    write_json(folder / "inputs.json", points)
    write_json(folder / "predictions.json", {
        "expected_outputs": [1.5] * 8,
        "stds": [0.2] * 8,
        "expected_best_outputs": [1.6] * 8,
    })
    return folder, points


def wait_for(predicate, process):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if predicate():
            return
        if process.poll() is not None:
            raise AssertionError(f"Watcher exited early: {process.communicate()}")
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for watcher")


def test_watcher_imports_new_cumulative_exports_and_reports_conflict(tmp_path):
    root = tmp_path / "data"
    folder, points = prepare_week(root)
    stdout_path, stderr_path = tmp_path / "stdout", tmp_path / "stderr"
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--data-dir", str(root), "--interval", "0.1"],
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        try:
            (folder / "inputs.txt").write_text(
                json.dumps([[0.3] * d for d in FUNCTION_SPECS.values()]) + "\n" + json.dumps(points),
                encoding="utf-8",
            )
            (folder / "outputs.txt").write_text(
                json.dumps([3.0] * 8) + "\n" + json.dumps([4.0] * 8), encoding="utf-8",
            )
            wait_for(lambda: "incorporated in the main database" in stdout_path.read_text(), process)
            for function_id in FUNCTION_SPECS:
                observations = load_observations(root, function_id)
                np.testing.assert_array_equal(observations.weeks, [0, 2])
                np.testing.assert_array_equal(observations.outputs, [1.0, 4.0])
            assert json.loads((folder / "outputs.json").read_text()) == [4.0] * 8

            (folder / "outputs.txt").write_text(
                json.dumps([3.0] * 8) + "\n" + json.dumps([5.0] * 8), encoding="utf-8",
            )
            wait_for(lambda: "exists with different values" in stderr_path.read_text(), process)
        finally:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=10)

    assert process.returncode == 0
    assert "your weekly observations have been incorporated in the main database" in stdout_path.read_text()
    assert json.loads((folder / "outputs.json").read_text()) == [4.0] * 8
    for function_id in FUNCTION_SPECS:
        np.testing.assert_array_equal(load_observations(root, function_id).outputs, [1.0, 4.0])


def test_import_rejects_mismatched_submission_and_incomplete_history(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    from watch_weekly_updates import import_week

    root = tmp_path / "data"
    folder, points = prepare_week(root)
    (folder / "inputs.txt").write_text(json.dumps(points), encoding="utf-8")
    (folder / "outputs.txt").write_text(json.dumps([4.0] * 8), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 2 cumulative weekly lists"):
        import_week(root, folder, 2)

    wrong_points = [[0.3] * d for d in FUNCTION_SPECS.values()]
    (folder / "inputs.txt").write_text(
        json.dumps(points) + "\n" + json.dumps(wrong_points), encoding="utf-8",
    )
    (folder / "outputs.txt").write_text(
        json.dumps([3.0] * 8) + "\n" + json.dumps([4.0] * 8), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differs from the submitted text inputs"):
        import_week(root, folder, 2)
    assert not (folder / "outputs.json").exists()
    assert not (root / "function_1" / "all_observations.npz").exists()

    (folder / "inputs.txt").write_text(json.dumps(points) + "\n" + json.dumps(points), encoding="utf-8")
    import_week(root, folder, 2)
    import_week(root, folder, 2)
    np.testing.assert_array_equal(load_observations(root, 1).weeks, [0, 2])
