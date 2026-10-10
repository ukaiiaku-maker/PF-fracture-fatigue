import json

import numpy as np

from arrhenius_fracture.run_state_checkpoint_v10230 import (
    MANIFEST,
    load_combined_checkpoint,
    write_combined_checkpoint,
)


def _write(root, generation):
    return write_combined_checkpoint(
        root,
        outer={"case": {"generation": generation}},
        arrays={"state": np.array([generation], dtype=float)},
        kinetic={"generation": generation},
        kinetic_vector=np.array([generation], dtype=float),
    )


def test_atomic_checkpoint_policy_keeps_current_and_predecessors(tmp_path, monkeypatch):
    monkeypatch.setenv("V10230_CHECKPOINT_GENERATIONS", "3")
    manifests = [_write(tmp_path, value) for value in range(5)]
    generations = sorted((tmp_path / "run_state_generations").iterdir())
    assert len(generations) == 3
    current = json.loads((tmp_path / MANIFEST).read_text())
    assert current["generation"] == manifests[-1]["generation"]
    assert (tmp_path / "run_state_generations" / current["generation"]).is_dir()
    outer, kinetic, arrays = load_combined_checkpoint(tmp_path)
    assert outer["case"]["generation"] == 4
    assert kinetic["generation"] == 4
    assert arrays["state"].tolist() == [4.0]
