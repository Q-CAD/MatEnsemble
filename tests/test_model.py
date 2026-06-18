import pickle

from pathlib import Path

import pytest

from matensemble.model import ChoreType, OutputReference, Resources


def test_resources_validates_basic_inputs():
    r = Resources(num_tasks=2, cores_per_task=4, gpus_per_task=1, mpi=True)
    assert r.num_tasks == 2
    assert r.cores_per_task == 4
    assert r.gpus_per_task == 1
    assert r.mpi is True


def test_resources_rejects_invalid_values():
    with pytest.raises(ValueError, match="num_tasks must be an integer >= 1"):
        Resources(num_tasks=0)


def test_output_reference_result_reads_pickled_value(tmp_path: Path):
    workdir = tmp_path / "chore-a"
    workdir.mkdir()
    with (workdir / "result.pickle").open("wb") as f:
        pickle.dump({"x": 1}, f)

    ref = OutputReference("chore-a", workdir)
    assert ref.result() == {"x": 1}


def test_output_reference_result_raises_for_missing_file(tmp_path: Path):
    ref = OutputReference("missing", tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        ref.result()


def test_choretype_enum_distinct():
    assert ChoreType.PYTHON != ChoreType.EXECUTABLE
