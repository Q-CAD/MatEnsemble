import pickle

from pathlib import Path

from matensemble.runtime_worker import _load_dep_result


def test_load_dep_result_reads_result_pickle(tmp_path: Path):
    dep_id = "chore-upstream-0001"
    dep_dir = tmp_path / "out" / dep_id
    dep_dir.mkdir(parents=True)

    expected = {"value": 7}
    with (dep_dir / "result.pickle").open("wb") as f:
        pickle.dump(expected, f)

    spec_file = tmp_path / "out" / "chore-downstream-0002" / "chore.pickle"
    spec_file.parent.mkdir(parents=True)
    spec_file.touch()

    loaded = _load_dep_result(spec_file, dep_id)
    assert loaded == expected
