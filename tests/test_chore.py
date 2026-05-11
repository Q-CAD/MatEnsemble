import json

from pathlib import Path

from matensemble.chore import Chore
from matensemble.model import ChoreType, Resources


def test_chore_normalizes_string_command():
    chore = Chore(
        id="chore-1",
        workdir=Path.cwd() / "tmp" / "chore-1",
        command="python -c 'print(1)'",
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(),
    )
    assert isinstance(chore.command, list)
    assert chore.spec_path.name == "chore.pickle"


def test_chore_write_metadata(tmp_path: Path):
    chore = Chore(
        id="chore-2",
        workdir=tmp_path / "chore-2",
        command=["echo", "ok"],
        chore_type=ChoreType.EXECUTABLE,
        resources=Resources(),
    )
    chore._write_metadata()
    metadata = json.loads((tmp_path / "chore-2" / "metadata.json").read_text())
    assert metadata["id"] == "chore-2"
