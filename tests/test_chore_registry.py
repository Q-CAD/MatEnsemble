import cloudpickle
import pytest

from matensemble.chore import ChoreRegistry, registry_entry_filename
from matensemble.model import Resources


def test_registry_decorator_registers_callable_with_resources():
    registry = ChoreRegistry()

    @registry.chore(name="evaluate", cores_per_task=2)
    def evaluate(candidate):
        return candidate

    entry = registry.get("evaluate")

    assert entry.func is evaluate
    assert entry.qualname == "evaluate"
    assert entry.id_name == "evaluate"
    assert entry.resources.cores_per_task == 2


def test_registry_rejects_duplicate_and_invalid_names():
    registry = ChoreRegistry()

    @registry.chore(name="evaluate")
    def evaluate():
        return None

    with pytest.raises(ValueError, match="already registered"):
        registry.register(evaluate, name="evaluate")

    with pytest.raises(ValueError, match="path separators"):
        registry.register(evaluate, name="../bad")


def test_registry_write_serializes_registered_callables(tmp_path):
    registry = ChoreRegistry()

    @registry.chore(name="evaluate", resources=Resources(cores_per_task=3))
    def evaluate(candidate):
        return {"candidate": candidate}

    registry.write(tmp_path)

    with (tmp_path / registry_entry_filename("evaluate")).open("rb") as fh:
        loaded = cloudpickle.load(fh)

    assert loaded("x") == {"candidate": "x"}
