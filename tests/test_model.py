from __future__ import annotations

import pytest

from matensemble.model import ChoreType, OutputReference, Resources


def test_output_reference_is_frozen_and_holds_chore_id():
    ref = OutputReference("chore-1")
    assert ref.chore_id == "chore-1"


@pytest.mark.parametrize(
    ("kwargs", "exc_type"),
    [
        ({"num_tasks": 0}, ValueError),
        ({"num_tasks": 1.5}, ValueError),
        ({"cores_per_task": 0}, ValueError),
        ({"gpus_per_task": -1}, ValueError),
        ({"mpi": "yes"}, TypeError),
        ({"env": []}, TypeError),
        ({"env": {1: "x"}}, TypeError),
        ({"env": {"A": 1}}, TypeError),
    ],
)
def test_resources_validation(kwargs, exc_type):
    with pytest.raises(exc_type):
        Resources(**kwargs)


def test_resources_accept_valid_values():
    resources = Resources(
        num_tasks=2,
        cores_per_task=4,
        gpus_per_task=1,
        mpi=True,
        env={"A": "B"},
    )
    assert resources.num_tasks == 2
    assert resources.cores_per_task == 4
    assert resources.gpus_per_task == 1
    assert resources.mpi is True
    assert resources.env == {"A": "B"}


def test_chore_type_string_values_exist():
    assert ChoreType.PYTHON
    assert ChoreType.EXECUTABLE
