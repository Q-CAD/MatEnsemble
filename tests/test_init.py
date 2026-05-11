import matensemble

from matensemble.model import ChoreType, OutputReference, Resources


def test_package_exports_core_types():
    assert matensemble.OutputReference is OutputReference
    assert matensemble.Resources is Resources
    assert matensemble.ChoreType is ChoreType
