from __future__ import annotations

from .examples import (
    API_GUIDANCE,
    get_example_source,
    how_to_build_container,
    list_examples,
    get_container_contents,
)
from .systems import get_system_profile, list_systems, render_environment_setup


def api_overview() -> str:
    return API_GUIDANCE


def examples_overview() -> list[dict[str, object]]:
    return list_examples()


def read_repo_example(name: str) -> str:
    return get_example_source(name)


def read_container_contents(name: str) -> str:
    return get_container_contents(name)


def container_build_instructions(name: str) -> str:
    return how_to_build_container(name)


def systems_overview() -> list[dict[str, object]]:
    return list_systems()


def environment_setup(name: str) -> str:
    return render_environment_setup(name)


def system_profile(name: str) -> dict[str, object]:
    return get_system_profile(name).to_dict()
