"""Tests for how the interaction-profiler service reaches a running node.

The v2 ``/profile_complex`` route answered 404 on LUMI for two weeks while the
code that serves it sat in the checkout, because two things went wrong at once
and each hid the other:

- ``run.sh`` bound the profiler *package* over the baked copy but not the
  service's ``app.py``, so the container kept the route table from whenever
  the image was last built
- every handler that returns a ``JSONResponse`` annotated it as such, and the
  module uses postponed annotations, so FastAPI received the unresolvable
  string ``"JSONResponse"``; the handlers still worked, but ``/openapi.json``
  answered 500 -- the one request that would have shown the route was missing

Both are checked by reading the sources, so this runs anywhere: no container,
no FastAPI, no cluster.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_APP = _ROOT / "services" / "interaction-profiler" / "app.py"
_RUN_SH = _ROOT / "services" / "singularity" / "run.sh"

_ROUTE_DECORATORS = {"get", "post", "put", "patch", "delete"}


def _route_handlers(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every function carrying an ``@<something>.<verb>("/path")`` decorator."""

    handlers: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            func = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(func, ast.Attribute) and func.attr in _ROUTE_DECORATORS:
                handlers.append(node)
                break
    return handlers


@pytest.fixture(scope="module")
def app_tree() -> ast.Module:
    return ast.parse(_APP.read_text(encoding="utf-8"))


class TestTheRouteTableSurvivesDeployment:
    def test_the_service_app_is_bound_over_the_baked_copy(self) -> None:
        # Without this bind the container serves the app.py from image build
        # time, so any route added since is a 404 no matter what ships.
        run_sh = _RUN_SH.read_text(encoding="utf-8")
        assert (
            "services/interaction-profiler/app.py:/app/app.py" in run_sh
        ), "run.sh must bind the profiler's app.py so new routes deploy without a rebuild"

    def test_the_v2_routes_are_registered_on_the_app(self, app_tree: ast.Module) -> None:
        paths = {
            dec.args[0].value
            for node in _route_handlers(app_tree)
            for dec in node.decorator_list
            if isinstance(dec, ast.Call) and dec.args
            and isinstance(dec.args[0], ast.Constant)
        }
        assert {"/profile_complex", "/characterize_site"} <= paths


class TestHandlersDoNotBreakSchemaGeneration:
    def test_no_handler_returns_an_annotation_the_module_cannot_resolve(
        self, app_tree: ast.Module
    ) -> None:
        # ``from __future__ import annotations`` hands FastAPI the annotation as
        # a string, which it resolves against module globals -- so a name that
        # only exists inside create_app() leaves pydantic with an unbuilt
        # TypeAdapter and /openapi.json 500s.
        module_names = {
            alias.asname or alias.name.split(".")[0]
            for node in ast.iter_child_nodes(app_tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        module_names |= {
            node.name
            for node in ast.iter_child_nodes(app_tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }

        unresolvable = [
            f"{node.name} -> {node.returns.id}"
            for node in _route_handlers(app_tree)
            if isinstance(node.returns, ast.Name) and node.returns.id not in module_names
        ]
        assert not unresolvable, (
            "route handlers annotate a return type that is not importable at "
            f"module scope, which breaks /openapi.json: {unresolvable}"
        )


class TestTheApptainerImageCarriesWhatTheDockerImageDoes:
    """The .def says it mirrors the Dockerfile; it had stopped.

    LUMI runs the Apptainer image, and that one was built without fpocket,
    requests or freesasa. None of the three is a hard dependency, so the
    service started fine and ``characterize_site_profile`` answered with an
    empty ``site_profile`` and the reasons buried in a ``warnings`` list --
    which reads as "this target has no druggable sub-pocket" rather than
    "no pocket geometry was computed".
    """

    @pytest.fixture(scope="class")
    def recipes(self) -> tuple[str, str]:
        return (
            (_ROOT / "services" / "singularity" / "interaction-profiler.def").read_text(),
            (_ROOT / "services" / "interaction-profiler" / "Dockerfile").read_text(),
        )

    @pytest.mark.parametrize("package", ["requests", "freesasa"])
    def test_an_annotation_dependency_is_installed_in_both(
        self, recipes: tuple[str, str], package: str
    ) -> None:
        apptainer, docker = recipes
        assert package in docker
        assert package in apptainer, (
            f"{package} is in the Dockerfile but not the .def, so the image "
            "LUMI runs degrades that code path to a warning"
        )

    def test_fpocket_is_built_in_both(self, recipes: tuple[str, str]) -> None:
        apptainer, docker = recipes
        for recipe in (docker, apptainer):
            assert "Discngine/fpocket.git" in recipe
            # Proof the build produced a binary on PATH, not just that the
            # clone was attempted.
            assert "command -v fpocket" in recipe
