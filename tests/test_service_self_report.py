"""Tests for services telling the truth about what they can currently do.

Each case here is a way a healthy-looking service was quietly doing less than
callers assumed:

- the ADMET predictor's ``/health`` returned a top-level ``endpoints`` list
  covering the drug stack only, with nothing to say so, which is how the agro
  stack came to look as though it were not deployed
- the profiler reported its Python versions but not ``fpocket``, a binary, so
  an image built before the Dockerfile grew that layer answered every
  subpocket request with an empty list and still called itself well
- ``run.sh`` bound some services' ``app.py`` over the baked copy and not
  others', so a one-line fix to those others shipped and did nothing
- InterPro's "valid accession, no entries" 204 was an empty list through the
  annotation service and an error on the direct path, so the same protein
  looked domain-free or broken depending on one environment variable

The source-reading cases run anywhere: no container, no cluster.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_RUN_SH = _ROOT / "services" / "singularity" / "run.sh"
_ADMET_APP = _ROOT / "services" / "admet-predictor" / "app.py"
_ANNOTATION_APP = _ROOT / "services" / "annotation" / "app.py"


def _health_keys(app_path: Path) -> set[str]:
    """Constant keys of the dict the module's ``health()`` returns."""

    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "health":
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
                return {
                    k.value for k in stmt.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                }
    raise AssertionError(f"no health() returning a dict literal in {app_path}")


class TestTheAdmetPredictorSaysWhichStackItIsDescribing:
    def test_health_offers_every_model_set(self) -> None:
        assert "model_sets" in _health_keys(_ADMET_APP)

    def test_the_legacy_endpoint_list_names_the_stack_it_belongs_to(self) -> None:
        # The bare list is one stack's. Unlabelled beside "status": "ok" it
        # reads as the whole service's, which is the misreading that made the
        # agro stack look missing.
        keys = _health_keys(_ADMET_APP)
        assert "endpoints" not in keys or "endpoints_model_set" in keys


@pytest.fixture(scope="module")
def versions() -> dict[str, str]:
    src = str(_ROOT / "interaction-profiler" / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from interaction_profiler.schemas import detect_tool_versions
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"interaction_profiler not importable: {exc}")
    return detect_tool_versions()


@pytest.fixture(scope="module")
def run_sh() -> str:
    return _RUN_SH.read_text(encoding="utf-8")


class TestTheProfilerAdmitsWhenFpocketIsNotThere:
    def test_fpocket_is_reported_either_way(self, versions: dict[str, str]) -> None:
        # Present or not, it has to appear: the old report simply omitted
        # missing dependencies, so absent and never-checked looked the same.
        assert "fpocket" in versions

    def test_an_absent_binary_says_so_rather_than_guessing_a_version(
        self, versions: dict[str, str]
    ) -> None:
        from interaction_profiler.site.fpocket import fpocket_available

        if fpocket_available():
            assert versions["fpocket"] != "absent"
        else:
            assert versions["fpocket"] == "absent"


class TestEveryBoundServiceDeploysWithoutARebuild:
    @pytest.mark.parametrize(
        "service", ["posebusters", "admet-predictor", "interaction-profiler"]
    )
    def test_the_service_app_is_bound_over_the_baked_copy(
        self, run_sh: str, service: str
    ) -> None:
        # Both spellings count: these two are bound through "${name}", which
        # is the same bind written once for the pair.
        literal = f"services/{service}/app.py:/app/app.py"
        templated = 'services/${name}/app.py:/app/app.py'
        assert literal in run_sh or (
            templated in run_sh and f'"{service}"' in run_sh
        ), f"{service}'s app.py is not bound, so edits to it need an image rebuild"


class TestInterProAgreesWithItselfOnAnEmptyResult:
    @pytest.fixture
    def direct_route(self, monkeypatch: pytest.MonkeyPatch):
        from reinvent4_mcp.tools import interpro

        monkeypatch.setattr(interpro, "_ANN_URL", "")
        monkeypatch.setattr(interpro, "_load_cached_interpro", lambda _acc: None)
        return interpro

    def test_a_204_is_no_entries_and_not_an_error(self, direct_route, monkeypatch) -> None:
        class _Resp:
            status_code = 204

            def json(self) -> dict:  # pragma: no cover - a 204 carries no body
                raise ValueError("no content")

        monkeypatch.setattr(direct_route.requests, "get", lambda *a, **k: _Resp())

        result = direct_route._fetch_interpro("P12345", None)

        assert "error" not in result
        assert result == {"uniprot_accession": "P12345", "interpro_entries": []}

    def test_the_annotation_service_normalises_it_the_same_way(self) -> None:
        # The two routes are separate code; this is what stops them drifting
        # back apart, since only one of them is exercised in any deployment.
        src = _ANNOTATION_APP.read_text(encoding="utf-8")
        assert "status_code == 204" in src
        assert '"interpro_entries": []' in src
