# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A TTP REINVENT4 could not start a run from is refused when it is saved."""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from reinvent4_mcp.jobs import toml_generator
from reinvent4_mcp.jobs.toml_generator import generate_reinvent_toml
from reinvent4_mcp.tools import ttp as ttp_tools
from reinvent4_mcp.ttp.store import TTPStore

_TTP = {
    "id": "jak3-design",
    "name": "JAK3 design",
    "scoring": {
        "objectives": [
            {
                "name": "Drug-likeness",
                "type": "native",
                "component": "QED",
                "transform": {"type": "sigmoid", "low": 0.3, "high": 0.8},
            },
        ],
    },
}


@pytest.fixture
def store(monkeypatch, tmp_path):
    ttp_tools.reset_store()
    store = TTPStore(tmp_path / "ttps")
    monkeypatch.setattr(ttp_tools, "_get_store", lambda: store)

    def no_reinvent() -> None:
        raise ImportError("REINVENT4 is not installed here")

    monkeypatch.setattr("reinvent4_mcp.utils.ensure_reinvent4_importable", no_reinvent)
    yield store
    ttp_tools.reset_store()


def _with(objective: dict) -> dict:
    data = copy.deepcopy(_TTP)
    data["scoring"]["objectives"].append(objective)
    return data


def test_a_transform_without_a_bound_is_refused_with_its_objective(store):
    answer = ttp_tools.create_ttp(_with({
        "name": "Hinge binder",
        "type": "native",
        "component": "QED",
        "transform": {"type": "sigmoid", "low": 0.3, "k": 10},
    }))

    assert "Objective 'Hinge binder'" in answer["error"]
    assert "high" in answer["error"]
    assert store.list_all() == []


def test_an_update_that_drops_a_bound_is_refused(store):
    assert ttp_tools.create_ttp(copy.deepcopy(_TTP))["status"] == "created"

    answer = ttp_tools.update_ttp("jak3-design", {
        "scoring": {"objectives": [{
            "name": "Drug-likeness", "type": "native", "component": "QED",
            "transform": {"type": "sigmoid", "low": 0.3, "high": None},
        }]},
    })

    assert "high" in answer["error"]
    assert store.get("jak3-design").version == 1


def test_group_count_takes_one_pattern(store):
    one = {"name": "Hinge motif", "type": "native", "component": "GroupCount", "params": {"smarts": ["c1ccc2[nH]ncc2c1"]}}
    two = {**one, "params": {"smarts": ["c1ccc2[nH]ncc2c1", "C=CC(=O)N"]}}

    assert "one smarts pattern per objective" in ttp_tools.create_ttp(_with(two))["error"]
    assert ttp_tools.create_ttp(_with(one))["status"] == "created"


def test_group_count_gets_its_pattern_as_a_string():
    scorer = {
        "type": "geometric_mean",
        "component": [
            {"GroupCount": {"endpoint": [{"name": "Hinge motif", "weight": 1.0, "params": {"smarts": ["c1ccc2[nH]ncc2c1"]}}]}},
            {"CustomAlerts": {"endpoint": [{"name": "Alerts", "weight": 1.0, "params": {"smarts": ["[#8][#8]"]}}]}},
        ],
    }

    toml = generate_reinvent_toml(scorer_config=scorer, prior_file="/model/prior.model")

    assert 'params.smarts = "c1ccc2[nH]ncc2c1"' in toml
    assert 'params.smarts = ["[#8][#8]"]' in toml


def _scorer(component: str) -> dict:
    return {
        "type": "geometric_mean",
        "component": [{component: {"endpoint": [{"name": "x", "weight": 1.0}]}}],
    }


class TestComponentNamesAreCheckedBeforeTheGpu:
    """The name becomes a TOML table verbatim, so nothing else checks it.

    A misspelling used to reach the node and fail inside REINVENT4 minutes
    later, reaching the agent as a dead run in a log file rather than as an
    answer to the call that caused it.
    """

    KNOWN = frozenset({"QED", "SlogP", "TanimotoSimilarity"})

    def test_an_unknown_component_is_refused_with_the_closest_known_one(self):
        with patch.object(
            toml_generator, "_known_component_names", return_value=self.KNOWN
        ):
            with pytest.raises(ValueError) as excinfo:
                generate_reinvent_toml(
                    scorer_config=_scorer("TanimotoSimilarty"),
                    prior_file="/model/prior.model",
                )

        assert "TanimotoSimilarty" in str(excinfo.value)
        assert "TanimotoSimilarity" in str(excinfo.value)

    def test_a_known_component_is_emitted(self):
        with patch.object(
            toml_generator, "_known_component_names", return_value=self.KNOWN
        ):
            toml = generate_reinvent_toml(
                scorer_config=_scorer("SlogP"), prior_file="/model/prior.model"
            )

        assert "[stage.scoring.component.SlogP]" in toml

    def test_a_class_name_matches_its_lowercased_registry_key(self):
        """REINVENT4 keys the registry on the lowercased class name, while the
        mapper emits the class name itself (ExternalRESTScorer, QED)."""

        registry = frozenset({"externalrestscorer", "qed", "sascore"})
        with patch.object(
            toml_generator, "_known_component_names", return_value=registry
        ):
            toml = generate_reinvent_toml(
                scorer_config=_scorer("ExternalRESTScorer"),
                prior_file="/model/prior.model",
            )

        assert "[stage.scoring.component.ExternalRESTScorer]" in toml

    def test_names_go_through_where_the_registry_cannot_be_read(self):
        """This module also renders where REINVENT4 is not installed, because
        the runner relays a TOML built elsewhere. Refusing there would reject
        every component rather than the misspelled one."""

        with patch.object(
            toml_generator, "_known_component_names", return_value=frozenset()
        ):
            toml = generate_reinvent_toml(
                scorer_config=_scorer("SomethingOnlyTheNodeHas"),
                prior_file="/model/prior.model",
            )

        assert "[stage.scoring.component.SomethingOnlyTheNodeHas]" in toml


def test_a_run_scoring_nothing_says_so_in_its_own_config():
    """REINVENT4 needs a component, so an unmapped config still has to run.

    Scored on drug-likeness alone it looks exactly like a healthy run for
    the hours it takes, and the TOML on disk is the only record of which it
    was.
    """

    toml = generate_reinvent_toml(
        scorer_config={"type": "geometric_mean", "component": []},
        prior_file="/model/prior.model",
    )

    assert "no scoring components were mapped" in toml
    assert "[stage.scoring.component.QED]" in toml
