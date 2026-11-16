# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What keeps a long-running fleet readable: one identity per supervisor, a
pattern parameter in the shape REINVENT4 reads it, and a campaign that
closes itself."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from reinvent4_agent.conductor import DEFAULT_CAMPAIGN_HOURS, DEFAULT_GRACE_HOURS, start_campaign
from reinvent4_mcp.tools.ttp import _endpoint_params_for_component

_SBATCH = Path(__file__).resolve().parent.parent / "services" / "lumi" / "supervisor.sbatch"


def test_a_supervisor_is_named_after_its_job_not_only_its_node():
    """Slurm puts two supervisors on one node; a hostname is then not an identity."""
    line = [one for one in _SBATCH.read_text(encoding="utf-8").splitlines() if "SUPERVISOR_ID" in one]

    assert line, "supervisor.sbatch must set SUPERVISOR_ID"
    assert "$SLURM_JOB_ID" in line[0]


class _Store:
    def __init__(self) -> None:
        self.asked: dict[str, Any] = {}

    def create_campaign(self, **kwargs: Any) -> dict[str, Any]:
        self.asked = kwargs
        return {"campaign_id": "c1", **kwargs}


def test_a_campaign_opened_without_a_window_still_gets_one():
    """With no deadline the store's phase stays 'dispatching', so nothing ever closes it."""
    store = _Store()

    start_campaign(store, "Design JAK3 ligands")

    assert store.asked["dispatch_seconds"] == DEFAULT_CAMPAIGN_HOURS * 3600
    assert store.asked["grace_seconds"] == DEFAULT_GRACE_HOURS * 3600


def test_a_window_that_was_asked_for_is_the_one_used():
    store = _Store()

    start_campaign(store, "Design JAK3 ligands", hours=3, grace_hours=0.5)

    assert store.asked["dispatch_seconds"] == 3 * 3600
    assert store.asked["grace_seconds"] == 0.5 * 3600


class TestPatternParameters:
    """REINVENT4 collects one list per endpoint, so the shape inside it matters."""

    def test_a_counted_group_takes_one_pattern(self):
        endpoint = _endpoint_params_for_component("GroupCount", {"smarts": ["c1ccccc1"]})

        assert endpoint["smarts"] == "c1ccccc1"

    def test_an_alert_filter_keeps_its_list(self):
        endpoint = _endpoint_params_for_component("CustomAlerts", {"smarts": ["[#8][#8]", "C#C"]})

        assert endpoint["smarts"] == ["[#8][#8]", "C#C"]

    def test_a_pattern_written_as_a_bare_string_is_read_either_way(self):
        assert _endpoint_params_for_component("GroupCount", {"smarts": "C=CC(=O)N"})["smarts"] == "C=CC(=O)N"
        assert _endpoint_params_for_component("CustomAlerts", {"smarts": "[#8][#8]"})["smarts"] == ["[#8][#8]"]

    @pytest.mark.parametrize(
        "component,params,says",
        [
            ("GroupCount", {"smarts": ["c1ccccc1", "C=CC(=O)N"]}, "one smarts pattern per objective"),
            ("GroupCount", {}, "must be one pattern"),
            ("CustomAlerts", {"smarts": []}, "non-empty list"),
            ("TanimotoSimilarity", {"smiles": [""]}, "non-empty list"),
            ("MMP", {}, "non-empty list"),
        ],
    )
    def test_what_reinvent_could_not_build_is_refused_here(self, component, params, says):
        with pytest.raises(ValueError, match=says):
            _endpoint_params_for_component(component, params)

    def test_a_component_with_no_pattern_of_its_own_is_left_alone(self):
        assert _endpoint_params_for_component("QED", {}) == {}
