# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Service discovery for schedulers that have no DNS.

The property that matters most here is that applying a registry can never
change behaviour on Kubernetes, where the environment is already complete.
"""

from __future__ import annotations

import json

from reinvent4_agent.service_registry import (
    apply_registry,
    load_registry,
    write_registry,
)


class TestLoadRegistry:

    def test_no_registry_configured_is_not_an_error(self):
        """The normal case on Kubernetes."""
        assert load_registry(None) == {}

    def test_a_missing_file_degrades_to_empty(self, tmp_path):
        assert load_registry(tmp_path / "absent.json") == {}

    def test_malformed_json_degrades_to_empty(self, tmp_path):
        bad = tmp_path / "registry.json"
        bad.write_text("{not json")

        assert load_registry(bad) == {}

    def test_a_json_list_is_rejected(self, tmp_path):
        bad = tmp_path / "registry.json"
        bad.write_text('["http://nid001:8088"]')

        assert load_registry(bad) == {}


class TestApplyRegistry:

    def test_it_sets_the_variables_the_tools_read(self, tmp_path):
        path = write_registry(
            {"synthesis": "http://nid001:8088", "posebusters": "http://nid002:8082"},
            tmp_path / "registry.json",
        )
        env: dict[str, str] = {}

        applied = apply_registry(path, env)

        assert env["SYNTHESIS_SERVICE_URL"] == "http://nid001:8088"
        assert env["POSEBUSTERS_SERVICE_URL"] == "http://nid002:8082"
        assert sorted(applied) == ["POSEBUSTERS_SERVICE_URL", "SYNTHESIS_SERVICE_URL"]

    def test_the_environment_always_wins(self, tmp_path):
        """This is what makes it safe to call unconditionally on Kubernetes."""
        path = write_registry(
            {"synthesis": "http://nid001:8088"}, tmp_path / "registry.json"
        )
        env = {"SYNTHESIS_SERVICE_URL": "http://pl-dd-agentic-synthesis-planner:8088"}

        applied = apply_registry(path, env)

        assert env["SYNTHESIS_SERVICE_URL"] == (
            "http://pl-dd-agentic-synthesis-planner:8088"
        )
        assert applied == []

    def test_an_empty_value_is_treated_as_unset(self, tmp_path):
        path = write_registry(
            {"synthesis": "http://nid001:8088"}, tmp_path / "registry.json"
        )
        env = {"SYNTHESIS_SERVICE_URL": ""}

        apply_registry(path, env)

        assert env["SYNTHESIS_SERVICE_URL"] == "http://nid001:8088"

    def test_an_unknown_service_is_skipped_not_fatal(self, tmp_path):
        path = write_registry(
            {"synthesis": "http://nid001:8088", "nonesuch": "http://nid003:9000"},
            tmp_path / "registry.json",
        )
        env: dict[str, str] = {}

        applied = apply_registry(path, env)

        assert applied == ["SYNTHESIS_SERVICE_URL"]
        assert "http://nid003:9000" not in env.values()

    def test_with_no_registry_it_changes_nothing(self):
        env = {"SYNTHESIS_SERVICE_URL": "http://kept:8088"}

        assert apply_registry(None, env) == []
        assert env == {"SYNTHESIS_SERVICE_URL": "http://kept:8088"}


class TestWriteRegistry:

    def test_it_round_trips(self, tmp_path):
        services = {"synthesis": "http://nid001:8088"}

        path = write_registry(services, tmp_path / "sub" / "registry.json")

        assert json.loads(path.read_text()) == services

    def test_no_partial_file_is_left_behind(self, tmp_path):
        """Readers must see the old registry or the new one, never half of one."""
        path = write_registry({"synthesis": "http://a:1"}, tmp_path / "registry.json")
        write_registry({"synthesis": "http://b:2"}, path)

        assert list(p.name for p in tmp_path.iterdir()) == ["registry.json"]
        assert json.loads(path.read_text())["synthesis"] == "http://b:2"
