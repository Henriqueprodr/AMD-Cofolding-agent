# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading a sub-agent's conversation back from the transcript it wrote."""

from __future__ import annotations

import json
import os
from datetime import datetime

from reinvent4_agent.web.subagent_transcript import read_conversation, valid_task_id


def _write(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_the_task_is_found_by_its_name_and_read_in_order(tmp_path):
    _write(tmp_path / "session_20261111_120008_subagent-05f31fcd.jsonl", [
        {"ts": "t1", "turn": 1, "event": "user_message", "content": "Read the JAK3 SAR"},
        {"ts": "t2", "turn": 1, "event": "tool_call", "name": "fetch_target_info", "arguments": {"query": "JAK3"}, "round": 0},
        {"ts": "t3", "turn": 1, "event": "tool_result", "name": "fetch_target_info", "elapsed_s": 0.4, "is_error": False, "content_length": 900},
        {"ts": "t4", "turn": 1, "event": "assistant_message", "content": "## Report\nCys909 sets JAK3 apart."},
    ])
    _write(tmp_path / "session_20261111_120008_driver-c813eb02.jsonl", [{"event": "user_message", "content": "other task"}])

    read = read_conversation(tmp_path, "05f31fcd")

    assert read["found"] is True
    assert [event["event"] for event in read["events"]] == ["user_message", "tool_call", "tool_result", "assistant_message"]
    assert read["events"][1]["arguments"] == {"query": "JAK3"}
    # Bookkeeping the browser has no use for stays behind.
    assert "turn" not in read["events"][0]
    # Kept, so the browser can tell an excerpt from the whole answer.
    assert read["events"][2]["content_length"] == 900


def test_a_task_claimed_again_reads_its_latest_run(tmp_path):
    old = tmp_path / "session_20261111_110000_driver-t1.jsonl"
    new = tmp_path / "session_20261111_120000_driver-t1.jsonl"
    _write(old, [{"event": "user_message", "content": "first attempt"}])
    _write(new, [{"event": "user_message", "content": "second attempt"}])
    os.utime(old, (1, 1))

    assert read_conversation(tmp_path, "t1")["events"][0]["content"] == "second attempt"


def test_the_reviewer_and_a_half_written_line_are_left_out(tmp_path):
    _write(tmp_path / "session_20261111_120000_reviewer-t1.jsonl", [{"event": "user_message", "content": "review"}])
    path = tmp_path / "session_20261111_120000_driver-t1.jsonl"
    _write(path, [{"event": "user_message", "content": "drive"}])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "tool_ca')

    read = read_conversation(tmp_path, "t1")
    assert [event["content"] for event in read["events"]] == ["drive"]


def test_no_transcript_is_said_so(tmp_path):
    assert read_conversation(tmp_path, "nope") == {"task_id": "nope", "found": False, "events": [], "truncated": False, "queued": []}


def test_a_long_thought_is_cut_for_display(tmp_path):
    _write(tmp_path / "session_x_subagent-t1.jsonl", [{"event": "thinking", "content": "x" * 50_000}])
    content = read_conversation(tmp_path, "t1")["events"][0]["content"]
    assert len(content) < 25_000 and content.endswith("cut for display.")


def test_a_time_without_a_zone_is_given_this_site_s(tmp_path):
    _write(tmp_path / "session_x_subagent-t1.jsonl", [
        {"ts": "2026-11-11T12:00:08.5", "event": "user_message"},
        {"ts": "2026-11-11T10:00:09+00:00", "event": "thinking"},
    ])
    first, second = (event["ts"] for event in read_conversation(tmp_path, "t1")["events"])
    assert datetime.fromisoformat(first).tzinfo is not None
    assert second == "2026-11-11T10:00:09+00:00"


def test_only_a_task_id_can_name_a_file():
    assert valid_task_id("05f31fcd")
    assert not valid_task_id("../web_sessions")
    assert not valid_task_id("a/b")
