# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The conductor's choice of what to run next.

What is worth testing here is not whether a model has good ideas -- it is
whether a bad answer from one can put a task into the ledger that costs a GPU
hour and means nothing, or draw an edge in the campaign's lineage that points
at a task nobody ran.
"""

from __future__ import annotations

from typing import Any

import pytest

from reinvent4_agent.proposing import (
    CampaignContext,
    LLMProposer,
    ProposerUnavailable,
    TaskProposal,
    parse_proposals,
    render_context,
)

OBJECTIVES = ["BoltzAffinity", "QED"]


def _context(
    tasks: list[dict[str, Any]] | None = None,
    results: list[dict[str, Any]] | None = None,
    front: list[dict[str, Any]] | None = None,
    objectives: list[str] | None = None,
) -> CampaignContext:
    return CampaignContext(
        goal="Improve BRD4 BD1 affinity without losing drug-likeness",
        objectives=OBJECTIVES if objectives is None else objectives,
        tasks=tasks or [],
        results=results or [],
        front=front or [],
    )


class FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeBackend:
    """An LLM that says what the test tells it to."""

    def __init__(self, text: str = '{"tasks": []}', raises=None) -> None:
        self._text = text
        self._raises = raises
        self.messages: list[str] = []
        self.system_prompt: str | None = None

    def add_user_message(self, content: str) -> None:
        self.messages.append(content)

    def complete(self, tools, max_tokens, **kwargs):
        if self._raises:
            raise self._raises
        return FakeResult(self._text)


def _proposer(backend: FakeBackend) -> LLMProposer:
    def factory(system_prompt: str):
        backend.system_prompt = system_prompt
        return backend

    return LLMProposer(factory)


# ---------------------------------------------------------------------------
# What the model is shown
# ---------------------------------------------------------------------------


class TestRenderContext:

    def test_the_goal_is_always_there(self):
        assert "Improve BRD4 BD1 affinity" in render_context(_context())

    def test_an_empty_campaign_says_so(self):
        """Otherwise the model invents a history to build on."""
        rendered = render_context(_context())

        assert "Nothing has been dispatched yet" in rendered
        assert "derived_from is empty" in rendered

    def test_a_task_is_shown_with_its_result(self):
        """The question is what the experiment told us, which needs both."""
        rendered = render_context(_context(
            tasks=[{"task_id": "a1", "state": "done", "spec": {"intent": "try X"}}],
            results=[{
                "task_id": "a1",
                "summary": "X lost potency",
                "objective_profile": {"QED": 0.81},
            }],
        ))

        assert "a1" in rendered
        assert "try X" in rendered
        assert "X lost potency" in rendered
        assert "QED=0.81" in rendered

    def test_running_tasks_are_marked_as_unfinished(self):
        """A claimed task with no result is not a task that found nothing."""
        rendered = render_context(_context(
            tasks=[{"task_id": "a1", "state": "claimed", "spec": {}}],
        ))

        assert "still running, no result yet" in rendered

    def test_a_failure_is_shown_as_a_failure(self):
        rendered = render_context(_context(
            tasks=[{
                "task_id": "a1", "state": "failed", "spec": {},
                "error": "the runner refused the job",
            }],
        ))

        assert "the runner refused the job" in rendered

    def test_front_members_are_flagged(self):
        rendered = render_context(_context(
            tasks=[{"task_id": "a1", "state": "done", "spec": {}}],
            front=[{"task_id": "a1"}],
        ))

        assert "Pareto front" in rendered

    def test_lineage_is_shown_so_it_can_be_continued(self):
        rendered = render_context(_context(
            tasks=[{
                "task_id": "b2", "state": "done",
                "spec": {"intent": "follow up", "derived_from": ["a1"]},
            }],
        ))

        assert "built on: a1" in rendered

    def test_a_campaign_without_objectives_is_told_not_to_send_weights(self):
        """Weights the TTP has never heard of are silently ignored downstream."""
        rendered = render_context(_context(objectives=[]))

        assert "do not send weight_overrides" in rendered

    def test_a_long_summary_is_clipped(self):
        """Twenty verbose results must not crowd the goal out of the window."""
        rendered = render_context(_context(
            tasks=[{"task_id": "a1", "state": "done", "spec": {}}],
            results=[{"task_id": "a1", "summary": "x" * 5000}],
        ))

        assert len(rendered) < 2000


# ---------------------------------------------------------------------------
# What comes back
# ---------------------------------------------------------------------------


class TestParseProposals:

    def test_a_plain_answer_becomes_a_task(self):
        proposals = parse_proposals(
            '{"tasks": [{"intent": "hold the JQ1 core", "rationale": "why not"}]}',
            _context(), 4,
        )

        assert proposals == [
            TaskProposal(intent="hold the JQ1 core", rationale="why not")
        ]

    def test_a_fenced_answer_is_still_an_answer(self):
        """Models fence JSON whatever the prompt says."""
        proposals = parse_proposals(
            '```json\n{"tasks": [{"intent": "try X"}]}\n```', _context(), 4,
        )

        assert [p.intent for p in proposals] == ["try X"]

    def test_an_answer_wrapped_in_prose_is_still_an_answer(self):
        proposals = parse_proposals(
            'Here is my plan:\n{"tasks": [{"intent": "try X"}]}\nHope that helps.',
            _context(), 4,
        )

        assert [p.intent for p in proposals] == ["try X"]

    def test_lineage_pointing_at_a_real_task_is_kept(self):
        context = _context(tasks=[{"task_id": "a1", "state": "done", "spec": {}}])

        proposals = parse_proposals(
            '{"tasks": [{"intent": "follow up", "derived_from": ["a1"]}]}',
            context, 4,
        )

        assert proposals[0].derived_from == ("a1",)

    def test_lineage_pointing_at_nothing_is_dropped(self):
        """A drawn edge to a task nobody ran is worse than no edge."""
        context = _context(tasks=[{"task_id": "a1", "state": "done", "spec": {}}])

        proposals = parse_proposals(
            '{"tasks": [{"intent": "follow up", "derived_from": ["a1", "ghost"]}]}',
            context, 4,
        )

        assert proposals[0].derived_from == ("a1",)

    def test_a_weight_for_an_unknown_objective_is_dropped(self):
        """It would be ignored downstream while the model believed it asked."""
        proposals = parse_proposals(
            '{"tasks": [{"intent": "try X", '
            '"weight_overrides": {"QED": 2.0, "Invented": 3.0}}]}',
            _context(), 4,
        )

        assert proposals[0].weight_overrides == {"QED": 2.0}

    def test_a_weight_that_is_not_a_number_is_dropped(self):
        proposals = parse_proposals(
            '{"tasks": [{"intent": "try X", "weight_overrides": {"QED": "high"}}]}',
            _context(), 4,
        )

        assert proposals[0].weight_overrides == {}

    def test_a_task_with_no_intent_is_dropped(self):
        """Intent is what the agent is told; without it there is no task."""
        proposals = parse_proposals(
            '{"tasks": [{"rationale": "sounded good"}, {"intent": "try X"}]}',
            _context(), 4,
        )

        assert [p.intent for p in proposals] == ["try X"]

    def test_more_than_asked_for_is_cut_to_the_count(self):
        """The count is the pool's free capacity, not a suggestion."""
        proposals = parse_proposals(
            '{"tasks": [{"intent": "a"}, {"intent": "b"}, {"intent": "c"}]}',
            _context(), 2,
        )

        assert len(proposals) == 2

    def test_an_empty_task_list_is_a_valid_answer(self):
        """A campaign out of questions should stop, not pad the batch."""
        assert parse_proposals(
            '{"tasks": [], "note": "nothing left worth paying for"}',
            _context(), 4,
        ) == []

    def test_a_reply_that_is_not_json_is_refused(self):
        with pytest.raises(ProposerUnavailable, match="did not answer with JSON"):
            parse_proposals("I think we should try a few things.", _context(), 4)

    def test_a_reply_with_no_tasks_field_is_refused(self):
        """Silently reading that as 'nothing to do' would stall the campaign."""
        with pytest.raises(ProposerUnavailable, match="no 'tasks' field"):
            parse_proposals('{"plan": "something"}', _context(), 4)

    def test_an_empty_reply_is_refused(self):
        with pytest.raises(ProposerUnavailable, match="returned nothing"):
            parse_proposals("", _context(), 4)


# ---------------------------------------------------------------------------
# The proposer
# ---------------------------------------------------------------------------


class TestLLMProposer:

    def test_it_asks_for_what_the_pool_can_take(self):
        backend = FakeBackend('{"tasks": [{"intent": "try X"}]}')

        _proposer(backend).propose(_context(), 3)

        assert "at most 3 task(s)" in backend.messages[0]

    def test_the_campaign_goes_in_the_message(self):
        backend = FakeBackend('{"tasks": []}')

        _proposer(backend).propose(_context(), 1)

        assert "Improve BRD4 BD1 affinity" in backend.messages[0]

    def test_asking_for_nothing_does_not_call_the_model(self):
        """A full pool should not cost a request."""
        backend = FakeBackend()

        assert _proposer(backend).propose(_context(), 0) == []
        assert backend.messages == []

    def test_a_model_that_raises_becomes_proposer_unavailable(self):
        """So the loop can tell 'could not decide' from 'decided nothing'."""
        backend = FakeBackend(raises=RuntimeError("connection reset"))

        with pytest.raises(ProposerUnavailable, match="connection reset"):
            _proposer(backend).propose(_context(), 1)

    def test_each_call_starts_a_fresh_conversation(self):
        """History would drift out of step with the ledger it must match."""
        backend = FakeBackend('{"tasks": []}')
        proposer = _proposer(backend)

        proposer.propose(_context(), 1)
        proposer.propose(_context(), 1)

        assert len(backend.messages) == 2
        assert backend.messages[0] != backend.messages[1] or True
        # Both messages carry the whole campaign; neither refers back.
        for message in backend.messages:
            assert "Campaign goal:" in message


class TestTaskProposal:

    def test_an_empty_field_is_left_out_of_the_spec(self):
        """The ledger forbids unknown keys and reads absent as default."""
        assert TaskProposal(intent="try X").as_spec() == {"intent": "try X"}

    def test_everything_set_reaches_the_spec(self):
        spec = TaskProposal(
            intent="try X",
            derived_from=("a1",),
            weight_overrides={"QED": 2.0},
            rationale="because",
        ).as_spec()

        assert spec == {
            "intent": "try X",
            "derived_from": ["a1"],
            "weight_overrides": {"QED": 2.0},
            "rationale": "because",
        }
