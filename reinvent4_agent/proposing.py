"""What the campaign should try next, decided by the conductor's own model.

This used to be a fixed sweep: seed one weight vector per objective, then
perturb whichever of them scored best, forever. It had the
virtue of being deterministic and the defect of being unable to have an idea.
Every task was a reweighting of the same objectives, so the only experiment
the campaign could run was "the same thing, but caring more about X" -- it
could not hold a chemotype, revisit a result it had walked past, combine what
two runs found, or say in words what a run was for, even though the
supervisor has always been able to act on exactly that.

So the decision is a model's now. The proposer is handed the goal, what has
been tried, and what came back, and answers with tasks in prose. The
structured fields remain available to it and are still the precise way to say
"weight QED at 2.0", but they are no longer the only thing it can say.

Two consequences worth stating plainly.

**A campaign is no longer replayable.** The old sweep promised that the same
ledger produced the same next proposal, so a restarted conductor did not
change course. A model does not promise that. What survives is the property
that actually mattered: nothing is decided from memory, so a restarted
conductor still reads the whole campaign out of the ledger before it decides,
and still cannot double-dispatch.

**A proposer that cannot decide dispatches nothing.** There is no mechanical
fallback, deliberately. A sweep running quietly in place of a model that was
unreachable would spend a GPU allocation measuring something nobody asked
for, and would look from outside exactly like a campaign that was working.
Dispatching nothing is visible, costs nothing, and is retried next tick.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: How much of a result summary the model is shown per task. Enough to see
#: what a run concluded, not so much that twenty of them crowd out the goal.
SUMMARY_CHARS = 600

PROPOSER_SYSTEM_PROMPT = """\
You direct a drug-design campaign. Several agents work for you in parallel; \
each one takes a task from you, decides how to run it, drives a REINVENT \
reinforcement-learning job, and reports the molecules it found.

You are given the campaign goal, every task dispatched so far, and what came \
back. You answer with the next tasks to run. You never run anything yourself.

What makes a good task:

* It is an experiment with an expected answer, not a direction. "Hold the \
JQ1 acetyl-lysine mimetic and vary the fused ring, to see whether potency \
survives losing the methyl-triazole" is a task. "Improve potency" is not.
* It is worth a GPU-hour. Runs are expensive and slow, so a task that \
repeats what a previous one already measured is waste, and a task whose \
result would not change what you do next is waste too.
* It builds on what you know. Say which earlier tasks informed it. Nothing \
at the start of a campaign has anything to build on, and that is fine.
* It plays to the agent's judgement. The agent can look compounds up and \
choose a generative prior; it cannot read your mind. Say what the run is \
for and let it decide how.

Spread the batch. Tasks you dispatch together run at the same time and cannot \
learn from each other, so proposing near-identical variants wastes the \
parallelism -- make them answer different questions.

Reply with JSON only, no prose around it, in this shape:

{"tasks": [{"intent": "...", "derived_from": ["task_id", ...],
            "weight_overrides": {"ObjectiveName": 2.0},
            "workstream": "workstream_id", "rationale": "..."}]}

`intent` is required and is handed to the agent verbatim; write it to be read \
by the agent, not by me. `derived_from` lists the task ids whose results led \
you here, and is empty when nothing did. `weight_overrides` is optional, uses \
only the campaign's own objective names, and is for when you want the \
objective weights changed from the campaign default -- omit it when the \
intent says everything. `workstream` names the plan's workstream the task \
serves, and is omitted when the campaign has no plan. `rationale` is one line for the human watching, \
explaining why this task and not another.

Propose at most the number asked for. Propose fewer, or none at all, if the \
campaign has run out of questions worth paying for -- say so in a `note` \
field alongside an empty task list rather than padding the batch.\
"""


@dataclass(frozen=True)
class TaskProposal:
    """One task the conductor has decided to run."""

    intent: str
    derived_from: tuple[str, ...] = ()
    weight_overrides: dict[str, float] = field(default_factory=dict)
    rationale: Optional[str] = None
    workstream: Optional[str] = None

    def as_spec(self) -> dict[str, Any]:
        """The task spec the ledger stores, without the empty fields."""

        spec: dict[str, Any] = {"intent": self.intent}
        if self.workstream:
            spec["workstream"] = self.workstream
        if self.derived_from:
            spec["derived_from"] = list(self.derived_from)
        if self.weight_overrides:
            spec["weight_overrides"] = dict(self.weight_overrides)
        if self.rationale:
            spec["rationale"] = self.rationale
        return spec


@dataclass
class CampaignContext:
    """Everything the proposer is allowed to decide from.

    Assembled by the conductor from one read of the ledger, so that what the
    model sees and what the loop acts on cannot disagree.
    """

    goal: str
    objectives: list[str]
    tasks: list[dict[str, Any]]
    results: list[dict[str, Any]]
    #: The campaign's plan, ``{"workstreams": [...]}``, when it has one.
    plan: Optional[dict[str, Any]] = None

    @property
    def task_ids(self) -> set[str]:
        return {str(task["task_id"]) for task in self.tasks if task.get("task_id")}

    @property
    def workstreams(self) -> list[dict[str, Any]]:
        return list((self.plan or {}).get("workstreams") or [])


@runtime_checkable
class Proposer(Protocol):
    """Decides the campaign's next tasks."""

    def propose(self, context: CampaignContext, count: int) -> list[TaskProposal]: ...


class ProposerUnavailable(Exception):
    """The proposer could not decide, and the campaign should wait."""


# ---------------------------------------------------------------------------
# Rendering the campaign for the model
# ---------------------------------------------------------------------------

def _result_for(task_id: str, results: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    for result in results:
        if result.get("task_id") == task_id:
            return result
    return None


def _clip(text: Any, limit: int = SUMMARY_CHARS) -> str:
    body = str(text or "").strip()
    return body if len(body) <= limit else body[: limit - 1] + "…"


def _render_plan(context: CampaignContext) -> list[str]:
    """The workstreams with how much each has had, so gaps are visible.

    Counted from the tasks rather than read from the store's coverage view,
    so the proposer and the ledger read it from the same snapshot.
    """

    counts: dict[str, dict[str, int]] = {}
    for task in context.tasks:
        stream = (task.get("spec") or {}).get("workstream")
        if stream:
            by_state = counts.setdefault(str(stream), {})
            state = str(task.get("state", "unknown"))
            by_state[state] = by_state.get(state, 0) + 1

    lines = ["The goal is split into these workstreams:"]
    for stream in context.workstreams:
        stream_id = str(stream.get("id"))
        status = stream.get("status") or "open"
        share = stream.get("share")
        seen = counts.get(stream_id, {})
        tally = ", ".join(f"{n} {state}" for state, n in sorted(seen.items())) or "no tasks yet"
        head = f"- {stream_id}: {stream.get('name') or stream_id} [{status}"
        head += f", share {share}]" if share is not None else "]"
        lines.append(f"{head} -- {tally}")
        if stream.get("question"):
            lines.append(f"    question: {_clip(stream['question'], 300)}")
    lines += [
        "Tag every task with the workstream it serves, and favour open "
        "workstreams that have had the least. A settled or dropped workstream "
        "gets no new tasks.",
        "",
    ]
    return lines


def render_context(context: CampaignContext) -> str:
    """The campaign as the model reads it.

    Tasks are rendered with their outcome attached rather than in separate
    sections, because the question the model is answering -- what did this
    experiment tell us, and what follows from it -- needs both at once.
    """

    lines = [f"Campaign goal: {context.goal}", ""]

    if context.objectives:
        lines += [
            "Objectives being scored: " + ", ".join(context.objectives),
            "These are the only names valid in weight_overrides.",
            "",
        ]
    else:
        lines += [
            "This campaign declares no fixed objective list, so do not send "
            "weight_overrides; say what you want in the intent.",
            "",
        ]

    if context.workstreams:
        lines += _render_plan(context)

    if not context.tasks:
        lines.append(
            "Nothing has been dispatched yet. These are the campaign's first "
            "tasks, so derived_from is empty for all of them, and between "
            "them they should establish what the easy wins and the hard "
            "constraints are rather than all chasing the same objective."
        )
        return "\n".join(lines)

    lines.append("What has been tried:")
    for task in context.tasks:
        task_id = str(task.get("task_id"))
        spec = task.get("spec") or {}
        state = task.get("state", "unknown")
        lines.append(f"- {task_id} [{state}]")
        if spec.get("intent"):
            lines.append(f"    asked: {_clip(spec['intent'], 300)}")
        if spec.get("weight_overrides"):
            weights = ", ".join(
                f"{k}={v}" for k, v in sorted(spec["weight_overrides"].items())
            )
            lines.append(f"    weights: {weights}")
        if spec.get("derived_from"):
            lines.append(f"    built on: {', '.join(spec['derived_from'])}")

        result = _result_for(task_id, context.results)
        if result:
            profile = result.get("objective_profile") or {}
            if profile:
                scores = ", ".join(
                    f"{k}={v:.3g}" if isinstance(v, (int, float)) else f"{k}={v}"
                    for k, v in sorted(profile.items())
                )
                lines.append(f"    scored: {scores}")
            if result.get("summary"):
                lines.append(f"    found: {_clip(result['summary'])}")
        elif state == "failed" and task.get("error"):
            lines.append(f"    failed: {_clip(task['error'], 300)}")
        elif state in {"queued", "claimed"}:
            lines.append("    still running, no result yet")

    lines += [
        "",
        "Tasks still running will report later. Do not propose a task whose "
        "only purpose is to wait for one of them, and do not re-ask a "
        "question one of them is already answering.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading the model's answer
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _json_object(text: str) -> dict[str, Any]:
    """The JSON object in a reply that may be wrapped in prose or a fence."""

    body = (text or "").strip()
    if not body:
        raise ProposerUnavailable("the proposer returned nothing")

    fenced = _FENCE.search(body)
    if fenced:
        body = fenced.group(1).strip()

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        # A model that explained itself before answering is still answering;
        # a model that produced no object at all is not.
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end <= start:
            raise ProposerUnavailable(
                f"the proposer did not answer with JSON: {_clip(body, 200)}"
            ) from None
        try:
            parsed = json.loads(body[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProposerUnavailable(
                f"the proposer's JSON would not parse: {exc}"
            ) from exc

    if not isinstance(parsed, dict):
        raise ProposerUnavailable("the proposer answered with a non-object")
    return parsed


def parse_proposals(
    text: str, context: CampaignContext, count: int
) -> list[TaskProposal]:
    """The model's reply as tasks, with anything unusable dropped.

    Lenient about shape and strict about meaning. A missing rationale is the
    model being terse; a `derived_from` naming a task that does not exist
    would put an edge in the campaign's lineage pointing at nothing, and an
    objective name the TTP has never heard of would be silently ignored
    downstream while the model believed it had asked for something. Those are
    dropped rather than honoured, and the reason is logged.
    """

    parsed = _json_object(text)
    raw = parsed.get("tasks")
    if raw is None:
        raise ProposerUnavailable("the proposer's answer has no 'tasks' field")
    if not isinstance(raw, list):
        raise ProposerUnavailable("the proposer's 'tasks' is not a list")

    if not raw and parsed.get("note"):
        logger.info("the proposer declined to dispatch: %s", _clip(parsed["note"], 300))

    known_tasks = context.task_ids
    known_objectives = set(context.objectives)
    known_streams = {str(stream.get("id")) for stream in context.workstreams}

    proposals: list[TaskProposal] = []
    for entry in raw:
        if len(proposals) >= count:
            logger.info("the proposer offered more than the %d asked for", count)
            break
        if not isinstance(entry, dict):
            logger.warning("skipping a proposed task that is not an object")
            continue

        intent = str(entry.get("intent") or "").strip()
        if not intent:
            logger.warning("skipping a proposed task with no intent")
            continue

        derived = []
        for ref in entry.get("derived_from") or []:
            if str(ref) in known_tasks:
                derived.append(str(ref))
            else:
                logger.warning(
                    "dropping derived_from '%s': no such task in this campaign", ref
                )

        weights: dict[str, float] = {}
        for name, value in (entry.get("weight_overrides") or {}).items():
            if not known_objectives:
                # Nothing to re-weight: the task writes its own TTP, so a
                # weight here would be dropped by the supervisor rather than
                # applied. Said plainly now, where it can be read, instead of
                # being discovered as a task that scored the wrong thing.
                logger.warning(
                    "dropping weight for '%s': this campaign has no base TTP, "
                    "so weights have nothing to apply to -- say it in the intent",
                    name,
                )
                continue
            if name not in known_objectives:
                logger.warning(
                    "dropping weight for '%s': not an objective of this campaign", name
                )
                continue
            try:
                weights[str(name)] = float(value)
            except (TypeError, ValueError):
                logger.warning("dropping weight for '%s': %r is not a number", name, value)

        stream = str(entry.get("workstream") or "").strip() or None
        if stream and stream not in known_streams:
            logger.warning(
                "dropping workstream '%s': not a workstream of this campaign", stream
            )
            stream = None

        rationale = str(entry.get("rationale") or "").strip() or None
        proposals.append(
            TaskProposal(
                intent=intent,
                derived_from=tuple(derived),
                weight_overrides=weights,
                rationale=rationale,
                workstream=stream,
            )
        )

    return proposals


# ---------------------------------------------------------------------------
# The proposer
# ---------------------------------------------------------------------------

class LLMProposer:
    """Asks a model what to run next.

    Stateless between calls on purpose. Each decision starts from a fresh
    conversation holding the whole campaign as the ledger currently has it,
    rather than from a history that would drift out of step with the ledger
    every time a task finished while the conductor was not looking.
    """

    def __init__(self, backend_factory: Any, *, max_tokens: int = 4096) -> None:
        """`backend_factory` takes a system prompt and returns an LLMBackend."""

        self._backend_factory = backend_factory
        self.max_tokens = max_tokens

    def propose(self, context: CampaignContext, count: int) -> list[TaskProposal]:
        if count <= 0:
            return []

        backend = self._backend_factory(PROPOSER_SYSTEM_PROMPT)
        backend.add_user_message(
            render_context(context)
            + f"\n\nPropose at most {count} task(s) to dispatch now."
        )

        try:
            result = backend.complete([], self.max_tokens)
        except Exception as exc:
            raise ProposerUnavailable(f"{type(exc).__name__}: {exc}") from exc

        proposals = parse_proposals(result.text or "", context, count)
        for proposal in proposals:
            logger.info(
                "proposed: %s%s",
                _clip(proposal.intent, 200),
                f" (from {', '.join(proposal.derived_from)})"
                if proposal.derived_from
                else "",
            )
        return proposals


def make_proposer(max_tokens: int = 4096) -> LLMProposer:
    """A proposer on the conductor's own model, configured from the environment.

    The same variables the agent reads, because the conductor is the agent:
    the model that talks to the human is the one that decides what the
    campaign tries, and a campaign silently directed by some other model
    would be a different system than the one the human thinks they are
    talking to.
    """

    from reinvent4_agent.backends import create_backend

    def factory(system_prompt: str):
        return create_backend(
            os.getenv("REINVENT4_AGENT_BACKEND", "anthropic"),
            model=os.getenv("REINVENT4_AGENT_MODEL", "claude-sonnet-4.6"),
            base_url=os.getenv("REINVENT4_AGENT_BASE_URL") or None,
            api_key=os.getenv("REINVENT4_AGENT_API_KEY") or None,
            gateway_key=os.getenv("LLM_GATEWAY_KEY") or None,
            api_version=os.getenv("REINVENT4_AGENT_API_VERSION") or None,
            verify_ssl=not os.getenv("REINVENT4_AGENT_NO_VERIFY_SSL"),
            system_prompt=system_prompt,
        )

    return LLMProposer(factory, max_tokens=max_tokens)
