# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Conversations that outlive the browser tab.

A session is one conversation with the agent.  It belongs to the gateway, not
to a WebSocket: a turn keeps running when the tab closes, and a tab that comes
back (or a new one) attaches to the session and receives every event it
missed.  Each session is a directory::

    <root>/<session id>/
        meta.json      title, timestamps, turn and token counts
        events.jsonl   every event sent to the browser, in order, numbered
        history.json   the LLM conversation, written after each turn
        campaign.json  the agent's campaign log (see campaign.py)

The events are what the UI replays; the history is what the agent needs to
carry on.  They are kept apart because they differ: tool results in the
history are the full text the model saw, while events carry an excerpt and
the panels derived from it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import secrets
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from reinvent4_agent.web import campaign as campaign_log
from reinvent4_agent.web.artifacts import extract_panels
from reinvent4_agent.web.events import WebSocketEventHandler
from reinvent4_agent.web.plaintext import plain

logger = logging.getLogger(__name__)

#: A campaign id as the ledger writes it: eight hex characters.
_CAMPAIGN_ID = re.compile(r'"campaign_id"\s*:\s*"([0-9a-f]{4,32})"')


def _campaign_id_in(result: str) -> Optional[str]:
    """The campaign a tool's answer is about, if it names one.

    Read out of the text rather than by re-parsing it as JSON: a tool result
    is whatever the tool chose to return, and a conversation's record of what
    it started should not depend on that staying a bare JSON object.
    """

    found = _CAMPAIGN_ID.search(result or "")
    return found.group(1) if found else None

#: Session ids are generated here and also arrive from the browser, where they
#: end up in a file path, so they are held to this shape.
_ID_PATTERN = re.compile(r"^s-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")

#: Characters of the first user message used as the default title.
_TITLE_CHARS = 80

#: "## Score: c1ccccc1" at the top of an answer, the agent's own name for a turn.
_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


#: Headings that name no subject, so never a title.
_EMPTY_HEADINGS = frozenset({"answer", "final answer", "the answer"})


def first_heading(markdown: str) -> str | None:
    """The first markdown heading of an answer, without its emphasis marks."""
    match = _HEADING_PATTERN.search(markdown or "")
    if match is None:
        return None
    text = re.sub(r"[*_`]", "", match.group(1)).strip()
    if text.rstrip(":").strip().lower() in _EMPTY_HEADINGS:
        return None
    return text or None

#: Idle sessions kept in memory.  Older ones are reloaded from disk on demand.
MAX_LIVE_SESSIONS = 20

#: Longest autonomy budget a browser may ask for, in minutes.
MAX_AUTONOMY_MINUTES = 480

#: Pause before looking again once the agent has said twice that it has
#: nothing left to do.  Long, because what is most likely to change is a run
#: or a sub-agent finishing, and the watcher wakes the loop the moment one
#: does.  It keeps looking rather than handing back: auto mode is the user
#: saying "carry on without me", so only they and the budget end it.
AUTO_IDLE_DELAY_S = 5 * 60.0

#: Pause before picking up a turn that failed, doubling to the second.  What
#: fails a turn is almost always a service or the endpoint, and those come
#: back; handing the campaign back instead would end the budget over a
#: minute's outage.
AUTO_RETRY_DELAY_S = 30.0
AUTO_RETRY_MAX_S = 10 * 60.0

#: Pause before carrying on when the agent is not waiting for any run: it is
#: in the middle of something, and the pause only keeps it from spinning.
AUTO_TURN_DELAY_S = 60.0

#: How often the gateway reads the state of the runs of conversations working
#: on their own.  Reading job metadata costs nothing; the model is what costs,
#: so it is woken only when something changed.
WATCH_INTERVAL_S = 30.0

#: Longest an agent goes without a turn while its runs are still going and
#: nothing has happened: long enough not to pay for "still running", short
#: enough that the time a run takes is spent on something.
AUTO_BACKSTOP_S = 15 * 60.0

#: Context, in input tokens, above which a turn starts from a brief and the
#: campaign log instead of re-reading the whole conversation.  Used only for
#: an endpoint that does not report its window.
COMPACT_ABOVE_TOKENS = 100_000

#: Share of a reported window a conversation may fill before the same
#: happens.  The rest is for the turn itself: a long tool result, the
#: model's reasoning and its answer all land on top of the history.
COMPACT_FRACTION = 0.7

#: Handing work to sub-agents and holding the pool that does it.  Withheld
#: below two sub-agents, where each of these is a queue nobody can take work
#: from: sub-agent mode exists only when the user has switched it on.
DISPATCH_TOOLS = frozenset({
    "enqueue_task",
    "enqueue_tasks",
    "conduct_campaign",
    "stop_conducting",
    "set_concurrency",
    "cancel_supervisor",
    "take_proposals",
    "drop_proposals",
})

#: Keeping the campaign itself: its goal, its plan, the experiments the agent
#: runs and what they found.  None of it needs a sub-agent, and withholding
#: it left a conversation working alone with no memory that outlives its
#: context, no report of what it had tried and no trajectory -- the user saw
#: a list of runs where a campaign of two agents showed a line of reasoning.
CAMPAIGN_TOOLS = frozenset({
    "start_campaign",
    "set_campaign_plan",
    "start_own_task",
    "finish_own_task",
    "close_campaign",
    "finalize_campaign",
})

#: Everything the fleet surface covers, for callers that mean all of it.
FLEET_TOOLS = DISPATCH_TOOLS | CAMPAIGN_TOOLS

#: What the conductor gives up once it has sub-agents.  Starting a run *is*
#: the task, and at two or more the tasks are theirs.  Left the choice, a
#: conductor whose campaign tool fails quietly runs the tracks itself and
#: reports them as the campaign's.
RUN_STARTERS = frozenset({
    "run_optimization",
    "run_transfer_learning",
    "run_generation_campaign",
    "create_generation_campaign",
})

#: Said to the model in place of each tool it may not call.
_DELEGATE_INSTEAD = (
    "The user allowed sub-agents, so runs are theirs: describe this one to "
    "enqueue_task and a sub-agent will start it."
)
_WORK_ALONE_INSTEAD = (
    "This conversation has no sub-agents, so there is nobody to dispatch "
    "to. Do the work yourself and record it with start_own_task and "
    "finish_own_task, or ask the user to allow sub-agents beside the "
    "message box."
)

#: Said on every turn of a conversation with no sub-agents. The campaign is
#: the memory: a conversation that kept none had nothing left once its
#: context was compacted past the work, and the user had a list of runs
#: where a campaign of two agents showed a line of reasoning.
_WORKING_ALONE = (
    "[Working alone. Keep every substantive turn in the campaign opened for "
    "this conversation. It is opened automatically, so do not call "
    "start_campaign and create a duplicate. Set out its workstreams with "
    "set_campaign_plan once its id is stated. Record "
    "each experiment you run "
    "with start_own_task before you start it and finish_own_task when it is "
    "done, with the molecules it put forward and the runs it started, and "
    "name in derived_from the earlier tasks it followed from. Write what "
    "you settle to the atlas with atlas_finding. This is what survives your "
    "context being compacted, and it is what the user reads in the Campaign "
    "tab. There is nobody to dispatch to, so do the work yourself.]"
)

#: Run states, as job metadata spells them.
ACTIVE_RUN_STATES = frozenset({"running", "queued"})
FINISHED_RUN_STATES = frozenset({"completed", "failed", "stopped"})

#: Alerts that are about a run ending; the run's own state says that first.
_END_ALERTS = frozenset({"job_completed", "job_failed"})

#: Every automatic turn opens with this.  It is not a user message: it is
#: never recorded as one, and the turn it opens is marked automatic.
AUTO_PREFIX = "[AUTO] You are running this campaign on your own; nobody is waiting for an answer."

#: What working on its own means, said on every automatic turn.
AUTO_POLICY = (
    "Decide the next step from the latest results and take it. When an "
    "experiment has finished, analyse it and run the experiment its results "
    "argue for. While a run is going, spend the time on work that does not "
    "compete with it -- score, cofold, profile or counter-screen the current "
    "best, or prepare the next TTP -- and start a second run only to test "
    "something different (other seeds, another chemotype, other weights). Treat "
    "a problem with a run as the [MONITOR] section describes. If you are "
    "conducting a campaign, no sub-agent should be sitting idle while the "
    "budget runs: read the reports that came in, and give every free "
    "sub-agent its next task. Record what you "
    "learned and the next step, with the result behind it, in campaign_log. "
    "Keep the reply to a few lines. End with AUTONOMY DONE only when you cannot "
    "name a result that justifies another step, or the next step needs something "
    "only the user can give. Next steps you could carry out yourself are not the "
    "user's call: choose the best one and start it. A run still going is never a "
    "reason to stop."
)

#: Why an automatic turn was taken, as the turn is told.
_WAKE_REASONS = {
    "start": (
        "The user has just left the campaign with you. Plan the whole campaign "
        "now, before you have results to react to: open it, set a plan with "
        "set_campaign_plan, and queue tasks for every sub-agent at once. The "
        "number you may hold is in the system note below. Queue a task per "
        "sub-agent in the first enqueue_tasks call -- do not queue only one "
        "and wait: a sub-agent not working is budget spent with nothing back."
    ),
    "continue": "Carry on from your last turn.",
    "confirm": (
        "You were about to hand the campaign back. Check that first: if any next "
        "step in your last answer or in the campaign log can be done with your "
        "tools -- fix a TTP, cofold, score or profile another compound, start or "
        "extend a run -- it is not the user's decision. Pick the one the latest "
        "results support most and start it now. Hand back with AUTONOMY DONE only "
        "if every remaining step needs something only the user can give: a target "
        "they have not named, data you cannot fetch, or a choice between their own "
        "goals."
    ),
    "idle": (
        "You have said twice that you have nothing left to do, and the budget is "
        "still running, so look again rather than wait. Read the campaign log, "
        "check every run and every sub-agent, and start the most useful thing you "
        "can do without the user -- a counter-screen, a profile, another seed, a "
        "task for a sub-agent that is free. Say so plainly if there is genuinely "
        "nothing; you will be asked again."
    ),
    "retry": (
        "Your last turn failed before it finished. Find out what state the work is "
        "actually in before redoing any of it -- a run you started may well be "
        "going -- then carry on."
    ),
}

#: Last answer and user requests a condensed history keeps, in characters.
_BRIEF_ANSWER_CHARS = 1500
_BRIEF_REQUEST_CHARS = 500
_BRIEF_REQUESTS = 6

#: How an answer says the autonomy loop has nothing left to do.
_DONE_PATTERN = re.compile(r"\bAUTONOMY\s+DONE\b", re.IGNORECASE)

#: The same phrase with the markdown and punctuation a model wraps it in: it
#: is addressed to the gateway, so it is cut out of what the browser shows.
_DONE_MARKUP = re.compile(r"\s*[*_#]*\bAUTONOMY\s+DONE\b[*_#.!]*", re.IGNORECASE)

#: Events that speak to whoever is watching now rather than to the
#: conversation, so they are neither numbered nor recorded.
_LIVE_ONLY = frozenset({"notice", "auto", "agents"})


def valid_session_id(session_id: str | None) -> bool:
    return bool(session_id) and _ID_PATTERN.match(session_id) is not None


def spoken_answer(answer: str | None) -> str:
    """The answer as the conversation keeps it, without the stop phrase.

    Also without LaTeX: the chat renders Markdown only, so ``$5.2 \\pm 0.3$``
    would otherwise reach the screen with its dollars and backslashes showing.
    """
    return plain(_DONE_MARKUP.sub("", answer or "").strip())


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"s-{stamp}-{secrets.token_hex(3)}"


def refresh_panels(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive the panels of recorded tool results that have none.

    Panels are derived when a result arrives, so a conversation recorded
    before the UI could show a kind of result (ADMET predictions, say) would
    never show it.  Results with panels are left as recorded.
    """
    arguments: dict[str, Any] = {}
    for event in events:
        kind = event.get("type")
        if kind == "tool_call":
            arguments[event.get("id")] = event.get("arguments")
        elif kind == "tool_result" and not event.get("panels") and not event.get("isError"):
            result = event.get("result")
            if isinstance(result, str):
                called = arguments.get(event.get("id"))
                event["panels"] = extract_panels(event.get("name", ""), result, called if isinstance(called, dict) else None)
    return events


def backfill_title(meta: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    """Rename a still-automatic title after the first answer's heading.

    Sessions recorded before titles came from the answer are named after the
    first line of the question, which is often a raw SMILES and tells the
    user nothing.  Returns whether the title changed, so the caller can save
    the metadata; clearing ``auto_title`` keeps it a one-off, and a title the
    user chose is never touched.
    """
    if "auto_title" in meta:
        if not meta["auto_title"]:
            return False
    elif meta.get("title", "") != _first_user_line(events):
        # Recorded before auto_title existed: only a title that still matches
        # the question verbatim can be assumed to be automatic.
        return False

    first_answer = next(
        (e for e in events if e.get("type") == "assistant_message"),
        None,
    )
    heading = first_heading(str((first_answer or {}).get("content") or ""))
    if not heading:
        return False
    meta["title"] = heading[:_TITLE_CHARS]
    meta["auto_title"] = False
    return True


def _first_user_line(events: list[dict[str, Any]]) -> str:
    """The title ``start_turn`` would have derived from the first question."""
    first = next((e for e in events if e.get("type") == "user_message"), None)
    content = str((first or {}).get("content") or "").strip()
    return content.splitlines()[0][:_TITLE_CHARS] if content else ""


class SessionStore:
    """Reads and writes session directories below *root*."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _dir(self, session_id: str) -> Path:
        if not valid_session_id(session_id):
            raise ValueError(f"Invalid session id: {session_id!r}")
        return self.root / session_id

    def exists(self, session_id: str) -> bool:
        return valid_session_id(session_id) and (self._dir(session_id) / "meta.json").is_file()

    def create(self, meta: dict[str, Any]) -> None:
        directory = self._dir(meta["id"])
        directory.mkdir(parents=True, exist_ok=False)
        self.save_meta(meta)

    def list_meta(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        metas = []
        for directory in self.root.iterdir():
            if not valid_session_id(directory.name):
                continue
            meta = self._read_json(directory / "meta.json")
            if isinstance(meta, dict):
                metas.append(meta)
        return sorted(metas, key=lambda m: m.get("updated", ""), reverse=True)

    def load_meta(self, session_id: str) -> dict[str, Any] | None:
        meta = self._read_json(self._dir(session_id) / "meta.json")
        return meta if isinstance(meta, dict) else None

    def save_meta(self, meta: dict[str, Any]) -> None:
        self._write_json(self._dir(meta["id"]) / "meta.json", meta)

    def load_history(self, session_id: str) -> list[dict[str, Any]]:
        history = self._read_json(self._dir(session_id) / "history.json")
        return history if isinstance(history, list) else []

    def save_history(self, session_id: str, history: list[dict[str, Any]]) -> None:
        self._write_json(self._dir(session_id) / "history.json", history)

    def load_campaign(self, session_id: str) -> dict[str, Any]:
        return campaign_log.normalise(self._read_json(self._dir(session_id) / "campaign.json"))

    def save_campaign(self, session_id: str, log: dict[str, Any]) -> None:
        self._write_json(self._dir(session_id) / "campaign.json", log)

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        path = self._dir(session_id) / "events.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        except OSError:
            logger.exception("Could not record an event of session %s", session_id)

    def load_events(self, session_id: str) -> list[dict[str, Any]]:
        path = self._dir(session_id) / "events.jsonl"
        events: list[dict[str, Any]] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    # A line cut short by a crash is skipped, not fatal.
                    with contextlib.suppress(ValueError):
                        events.append(json.loads(line))
        except FileNotFoundError:
            pass
        return refresh_panels(events)

    def delete(self, session_id: str) -> None:
        shutil.rmtree(self._dir(session_id), ignore_errors=True)

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        # Write then rename, so a crash never leaves half a file behind.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, default=str), encoding="utf-8")
        tmp.replace(path)


@dataclass
class LiveSession:
    """A session loaded in memory: its agent, its event log, its listeners."""

    meta: dict[str, Any]
    store: SessionStore
    events: list[dict[str, Any]] = field(default_factory=list)
    agent: Any = None
    handler: WebSocketEventHandler | None = None
    running: asyncio.Task[None] | None = None
    subscribers: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)
    #: End of the autonomy budget; None when the agent only answers requests.
    auto_until: datetime | None = None
    #: Sub-agents the user allows this conversation to run at once.  One means
    #: the agent works alone, which is the default; more switches campaign
    #: mode on.  A ceiling the user set, not a promise the scheduler made.
    agents_allowed: int = 1
    #: The continuation turn waiting out its pause, if one is queued.
    auto_task: asyncio.Task[None] | None = None
    #: Automatic turns taken since the last user message.
    auto_turns: int = 0
    #: Turns that have failed in a row, which is how long the retry waits.
    auto_failures: int = 0
    #: Pause before a continuation; read per session so tests can shorten it.
    auto_delay: float = field(default_factory=lambda: AUTO_TURN_DELAY_S)
    #: The longer pause once it has twice found nothing left to do.
    auto_idle_delay: float = field(default_factory=lambda: AUTO_IDLE_DELAY_S)
    #: Asks the gateway what this conversation's runs are doing.  None where
    #: nothing watches runs (the tests, the CLI), and then the agent is simply
    #: never waiting for one.
    observe: Callable[[LiveSession], Awaitable[None]] | None = None
    #: Opens or restores the one-agent campaign associated with this session.
    #: Kept injectable so local and CLI sessions do not depend on the ledger.
    bootstrap_campaign: Callable[[str, str], Awaitable[str | None]] | None = None
    campaign_bootstrap_task: asyncio.Task[str | None] | None = None
    #: Last state seen of each run this conversation launched.
    runs_seen: dict[str, str] = field(default_factory=dict)
    #: Problems already reported, so a plateau is news once.
    alerts_seen: set[tuple[str, str]] = field(default_factory=set)
    #: What happened since the agent last looked, for its next turn to say.
    pending: list[str] = field(default_factory=list)
    #: Messages the user sent while a turn ran: read after its current step,
    #: or answered by the next turn if it ends first.
    waiting: list[str] = field(default_factory=list)
    #: Campaigns whose ready queue was under target at the last look.
    queue_low: set[str] = field(default_factory=set)
    #: Campaigns left with idle sub-agents and nothing queued, already told.
    stalled_told: set[str] = field(default_factory=set)
    #: When the agent last finished a turn, on the monotonic clock: the start
    #: of the quiet spell a long wait is measured from.
    last_turn: float = field(default_factory=time.monotonic)
    #: The agent has said it is done once since it last did any work; the
    #: first time is checked with a confirm turn, the second time is final.
    done_once: bool = False
    #: The agent's campaign log.
    campaign: dict[str, Any] = field(default_factory=campaign_log.empty)
    #: Fleet campaigns this conversation owns, and the last ledger event read
    #: of each.  A sub-agent reports into the ledger, not into this
    #: conversation, so without reading it the conductor would never hear
    #: back from the work it dispatched.
    campaigns_seen: dict[str, int] = field(default_factory=dict)
    #: Those of them the ledger last said were still open.
    campaigns_open: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.campaign = self.store.load_campaign(self.id)
        self.install_tools()

    @property
    def id(self) -> str:
        return self.meta["id"]

    def install_tools(self) -> None:
        """Give the agent the tools that belong to this conversation.

        Called again whenever an agent is attached, because the tools belong
        to the conversation and the agent is the replaceable part. Nothing to
        do before there is one -- which is a session not yet started, not an
        agent that cannot hold tools. It used to be written as a getattr
        against the agent, and since Agent had no such attribute the tool was
        never installed anywhere and every call of it answered "unknown tool".
        """

        if self.agent is None:
            return
        self.agent.local_tools[campaign_log.TOOL_NAME] = (
            campaign_log.TOOL_DEFINITION, self._campaign_tool,
        )
        self.agent.inbox = self._take_waiting
        # What the agent may do itself is part of this conversation too, and
        # an agent attached after the user moved the slider would otherwise
        # start with the catalog of a conversation working alone.
        self.apply_allowance()

    #: Tools whose answer names a campaign this conversation now owns.
    #: conduct_campaign as well as start_campaign, because a conversation
    #: that picked up a campaign opened elsewhere and began conducting it is
    #: the one driving it now.
    CAMPAIGN_TOOLS = ("start_campaign", "conduct_campaign")

    def note_tool_result(self, name: str, result: str, is_error: bool) -> None:
        """Watch the conversation's own tool calls for things only it can know.

        The campaign tools run inside the MCP server, which is one process
        behind one pipe for every conversation at once and therefore cannot
        tell which one called it. This end can, so the association is made
        here and written to the ledger, where it outlives the browser.
        """

        if is_error or name not in self.CAMPAIGN_TOOLS:
            return
        campaign_id = _campaign_id_in(result)
        if campaign_id:
            self.campaigns_seen.setdefault(campaign_id, 0)
            self.campaigns_open.add(campaign_id)
            self._claim_campaign(campaign_id)

    def _claim_campaign(self, campaign_id: str) -> None:
        from reinvent4_agent.campaign_client import (
            CampaignStoreClient,
            StoreUnavailable,
        )

        try:
            CampaignStoreClient().attach_session(campaign_id, self.id)
        except StoreUnavailable as exc:
            # The campaign is running either way; all that is lost is this
            # conversation's claim on it, which the next call re-attempts.
            logger.warning(
                "could not record campaign %s against session %s: %s",
                campaign_id, self.id, exc,
            )
        except Exception:
            logger.exception("could not record campaign %s", campaign_id)

    def _campaign_tool(self, arguments: dict[str, Any]) -> str:
        self.campaign = campaign_log.apply(self.campaign, arguments)
        self.store.save_campaign(self.id, self.campaign)
        self.publish({"type": "campaign", "campaign": self.campaign})
        return campaign_log.as_tool_result(self.campaign)

    @property
    def busy(self) -> bool:
        return self.running is not None and not self.running.done()

    @property
    def last_seq(self) -> int:
        return self.events[-1]["seq"] if self.events else 0

    def publish(self, event: dict[str, Any]) -> None:
        """Number, record and fan out one event.

        A notice, and the state of the autonomy budget, speak to whoever is
        watching now rather than to the conversation, so they are neither
        numbered nor recorded: replayed on a reconnect they would announce a
        retry that finished long ago, or a budget that has run out.  A
        browser that comes back reads the budget from ``summary()``.
        """
        if event.get("type") in _LIVE_ONLY:
            for queue in list(self.subscribers):
                queue.put_nowait(event)
            return
        event = {**event, "seq": self.last_seq + 1, "ts": _now()}
        self.events.append(event)
        self.store.append_event(self.id, event)
        for queue in list(self.subscribers):
            queue.put_nowait(event)

    def events_after(self, seq: int) -> list[dict[str, Any]]:
        return [event for event in self.events if event["seq"] > seq]

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.subscribers.discard(queue)

    def start_turn(self, content: str, effort: str | None = None) -> bool:
        """Start a turn, or hand *content* to the one already running.

        *effort* sets the reasoning effort of this turn and the ones after
        it, on backends that have one.
        """
        backend = getattr(self.agent, "backend", None)
        if effort and hasattr(backend, "reasoning_effort"):
            backend.reasoning_effort = effort
        effort_field = {"effort": effort} if effort else {}
        if self.busy:
            # Refusing it lost the message: the browser has cleared its box by
            # the time the refusal arrives, and with sub-agents reporting in,
            # the conductor is mid-turn most of the time.
            self.waiting.append(content)
            self.publish({"type": "user_message", "content": content, "interjected": True, **effort_field})
            return True
        sent = self._lead(content)
        if not self.meta.get("title"):
            first_line = content.strip().splitlines()[0] if content.strip() else ""
            self.meta["title"] = first_line[:_TITLE_CHARS]
            self.meta["auto_title"] = True
        self.publish({"type": "user_message", "content": content, **effort_field})
        if campaign_log.is_empty(self.campaign) and content.strip():
            # Always give the overview a durable goal. Findings and next steps
            # stay with the agent because they require scientific judgement.
            self.campaign = campaign_log.apply(self.campaign, {"goal": content})
            self.store.save_campaign(self.id, self.campaign)
            self.publish({"type": "campaign", "campaign": self.campaign})
        self._begin(sent)
        return True

    def _lead(self, content: str) -> str:
        """What the agent is sent for the user's *content*, the user now leading."""
        # The allowance rides with the turn rather than the system prompt: the
        # user can move the slider between turns, and the agent has to be told
        # the number in force for this one.  Published without it, so the
        # user's own message reads back as they typed it.
        sent = self._with_allowance(content)
        # A conversation the user drives can outgrow the context as surely as
        # one the agent drives, and it is the user's next question that would
        # be refused.
        brief = self._condense()
        if brief:
            sent = f"{brief}\n\n{sent}"
        # The user leads: whatever the agent was about to do on its own is dropped.
        self.cancel_continuation()
        self.auto_turns = self.auto_failures = 0
        return sent

    def _take_waiting(self) -> list[str]:
        """The agent's inbox: what the user sent since it last looked."""
        taken, self.waiting = self.waiting, []
        return taken

    def cancel(self) -> bool:
        """Stop the running turn and any continuation.  False when none runs.

        Stop means stop: the budget goes with the turn, or the next
        continuation would start the agent up again a minute later.
        """
        self.end_autonomy()
        if not self.busy:
            return False
        assert self.running is not None
        self.running.cancel()
        return True

    # -- sub-agents ---------------------------------------------------------

    def set_agents(self, count: int) -> int:
        """Allow this conversation up to *count* sub-agents at once.

        Nothing is started here.  One means the agent works alone and the
        fleet tools are withheld; two or more makes the conversation a
        campaign and takes the run starters away instead.  Published so
        every tab watching the conversation shows the same setting.
        """

        self.agents_allowed = max(1, int(count))
        self.apply_allowance()
        self.publish({"type": "agents", "count": self.agents_allowed})
        return self.agents_allowed

    def apply_allowance(self) -> None:
        """Withhold the tools this allowance does not permit.

        Said to the model as well, on every turn, but saying it is not
        enough: told it may spend sub-agents, it reliably decides the work
        is linear and keeps it.  Withholding is what makes delegation the
        default rather than a preference.
        """

        if self.agent is None:
            return
        if self.agents_allowed > 1:
            self.agent.withheld = {name: _DELEGATE_INSTEAD for name in RUN_STARTERS}
        else:
            # Only what needs somebody to dispatch to. The campaign itself --
            # its goal, its plan, the experiments the agent runs and what
            # they found -- is what gives a conversation working alone the
            # memory, the reports and the trajectory a campaign of two has.
            self.agent.withheld = {name: _WORK_ALONE_INSTEAD for name in DISPATCH_TOOLS}

    def _with_allowance(self, content: str) -> str:
        """*content* with how the work is to be divided appended.

        At one there is nobody to divide it with, so what is said is how to
        keep the campaign rather than how to dispatch: a line offering one
        sub-agent would read as a standing instruction to run campaigns.
        """

        if not content.strip():
            return content
        if self.agents_allowed <= 1:
            return f"{content}\n\n{_WORKING_ALONE}"
        n = self.agents_allowed
        # half as many again, rounded up, is the target queue depth
        target_q = (n * 3 + 1) // 2
        return (
            f"{content}\n\n"
            f"[Team size: {n} sub-agent{'s' if n > 1 else ''} at once. "
            "Open a campaign with start_campaign, split its goal into "
            "workstreams with set_campaign_plan, hold the pool with "
            "conduct_campaign. First turn: queue a task for every "
            f"sub-agent in one enqueue_tasks call ({n} tasks), immediately — "
            "do not queue one and wait for it to finish. Keep the queue at "
            f"about {target_q} tasks ({n} working + backlog) so a sub-agent "
            "that finishes starts the next task at once. Use get_campaign_coverage "
            "to find workstreams nobody is working on, and atlas_overview to "
            "see what each one has settled -- the sub-agents write their "
            "findings there, and it is still there after your own context "
            "has been compacted past the report. Where a hypothesis is "
            "open give the alternatives to different sub-agents. With more "
            "workstreams than you can follow, use a 'lead' task with its own "
            "concurrency. Anything that takes minutes or longer is theirs "
            "(a design run, a deep search, a screen); run starters are "
            "withheld from you. If a campaign tool fails, stop and say so.]"
        )

    # -- working on its own ------------------------------------------------

    @property
    def auto_on(self) -> bool:
        """Whether the autonomy budget still has time left."""
        return self.auto_until is not None and self.auto_until > datetime.now(timezone.utc)

    def set_autonomy(self, minutes: float) -> datetime | None:
        """Start or extend the autonomy budget; 0 minutes stops it.

        The deadline is absolute, so a browser that reconnects reads the time
        left from ``summary()`` instead of counting from its own clock.
        """
        if minutes > 0:
            starting = not self.auto_on
            self.auto_until = datetime.now(timezone.utc) + timedelta(minutes=min(float(minutes), float(MAX_AUTONOMY_MINUTES)))
            self.auto_turns = self.auto_failures = 0
            if starting:
                # What happened while nobody was watching is not news: the
                # first look sets what counts as unchanged.
                self.runs_seen.clear()
                self.alerts_seen.clear()
                self.pending.clear()
                self.done_once = False
            # Turned on between two turns: pick the campaign up at once rather
            # than waiting for a message the user does not intend to send.
            if not self.busy and self.meta.get("turns"):
                self._schedule_continuation(0.0, reason="start")
        else:
            self.auto_until = None
            self.cancel_continuation()
        self.publish({"type": "auto", "until": self._auto_iso()})
        return self.auto_until

    def cancel_continuation(self) -> None:
        """Drop a continuation that has not started yet."""
        if self.auto_task is not None:
            self.auto_task.cancel()
            self.auto_task = None

    def end_autonomy(self) -> None:
        """Give the budget back once the loop stops taking turns.

        The browser counts the budget down from the deadline, so a loop that
        has stopped -- the user hit stop, or the budget ran out -- must say
        so, or the chip keeps promising work that nobody is doing.  Those
        are the only two ways it stops: see ``_run_turn``.
        """
        running = self.auto_until is not None
        self.auto_until = None
        self.cancel_continuation()
        if running:
            self.publish({"type": "auto", "until": None})

    def _auto_iso(self) -> str | None:
        return self.auto_until.isoformat(timespec="seconds") if self.auto_on else None

    def _schedule_continuation(self, delay: float, reason: str = "continue") -> None:
        """Queue the next automatic turn.  The task checks again before it starts."""
        if not self.auto_on:
            # Only here does a budget that simply ran out become visible; the
            # browser would otherwise count down to zero and keep the chip.
            if self.auto_until is not None:
                self.end_autonomy()
            return
        self.cancel_continuation()
        self.auto_task = asyncio.create_task(self._continue_after(delay, reason))

    async def _continue_after(self, delay: float, reason: str = "continue") -> None:
        """Wait out *delay*, then take another turn unless it should wait for a run.

        A turn that left runs going and nothing new to look at hands over to
        the watcher, which wakes the agent when one of them has news.  The
        user handing over the campaign is a reason on its own.
        """
        if delay > 0:
            await asyncio.sleep(delay)
        if self.observe is not None:
            try:
                await self.observe(self)
            except Exception:
                logger.exception("Could not read the runs of session %s", self.id)
        self.auto_task = None
        if self.busy or not self.auto_on:
            return
        if reason not in ("start", "confirm", "retry") and self.running_runs and not self.pending:
            return
        self._wake(reason)

    # -- what the watcher sees -------------------------------------------------

    @property
    def running_runs(self) -> list[str]:
        """Runs this conversation launched that are queued or running."""
        return [job_id for job_id, state in self.runs_seen.items() if state in ACTIVE_RUN_STATES]

    def note_runs(self, states: dict[str, str]) -> None:
        """Take the latest state of this conversation's runs; a run that ended is news."""
        for job_id, state in states.items():
            before = self.runs_seen.get(job_id)
            if before in ACTIVE_RUN_STATES and state in FINISHED_RUN_STATES:
                self.pending.append(f"Run {job_id} {state}.")
            self.runs_seen[job_id] = state

    def note_campaign(self, campaign_id: str, news: dict[str, Any]) -> None:
        """Take what a campaign's sub-agents have done since the last look.

        A report is news the conductor has to act on: it is the only thing a
        sub-agent sends back, and the slot that sub-agent was holding is now
        free. The two ways a campaign quietly stalls are news as well, since
        the budget runs either way: a pool with nothing left to do, and work
        with no pool to do it.
        """

        for event in news.get("events") or []:
            kind = event.get("kind")
            payload = event.get("payload") or {}
            what = payload.get("title") or payload.get("task_kind") or "task"
            if kind == "task_done":
                summary = str(payload.get("summary") or "").strip()
                proposed = payload.get("n_follow_ups") or 0
                self.pending.append(
                    f"Sub-agent reported on {what} (task {event.get('task_id')})"
                    + (f": {_clip(summary, 300)}" if summary else "")
                    + ". Read it with get_task_reports, and give the sub-agent it "
                    "freed its next task."
                    + (
                        f" It proposed {proposed} next task(s): see list_proposals, "
                        "and take_proposals the ones worth running."
                        if proposed else ""
                    )
                )
            elif kind in ("task_failed", "task_abandoned"):
                reason = str(payload.get("error") or kind).strip()
                self.pending.append(
                    f"Task {event.get('task_id')} ({what}) did not finish: "
                    f"{_clip(reason, 200)}"
                )

        idle = news.get("idle")
        queued = news.get("queued")
        # Explicitly none alive, rather than a reader that did not say: a
        # missing count must not be reported as an empty pool.
        if queued and news.get("alive") == 0:
            # Tasks nobody can pick up. The pool rests at zero and is scaled
            # up for a campaign, so this is what a scale-up that did not
            # happen looks like from here -- silent, with the budget running.
            self.pending.append(
                f"{queued} task(s) are queued on campaign {campaign_id} and no "
                "sub-agent is running to take them. Check the campaign with "
                "get_campaign_status before queueing more."
            )
        # Told once while it lasts, like the backlog below: the conductor that
        # leaves a sub-agent idle is usually waiting on another one, and on
        # every look this woke it every half minute to say so again. The
        # quiet backstop still brings it back to a campaign that stays stalled.
        stalled = bool(idle) and not queued and not news.get("events")
        if stalled and campaign_id not in self.stalled_told:
            self.pending.append(
                f"{idle} sub-agent(s) are idle on campaign {campaign_id} with "
                "nothing queued. Give them work or close the campaign."
            )
        if stalled:
            self.stalled_told.add(campaign_id)
        else:
            self.stalled_told.discard(campaign_id)

        # Told once as it runs low rather than on every look, or a queue kept
        # just under target would wake the conductor every half minute.
        backlog = news.get("backlog")
        low = (
            isinstance(backlog, int) and isinstance(queued, int)
            and queued < backlog and news.get("open") is not False
        )
        if low and campaign_id not in self.queue_low and not idle:
            self.pending.append(
                f"Only {queued} task(s) are queued on campaign {campaign_id}, "
                f"under the {backlog} to keep ready. Queue the next ones with "
                "enqueue_tasks, so no sub-agent waits when it finishes."
            )
        if low:
            self.queue_low.add(campaign_id)
        else:
            self.queue_low.discard(campaign_id)

        last = news.get("last_event_id")
        if isinstance(last, int):
            self.campaigns_seen[campaign_id] = last
        if news.get("open") is False:
            self.campaigns_open.discard(campaign_id)

    def note_alerts(self, alerts: list[dict[str, Any]]) -> None:
        """Take the problems reported for running jobs; each is news once."""
        running = set(self.running_runs)
        for alert in alerts:
            job_id = str(alert.get("job_id") or "")
            kind = str(alert.get("type") or "")
            if job_id not in running or kind in _END_ALERTS or (job_id, kind) in self.alerts_seen:
                continue
            self.alerts_seen.add((job_id, kind))
            message = str(alert.get("message") or kind).strip()
            self.pending.append(f"Run {job_id} ({alert.get('severity') or 'medium'}): {message}")

    @property
    def conducting(self) -> bool:
        """Whether a campaign this conversation opened is still open."""
        return bool(self.campaigns_open)

    def wake_if_due(self) -> None:
        """Take a turn if something happened, or if the runs have been quiet too long.

        Without autonomy, only while a campaign of this conversation is open
        and only on news: a sub-agent that has reported is idle until the
        conductor reads it, and nobody else will give it its next task. The
        turn is one; with autonomy off nothing follows it.

        News cuts a pause short: after an answer that said there was nothing
        left to do the next look is minutes away, and a sub-agent that
        reports meanwhile would sit idle for all of them.
        """
        if self.busy:
            return
        if self.auto_task is not None:
            if not self.pending:
                return
            self.cancel_continuation()
        if not self.auto_on:
            if self.conducting and self.pending:
                self._wake("events")
            return
        if self.pending:
            self._wake("events")
        elif self.running_runs and time.monotonic() - self.last_turn >= AUTO_BACKSTOP_S:
            self._wake("quiet")
        elif self.campaigns_seen and time.monotonic() - self.last_turn >= AUTO_BACKSTOP_S:
            # A campaign whose sub-agents are all mid-task launches no runs of
            # the conversation's own, so `running_runs` is empty and the
            # backstop above never fires -- the conductor would sit out the
            # whole campaign without looking at it once.
            self._wake("quiet")

    def _wake(self, reason: str) -> None:
        self.auto_turns += 1
        events, self.pending = self.pending, []
        brief = self._condense()
        self._begin(self._wake_prompt(reason, events, brief), auto=True)

    def _wake_prompt(self, reason: str, events: list[str], brief: str) -> str:
        parts = [AUTO_PREFIX]
        if brief:
            parts.append(brief)
        if events:
            parts.append("Since you last looked:\n" + "\n".join(f"- {event}" for event in events))
        elif reason == "quiet":
            minutes = round(AUTO_BACKSTOP_S / 60)
            parts.append(
                f"Runs still going: {', '.join(self.running_runs)}. Nothing has changed "
                f"for {minutes} minutes; use the time."
            )
        else:
            parts.append(_WAKE_REASONS.get(reason, _WAKE_REASONS["continue"]))
        parts.append(
            campaign_log.render(self.campaign)
            or "You have no campaign log yet: set the goal with campaign_log before anything else."
        )
        parts.append(AUTO_POLICY)
        return "\n\n".join(parts)

    def _compact_above(self) -> int:
        """Context, in tokens, past which the next turn starts from a brief.

        Sized to what the endpoint actually serves, which differs by an
        order of magnitude between deployments of the same model.
        """
        window = getattr(self.agent, "context_window", None)
        return int(window * COMPACT_FRACTION) if window else COMPACT_ABOVE_TOKENS

    def _condense(self) -> str:
        """Start the agent afresh from a brief once its context has grown too long.

        Returns the brief, which the turn then opens with, or "" when the
        history is short enough to keep.  The browser's record of the
        conversation is untouched: only what the model re-reads shrinks.
        """
        tokens = getattr(self.agent, "context_tokens", 0) or 0
        reset = getattr(self.agent, "reset_history", None)
        # Asked in this order because reading the window can cost a call to
        # the endpoint, and a conversation that has not had a completion yet
        # is under any threshold there could be.
        if not tokens or not callable(reset) or tokens <= self._compact_above():
            return ""
        requests = [e.get("content", "") for e in self.events if e.get("type") == "user_message"]
        answers = [e.get("content", "") for e in self.events if e.get("type") == "assistant_message"]
        lines = [
            "Your earlier conversation was condensed to keep turns fast; what matters "
            "of it is below and in the campaign log. Look up anything else with your "
            "tools (list_jobs, get_optimization_status, list_ttps, list_shared_assets).",
            "What the user asked, oldest first:",
            *(f"- {_clip(text, _BRIEF_REQUEST_CHARS)}" for text in requests[-_BRIEF_REQUESTS:]),
        ]
        if self.runs_seen:
            lines.append("Runs of this conversation: " + ", ".join(f"{job} ({state})" for job, state in self.runs_seen.items()))
        if answers:
            lines.append("Your last answer:\n" + _clip(answers[-1], _BRIEF_ANSWER_CHARS))
        reset()
        self.agent.context_tokens = 0
        logger.info("Session %s: condensed a %d-token history before the next turn", self.id, tokens)
        return "\n".join(lines)

    def _begin(self, content: str, auto: bool = False) -> None:
        """Record the start of a turn and run it."""
        self.publish({"type": "status", "state": "busy", **({"auto": True} if auto else {})})
        self.meta["turns"] = sum(1 for event in self.events if event.get("type") == "user_message")
        self.meta["updated"] = _now()
        self.store.save_meta(self.meta)
        self.running = asyncio.create_task(self._run_turn(content))

    async def _run_turn(self, content: str) -> None:
        unfinished = done = failed = False
        try:
            content = await self._with_solo_campaign(content)
            answer = await self.agent.chat(content)
            spoken = spoken_answer(answer)
            self._title_from_answer(spoken)
            self.publish(
                {
                    "type": "assistant_message",
                    "content": spoken,
                    "inputTokens": getattr(self.agent, "total_input_tokens", 0),
                    "outputTokens": getattr(self.agent, "total_output_tokens", 0),
                }
            )
            done = _DONE_PATTERN.search(answer or "") is not None
            unfinished = not done
        except asyncio.CancelledError:
            self.publish({"type": "cancelled"})
        except Exception as exc:
            failed = True
            logger.exception("Agent turn failed in session %s", self.id)
            self.publish({"type": "error", "message": str(exc)})
        finally:
            self.last_turn = time.monotonic()
            self.publish({"type": "status", "state": "idle"})
            self.save()
            # Auto mode ends two ways and no others: the user stops it, or
            # the budget they set runs out. Nothing here hands it back -- not
            # a failed turn, and not the agent saying it is finished, which
            # it says while runs it started are still going.
            if self.waiting:
                # Sent after the agent's last look, so this turn never read
                # it. Answered after a Stop too: Stop ends the work in hand,
                # not what the user asked for next.
                self._begin(self._lead("\n\n".join(self._take_waiting())))
            elif failed:
                self.auto_failures += 1
                self._schedule_continuation(self._retry_delay(), reason="retry")
            elif unfinished:
                self.auto_failures = 0
                self.done_once = False
                self._schedule_continuation(self.auto_delay)
            elif done:
                self.auto_failures = 0
                checked, self.done_once = self.done_once, True
                self._schedule_continuation(
                    self.auto_idle_delay if checked else self.auto_delay,
                    reason="idle" if checked else "confirm",
                )

    async def _with_solo_campaign(self, content: str) -> str:
        """Ensure a one-agent conversation has a durable campaign ledger."""

        if self.agents_allowed > 1:
            return content
        campaign_id = next(iter(self.campaigns_open or self.campaigns_seen), None)
        if campaign_id is None and self.bootstrap_campaign is not None:
            if self.campaign_bootstrap_task is None:
                self.campaign_bootstrap_task = asyncio.create_task(
                    self.bootstrap_campaign(
                        self.id,
                        str(self.campaign.get("goal") or self.meta.get("title") or "Campaign"),
                    )
                )
                self.campaign_bootstrap_task.add_done_callback(
                    self._campaign_bootstrapped
                )
            return (
                f"{content}\n\n[The campaign for this conversation is being "
                "opened automatically. Do not call start_campaign and create "
                "a duplicate. Keep immediate memory with campaign_log; its id "
                "will be stated on the next turn.]"
            )
        if campaign_id is None:
            return content
        self.campaigns_seen.setdefault(campaign_id, 0)
        self.campaigns_open.add(campaign_id)
        return (
            f"{content}\n\n[The campaign for this conversation is already open: "
            f"{campaign_id}. Use this id for set_campaign_plan, start_own_task "
            "and atlas_finding; do not open a duplicate.]"
        )

    def _campaign_bootstrapped(
        self, task: asyncio.Task[str | None]
    ) -> None:
        """Remember a campaign opened beside the model turn."""

        self.campaign_bootstrap_task = None
        try:
            campaign_id = task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning(
                "could not open the single-agent campaign for session %s: %s",
                self.id,
                exc,
            )
            return
        if campaign_id:
            self.campaigns_seen.setdefault(campaign_id, 0)
            self.campaigns_open.add(campaign_id)

    def _retry_delay(self) -> float:
        """Wait before picking a failed turn up, longer each time it fails."""
        return min(AUTO_RETRY_DELAY_S * 2 ** (self.auto_failures - 1), AUTO_RETRY_MAX_S)

    def save(self) -> None:
        """Write the metadata and the LLM history."""
        self.meta["updated"] = _now()
        self.meta["turns"] = sum(1 for event in self.events if event.get("type") == "user_message")
        if self.agent is not None:
            self.meta["inputTokens"] = getattr(self.agent, "total_input_tokens", 0)
            self.meta["outputTokens"] = getattr(self.agent, "total_output_tokens", 0)
            try:
                self.store.save_history(self.id, self.agent.backend.export_history())
            except Exception:
                logger.exception("Could not save the history of session %s", self.id)
        self.store.save_meta(self.meta)

    def _title_from_answer(self, answer: str) -> None:
        """Name a session after the first answer's heading, which beats a raw SMILES."""
        if not self.meta.get("auto_title"):
            return
        heading = first_heading(answer)
        if heading:
            self.meta["title"] = heading[:_TITLE_CHARS]
            self.meta["auto_title"] = False

    def summary(self) -> dict[str, Any]:
        return {
            **self.meta,
            "busy": self.busy,
            "auto_until": self._auto_iso(),
            "agents_allowed": self.agents_allowed,
        }


#: Builds an agent whose events go to the given sink.
AgentFactory = Callable[[WebSocketEventHandler], Any]

#: ``{session id: {job id: state}}`` for the runs each conversation launched.
RunStates = Callable[[list[str]], Awaitable[dict[str, dict[str, str]]]]

#: The problems the job manager currently reports, one dict per alert.
JobAlerts = Callable[[], Awaitable[list[dict[str, Any]]]]

#: Takes ``{campaign id: last event id seen}`` and answers with what has
#: happened in each since: ``{campaign id: {"events": [...],
#: "last_event_id": int, "idle": int, "queued": int}}``.
CampaignNews = Callable[[dict[str, int]], Awaitable[dict[str, dict[str, Any]]]]
CampaignBootstrap = Callable[[str, str], Awaitable[str | None]]


class SessionManager:
    """Creates, loads and forgets live sessions, and watches the runs of busy ones."""

    def __init__(self, store: SessionStore, agent_factory: AgentFactory) -> None:
        self.store = store
        self._agent_factory = agent_factory
        self._live: dict[str, LiveSession] = {}
        self._run_states: RunStates | None = None
        self._job_alerts: JobAlerts | None = None
        self._campaign_news: CampaignNews | None = None
        self._campaign_bootstrap: CampaignBootstrap | None = None
        self._watcher: asyncio.Task[None] | None = None
        self.watch_interval = WATCH_INTERVAL_S

    def _attach_agent(self, live: LiveSession) -> None:
        # A fresh id prefix per load keeps call ids unique across restores.
        handler = WebSocketEventHandler(
            sink=live.publish,
            id_prefix=f"call-{secrets.token_hex(3)}",
            on_result=live.note_tool_result,
        )
        live.handler = handler
        live.agent = self._agent_factory(handler)
        live.install_tools()
        if self._run_states is not None:
            live.observe = self._observe
        live.bootstrap_campaign = self._campaign_bootstrap

    # -- watching runs ---------------------------------------------------------

    def watch(
        self,
        run_states: RunStates,
        job_alerts: JobAlerts,
        campaign_news: CampaignNews | None = None,
        campaign_bootstrap: CampaignBootstrap | None = None,
    ) -> None:
        """Follow the runs of conversations working on their own, without the model.

        Every few seconds the state of their runs is read; a run that ended,
        a new problem with one, or a sub-agent's report wakes the agent with
        the news. Reading all of this costs nothing -- the model is what
        costs, so it is woken only when something changed.
        """
        self._run_states = run_states
        self._job_alerts = job_alerts
        self._campaign_news = campaign_news
        self._campaign_bootstrap = campaign_bootstrap
        for live in self._live.values():
            live.observe = self._observe
            live.bootstrap_campaign = campaign_bootstrap
        if self._watcher is None:
            self._watcher = asyncio.create_task(self._watch_loop())

    async def _observe(self, live: LiveSession) -> None:
        await self._look([live])

    async def _look(self, sessions: list[LiveSession]) -> None:
        """Bring the sessions' view of their runs up to date."""
        if self._run_states is None or not sessions:
            return
        states = await self._run_states([live.id for live in sessions])
        for live in sessions:
            live.note_runs(states.get(live.id, {}))
        if self._job_alerts is not None and any(live.running_runs for live in sessions):
            alerts = await self._job_alerts()
            for live in sessions:
                live.note_alerts(alerts)
        await self._look_at_campaigns(sessions)

    async def _look_at_campaigns(self, sessions: list[LiveSession]) -> None:
        """Read the ledger for the campaigns these conversations own."""
        if self._campaign_news is None:
            return
        since: dict[str, int] = {}
        for live in sessions:
            for campaign_id, seen in live.campaigns_seen.items():
                # One read per campaign even when two conversations own it:
                # each is told the same news and keeps its own cursor.
                since[campaign_id] = min(since.get(campaign_id, seen), seen)
        if not since:
            return
        news = await self._campaign_news(since)
        for live in sessions:
            for campaign_id in list(live.campaigns_seen):
                if campaign_id in news:
                    live.note_campaign(campaign_id, news[campaign_id])

    async def _watch_loop(self) -> None:
        while True:
            await asyncio.sleep(self.watch_interval)
            working = [live for live in self._live.values() if (live.auto_on or live.conducting) and not live.busy]
            try:
                await self._look(working)
            except Exception:
                logger.exception("Could not read the state of the runs")
                continue
            for live in working:
                live.wake_if_due()

    def create(self) -> LiveSession:
        stamp = _now()
        meta = {"id": _new_id(), "title": "", "created": stamp, "updated": stamp, "turns": 0}
        self.store.create(meta)
        live = LiveSession(meta=meta, store=self.store)
        self._attach_agent(live)
        self._remember(live)
        return live

    def get(self, session_id: str) -> LiveSession | None:
        """The live session, loading it from disk if needed."""
        if session_id in self._live:
            return self._live[session_id]
        if not self.store.exists(session_id):
            return None
        meta = self.store.load_meta(session_id)
        if meta is None:
            return None
        live = LiveSession(meta=meta, store=self.store, events=self.store.load_events(session_id))
        if backfill_title(live.meta, live.events):
            self.store.save_meta(live.meta)
        self._attach_agent(live)
        history = self.store.load_history(session_id)
        if history:
            live.agent.backend.import_history(history)
        live.agent.total_input_tokens = meta.get("inputTokens", 0)
        live.agent.total_output_tokens = meta.get("outputTokens", 0)
        # A turn that was running when the gateway stopped never finished.
        statuses = [event for event in live.events if event.get("type") == "status"]
        if statuses and statuses[-1].get("state") == "busy":
            live.publish({"type": "error", "message": "The gateway restarted while this turn was running."})
            live.publish({"type": "status", "state": "idle"})
        self._remember(live)
        return live

    def list(self) -> list[dict[str, Any]]:
        """Sessions with at least one message, newest first.

        Every new browser tab gets a session before anyone types, so the
        empty ones are left out rather than filling the list.
        """
        summaries = {meta["id"]: {**meta, "busy": False} for meta in self.store.list_meta()}
        for live in self._live.values():
            summaries[live.id] = live.summary()
        shown = [s for s in summaries.values() if s.get("turns") or s.get("busy")]
        return sorted(shown, key=lambda m: m.get("updated", ""), reverse=True)

    def busy_ids(self) -> list[str]:
        """Sessions with a turn running, so a caller can refuse to disturb them."""
        return [live.id for live in self._live.values() if live.busy]

    def rename(self, session_id: str, title: str) -> dict[str, Any] | None:
        live = self.get(session_id)
        if live is None:
            return None
        live.meta["title"] = title.strip()[:200]
        live.meta["auto_title"] = False
        live.store.save_meta(live.meta)
        return live.summary()

    def delete(self, session_id: str) -> bool:
        """Delete a session.  Refuses one with a turn running."""
        live = self._live.get(session_id)
        if live is not None:
            if live.busy:
                return False
            live.cancel_continuation()
            del self._live[session_id]
        self.store.delete(session_id)
        return True

    async def close(self) -> None:
        """Cancel running turns and queued continuations; used at shutdown."""
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher
            self._watcher = None
        for live in list(self._live.values()):
            live.cancel_continuation()
            if live.busy and live.running is not None:
                live.running.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await live.running

    def _remember(self, live: LiveSession) -> None:
        self._live[live.id] = live
        # A session working on its own is not idle, whoever is watching.
        idle = [s for s in self._live.values() if not s.busy and not s.auto_on and not s.subscribers and s is not live]
        while len(self._live) > MAX_LIVE_SESSIONS and idle:
            del self._live[idle.pop(0).id]
