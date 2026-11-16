"""HTTP client for ``campaign-store``, shared by the supervisor and conductor.

One module so there is a single definition of the ledger's wire format. The
two sides use disjoint halves of it -- supervisors claim and report, the
conductor opens campaigns, enqueues work and reads the results -- but they have
to agree on what a task looks like, and a second copy would drift.

The error types matter as much as the calls. ``LeaseLost`` and
``StoreUnavailable`` are separated because they demand opposite responses: one
means stop touching this task, the other means wait and try again.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

STORE_URL = os.getenv("CAMPAIGN_STORE_URL", "http://dd-agentic-campaign-store:8087")

# Long enough that a request waiting on the store's write lock is not mistaken
# for the store being down, short enough that a hung call does not stall the
# caller for the whole lease.
STORE_TIMEOUT_SECONDS = 60.0


def supervisor_id() -> str:
    """Identity in the ledger. The pod name, when there is one."""

    return os.getenv("SUPERVISOR_ID") or socket.gethostname()


class StoreUnavailable(Exception):
    """The ledger could not be reached or refused the call."""


class LeaseLost(Exception):
    """This supervisor no longer holds the task it is working on.

    Raised rather than swallowed: continuing would mean spending a GPU on work
    the store has already handed to someone else.
    """


@dataclass
class TaskOutcome:
    """What a finished task reports back to the ledger."""

    objective_profile: dict[str, float] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    job_id: Optional[str] = None
    summary: Optional[str] = None
    #: The sub-agent's account of the task, for the conductor.
    report: Optional[str] = None
    #: Every run the task started; ``job_id`` is the one its molecules came from.
    job_ids: list[str] = field(default_factory=list)
    #: Next tasks the sub-agent would run: ``{"intent", "title", "kind", "rationale"}``.
    follow_ups: list[dict[str, Any]] = field(default_factory=list)


def _detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        return str(body.get("detail") or body.get("error") or body)[:300]
    return str(body)[:300]


class CampaignStoreClient:
    """The ledger's HTTP surface. Blocking; async callers use a thread."""

    def __init__(
        self,
        base_url: str = STORE_URL,
        owner: Optional[str] = None,
        timeout: float = STORE_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.owner = owner or supervisor_id()
        self.timeout = timeout

    def _call(self, method: str, path: str, **json_body: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method, url, json=json_body or None, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise StoreUnavailable(f"{url} is unreachable: {exc}") from exc

        # 409 is the store telling us someone else owns this task now, which
        # is a different problem from the store being broken.
        if response.status_code == 409:
            raise LeaseLost(_detail(response))
        if not response.ok:
            raise StoreUnavailable(
                f"{url} returned {response.status_code}: {_detail(response)}"
            )
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def _get(self, path: str) -> Any:
        return self._call("GET", path)

    # -- supervisor side ---------------------------------------------------

    def claim(
        self, campaign_id: Optional[str] = None, lease_seconds: Optional[float] = None
    ) -> Optional[dict[str, Any]]:
        """The next task, or None when the queue is empty."""

        body: dict[str, Any] = {"owner": self.owner}
        if campaign_id:
            body["campaign_id"] = campaign_id
        if lease_seconds:
            body["lease_seconds"] = lease_seconds
        return self._call("POST", "/tasks/claim", **body)

    def claim_named(
        self, task_id: str, lease_seconds: Optional[float] = None
    ) -> dict[str, Any]:
        """Take one named task, rather than whatever is next in the queue.

        Without a lease the task is held until a result is posted: a holder
        that is not a pool member has nobody to hand it back to.
        """

        body: dict[str, Any] = {"owner": self.owner}
        if lease_seconds:
            body["lease_seconds"] = lease_seconds
        return self._call("POST", f"/tasks/{task_id}/claim", **body)

    def heartbeat(
        self,
        task_id: str,
        lease_seconds: Optional[float] = None,
        job_id: Optional[str] = None,
    ) -> None:
        body: dict[str, Any] = {"owner": self.owner}
        if lease_seconds:
            body["lease_seconds"] = lease_seconds
        if job_id:
            body["job_id"] = job_id
        self._call("POST", f"/tasks/{task_id}/heartbeat", **body)

    def post_result(self, task_id: str, outcome: TaskOutcome) -> dict[str, Any]:
        body: dict[str, Any] = {
            "owner": self.owner,
            "objective_profile": outcome.objective_profile,
            "candidates": outcome.candidates,
            "job_id": outcome.job_id,
            "summary": outcome.summary,
        }
        if outcome.report:
            body["report"] = outcome.report
        if outcome.job_ids:
            body["job_ids"] = outcome.job_ids
        if outcome.follow_ups:
            body["follow_ups"] = outcome.follow_ups
        try:
            return self._call("POST", f"/tasks/{task_id}/result", **body)
        except StoreUnavailable as exc:
            if "422" not in str(exc) or not ({"report", "job_ids", "follow_ups"} & set(body)):
                raise
            if "follow_ups" in body:
                # Proposals are worth less than the result they would cost.
                logger.warning("the ledger does not take follow-ups yet; posting %s without them", task_id)
                body.pop("follow_ups")
                try:
                    return self._call("POST", f"/tasks/{task_id}/result", **body)
                except StoreUnavailable as again:
                    if "422" not in str(again):
                        raise
        # A ledger older than reports refuses the fields it does not know,
        # and the task would be lost over a column. The supervisors and the
        # ledger restart separately, so the two are out of step for a while
        # on every upgrade; the report survives inside the summary instead.
        logger.warning("the ledger does not take reports yet; posting %s without one", task_id)
        legacy = {k: v for k, v in body.items() if k not in ("report", "job_ids")}
        if outcome.report:
            legacy["summary"] = "\n\n".join(filter(None, [outcome.summary, outcome.report]))
        return self._call("POST", f"/tasks/{task_id}/result", **legacy)

    def fail(self, task_id: str, error: str, requeue: bool = True) -> dict[str, Any]:
        return self._call(
            "POST", f"/tasks/{task_id}/fail",
            owner=self.owner, error=error, requeue=requeue,
        )

    def report_presence(
        self,
        state: str,
        campaign_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tasks_completed: int = 0,
        tasks_failed: int = 0,
        detail: Optional[str] = None,
    ) -> dict[str, Any]:
        """Tell the ledger this supervisor exists and whether it is free.

        Claiming is the only other thing that puts a supervisor in the ledger,
        so without this an idle one is indistinguishable from one that never
        started.
        """

        return self._call(
            "POST", f"/supervisors/{self.owner}/presence",
            state=state, campaign_id=campaign_id, task_id=task_id,
            tasks_completed=tasks_completed, tasks_failed=tasks_failed,
            detail=detail,
        )

    def drop_presence(self) -> None:
        """Leave the pool on purpose, rather than being timed out of it."""

        self._call("DELETE", f"/supervisors/{self.owner}")

    # -- conductor side ----------------------------------------------------

    def create_campaign(
        self,
        goal: str,
        base_ttp_id: Optional[str] = None,
        objectives: Optional[list[str]] = None,
        concurrency: int = 1,
        dispatch_seconds: Optional[float] = None,
        grace_seconds: float = 0.0,
        parent_campaign_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
    ) -> dict[str, Any]:
        lineage = {
            key: value for key, value in (
                ("parent_campaign_id", parent_campaign_id),
                ("parent_task_id", parent_task_id),
            ) if value
        }
        return self._call(
            "POST", "/campaigns",
            goal=goal, base_ttp_id=base_ttp_id, objectives=objectives or [],
            concurrency=concurrency, dispatch_seconds=dispatch_seconds,
            grace_seconds=grace_seconds, **lineage,
        )

    def enqueue_task(self, campaign_id: str, **spec: Any) -> dict[str, Any]:
        return self._call("POST", f"/campaigns/{campaign_id}/tasks", **spec)

    def enqueue_tasks(self, campaign_id: str, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Queue several tasks in one transaction: all of them or none."""
        return self._call("POST", f"/campaigns/{campaign_id}/tasks/batch", tasks=tasks)

    def set_plan(self, campaign_id: str, workstreams: list[dict[str, Any]]) -> dict[str, Any]:
        return self._call("PUT", f"/campaigns/{campaign_id}/plan", workstreams=workstreams)

    def coverage(self, campaign_id: str) -> dict[str, Any]:
        return self._get(f"/campaigns/{campaign_id}/coverage")

    def metrics(self, campaign_id: str) -> dict[str, Any]:
        return self._get(f"/campaigns/{campaign_id}/metrics")

    def proposals(self, campaign_id: str, state: Optional[str] = "open") -> list[dict[str, Any]]:
        query = f"?state={state}" if state else ""
        return self._get(f"/campaigns/{campaign_id}/proposals{query}")

    def take_proposals(
        self, campaign_id: str, proposal_ids: list[str], priority: int = 0,
    ) -> list[dict[str, Any]]:
        return self._call(
            "POST", f"/campaigns/{campaign_id}/proposals/take",
            proposal_ids=proposal_ids, priority=priority,
        )

    def drop_proposals(self, campaign_id: str, proposal_ids: list[str]) -> dict[str, Any]:
        return self._call(
            "POST", f"/campaigns/{campaign_id}/proposals/drop", proposal_ids=proposal_ids,
        )

    def record_finding(
        self, campaign_id: str, author: str, branch: str, claim: str, **fields: Any
    ) -> dict[str, Any]:
        """Add to what the campaign knows."""

        return self._call(
            "POST", f"/campaigns/{campaign_id}/findings",
            author=author, branch=branch, claim=claim, **fields,
        )

    def note_finding(
        self, campaign_id: str, finding_id: str, author: str, body: str, **fields: Any
    ) -> dict[str, Any]:
        return self._call(
            "POST", f"/campaigns/{campaign_id}/findings/{finding_id}/notes",
            author=author, body=body, **fields,
        )

    def findings(
        self,
        campaign_id: str,
        branch: Optional[str] = None,
        q: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        query = {"limit": str(limit)}
        if branch:
            query["branch"] = branch
        if q:
            query["q"] = q
        if tag:
            query["tag"] = tag
        return self._get(f"/campaigns/{campaign_id}/findings?{urlencode(query)}")

    def atlas(self, campaign_id: str) -> dict[str, Any]:
        """The map of what each branch of the campaign knows."""

        return self._get(f"/campaigns/{campaign_id}/atlas")

    def campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._get(f"/campaigns/{campaign_id}")

    def campaigns(self, session_id: Optional[str] = None) -> list[dict[str, Any]]:
        query = f"?session_id={session_id}" if session_id else ""
        return self._get(f"/campaigns{query}")

    def attach_session(self, campaign_id: str, session_id: str) -> dict[str, Any]:
        """Record which conversation a campaign belongs to."""

        return self._call(
            "POST", f"/campaigns/{campaign_id}/session", session_id=session_id
        )

    def supervisors(self, campaign_id: Optional[str] = None) -> dict[str, Any]:
        """The live pool, with dead presences swept first."""

        query = f"?campaign_id={campaign_id}" if campaign_id else ""
        return self._get(f"/supervisors{query}")

    def status(self, campaign_id: str) -> dict[str, Any]:
        return self._get(f"/campaigns/{campaign_id}/status")

    def results(
        self, campaign_id: str, full: bool = False
    ) -> list[dict[str, Any]]:
        """Finished results. ``full`` keeps every scorer diagnostic."""

        query = "?full=1" if full else ""
        return self._get(f"/campaigns/{campaign_id}/results{query}")

    def tasks(
        self, campaign_id: str, state: Optional[str] = None
    ) -> list[dict[str, Any]]:
        query = f"?state={state}" if state else ""
        return self._get(f"/campaigns/{campaign_id}/tasks{query}")

    def events(self, campaign_id: str, since: int = 0) -> dict[str, Any]:
        return self._get(f"/campaigns/{campaign_id}/events?since={since}")

    def set_concurrency(self, campaign_id: str, concurrency: int) -> dict[str, Any]:
        return self._call(
            "POST", f"/campaigns/{campaign_id}/concurrency", concurrency=concurrency
        )

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Take a task back from whoever holds it."""

        return self._call("POST", f"/tasks/{task_id}/cancel")

    def close_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._call("POST", f"/campaigns/{campaign_id}/close")

    def conducted(self, campaign_id: str) -> dict[str, Any]:
        """Say this conductor holds the campaign, which keeps its tasks claimable."""
        return self._call("POST", f"/campaigns/{campaign_id}/conducted")

    def finalize_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._call("POST", f"/campaigns/{campaign_id}/finalize")
