# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Living inside the model's context window.

A long campaign is the normal case here, not the edge one: the conductor
reads reports from six sub-agents, a results table runs to tens of
thousands of tokens, and a turn can call tools for hours.  Three things
keep that inside the window -- the agent bounds one tool result, the
backend asks the endpoint how much room there is and gives some back when
it runs out, and the gateway restarts the conversation from a brief.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from reinvent4_agent import agent as agent_module
from reinvent4_agent.agent import Agent
from reinvent4_agent.backends import ChatResult, EndpointError, LLMBackend, ToolCall, ToolResult
from reinvent4_agent.backends.anthropic_backend import AnthropicBackend
from reinvent4_agent.backends.openai_backend import OpenAIBackend


# ---------------------------------------------------------------------------
# One tool result cannot swamp the conversation
# ---------------------------------------------------------------------------


class _OneCallBackend(LLMBackend):
    """Calls one tool, then answers; keeps what the tool said."""

    def __init__(self, tool_name: str = "get_campaign_files") -> None:
        self.tool_name = tool_name
        self.rounds = 0
        self.tool_results: list[ToolResult] = []

    def complete(self, tools: list[dict[str, Any]], max_tokens: int, *, force_tool_use: bool = False) -> ChatResult:
        self.rounds += 1
        if self.rounds == 1:
            return ChatResult(
                tool_calls=[ToolCall(id="call_1", name=self.tool_name, arguments={})],
                needs_tool_response=True,
            )
        return ChatResult(text="done", needs_tool_response=False)

    def add_user_message(self, content: str) -> None:
        pass

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.tool_results = results

    def repair_dangling_tool_use(self) -> None:
        pass

    def reset(self, system_prompt: str) -> None:
        pass

    def export_history(self) -> list[dict[str, Any]]:
        return []

    def import_history(self, messages: list[dict[str, Any]]) -> None:
        pass


class _BigResultMCP:
    def __init__(self, size: int) -> None:
        self.text = "HEAD" + ("x" * size) + "TAIL"

    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": "get_campaign_files", "description": "", "input_schema": {}}]

    def resolve_tool_name(self, name: str) -> str | None:
        return name if name == "get_campaign_files" else None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return self.text


class _Watcher:
    """Stands in for the browser, which is shown everything."""

    def __init__(self) -> None:
        self.results: list[str] = []

    def on_thinking(self, text: str) -> None: pass
    def on_tool_call(self, name: str, arguments: dict[str, Any]) -> None: pass
    def on_tool_result(self, name: str, result: str, is_error: bool) -> None: self.results.append(result)
    def on_error(self, message: str) -> None: pass
    def on_monitor_alert(self, alert: dict[str, Any]) -> None: pass
    def on_monitor_response(self, text: str) -> None: pass


class TestToolResultSize:

    def test_a_huge_result_is_cut_down_for_the_conversation(self, monkeypatch) -> None:
        monkeypatch.setattr(agent_module, "MAX_TOOL_RESULT_CHARS", 1_000)
        backend = _OneCallBackend()
        agent = Agent(_BigResultMCP(50_000), backend)

        asyncio.run(agent.chat("what is in the campaign"))

        kept = backend.tool_results[0].content
        assert len(kept) < 1_500
        # Both ends survive: a table's columns are at the top and its
        # summary at the bottom.
        assert kept.startswith("HEAD")
        assert kept.endswith("TAIL")
        assert "cut from the middle" in kept
        assert "get_campaign_files" in kept

    def test_the_browser_still_gets_the_whole_thing(self, monkeypatch) -> None:
        monkeypatch.setattr(agent_module, "MAX_TOOL_RESULT_CHARS", 1_000)
        watcher = _Watcher()
        agent = Agent(_BigResultMCP(50_000), _OneCallBackend(), event_handler=watcher)

        asyncio.run(agent.chat("what is in the campaign"))

        assert len(watcher.results[0]) > 50_000

    def test_an_ordinary_result_is_passed_through_untouched(self, monkeypatch) -> None:
        monkeypatch.setattr(agent_module, "MAX_TOOL_RESULT_CHARS", 1_000)
        backend = _OneCallBackend()
        agent = Agent(_BigResultMCP(10), backend)

        asyncio.run(agent.chat("what is in the campaign"))

        assert backend.tool_results[0].content == "HEAD" + "x" * 10 + "TAIL"


def test_a_turn_may_call_tools_two_thousand_times() -> None:
    """A supervisor works a task for hours; 200 rounds was a short afternoon."""
    assert agent_module.MAX_TOOL_ROUNDS == 2_000


# ---------------------------------------------------------------------------
# Giving room back when the endpoint says there is none
# ---------------------------------------------------------------------------


class _Model:
    def __init__(self, id: str, max_model_len: int | None) -> None:
        self.id = id
        if max_model_len is not None:
            self.max_model_len = max_model_len


class _Listing:
    def __init__(self, models: list[_Model]) -> None:
        self.data = models


class _Models:
    def __init__(self, listing: _Listing | Exception) -> None:
        self.listing = listing
        self.calls = 0

    def list(self) -> _Listing:
        self.calls += 1
        if isinstance(self.listing, Exception):
            raise self.listing
        return self.listing


def _backend(model: str = "Qwen/Qwen3-32B-FP8") -> OpenAIBackend:
    return OpenAIBackend(
        model=model,
        api_key="dummy",
        base_url="http://aim/v1",
        system_prompt="sp",
    )


def _serves(backend: OpenAIBackend, listing: _Listing | Exception) -> _Models:
    models = _Models(listing)
    backend.client.with_options = lambda **kwargs: type("C", (), {"models": models})()  # type: ignore[assignment]
    return models


class TestTheServedWindow:

    def test_it_is_read_from_the_endpoint_not_guessed_from_the_model(self) -> None:
        """The same weights are served at 32k here and 128k there."""
        backend = _backend()
        _serves(backend, _Listing([_Model("other", 8192), _Model("Qwen/Qwen3-32B-FP8", 65536)]))

        assert backend.context_window == 65536

    def test_the_endpoint_is_only_asked_once(self) -> None:
        backend = _backend()
        models = _serves(backend, _Listing([_Model("Qwen/Qwen3-32B-FP8", 65536)]))

        assert backend.context_window == 65536
        assert backend.context_window == 65536
        assert models.calls == 1

    def test_an_endpoint_that_does_not_say_leaves_it_unknown(self) -> None:
        backend = _backend()
        _serves(backend, _Listing([_Model("Qwen/Qwen3-32B-FP8", None)]))

        assert backend.context_window is None

    def test_a_listing_that_fails_is_not_an_error(self) -> None:
        """It is a nicety, asked for on the way to a turn the user wants."""
        backend = _backend()
        _serves(backend, RuntimeError("404"))

        assert backend.context_window is None

    def test_the_environment_wins_and_nothing_is_asked(self, monkeypatch) -> None:
        monkeypatch.setenv("REINVENT4_AGENT_CONTEXT_TOKENS", "40000")
        backend = _backend()
        models = _serves(backend, _Listing([_Model("Qwen/Qwen3-32B-FP8", 65536)]))

        assert backend.context_window == 40000
        assert models.calls == 0


class TestTheReplyCapFitsTheWindow:

    def test_a_cap_that_would_not_fit_beside_the_prompt_is_lowered(self) -> None:
        backend = _backend()
        _serves(backend, _Listing([_Model("Qwen/Qwen3-32B-FP8", 32768)]))
        backend._last_prompt_tokens = 30_000

        assert backend._token_cap(16384) == 2768

    def test_a_cap_that_fits_is_left_alone(self) -> None:
        backend = _backend()
        _serves(backend, _Listing([_Model("Qwen/Qwen3-32B-FP8", 131072)]))
        backend._last_prompt_tokens = 30_000

        assert backend._token_cap(16384) == 16384

    def test_an_unknown_window_changes_nothing(self) -> None:
        backend = _backend()
        _serves(backend, _Listing([]))
        backend._last_prompt_tokens = 30_000

        assert backend._token_cap(16384) == 16384


class _Rejection(Exception):
    """Stands in for openai.BadRequestError, which the code matches on text."""


class TestMakingRoom:

    def _overflow(self) -> Any:
        return _Rejection(
            "This model's maximum context length is 32768 tokens, however you "
            "requested 41000 tokens."
        )

    def test_the_reply_is_given_up_before_the_history(self) -> None:
        backend = _backend()
        kwargs = {"max_tokens": 16384, "messages": []}

        assert backend._adapt_to_rejection(self._overflow(), kwargs) is True
        assert kwargs["max_tokens"] == 1024

    def test_then_the_oldest_exchange_goes(self) -> None:
        backend = _backend()
        backend.add_user_message("first question")
        backend._messages.append({"role": "assistant", "content": "first answer"})
        backend.add_user_message("second question")
        kwargs = {"max_tokens": 1024, "messages": []}

        assert backend._adapt_to_rejection(self._overflow(), kwargs) is True
        assert [m["role"] for m in backend._messages] == ["system", "user"]
        assert backend._messages[1]["content"] == "second question"
        assert kwargs["messages"][1]["content"] == "second question"

    def test_a_conversation_with_nothing_left_to_drop_reports_the_error(self) -> None:
        backend = _backend()
        backend.add_user_message("the only question")
        kwargs = {"max_tokens": 1024, "messages": []}

        assert backend._adapt_to_rejection(self._overflow(), kwargs) is False

    def _one_long_turn(self) -> OpenAIBackend:
        backend = _backend()
        backend.add_user_message("Do target intelligence for JAK3")
        for i, size in enumerate((190_000, 500, 150_000)):
            backend._messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": f"c{i}", "type": "function",
                                "function": {"name": "t", "arguments": "{}"}}],
            })
            backend._messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * size})
        return backend

    def test_one_long_turn_shortens_the_results_already_read(self) -> None:
        backend = self._one_long_turn()
        kwargs = {"max_tokens": 1024, "messages": []}

        assert backend._adapt_to_rejection(self._overflow(), kwargs) is True

        sizes = [len(m["content"]) for m in backend._messages if m["role"] == "tool"]
        assert sizes[0] < 2500
        assert sizes[1] == 500
        assert sizes[2] == 150_000, "the latest result has not been answered yet"
        assert "call the tool again" in backend._messages[3]["content"]
        assert [m["role"] for m in backend._messages][:2] == ["system", "user"]

    def test_once_they_are_short_it_gives_up(self) -> None:
        backend = self._one_long_turn()
        backend._adapt_to_rejection(self._overflow(), {"max_tokens": 1024, "messages": []})

        assert backend._adapt_to_rejection(self._overflow(), {"max_tokens": 1024, "messages": []}) is False

    def test_an_overflow_it_cannot_fix_is_not_reported_as_a_routing_problem(self) -> None:
        import httpx
        import openai

        backend = _backend()
        backend.add_user_message("the only question")
        response = httpx.Response(400, request=httpx.Request("POST", "http://127.0.0.1:8001/v1/chat/completions"))

        def refuse(kwargs: dict) -> Any:
            raise openai.BadRequestError(str(self._overflow()), response=response, body=None)

        backend._request = refuse
        with pytest.raises(EndpointError) as caught:
            backend._request_adapting({"max_tokens": 1024, "messages": []})

        report = str(caught.value)
        assert "context window" in report
        assert "Resource not found" not in report
        assert "deployment" not in report

    def test_vllms_negative_cap_is_read_as_an_overflow(self) -> None:
        """It subtracts the prompt from max-model-len and reports the remainder."""
        backend = _backend()
        kwargs = {"max_tokens": 16384, "messages": []}
        exc = _Rejection("max_tokens must be at least 1, got -72257.")

        assert backend._adapt_to_rejection(exc, kwargs) is True
        assert kwargs["max_tokens"] == 1024

    def test_that_message_does_not_trigger_the_token_parameter_swap(self) -> None:
        """"max_tokens" appearing in it reads as a request for the other name."""
        backend = _backend("gpt-5")
        assert backend._token_param == "max_completion_tokens"
        kwargs = {"max_completion_tokens": 32768, "messages": []}

        backend._adapt_to_rejection(_Rejection("max_tokens must be at least 1, got -5"), kwargs)

        assert backend._token_param == "max_completion_tokens"
        assert kwargs["max_completion_tokens"] == 1024


class TestForgettingTheOldest:

    def test_a_tool_call_is_never_separated_from_its_result(self) -> None:
        """A history that starts on an orphan tool result is rejected outright."""
        backend = _backend()
        backend.add_user_message("first")
        backend._messages.append({"role": "assistant", "tool_calls": [{"id": "c1"}]})
        backend.add_tool_results([ToolResult(tool_call_id="c1", content="ok")])
        backend.add_user_message("second")

        dropped = backend.forget_oldest()

        assert dropped == 3
        assert [m["role"] for m in backend._messages] == ["system", "user"]

    def test_the_system_prompt_stays(self) -> None:
        backend = _backend()
        backend.add_user_message("first")
        backend.add_user_message("second")

        backend.forget_oldest()

        assert backend._messages[0]["role"] == "system"
        assert backend._messages[0]["content"] == "sp"

    def test_nothing_to_drop_is_said_rather_than_done(self) -> None:
        backend = _backend()
        backend.add_user_message("only")

        assert backend.forget_oldest() == 0
        assert len(backend._messages) == 2


class TestAnthropicForgetsTheOldest:

    def _backend(self) -> AnthropicBackend:
        return AnthropicBackend(model="claude-x", api_key="dummy", system_prompt="sp")

    def test_it_cuts_at_a_question_not_at_a_tool_result(self) -> None:
        """Tool results are sent as user messages there; starting on one orphans a call."""
        backend = self._backend()
        backend.add_user_message("first")
        backend._messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": "t1"}]})
        backend.add_tool_results([ToolResult(tool_call_id="t1", content="ok")])
        backend.add_user_message("second")

        dropped = backend.forget_oldest()

        assert dropped == 3
        assert backend._messages == [{"role": "user", "content": "second"}]

    def test_the_system_prompt_is_outside_the_history_and_unaffected(self) -> None:
        backend = self._backend()
        backend.add_user_message("first")
        backend.add_user_message("second")

        backend.forget_oldest()

        assert backend._system_prompt == "sp"
        assert len(backend._messages) == 1


def test_a_backend_that_cannot_forget_says_so() -> None:
    """The default keeps every message, so a fake backend is not asked to lie."""

    class _Plain(LLMBackend):
        def complete(self, tools, max_tokens, *, force_tool_use=False): ...
        def add_user_message(self, content): ...
        def add_tool_results(self, results): ...
        def repair_dangling_tool_use(self): ...
        def reset(self, system_prompt): ...
        def export_history(self): ...
        def import_history(self, messages): ...

    plain = _Plain()

    assert plain.forget_oldest() == 0
    assert plain.context_window is None


@pytest.mark.parametrize(
    "message",
    [
        "This model's maximum context length is 32768 tokens",
        "context length exceeded",
        "Please reduce the length of the messages",
        "max_tokens must be at least 1, got -72257",
        "The prompt is longer than the maximum model length of 65536",
    ],
)
def test_every_wording_of_a_full_window_is_recognised(message: str) -> None:
    from reinvent4_agent.backends.openai_backend import _CONTEXT_OVERFLOW_RE

    assert _CONTEXT_OVERFLOW_RE.search(message) is not None
