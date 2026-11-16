# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Carrying a conversation across a restart of the gateway.

The gateway saves ``export_history()`` after every turn and hands it back
through ``import_history()`` when a session is reopened, so a conversation
survives the restart that a code change needs.  Both were missing from the
real backends for a while: the save is wrapped in a ``try``, so the only
sign was a logged ``AttributeError`` and a conductor that had forgotten its
campaign the next time anyone opened it.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

from reinvent4_agent.backends import LLMBackend, ToolResult
from reinvent4_agent.backends.anthropic_backend import AnthropicBackend
from reinvent4_agent.backends.openai_backend import OpenAIBackend


def _openai() -> OpenAIBackend:
    return OpenAIBackend(model="m", api_key="dummy", system_prompt="sp")


def _anthropic() -> AnthropicBackend:
    return AnthropicBackend(model="claude-x", api_key="dummy", system_prompt="sp")


class _Block:
    """Stands in for an SDK content block, which is a pydantic model."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {
            k: v
            for k, v in self._fields.items()
            if not exclude_none or v is not None
        }


class TestTheOpenAIHistoryTravels:

    def test_a_conversation_comes_back_as_it_went(self) -> None:
        backend = _openai()
        backend.add_user_message("design something")
        backend._messages.append({"role": "assistant", "tool_calls": [{"id": "c1"}]})
        backend.add_tool_results([ToolResult(tool_call_id="c1", content="ok")])

        exported = backend.export_history()
        fresh = _openai()
        fresh.import_history(exported)

        assert fresh._messages == backend._messages

    def test_the_system_prompt_is_this_process_not_the_saved_one(self) -> None:
        """It carries the tools and limits of the session running now."""
        backend = _openai()
        backend.add_user_message("design something")

        fresh = OpenAIBackend(model="m", api_key="dummy", system_prompt="newer")
        fresh.import_history(backend.export_history())

        assert fresh._messages[0] == {"role": "system", "content": "newer"}
        assert [m["role"] for m in fresh._messages] == ["system", "user"]

    def test_importing_twice_does_not_say_everything_twice(self) -> None:
        backend = _openai()
        backend.add_user_message("design something")
        history = backend.export_history()

        fresh = _openai()
        fresh.import_history(history)
        fresh.import_history(history)

        assert [m["role"] for m in fresh._messages] == ["system", "user"]

    def test_the_history_is_not_a_live_view_of_the_backend(self) -> None:
        """Saved after the turn, written later; a shared list would change under the writer."""
        backend = _openai()
        backend.add_user_message("design something")

        exported = backend.export_history()
        backend.add_user_message("and another")

        assert len(exported) == 1


class TestTheAnthropicHistoryTravels:

    def test_an_assistant_turn_can_be_written_to_disk(self) -> None:
        """Its blocks are SDK objects in flight, and JSON cannot hold those."""
        backend = _anthropic()
        backend.add_user_message("design something")
        backend._messages.append(
            {
                "role": "assistant",
                "content": [_Block(type="tool_use", id="t1", input={}, name="x")],
            }
        )

        exported = backend.export_history()

        assert json.loads(json.dumps(exported)) == exported
        assert exported[1]["content"][0]["type"] == "tool_use"

    def test_a_conversation_comes_back_as_data(self) -> None:
        backend = _anthropic()
        backend.add_user_message("design something")
        backend.add_tool_results([ToolResult(tool_call_id="t1", content="ok")])

        fresh = _anthropic()
        fresh.import_history(backend.export_history())

        assert fresh._messages == backend._messages

    def test_an_unanswered_call_in_an_imported_history_is_still_healed(self) -> None:
        """The next request is refused outright if the call is left dangling."""
        backend = _anthropic()
        backend.import_history(
            [
                {"role": "user", "content": "design something"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "x"}],
                },
            ]
        )

        backend.repair_dangling_tool_use()

        healed = backend._messages[-1]
        assert healed["role"] == "user"
        assert [b["tool_use_id"] for b in healed["content"]] == ["t1"]


@pytest.mark.parametrize(
    "backend_class", [cls.__name__ for cls in LLMBackend.__subclasses__()]
)
def test_no_backend_can_forget_how_to_save_a_conversation(backend_class: str) -> None:
    """Abstract, so a backend that cannot do this cannot be built at all.

    Asserted over the subclasses rather than the two known names: the point
    is that the next backend is held to it too.
    """
    cls = next(c for c in LLMBackend.__subclasses__() if c.__name__ == backend_class)

    for name in ("export_history", "import_history"):
        assert not getattr(getattr(cls, name), "__isabstractmethod__", False)
        assert inspect.isfunction(getattr(cls, name))
