"""The agent loop with a scripted provider: termination, fallback, retries, reflection."""
import json

import pytest

from app.core.base_provider import BaseAIProvider, ParsedResponse
from app.core.tool_manager import initialize_tools
from app.core.tools import tool_registry
from app.core.trajectory import TrajectoryRecorder
from app.models.requests import ChatMessage, ToolCallRequest
from app.models.responses import TokenUsage
from app.utils.repo_context import current_repo_path

DIFF_ANSWER = "Fix:\n\na.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"
UNIFIED_DIFF_ANSWER = "Fix:\n```diff\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n```"


class Overloaded(Exception):
    """Stands in for a transient provider error."""


class ScriptedProvider(BaseAIProvider):
    """Returns pre-scripted responses and records every call it receives."""
    retry_base_seconds = 0.0

    def __init__(self, script):
        super().__init__(api_key="test", model="scripted-model")
        self.script = list(script)
        self.calls = []

    @property
    def provider_name(self):
        return "scripted"

    def _validate_api_key(self):
        pass

    def _is_retryable(self, exc):
        return isinstance(exc, Overloaded)

    async def _make_api_call(self, messages, temperature=0.7, max_tokens=1000, tools=None, tool_choice="auto", **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools, "tool_choice": tool_choice, "max_tokens": max_tokens})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _parse_response(self, raw_response):
        return raw_response


def reply(text="", calls=()):
    return ParsedResponse(
        text=text,
        tool_calls=[ToolCallRequest(id=f"c{i}", name=name, arguments=args) for i, (name, args) in enumerate(calls)],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        stop_reason="stop", api_model_id="scripted-model-001",
    )


@pytest.fixture(autouse=True)
def repo(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    initialize_tools()
    token = current_repo_path.set(str(tmp_path))
    yield tmp_path
    current_repo_path.reset(token)


def start():
    return [ChatMessage(role="system", content="S"), ChatMessage(role="user", content="U")]


async def test_stops_with_diff_and_keeps_native_tool_turns():
    provider = ScriptedProvider([reply("reading", [("read_file", {"file_path": "a.py"})]), reply(DIFF_ANSWER)])
    response = await provider.generate_response(start(), max_tool_iterations=10)

    assert response.success and response.final_answer == DIFF_ANSWER
    assert response.run_meta["termination_reason"] == "model_stopped"
    assert response.run_meta["fallback_used"] is False and response.run_meta["iterations"] == 2
    assert response.run_meta["api_model_ids"] == ["scripted-model-001"]

    # second call sees: system, user, assistant(with tool call), tool(result)
    second = provider.calls[1]["messages"]
    assert [m.role for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2].tool_calls[0].name == "read_file"
    assert second[3].tool_results[0].tool_call_id == "c0"
    assert "x = 1" in second[3].tool_results[0].content
    assert all("Calling tools:" not in (m.content or "") for m in second)


async def test_draft_diff_during_exploration_is_not_the_final_answer():
    """A diff written while still calling tools does not count; the final message does."""
    provider = ScriptedProvider([
        reply(DIFF_ANSWER, [("read_file", {"file_path": "a.py"})]),
        reply("I am done."),
        reply("Final:\n\na.py\n<<<<<<< SEARCH\ny = 2\n=======\ny = 3\n>>>>>>> REPLACE"),
    ])
    response = await provider.generate_response(start())
    assert response.run_meta["termination_reason"] == "fallback_used"
    assert "y = 3" in response.final_answer and "x = 2" not in response.final_answer
    assert provider.calls[2]["tool_choice"] == "none" and provider.calls[2]["tools"] is not None


async def test_cap_hit_then_fallback_with_tools_disabled():
    script = [reply("", [("list_directory", {"directory_path": "."})]) for _ in range(3)] + [reply(DIFF_ANSWER)]
    provider = ScriptedProvider(script)
    response = await provider.generate_response(start(), max_tool_iterations=3)

    assert response.run_meta["termination_reason"] == "cap_hit" and response.run_meta["cap_hit"] is True
    assert response.run_meta["fallback_used"] is True and response.run_meta["iterations"] == 3
    fallback = provider.calls[3]
    assert fallback["tool_choice"] == "none" and fallback["tools"] is not None
    assert fallback["messages"][-1].role == "user" and "FINAL ANSWER REQUIRED" in fallback["messages"][-1].content
    assert fallback["messages"][-2].role == "tool"  # the last tool results were delivered
    assert response.final_answer == DIFF_ANSWER


async def test_final_answer_prompts_differ_only_in_the_abstention_option():
    from app.core.edit_format import EDIT_FORMAT_INSTRUCTIONS
    prompts = {}
    for mode in ("baseline", "forced"):
        provider = ScriptedProvider([reply("no idea"), reply(DIFF_ANSWER)])
        await provider.generate_response(start(), experiment_mode=mode)
        prompts[mode] = provider.calls[1]["messages"][-1].content
    assert "NO ESCAPE OPTION" in prompts["forced"] and "<<<CANNOT_SOLVE>>>" not in prompts["forced"]
    assert "<<<CANNOT_SOLVE>>>" in prompts["baseline"]
    for text in prompts.values():
        assert text.endswith(EDIT_FORMAT_INSTRUCTIONS)   # the same format block in both
        assert "@@" not in text and "```diff" not in text  # no diff template to copy


async def test_a_unified_diff_answer_is_still_accepted_as_an_answer():
    provider = ScriptedProvider([reply(UNIFIED_DIFF_ANSWER)])
    response = await provider.generate_response(start())
    assert response.run_meta["fallback_used"] is False and response.final_answer == UNIFIED_DIFF_ANSWER


async def test_abstention_in_final_answer_needs_no_fallback():
    provider = ScriptedProvider([reply("<<<CANNOT_SOLVE>>>\nreason: too_complex\nexplanation: hard")])
    response = await provider.generate_response(start())
    assert response.run_meta["fallback_used"] is False and len(provider.calls) == 1


async def test_empty_first_response_still_triggers_fallback():
    provider = ScriptedProvider([reply(""), reply(DIFF_ANSWER)])
    response = await provider.generate_response(start())
    assert response.run_meta["fallback_used"] is True and response.final_answer == DIFF_ANSWER


async def test_gemini_style_float_arguments_are_coerced():
    provider = ScriptedProvider([
        reply("", [("read_file", {"file_path": "a.py", "start_line": 1.0, "max_lines": "1"})]),
        reply(DIFF_ANSWER),
    ])
    response = await provider.generate_response(start())
    call = response.tool_calls[0]
    assert call.success, call.error
    assert call.result["data"]["content"] == "x = 1"


async def test_malformed_tool_arguments_are_reported_to_the_model():
    bad = reply("", [("read_file", {})])
    bad.tool_calls[0].parse_error = "arguments are not valid JSON"
    provider = ScriptedProvider([bad, reply(DIFF_ANSWER)])
    response = await provider.generate_response(start())
    assert response.tool_calls[0].success is False
    result = provider.calls[1]["messages"][-1].tool_results[0]
    assert result.is_error and "Invalid tool arguments" in result.content


async def test_transient_error_is_retried_and_logged(tmp_path):
    recorder = TrajectoryRecorder(tmp_path / "traj" / "inst.jsonl")
    provider = ScriptedProvider([Overloaded("529"), reply(DIFF_ANSWER)])
    response = await provider.generate_response(start(), recorder=recorder)
    recorder.summary(result_status="success_patch_generated")

    assert response.success and response.run_meta["api_retry_count"] == 1 and response.run_meta["api_call_count"] == 1
    lines = [json.loads(line) for line in (tmp_path / "traj" / "inst.jsonl").read_text().splitlines()]
    api_calls = [l for l in lines if l["type"] == "api_call"]
    assert [bool(l.get("error")) for l in api_calls] == [True, False]
    assert api_calls[1]["api_model_id"] == "scripted-model-001" and api_calls[1]["phase"] == "loop"
    assert lines[-1]["type"] == "summary"


async def test_non_retryable_error_is_an_api_error_not_a_model_failure():
    provider = ScriptedProvider([ValueError("credit balance is too low")])
    response = await provider.generate_response(start())
    assert response.success is False and response.error_kind == "api_error"
    assert "credit balance" in response.error_message and len(provider.calls) == 1


async def test_retries_are_bounded():
    provider = ScriptedProvider([Overloaded("x")] * 6)
    response = await provider.generate_response(start())
    assert response.error_kind == "api_error" and len(provider.calls) == provider.max_api_attempts


async def test_context_overflow_is_its_own_outcome():
    provider = ScriptedProvider([ValueError("prompt is too long: 250000 tokens > 200000 maximum")])
    response = await provider.generate_response(start())
    assert response.error_kind == "context_overflow"


async def test_reflection_sees_full_history_and_cannot_call_tools():
    provider = ScriptedProvider([
        reply("reading", [("read_file", {"file_path": "a.py"})]),
        reply(DIFF_ANSWER),
        reply("<<<PATCH_APPROVED>>>\nLooks right."),
    ])
    response = await provider.generate_response(start(), max_tokens=8000, experiment_mode="self_reflect")

    reflection_call = provider.calls[2]
    roles = [m.role for m in reflection_call["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]  # nothing was dropped
    reflection_prompt = reflection_call["messages"][-1].content
    assert "SELF-REFLECTION PHASE" in reflection_prompt
    assert "TEST" not in reflection_prompt  # no test feedback of any kind
    assert "a.py\n<<<<<<< SEARCH\nx = 1" in reflection_prompt  # the patch under review is quoted as written
    assert "line numbers" not in reflection_prompt and "```diff" not in reflection_prompt
    assert reflection_call["tool_choice"] == "none" and reflection_call["max_tokens"] == 8000
    assert response.reflection.outcome == "approved" and response.reflection_text.startswith("<<<PATCH_APPROVED>>>")
    assert response.final_answer == DIFF_ANSWER  # the first answer is kept for comparison
    # reflection tokens are tracked apart from the main loop
    assert response.usage.total_tokens == 30 and response.reflection.total_tokens == 15


async def test_reflection_is_skipped_without_a_patch():
    provider = ScriptedProvider([reply("<<<CANNOT_SOLVE>>>\nreason: too_complex\nexplanation: x")])
    response = await provider.generate_response(start(), experiment_mode="self_reflect")
    assert response.reflection.outcome == "skipped" and response.reflection_text is None and len(provider.calls) == 1


async def test_failed_reflection_call_fails_the_instance():
    provider = ScriptedProvider([reply(DIFF_ANSWER), ValueError("boom")])
    response = await provider.generate_response(start(), experiment_mode="self_reflect")
    assert response.success is False and response.error_kind == "api_error"
