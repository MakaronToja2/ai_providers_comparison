"""Provider responses are parsed into the neutral form, including the awkward cases."""
from types import SimpleNamespace

import pytest

from google.genai import types

from app.providers.anthropic_provider import AnthropicProvider
from app.providers.google_provider import GoogleProvider
from app.providers.openai_provider import OpenAIProvider


def openai_response(arguments: str):
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120,
                            model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
    call = SimpleNamespace(id="call_9", function=SimpleNamespace(name="read_file", arguments=arguments))
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(id="chatcmpl-1", model="gpt-4.1-mini-2025-04-14", usage=usage,
                           choices=[SimpleNamespace(message=message, finish_reason="tool_calls")])


def test_openai_parses_tool_call_and_model_id():
    parsed = OpenAIProvider(api_key="test")._parse_response(openai_response('{"file_path": "a.py"}'))
    assert parsed.text == ""
    assert parsed.tool_calls[0].id == "call_9"
    assert parsed.tool_calls[0].arguments == {"file_path": "a.py"}
    assert parsed.tool_calls[0].raw_arguments == '{"file_path": "a.py"}'
    assert parsed.api_model_id == "gpt-4.1-mini-2025-04-14"
    assert parsed.stop_reason == "tool_calls"
    assert (parsed.usage.prompt_tokens, parsed.usage.completion_tokens) == (100, 20)


def test_openai_malformed_arguments_become_a_tool_error_not_a_crash():
    parsed = OpenAIProvider(api_key="test")._parse_response(openai_response('{"file_path": '))
    assert parsed.tool_calls[0].arguments == {}
    assert parsed.tool_calls[0].parse_error


def test_anthropic_parses_text_and_tool_use():
    usage = SimpleNamespace(input_tokens=300, output_tokens=40, model_dump=lambda: {"input_tokens": 300, "output_tokens": 40})
    raw = SimpleNamespace(
        id="msg_1", model="claude-haiku-4-5-20251001", stop_reason="tool_use", usage=usage,
        content=[
            SimpleNamespace(type="text", text="Looking."),
            SimpleNamespace(type="tool_use", id="toolu_1", name="search_code", input={"pattern": "def f"}),
        ],
    )
    parsed = AnthropicProvider(api_key="test")._parse_response(raw)
    assert parsed.text == "Looking."
    assert parsed.tool_calls[0].id == "toolu_1" and parsed.tool_calls[0].arguments == {"pattern": "def f"}
    assert parsed.api_model_id == "claude-haiku-4-5-20251001"
    assert parsed.usage.total_tokens == 340


def google_response(parts, finish_reason="STOP"):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish_reason)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=50, candidates_token_count=7, thoughts_token_count=3),
        model_version="gemini-2.5-flash-lite", response_id="resp_1",
    )


def test_google_parses_function_call_with_float_arguments():
    raw = google_response([types.Part(function_call=types.FunctionCall(name="read_file", args={"file_path": "a.py", "max_lines": 500.0}))])
    parsed = GoogleProvider(api_key="test")._parse_response(raw)
    call = parsed.tool_calls[0]
    assert call.name == "read_file" and call.id  # an id is synthesised when Gemini gives none
    assert call.arguments == {"file_path": "a.py", "max_lines": 500.0}
    assert parsed.native is raw.candidates[0].content
    assert parsed.api_model_id == "gemini-2.5-flash-lite"
    # thinking tokens are billed as output
    assert (parsed.usage.prompt_tokens, parsed.usage.completion_tokens) == (50, 10)


def test_google_empty_candidate_content_is_an_empty_response():
    raw = types.GenerateContentResponse(
        candidates=[types.Candidate(content=None, finish_reason="MALFORMED_FUNCTION_CALL")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=5, candidates_token_count=0),
    )
    parsed = GoogleProvider(api_key="test")._parse_response(raw)
    assert parsed.text == "" and parsed.tool_calls == [] and parsed.native is None
    assert parsed.stop_reason == "MALFORMED_FUNCTION_CALL"


def test_provider_reported_cached_tokens_are_captured():
    raw = openai_response('{"file_path": "a.py"}')
    raw.usage.prompt_tokens_details = SimpleNamespace(cached_tokens=64)
    assert OpenAIProvider(api_key="test")._parse_response(raw).cached_tokens == 64

    gemini = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text="ok")]), finish_reason="STOP")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=50, candidates_token_count=7, cached_content_token_count=30),
    )
    assert GoogleProvider(api_key="test")._parse_response(gemini).cached_tokens == 30


def test_cost_at_list_price_and_as_billed():
    from app.benchmark.pricing import billed_cost_usd, cost_usd
    assert cost_usd("gpt-4.1-mini", 1_000_000, 100_000) == pytest.approx(0.40 + 0.16)
    # 750k of the million input tokens came from the cache: 250k at $0.40, 750k at $0.10
    assert billed_cost_usd("gpt-4.1-mini", 1_000_000, 100_000, 750_000) == pytest.approx(0.10 + 0.075 + 0.16)
    assert billed_cost_usd("claude-haiku-4-5-20251001", 1_000_000, 0, 0) == cost_usd("claude-haiku-4-5-20251001", 1_000_000, 0)
    with pytest.raises(KeyError):
        cost_usd("some-other-model", 1, 1)
