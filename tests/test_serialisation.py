"""The neutral conversation must reach each provider as native tool-call / tool-result turns."""
import json

from app.models.requests import ChatMessage, ToolCallRequest, ToolResultMessage
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.google_provider import GoogleProvider
from app.providers.openai_provider import OpenAIProvider


def conversation():
    """System, user, assistant with one tool call, its result, then a user message
    (the shape of a forced final answer after the iteration cap)."""
    return [
        ChatMessage(role="system", content="SYSTEM"),
        ChatMessage(role="user", content="BUG REPORT"),
        ChatMessage(
            role="assistant",
            content="Let me read the file.",
            tool_calls=[ToolCallRequest(id="call_1", name="read_file",
                                        arguments={"file_path": "a.py", "max_lines": 50},
                                        raw_arguments='{"file_path": "a.py", "max_lines": 50}')],
        ),
        ChatMessage(role="tool", tool_results=[
            ToolResultMessage(tool_call_id="call_1", name="read_file", content='{"content": "x = 1"}'),
        ]),
        ChatMessage(role="user", content="FINAL ANSWER REQUIRED"),
    ]


def no_placeholder(serialised) -> bool:
    return "Calling tools:" not in json.dumps(serialised, default=str)


def test_openai_native_tool_turns():
    messages = OpenAIProvider(api_key="test")._serialise_messages(conversation())
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
    call = messages[2]["tool_calls"][0]
    assert call == {"id": "call_1", "type": "function",
                    "function": {"name": "read_file", "arguments": '{"file_path": "a.py", "max_lines": 50}'}}
    assert messages[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"content": "x = 1"}'}
    assert no_placeholder(messages)


def test_openai_assistant_without_text_or_calls_has_string_content():
    messages = OpenAIProvider(api_key="test")._serialise_messages([ChatMessage(role="assistant", content="")])
    assert messages == [{"role": "assistant", "content": ""}]


def test_anthropic_native_tool_turns_and_merged_user_turn():
    system, messages = AnthropicProvider(api_key="test")._serialise_messages(conversation())
    assert system == "SYSTEM"
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == [
        {"type": "text", "text": "Let me read the file."},
        {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"file_path": "a.py", "max_lines": 50}},
    ]
    # tool result first, then the following user text in the same turn (roles must alternate)
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": '{"content": "x = 1"}', "is_error": False},
        {"type": "text", "text": "FINAL ANSWER REQUIRED"},
    ]
    assert no_placeholder(messages)


def test_anthropic_empty_assistant_turn_gets_placeholder():
    _, messages = AnthropicProvider(api_key="test")._serialise_messages([
        ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="  "),
    ])
    assert messages[1]["content"] == [{"type": "text", "text": "(empty response)"}]


def test_google_native_function_turns_and_system_instruction():
    system, contents = GoogleProvider(api_key="test")._serialise_messages(conversation())
    assert system == "SYSTEM"  # a real system instruction, not an extra user turn
    assert [c.role for c in contents] == ["user", "model", "user"]
    model_parts = contents[1].parts
    assert model_parts[0].text == "Let me read the file."
    assert model_parts[1].function_call.name == "read_file"
    assert model_parts[1].function_call.args == {"file_path": "a.py", "max_lines": 50}
    response_parts = contents[2].parts
    assert response_parts[0].function_response.name == "read_file"
    assert response_parts[0].function_response.response == {"output": '{"content": "x = 1"}'}
    assert response_parts[1].text == "FINAL ANSWER REQUIRED"
    assert no_placeholder([c.model_dump(exclude_none=True) for c in contents])


def test_google_echoes_native_model_turn():
    from google.genai import types
    native = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="list_directory", args={"directory_path": "."}))])
    _, contents = GoogleProvider(api_key="test")._serialise_messages([
        ChatMessage(role="user", content="go"),
        ChatMessage(role="assistant", content="", native=native,
                    tool_calls=[ToolCallRequest(id="x", name="list_directory", arguments={"directory_path": "."})]),
    ])
    assert contents[1] is native


def test_google_error_result_uses_error_key():
    _, contents = GoogleProvider(api_key="test")._serialise_messages([
        ChatMessage(role="tool", tool_results=[ToolResultMessage(tool_call_id="c", name="read_file", content="Error: nope", is_error=True)]),
    ])
    assert contents[0].parts[0].function_response.response == {"error": "Error: nope"}


async def test_google_answer_only_call_declares_no_tools(monkeypatch):
    provider = GoogleProvider(api_key="test")
    captured = {}

    async def fake_generate_content(model, contents, config):
        captured["config"] = config
        return "response"

    monkeypatch.setattr(provider.client.aio.models, "generate_content", fake_generate_content)
    tools = [{"name": "read_file", "description": "d", "parameters": {"type": "object", "properties": {}}}]

    await provider._make_api_call(conversation(), tools=tools, tool_choice="auto")
    assert captured["config"].tools and captured["config"].tools[0].function_declarations[0].name == "read_file"

    await provider._make_api_call(conversation(), tools=tools, tool_choice="none")
    assert not captured["config"].tools and captured["config"].tool_config is None
