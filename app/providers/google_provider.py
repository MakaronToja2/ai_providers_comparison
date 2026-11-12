from typing import List, Dict, Any, Optional, Tuple
import os
import uuid

from google import genai
from google.genai import types, errors

from ..core.base_provider import BaseAIProvider, ParsedResponse
from ..models.requests import ChatMessage, ToolCallRequest
from ..models.responses import TokenUsage

# Default model from environment or fallback
DEFAULT_GOOGLE_MODEL = os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-2.5-flash-lite")

# Stands in for a model turn that came back with no parts (for example a malformed
# function call), so the conversation still alternates between user and model.
EMPTY_TURN_PLACEHOLDER = "(empty response)"


class GoogleProvider(BaseAIProvider):
    """Google Gemini provider (google-genai SDK) with native function-call turns."""

    def __init__(self, api_key: str, model: str = None, **kwargs):
        model = model or DEFAULT_GOOGLE_MODEL
        super().__init__(api_key, model, **kwargs)
        # Retries are handled (and logged) by BaseAIProvider._call_model
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(self.request_timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def _serialise_messages(self, messages: List[ChatMessage]) -> Tuple[Optional[str], List[types.Content]]:
        """Convert the neutral conversation to Gemini contents with native function turns."""
        system_instruction = None
        contents: List[types.Content] = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "assistant":
                if msg.native is not None:
                    # Echo the model's own turn exactly as it was returned
                    contents.append(msg.native)
                    continue
                parts = []
                if msg.content:
                    parts.append(types.Part(text=msg.content))
                for tc in msg.tool_calls or []:
                    parts.append(types.Part(function_call=types.FunctionCall(name=tc.name, args=tc.arguments)))
                if not parts:
                    parts.append(types.Part(text=EMPTY_TURN_PLACEHOLDER))
                contents.append(types.Content(role="model", parts=parts))
            elif msg.role == "tool":
                contents.append(types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=tr.name,
                            response={"error": tr.content} if tr.is_error else {"output": tr.content},
                        )
                        for tr in msg.tool_results or []
                    ],
                ))
            else:
                text_part = types.Part(text=msg.content)
                previous = contents[-1] if contents else None
                if previous is not None and previous.role == "user" and any(p.function_response for p in previous.parts):
                    # A user message right after function responses joins the same user turn
                    previous.parts.append(text_part)
                else:
                    contents.append(types.Content(role="user", parts=[text_part]))

        return system_instruction, contents

    async def _make_api_call(self, messages: List[ChatMessage], temperature: float = 0.7,
                           max_tokens: int = 1000, tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: str = "auto", **kwargs) -> Any:
        system_instruction, contents = self._serialise_messages(messages)

        config_args: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            # The agent loop executes tools itself
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if system_instruction:
            config_args["system_instruction"] = system_instruction

        # On answer-only calls (tool_choice "none") the tools are not declared at all. With the
        # tools declared and function calling set to NONE, Gemini still attempts a call in a
        # share of responses and the API returns an empty UNEXPECTED_TOOL_CALL instead of text.
        # The history keeps its native function-call turns either way.
        if tools and tool_choice != "none":
            config_args["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters_json_schema=tool["parameters"],
                )
                for tool in tools
            ])]

        return await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_args),
        )

    def _parse_response(self, raw_response: Any) -> ParsedResponse:
        text_content = ""
        tool_calls: List[ToolCallRequest] = []
        native = None
        stop_reason = None

        candidates = raw_response.candidates or []
        if candidates:
            candidate = candidates[0]
            if candidate.finish_reason is not None:
                stop_reason = getattr(candidate.finish_reason, "name", None) or str(candidate.finish_reason)
            content = candidate.content
            if content is not None and content.parts:
                native = content
                for part in content.parts:
                    if part.function_call is not None:
                        function_call = part.function_call
                        tool_calls.append(ToolCallRequest(
                            # Gemini does not always assign call ids
                            id=function_call.id or f"call_{uuid.uuid4().hex[:16]}",
                            name=function_call.name or "",
                            arguments=dict(function_call.args or {}),
                        ))
                    elif part.text and not part.thought:
                        text_content += part.text

        usage = None
        raw_usage = None
        cached_tokens = 0
        usage_data = raw_response.usage_metadata
        if usage_data is not None:
            # Gemini 2.5 caches implicitly; cached tokens are part of prompt_token_count
            cached_tokens = usage_data.cached_content_token_count or 0
            raw_usage = usage_data.model_dump(exclude_none=True)
            prompt_tokens = usage_data.prompt_token_count or 0
            # Thinking tokens are billed as output
            completion_tokens = (usage_data.candidates_token_count or 0) + (usage_data.thoughts_token_count or 0)
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        else:
            print("[Google] Warning: No usage_metadata in response")

        return ParsedResponse(
            text=text_content,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=stop_reason,
            api_model_id=raw_response.model_version,
            response_id=raw_response.response_id,
            raw_usage=raw_usage,
            cached_tokens=cached_tokens,
            native=native,
        )

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, errors.ServerError):
            return True
        if isinstance(exc, errors.ClientError):
            return exc.code in (408, 429)
        return super()._is_retryable(exc)

    @property
    def provider_name(self) -> str:
        return "google"

    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("Google API key is required")
