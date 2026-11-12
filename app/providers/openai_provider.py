from typing import List, Dict, Any, Optional
import os
import json
import openai
from openai import AsyncOpenAI

from ..core.base_provider import BaseAIProvider, ParsedResponse
from ..models.requests import ChatMessage, ToolCallRequest
from ..models.responses import TokenUsage

# Default model from environment or fallback
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini")


class OpenAIProvider(BaseAIProvider):
    def __init__(self, api_key: str, model: str = None, **kwargs):
        model = model or DEFAULT_OPENAI_MODEL
        super().__init__(api_key, model, **kwargs)
        # Retries are handled (and logged) by BaseAIProvider._call_model
        self.client = AsyncOpenAI(api_key=api_key, max_retries=0, timeout=self.request_timeout_seconds)

    def _serialise_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert the neutral conversation to Chat Completions messages with native tool turns."""
        openai_messages: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.role == "assistant":
                entry: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    entry["content"] = msg.content or None
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                # Echo the model's own argument string when we have it
                                "arguments": tc.raw_arguments if tc.raw_arguments is not None else json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                openai_messages.append(entry)
            elif msg.role == "tool":
                # One tool message per call, directly after the assistant message
                for tr in msg.tool_results or []:
                    openai_messages.append({"role": "tool", "tool_call_id": tr.tool_call_id, "content": tr.content})
            else:
                openai_messages.append({"role": msg.role, "content": msg.content})
        return openai_messages

    async def _make_api_call(self, messages: List[ChatMessage], temperature: float = 0.7,
                           max_tokens: int = 1000, tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: str = "auto", **kwargs) -> Any:
        # Remove any tools from kwargs to avoid conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'tools'}

        params = {
            "model": self.model,
            "messages": self._serialise_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            **filtered_kwargs
        }

        # Add tools if provided
        if tools:
            # Convert our tool format to OpenAI's format
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                })
            params["tools"] = openai_tools
            if tool_choice == "none":
                params["tool_choice"] = "none"

        response = await self.client.chat.completions.create(**params)
        return response

    def _parse_response(self, raw_response: Any) -> ParsedResponse:
        choice = raw_response.choices[0]
        message = choice.message
        usage = raw_response.usage

        usage_obj = TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        ) if usage else None

        # Extract text content and tool calls
        tool_calls: List[ToolCallRequest] = []
        for tool_call in (getattr(message, 'tool_calls', None) or []):
            raw_arguments = tool_call.function.arguments or ""
            arguments: Dict[str, Any] = {}
            parse_error = None
            try:
                loaded = json.loads(raw_arguments) if raw_arguments.strip() else {}
                if isinstance(loaded, dict):
                    arguments = loaded
                else:
                    parse_error = "arguments are not a JSON object"
            except json.JSONDecodeError as e:
                parse_error = f"arguments are not valid JSON ({e})"
            tool_calls.append(ToolCallRequest(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=arguments,
                raw_arguments=raw_arguments,
                parse_error=parse_error,
            ))

        return ParsedResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            usage=usage_obj,
            stop_reason=choice.finish_reason,
            api_model_id=getattr(raw_response, "model", None),
            response_id=getattr(raw_response, "id", None),
            raw_usage=usage.model_dump() if usage else None,
            # OpenAI caches prompts automatically; cached tokens are part of prompt_tokens
            cached_tokens=getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
        )

    def _is_retryable(self, exc: Exception) -> bool:
        # An exhausted account is reported as HTTP 429 but will not recover by waiting
        if getattr(exc, "code", None) == "insufficient_quota":
            return False
        if isinstance(exc, (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code in (408, 409) or exc.status_code >= 500
        return super()._is_retryable(exc)

    @property
    def provider_name(self) -> str:
        return "openai"

    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
