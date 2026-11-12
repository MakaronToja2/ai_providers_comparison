from typing import List, Dict, Any, Optional, Tuple
import os
import anthropic

from ..core.base_provider import BaseAIProvider, ParsedResponse
from ..models.requests import ChatMessage, ToolCallRequest
from ..models.responses import TokenUsage

# Default model from environment or fallback
DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-haiku-4-5-20251001")

# The API rejects empty text blocks, so an assistant turn with no text and no tool call
# (an empty model response) is represented by this placeholder.
EMPTY_TURN_PLACEHOLDER = "(empty response)"


class AnthropicProvider(BaseAIProvider):
    def __init__(self, api_key: str, model: str = None, **kwargs):
        model = model or DEFAULT_ANTHROPIC_MODEL
        super().__init__(api_key, model, **kwargs)
        # Retries are handled (and logged) by BaseAIProvider._call_model
        self.client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0, timeout=self.request_timeout_seconds)

    def _serialise_messages(self, messages: List[ChatMessage]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert the neutral conversation to Messages API turns with native tool blocks."""
        # Claude expects system message separate from conversation
        system_message = None
        conversation_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            elif msg.role == "assistant":
                blocks: List[Dict[str, Any]] = []
                if msg.content and msg.content.strip():
                    blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls or []:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                if not blocks:
                    blocks.append({"type": "text", "text": EMPTY_TURN_PLACEHOLDER})
                conversation_messages.append({"role": "assistant", "content": blocks})
            elif msg.role == "tool":
                conversation_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tr.tool_call_id,
                            "content": tr.content,
                            "is_error": tr.is_error,
                        }
                        for tr in msg.tool_results or []
                    ],
                })
            else:
                previous = conversation_messages[-1] if conversation_messages else None
                if previous and previous["role"] == "user" and isinstance(previous["content"], list):
                    # A user message right after tool results joins the same user turn
                    previous["content"].append({"type": "text", "text": msg.content})
                else:
                    conversation_messages.append({"role": msg.role, "content": msg.content})

        return system_message, conversation_messages

    async def _make_api_call(self, messages: List[ChatMessage], temperature: float = 0.7,
                           max_tokens: int = 1000, tools: Optional[List[Dict[str, Any]]] = None,
                           tool_choice: str = "auto", **kwargs) -> Any:
        system_message, conversation_messages = self._serialise_messages(messages)

        # Remove any tools from kwargs to avoid conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'tools'}

        params = {
            "model": self.model,
            "messages": conversation_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **filtered_kwargs
        }

        if system_message:
            params["system"] = system_message

        # Add tools if provided
        if tools:
            # Convert our tool format to Anthropic's format
            anthropic_tools = []
            for tool in tools:
                anthropic_tools.append({
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"]
                })
            params["tools"] = anthropic_tools
            if tool_choice == "none":
                params["tool_choice"] = {"type": "none"}

        response = await self.client.messages.create(**params)
        return response

    def _parse_response(self, raw_response: Any) -> ParsedResponse:
        # Extract text content and tool calls
        text_content = ""
        tool_calls: List[ToolCallRequest] = []

        for block in raw_response.content or []:
            block_type = getattr(block, 'type', None)
            if block_type == 'tool_use':
                # Anthropic tool call format
                tool_calls.append(ToolCallRequest(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.input or {}),
                ))
            elif getattr(block, 'text', None):
                text_content += block.text

        usage = raw_response.usage
        usage_obj = TokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens
        ) if usage else None

        return ParsedResponse(
            text=text_content,
            tool_calls=tool_calls,
            usage=usage_obj,
            stop_reason=getattr(raw_response, "stop_reason", None),
            api_model_id=getattr(raw_response, "model", None),
            response_id=getattr(raw_response, "id", None),
            raw_usage=usage.model_dump() if usage else None,
            # Only non-zero if caching is requested, which this harness never does
            cached_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)):
            return True
        if isinstance(exc, anthropic.APIStatusError):
            return exc.status_code in (408, 409) or exc.status_code >= 500
        return super()._is_retryable(exc)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
