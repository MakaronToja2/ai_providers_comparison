from typing import List, Dict, Any, Optional
import os
import time
import anthropic

from ..core.base_provider import BaseAIProvider
from ..models.requests import ChatMessage
from ..models.responses import LLMResponse, TokenUsage

# Default model from environment or fallback
DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-haiku-4-5-20251001")


class AnthropicProvider(BaseAIProvider):
    def __init__(self, api_key: str, model: str = None, **kwargs):
        model = model or DEFAULT_ANTHROPIC_MODEL
        super().__init__(api_key, model, **kwargs)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def _make_api_call(self, messages: List[ChatMessage], temperature: float = 0.7, 
                           max_tokens: int = 1000, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        # Claude expects system message separate from conversation
        system_message = None
        conversation_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
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
        
        response = await self.client.messages.create(**params)
        return response
    
    def _parse_response(self, raw_response: Any) -> tuple[str, Optional[TokenUsage], Optional[List[Dict[str, Any]]]]:
        # Extract text content and tool calls
        text_content = ""
        tool_calls = []
        
        if raw_response.content:
            for block in raw_response.content:
                if hasattr(block, 'text') and block.text:
                    text_content += block.text
                elif hasattr(block, 'type') and block.type == 'tool_use':
                    # Anthropic tool call format
                    tool_calls.append({
                        "name": block.name,
                        "parameters": block.input
                    })
        
        usage = raw_response.usage
        usage_obj = TokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens
        ) if usage else None
        
        return text_content, usage_obj, tool_calls if tool_calls else None
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("Anthropic API key is required")