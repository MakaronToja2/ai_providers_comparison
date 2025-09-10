from typing import List, Dict, Any, Optional
import time
import json
import openai
from openai import AsyncOpenAI

from ..core.base_provider import BaseAIProvider
from ..models.requests import ChatMessage
from ..models.responses import LLMResponse, TokenUsage


class OpenAIProvider(BaseAIProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def _make_api_call(self, messages: List[ChatMessage], temperature: float = 0.7, 
                           max_tokens: int = 1000, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        # Remove any tools from kwargs to avoid conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'tools'}
        
        openai_messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in messages
        ]
        
        params = {
            "model": self.model,
            "messages": openai_messages,
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
        
        response = await self.client.chat.completions.create(**params)
        return response
    
    def _parse_response(self, raw_response: Any) -> tuple[str, Optional[TokenUsage], Optional[List[Dict[str, Any]]]]:
        message = raw_response.choices[0].message
        usage = raw_response.usage
        
        usage_obj = TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        ) if usage else None
        
        # Extract text content and tool calls
        text_content = message.content or ""
        tool_calls = []
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                # OpenAI tool call format
                tool_calls.append({
                    "name": tool_call.function.name,
                    "parameters": json.loads(tool_call.function.arguments)
                })
        
        return text_content, usage_obj, tool_calls if tool_calls else None
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("OpenAI API key is required")