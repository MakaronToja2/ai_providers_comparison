from typing import List, Dict, Any, Optional
import time
import anthropic

from ..core.base_provider import BaseAIProvider
from ..models.requests import ChatMessage
from ..models.responses import LLMResponse, TokenUsage


class AnthropicProvider(BaseAIProvider):
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def _make_api_call(self, messages: List[ChatMessage], temperature: float = 0.7, 
                           max_tokens: int = 1000, **kwargs) -> Any:
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
        
        params = {
            "model": self.model,
            "messages": conversation_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        if system_message:
            params["system"] = system_message
        
        response = await self.client.messages.create(**params)
        return response
    
    def _parse_response(self, raw_response: Any) -> tuple[str, Optional[TokenUsage]]:
        content = raw_response.content[0].text if raw_response.content else ""
        usage = raw_response.usage
        
        usage_obj = TokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens
        ) if usage else None
        
        return content, usage_obj
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("Anthropic API key is required")