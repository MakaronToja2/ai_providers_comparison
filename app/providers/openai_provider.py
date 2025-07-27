from typing import List, Dict, Any, Optional
import time
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
                           max_tokens: int = 1000, **kwargs) -> Any:
        openai_messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in messages
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response
    
    def _parse_response(self, raw_response: Any) -> tuple[str, Optional[TokenUsage]]:
        message = raw_response.choices[0].message
        usage = raw_response.usage
        
        usage_obj = TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        ) if usage else None
        
        return message.content, usage_obj
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("OpenAI API key is required")