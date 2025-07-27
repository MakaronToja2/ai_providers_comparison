from typing import List, Dict, Any, Optional
import time
import google.generativeai as genai

from ..core.base_provider import BaseAIProvider
from ..models.requests import ChatMessage
from ..models.responses import LLMResponse, TokenUsage


class GoogleProvider(BaseAIProvider):
    def __init__(self, api_key: str, model: str = "gemini2.5-flash-lite", **kwargs):
        super().__init__(api_key, model, **kwargs)
        if api_key:
            genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
    
    async def _make_api_call(self, messages: List[ChatMessage], temperature: float = 0.7, 
                           max_tokens: int = 1000, **kwargs) -> Any:
        # Convert messages to Gemini format
        conversation_parts = []
        
        for msg in messages:
            if msg.role == "system":
                # Gemini doesn't have system role, prepend to first user message
                if conversation_parts and conversation_parts[0].get("role") == "user":
                    conversation_parts[0]["parts"][0] = f"{msg.content}\n\n{conversation_parts[0]['parts'][0]}"
                else:
                    conversation_parts.insert(0, {
                        "role": "user",
                        "parts": [msg.content]
                    })
            else:
                role = "model" if msg.role == "assistant" else msg.role
                conversation_parts.append({
                    "role": role,
                    "parts": [msg.content]
                })
        
        generation_config = {
            "temperature": temperature,
        }
        generation_config["max_output_tokens"] = max_tokens
        
        # Google Generative AI doesn't have native async support yet
        # We'll use the sync method but wrap it
        import asyncio
        response = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self.client.generate_content(
                conversation_parts,
                generation_config=generation_config
            )
        )
        return response
    
    def _parse_response(self, raw_response: Any) -> tuple[str, Optional[TokenUsage]]:
        content = raw_response.text if hasattr(raw_response, 'text') else ""
        
        # Google's usage information might be limited
        usage = None
        if hasattr(raw_response, 'usage_metadata'):
            usage_data = raw_response.usage_metadata
            usage = TokenUsage(
                prompt_tokens=getattr(usage_data, 'prompt_token_count', 0),
                completion_tokens=getattr(usage_data, 'candidates_token_count', 0),
                total_tokens=getattr(usage_data, 'total_token_count', 0)
            )
        
        return content, usage
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("Google API key is required")