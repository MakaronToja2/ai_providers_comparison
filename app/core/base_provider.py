from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import time
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models.requests import ChatMessage, LLMRequest
from ..models.responses import LLMResponse, TokenUsage


class BaseAIProvider(ABC):
    """Abstract base class for all AI providers"""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.config = kwargs
        self._validate_api_key()
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider"""
        pass
    
    @abstractmethod
    def _validate_api_key(self) -> None:
        """Validate the API key"""
        pass
    
    @abstractmethod
    async def _make_api_call(
        self, 
        messages: List[ChatMessage], 
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Make the actual API call to the provider"""
        pass
    
    @abstractmethod
    def _parse_response(self, raw_response: Dict[str, Any]) -> tuple[str, Optional[TokenUsage]]:
        """Parse the provider's response into standardized format"""
        pass
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate response with retry logic and standardized output"""
        start_time = time.time()
        
        try:
            raw_response = await self._make_api_call(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            content, usage = self._parse_response(raw_response)
            response_time = time.time() - start_time
            
            # Convert raw_response to dict if it's not already
            raw_response_dict = raw_response
            if hasattr(raw_response, 'model_dump'):
                raw_response_dict = raw_response.model_dump()
            elif hasattr(raw_response, 'dict'):
                raw_response_dict = raw_response.dict()
            elif not isinstance(raw_response, dict):
                raw_response_dict = str(raw_response)
            
            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                content=content,
                usage=usage,
                response_time=response_time,
                success=True,
                raw_response=raw_response_dict
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                content="",
                response_time=response_time,
                success=False,
                error_message=str(e)
            )
    
    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible"""
        try:
            test_messages = [ChatMessage(role="user", content="Hi")]
            response = await self.generate_response(
                messages=test_messages,
                max_tokens=10
            )
            return response.success
        except:
            return False