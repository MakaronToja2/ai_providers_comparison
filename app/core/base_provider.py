from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import time
import asyncio
import json
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models.requests import ChatMessage, LLMRequest
from ..models.responses import LLMResponse, TokenUsage, ToolCall
from .tools import tool_registry
from ..config.settings import get_settings


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
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make the actual API call to the provider"""
        pass
    
    @abstractmethod
    def _parse_response(self, raw_response: Dict[str, Any]) -> tuple[str, Optional[TokenUsage], Optional[List[Dict[str, Any]]]]:
        """Parse the provider's response into standardized format"""
        pass
    
    async def _execute_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> ToolCall:
        """Execute a tool call and return the result"""
        settings = get_settings()
        
        # Get the tool from registry
        tool = tool_registry.get_tool(tool_name)
        if not tool:
            return ToolCall(
                name=tool_name,
                parameters=parameters,
                success=False,
                error=f"Tool '{tool_name}' not found or not enabled"
            )
        
        try:
            # Execute the tool
            result = await tool.execute(**parameters)
            
            # Log tool call if debugging is enabled
            if settings.tool_debug:
                print(f"Tool call: {tool_name} with {parameters}")
                print(f"Tool result: {result.model_dump()}")
            
            return ToolCall(
                name=tool_name,
                parameters=parameters,
                result=result.model_dump(),
                success=result.success,
                error=result.error
            )
            
        except Exception as e:
            return ToolCall(
                name=tool_name,
                parameters=parameters,
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_tools: bool = True,
        **kwargs
    ) -> LLMResponse:
        """Generate response with retry logic and standardized output"""
        start_time = time.time()
        
        try:
            # Get available tools if enabled
            tools_spec = None
            if use_tools:
                tool_definitions = tool_registry.get_tool_definitions()
                if tool_definitions:
                    tools_spec = [tool.model_dump() for tool in tool_definitions]
            
            raw_response = await self._make_api_call(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools_spec,
                **kwargs
            )
            
            content, usage, tool_calls_data = self._parse_response(raw_response)
            
            # Execute any tool calls
            executed_tool_calls = []
            if tool_calls_data:
                for tool_call_data in tool_calls_data:
                    tool_name = tool_call_data.get("name")
                    parameters = tool_call_data.get("parameters", {})
                    
                    executed_call = await self._execute_tool_call(tool_name, parameters)
                    executed_tool_calls.append(executed_call)
            
            response_time = time.time() - start_time
            
            # Convert raw_response to dict if it's not already
            raw_response_dict = None
            if isinstance(raw_response, dict):
                raw_response_dict = raw_response
            elif hasattr(raw_response, 'model_dump'):
                try:
                    raw_response_dict = raw_response.model_dump()
                except:
                    raw_response_dict = None
            elif hasattr(raw_response, 'dict'):
                try:
                    raw_response_dict = raw_response.dict()
                except:
                    raw_response_dict = None
            elif hasattr(raw_response, '__dict__'):
                try:
                    raw_response_dict = {k: str(v) for k, v in raw_response.__dict__.items()}
                except:
                    raw_response_dict = None
            
            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                content=content,
                usage=usage,
                response_time=response_time,
                success=True,
                raw_response=raw_response_dict,
                tool_calls=executed_tool_calls if executed_tool_calls else None
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
            print(f"Health check for {self.provider_name}: success={response.success}, content='{response.content[:50]}...', error={response.error_message}")
            return response.success
        except Exception as e:
            print(f"Health check exception for {self.provider_name}: {str(e)}")
            return False