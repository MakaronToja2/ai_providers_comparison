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
        max_tool_iterations: int = 10,
        **kwargs
    ) -> LLMResponse:
        """Generate response with agentic tool loop and standardized output"""
        start_time = time.time()
        all_tool_calls = []
        total_usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        conversation = list(messages)  # Copy messages for the loop

        try:
            # Get available tools if enabled
            tools_spec = None
            if use_tools:
                tool_definitions = tool_registry.get_tool_definitions()
                if tool_definitions:
                    tools_spec = [tool.model_dump() for tool in tool_definitions]

            all_content_parts = []  # Accumulate ALL content throughout iterations
            iteration = 0

            while iteration < max_tool_iterations:
                iteration += 1

                raw_response = await self._make_api_call(
                    messages=conversation,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools_spec,
                    **kwargs
                )

                content, usage, tool_calls_data = self._parse_response(raw_response)

                # Accumulate usage
                if usage:
                    total_usage.prompt_tokens += usage.prompt_tokens
                    total_usage.completion_tokens += usage.completion_tokens
                    total_usage.total_tokens += usage.total_tokens

                # Always accumulate content (fixes bug where content was lost during tool iterations)
                if content:
                    all_content_parts.append(content)

                # If no tool calls, we're done
                if not tool_calls_data:
                    break

                # Execute tool calls
                tool_results = []
                for tool_call_data in tool_calls_data:
                    tool_name = tool_call_data.get("name")
                    parameters = tool_call_data.get("parameters", {})

                    executed_call = await self._execute_tool_call(tool_name, parameters)
                    all_tool_calls.append(executed_call)
                    tool_results.append(executed_call)

                # Add assistant message with tool calls to conversation
                assistant_content = content or f"Calling tools: {', '.join(tc.name for tc in tool_results)}"
                conversation.append(ChatMessage(role="assistant", content=assistant_content))

                # Add tool results to conversation
                tool_result_text = self._format_tool_results(tool_results)
                conversation.append(ChatMessage(role="user", content=tool_result_text))

            response_time = time.time() - start_time

            # Combine all accumulated content
            final_content = "\n\n".join(all_content_parts) if all_content_parts else ""

            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                content=final_content,
                usage=total_usage if total_usage.total_tokens > 0 else None,
                response_time=response_time,
                success=True,
                raw_response=None,  # Don't store all raw responses
                tool_calls=all_tool_calls if all_tool_calls else None
            )

        except Exception as e:
            response_time = time.time() - start_time
            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                content="",
                response_time=response_time,
                success=False,
                error_message=str(e),
                tool_calls=all_tool_calls if all_tool_calls else None
            )

    def _format_tool_results(self, tool_calls: List[ToolCall]) -> str:
        """Format tool call results for sending back to the LLM."""
        results = []
        for tc in tool_calls:
            if tc.success:
                result_str = json.dumps(tc.result, indent=2) if tc.result else "Success"
                results.append(f"## Tool: {tc.name}\n{result_str}")
            else:
                results.append(f"## Tool: {tc.name}\nError: {tc.error}")
        return "Here are the tool results:\n\n" + "\n\n".join(results)
    
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