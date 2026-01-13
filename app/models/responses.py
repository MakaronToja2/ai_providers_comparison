from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(..., description="Tokens in the prompt")
    completion_tokens: int = Field(..., description="Tokens in the completion")
    total_tokens: int = Field(..., description="Total tokens used")


class ToolCall(BaseModel):
    name: str = Field(..., description="Tool name")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters")
    result: Optional[Dict[str, Any]] = Field(None, description="Tool execution result")
    success: bool = Field(..., description="Whether tool call succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


class ReflectionMetrics(BaseModel):
    """Metrics from self-reflection phase (Experiment 3)."""
    outcome: str = Field(..., description="approved, revised, skipped, or failed")
    prompt_tokens: int = Field(0, description="Tokens in reflection prompt")
    completion_tokens: int = Field(0, description="Tokens in reflection response")
    total_tokens: int = Field(0, description="Total reflection tokens")


class LLMResponse(BaseModel):
    provider: str = Field(..., description="AI provider used")
    model: str = Field(..., description="Model used")
    content: str = Field(..., description="Generated content")
    usage: Optional[TokenUsage] = Field(None, description="Token usage information (base generation)")
    reflection: Optional[ReflectionMetrics] = Field(None, description="Self-reflection metrics if applicable")
    response_time: float = Field(..., description="Response time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = Field(True, description="Whether the request was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Raw provider response")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls made during generation")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Full conversation history including tool results (for self-reflection)")


class ComparisonResponse(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    responses: List[LLMResponse] = Field(..., description="Responses from each provider")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    timestamp: datetime = Field(default_factory=datetime.now)


class ProviderHealth(BaseModel):
    healthy: bool = Field(..., description="Whether the provider is healthy")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    providers: Dict[str, ProviderHealth] = Field(..., description="Provider availability status")
    timestamp: datetime = Field(default_factory=datetime.now)