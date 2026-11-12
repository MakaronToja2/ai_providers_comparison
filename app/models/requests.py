from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class AIProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ToolCallRequest(BaseModel):
    """A tool call requested by the model, in provider-neutral form."""
    id: str = Field(..., description="Provider call id (synthesised when the provider gives none)")
    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Parsed arguments")
    raw_arguments: Optional[str] = Field(None, description="Argument string exactly as the provider sent it")
    parse_error: Optional[str] = Field(None, description="Set when the provider's argument string was not valid JSON")


class ToolResultMessage(BaseModel):
    """The result of one tool call, sent back to the model as a native tool-result turn."""
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, assistant, or tool")
    content: str = Field("", description="Message content")
    tool_calls: Optional[List[ToolCallRequest]] = Field(None, description="Tool calls made in an assistant turn")
    tool_results: Optional[List[ToolResultMessage]] = Field(None, description="Tool results carried by a tool turn")
    # Provider's own representation of an assistant turn, echoed back verbatim where the
    # provider needs it (Gemini). Never serialised.
    native: Optional[Any] = Field(None, exclude=True)


class LLMRequest(BaseModel):
    provider: AIProvider = Field(..., description="AI provider to use")
    model: str = Field(..., description="Model name (e.g., gpt-4, claude-3-sonnet)")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(1000, gt=0, description="Maximum tokens to generate")
    system_prompt: Optional[str] = Field(None, description="System prompt override")


class ComparisonRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    providers: List[AIProvider] = Field(..., description="List of providers to compare")
    model_mapping: Dict[str, str] = Field(..., description="Provider to model mapping")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(1000, gt=0)