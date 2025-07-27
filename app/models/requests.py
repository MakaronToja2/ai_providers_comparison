from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class AIProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


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