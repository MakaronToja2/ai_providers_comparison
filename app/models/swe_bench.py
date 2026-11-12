from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

from .requests import AIProvider
from .responses import LLMResponse


class SWEBenchInstance(BaseModel):
    """SWE-bench dataset instance model"""
    instance_id: str = Field(..., description="Unique identifier for the instance")
    repo: str = Field(..., description="Repository name")
    base_commit: str = Field(..., description="Base commit hash")
    patch: str = Field(..., description="The patch/diff to be analyzed") 
    test_patch: str = Field(..., description="Test patch for validation")
    problem_statement: str = Field(..., description="Description of the problem/issue")
    hints_text: Optional[str] = Field(None, description="Additional hints for solving")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    version: Optional[str] = Field(None, description="Version information")
    FAIL_TO_PASS: Optional[List[str]] = Field(None, description="Tests that should pass")
    PASS_TO_PASS: Optional[List[str]] = Field(None, description="Tests that should continue passing")
    environment_setup_commit: Optional[str] = Field(None, description="Environment setup commit")


class SWEBenchAnalysisRequest(BaseModel):
    """Request model for SWE-bench analysis"""
    instance_id: str = Field(..., description="SWE-bench instance ID to analyze")
    provider: AIProvider = Field(..., description="AI provider to use")
    model: Optional[str] = Field(None, description="Specific model to use (uses default if not provided)")
    analysis_type: str = Field(
        default="bug_analysis", 
        description="Type of analysis: bug_analysis, solution_generation, code_review"
    )
    include_patch: bool = Field(True, description="Whether to include the patch in analysis")
    include_tests: bool = Field(True, description="Whether to include test information")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt override")


class SWEBenchAnalysisResponse(BaseModel):
    """Response model for SWE-bench analysis"""
    instance_id: str = Field(..., description="SWE-bench instance ID analyzed")
    analysis_type: str = Field(..., description="Type of analysis performed")
    llm_response: LLMResponse = Field(..., description="Raw LLM response")
    analysis_summary: Dict[str, Any] = Field(..., description="Structured analysis summary")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


class SWEBenchDatasetInfo(BaseModel):
    """Information about loaded SWE-bench dataset"""
    split: str = Field(..., description="Dataset split (dev/test)")
    total_instances: int = Field(..., description="Total number of instances")
    loaded_at: datetime = Field(default_factory=datetime.now, description="Load timestamp")
    sample_instance_ids: List[str] = Field(..., description="Sample of instance IDs")