"""
Pydantic models for benchmark experiments and results.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolSetConfig(BaseModel):
    """Configuration for a set of tools to test."""
    name: str
    enabled_tools: List[str]


class ExperimentConfig(BaseModel):
    """Configuration for creating an experiment."""
    name: str
    description: Optional[str] = None
    providers: List[str]  # e.g., ["openai", "anthropic", "google"]
    models: Dict[str, str]  # provider -> model mapping
    tool_sets: List[ToolSetConfig] = Field(default_factory=lambda: [
        ToolSetConfig(name="all_tools", enabled_tools=["read_file", "search_code", "list_directory"])
    ])
    context_limit: Optional[int] = None  # Token limit for context
    instance_ids: Optional[List[str]] = None  # Specific instances to run
    repos: Optional[List[str]] = None  # Filter by repos
    limit: Optional[int] = None  # Limit number of instances
    max_concurrent: int = 3


class Experiment(BaseModel):
    """A benchmark experiment."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    config: ExperimentConfig
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_instances: int = 0
    completed_instances: int = 0


class ToolCallRecord(BaseModel):
    """Record of a single tool call."""
    name: str
    parameters: Dict[str, Any]
    result: Optional[Any] = None
    success: bool = True
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


class ExperimentResult(BaseModel):
    """Result of running a single instance with a provider."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str
    instance_id: str
    provider: str
    model: str
    tool_set: str  # Name of the tool set used

    # Response data
    response_content: Optional[str] = None
    generated_patch: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    # Timing
    response_time_seconds: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Token metrics
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Tool usage
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    tool_call_count: int = 0
    successful_tool_calls: int = 0

    # Context metrics
    context_size_chars: Optional[int] = None
    context_size_tokens: Optional[int] = None

    # Status
    success: bool = False
    error_message: Optional[str] = None


class TestResult(BaseModel):
    """Result of running tests on a generated patch."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    result_id: str  # Links to ExperimentResult
    experiment_id: str
    instance_id: str

    # Patch application
    patch_applied: bool = False
    patch_error: Optional[str] = None

    # Test execution
    fail_to_pass_total: int = 0
    fail_to_pass_passed: int = 0
    pass_to_pass_total: int = 0
    pass_to_pass_passed: int = 0

    # Overall result
    resolved: bool = False  # True if all tests pass

    # Execution details
    test_output: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    executed_at: Optional[datetime] = None


class ExperimentMetrics(BaseModel):
    """Aggregated metrics for an experiment."""
    experiment_id: str

    # Success rates
    overall_success_rate: float = 0.0
    success_rate_by_provider: Dict[str, float] = Field(default_factory=dict)
    success_rate_by_tool_set: Dict[str, float] = Field(default_factory=dict)

    # Tool usage patterns
    tool_usage_by_provider: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    avg_tool_calls_per_instance: Dict[str, float] = Field(default_factory=dict)

    # Token/context metrics
    avg_tokens_by_provider: Dict[str, float] = Field(default_factory=dict)
    avg_context_size_by_provider: Dict[str, float] = Field(default_factory=dict)

    # Timing
    avg_response_time_by_provider: Dict[str, float] = Field(default_factory=dict)
    total_runtime_seconds: float = 0.0

    # Breakdown by repo
    success_by_repo: Dict[str, float] = Field(default_factory=dict)

    updated_at: datetime = Field(default_factory=datetime.utcnow)


# API Request/Response models
class CreateExperimentRequest(BaseModel):
    """Request to create a new experiment."""
    name: str
    description: Optional[str] = None
    providers: List[str] = ["openai", "anthropic", "google"]
    models: Optional[Dict[str, str]] = None  # If None, use defaults
    tool_sets: Optional[List[ToolSetConfig]] = None
    context_limit: Optional[int] = None
    instance_ids: Optional[List[str]] = None
    repos: Optional[List[str]] = None
    limit: Optional[int] = None
    max_concurrent: int = 3


class ExperimentResponse(BaseModel):
    """Response with experiment details."""
    experiment: Experiment
    results_count: int = 0
    test_results_count: int = 0


class ExperimentListResponse(BaseModel):
    """Response with list of experiments."""
    experiments: List[Experiment]
    total: int


class ResultsListResponse(BaseModel):
    """Response with paginated results."""
    results: List[ExperimentResult]
    total: int
    page: int
    page_size: int


class ProgressUpdate(BaseModel):
    """Real-time progress update for streaming."""
    experiment_id: str
    instance_id: str
    provider: str
    status: str  # "started", "completed", "failed"
    success: Optional[bool] = None
    error: Optional[str] = None
    completed_count: int = 0
    total_count: int = 0
