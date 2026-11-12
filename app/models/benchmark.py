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


class ResultStatus(str, Enum):
    """Status of an individual result - what the model produced."""
    SUCCESS_PATCH_GENERATED = "success_patch_generated"  # Valid patch extracted
    FAILURE_EXPLICIT = "failure_explicit"  # Model admitted it cannot solve (<<<CANNOT_SOLVE>>>)
    HALLUCINATION_FORMAT_ERROR = "hallucination_format_error"  # Model responded but no valid output
    EDIT_NOT_APPLICABLE = "edit_not_applicable"  # Edits given, but they do not match the repository's code
    CONTEXT_OVERFLOW = "context_overflow"  # Conversation exceeded the model's context window
    API_ERROR = "api_error"  # Provider API error (not a model outcome; re-run on resume)
    INFRA_ERROR = "infra_error"  # Harness-side failure such as a clone error (re-run on resume)


# Statuses that describe what the model did. Everything else is re-run on resume and
# must not reach the analysis.
TERMINAL_RESULT_STATUSES = frozenset({
    ResultStatus.SUCCESS_PATCH_GENERATED,
    ResultStatus.FAILURE_EXPLICIT,
    ResultStatus.HALLUCINATION_FORMAT_ERROR,
    ResultStatus.EDIT_NOT_APPLICABLE,
    ResultStatus.CONTEXT_OVERFLOW,
})


class ExperimentMode(str, Enum):
    """Mode for how the experiment handles model output."""
    BASELINE = "baseline"  # Exp 1: Allow <<<CANNOT_SOLVE>>> escape hatch
    FORCED = "forced"  # Exp 2: Must produce patch, no escape hatch
    SELF_REFLECT = "self_reflect"  # Exp 3: Produce patch, then self-review


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
    split: str = "dev"  # SWE-bench split: "dev", "lite", "test", "verified"
    experiment_mode: ExperimentMode = ExperimentMode.BASELINE  # How to handle model output
    max_tool_iterations: int = 10  # Maximum tool calls before forcing final answer
    context_limit: Optional[int] = None  # Token limit for context
    instance_ids: Optional[List[str]] = None  # Specific instances to run
    repos: Optional[List[str]] = None  # Filter by repos
    limit: Optional[int] = None  # Limit number of instances
    max_concurrent: int = 3
    # Provenance of a rerun job (one provider, one mode, one repetition)
    run_tag: Optional[str] = None
    run_idx: Optional[int] = None  # Repetition index, 1-based
    instances_file: Optional[str] = None
    instances_sha256: Optional[str] = None
    dataset_revision: Optional[str] = None
    harness_commit: Optional[str] = None


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
    split: str = "dev"  # SWE-bench split used

    # Response data
    response_content: Optional[str] = None
    generated_patch: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    # Timing
    response_time_seconds: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Token metrics (base generation)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Self-reflection token metrics (separate tracking for cost analysis)
    reflection_prompt_tokens: Optional[int] = None
    reflection_completion_tokens: Optional[int] = None
    reflection_total_tokens: Optional[int] = None
    reflection_outcome: Optional[str] = None  # "approved", "revised", "unstructured", "skipped"

    # Self-reflection: the patch before the reflection call (generated_patch holds the final one)
    original_patch: Optional[str] = None

    # Loop statistics and provenance: run_idx, experiment_mode, iterations, termination_reason,
    # fallback_used, abstained, api_call_count, api_retry_count, api_model_ids,
    # max_input_tokens, final_diff_blocks, trajectory_path
    run_meta: Optional[Dict[str, Any]] = None

    # Tool usage
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    tool_call_count: int = 0
    successful_tool_calls: int = 0

    # Context metrics
    context_size_chars: Optional[int] = None
    context_size_tokens: Optional[int] = None

    # Status
    success: bool = False
    result_status: Optional[ResultStatus] = None  # Detailed outcome classification
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

    # Success rates (API call success - less meaningful)
    overall_success_rate: float = 0.0
    success_rate_by_provider: Dict[str, float] = Field(default_factory=dict)
    success_rate_by_tool_set: Dict[str, float] = Field(default_factory=dict)

    # Patch generation rates (more meaningful metric)
    overall_patch_rate: float = 0.0
    patch_rate_by_provider: Dict[str, float] = Field(default_factory=dict)

    # Result status breakdown (thesis-relevant)
    status_counts: Dict[str, int] = Field(default_factory=dict)  # ResultStatus -> count
    status_by_provider: Dict[str, Dict[str, int]] = Field(default_factory=dict)  # provider -> status -> count

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
    split: str = "dev"  # SWE-bench split: "dev", "lite", "test", "verified"
    experiment_mode: ExperimentMode = ExperimentMode.BASELINE  # baseline, forced, or self_reflect
    max_tool_iterations: int = 10  # Maximum tool calls before forcing final answer
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
