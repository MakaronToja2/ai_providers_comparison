"""
API endpoints for benchmark experiments.
"""
import io
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, Response
import pandas as pd

from ..benchmark.storage import Storage
from ..benchmark.runner import ExperimentRunner, get_runner
from ..benchmark.evaluator import TestEvaluator
from ..benchmark.metrics import MetricsCalculator
from ..models.benchmark import (
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    CreateExperimentRequest,
    ExperimentResponse,
    ExperimentListResponse,
    ResultsListResponse,
    ToolSetConfig,
)


router = APIRouter(prefix="/benchmark", tags=["benchmark"])

# Global instances
_storage: Optional[Storage] = None
_runner: Optional[ExperimentRunner] = None
_evaluator: Optional[TestEvaluator] = None
_metrics: Optional[MetricsCalculator] = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage


def get_experiment_runner() -> ExperimentRunner:
    global _runner
    if _runner is None:
        _runner = ExperimentRunner(get_storage())
    return _runner


def get_evaluator() -> TestEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = TestEvaluator(get_storage())
    return _evaluator


def get_metrics_calculator() -> MetricsCalculator:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCalculator(get_storage())
    return _metrics


# ==================== Experiments ====================

@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(request: CreateExperimentRequest):
    """Create a new benchmark experiment."""
    storage = get_storage()

    # Build config with defaults
    models = request.models or {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "google": "gemini-2.5-flash-lite",
    }

    tool_sets = request.tool_sets or [
        ToolSetConfig(
            name="all_tools",
            enabled_tools=["read_file", "search_code", "list_directory"]
        )
    ]

    config = ExperimentConfig(
        name=request.name,
        description=request.description,
        providers=request.providers,
        models=models,
        tool_sets=tool_sets,
        context_limit=request.context_limit,
        instance_ids=request.instance_ids,
        repos=request.repos,
        limit=request.limit,
        max_concurrent=request.max_concurrent,
    )

    experiment = Experiment(
        name=request.name,
        description=request.description,
        config=config,
    )

    await storage.create_experiment(experiment)

    return ExperimentResponse(experiment=experiment)


@router.get("/experiments", response_model=ExperimentListResponse)
async def list_experiments(limit: int = 100, offset: int = 0):
    """List all experiments."""
    storage = get_storage()
    experiments = await storage.list_experiments(limit, offset)
    return ExperimentListResponse(experiments=experiments, total=len(experiments))


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """Get experiment details."""
    storage = get_storage()
    experiment = await storage.get_experiment(experiment_id)

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    results_count = await storage.count_results(experiment_id)
    test_count = await storage.count_test_results(experiment_id)

    return ExperimentResponse(
        experiment=experiment,
        results_count=results_count,
        test_results_count=test_count,
    )


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete an experiment and all related data."""
    storage = get_storage()
    runner = get_experiment_runner()

    if runner.is_running(experiment_id):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running experiment. Stop it first."
        )

    await storage.delete_experiment(experiment_id)
    return {"status": "deleted", "experiment_id": experiment_id}


# ==================== Experiment Control ====================

@router.post("/experiments/{experiment_id}/start")
async def start_experiment(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    max_concurrent: int = 3,
):
    """Start or resume an experiment."""
    storage = get_storage()
    runner = get_experiment_runner()

    experiment = await storage.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if runner.is_running(experiment_id):
        raise HTTPException(status_code=400, detail="Experiment is already running")

    # Start in background
    async def run_experiment():
        async for _ in runner.start_experiment(experiment_id, max_concurrent):
            pass  # Progress is saved to storage

    background_tasks.add_task(run_experiment)

    return {"status": "started", "experiment_id": experiment_id}


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Stop a running experiment."""
    runner = get_experiment_runner()

    if not runner.is_running(experiment_id):
        raise HTTPException(status_code=400, detail="Experiment is not running")

    await runner.stop_experiment(experiment_id)
    return {"status": "stopping", "experiment_id": experiment_id}


@router.get("/experiments/{experiment_id}/stream")
async def stream_experiment_progress(experiment_id: str, max_concurrent: int = 3):
    """Stream experiment progress via Server-Sent Events."""
    storage = get_storage()
    runner = get_experiment_runner()

    experiment = await storage.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    async def event_generator():
        try:
            async for update in runner.start_experiment(experiment_id, max_concurrent):
                data = update.model_dump_json()
                yield f"data: {data}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        yield "data: {\"status\": \"complete\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ==================== Results ====================

@router.get("/experiments/{experiment_id}/results", response_model=ResultsListResponse)
async def get_experiment_results(
    experiment_id: str,
    provider: Optional[str] = None,
    success_only: bool = False,
    page: int = 1,
    page_size: int = 50,
):
    """Get paginated results for an experiment."""
    storage = get_storage()

    offset = (page - 1) * page_size
    results = await storage.get_results_for_experiment(
        experiment_id,
        provider=provider,
        success_only=success_only,
        limit=page_size,
        offset=offset,
    )

    total = await storage.count_results(experiment_id, success_only=success_only)

    return ResultsListResponse(
        results=results,
        total=total,
        page=page,
        page_size=page_size,
    )


# ==================== Test Evaluation ====================

@router.post("/experiments/{experiment_id}/evaluate")
async def evaluate_experiment(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    max_concurrent: int = 2,
):
    """Run tests on all generated patches in the experiment."""
    storage = get_storage()
    evaluator = get_evaluator()

    experiment = await storage.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Run evaluation in background
    background_tasks.add_task(
        evaluator.evaluate_experiment,
        experiment_id,
        max_concurrent,
    )

    return {"status": "evaluation_started", "experiment_id": experiment_id}


@router.get("/experiments/{experiment_id}/test-results")
async def get_test_results(
    experiment_id: str,
    resolved_only: bool = False,
):
    """Get test results for an experiment."""
    storage = get_storage()
    test_results = await storage.get_test_results_for_experiment(
        experiment_id,
        resolved_only=resolved_only,
    )
    return {"test_results": [tr.model_dump() for tr in test_results]}


# ==================== Metrics ====================

@router.get("/experiments/{experiment_id}/metrics")
async def get_experiment_metrics(experiment_id: str):
    """Get aggregated metrics for an experiment."""
    calculator = get_metrics_calculator()
    metrics = await calculator.calculate_metrics(experiment_id)
    return metrics.model_dump()


@router.get("/experiments/{experiment_id}/stats")
async def get_experiment_stats(experiment_id: str):
    """Get detailed statistics for an experiment."""
    calculator = get_metrics_calculator()
    stats = await calculator.get_detailed_stats(experiment_id)
    return stats


@router.get("/experiments/{experiment_id}/compare-providers")
async def compare_providers(experiment_id: str):
    """Get provider comparison data for visualization."""
    calculator = get_metrics_calculator()
    comparison = await calculator.compare_providers(experiment_id)
    return comparison


@router.get("/experiments/{experiment_id}/tool-analysis")
async def get_tool_analysis(experiment_id: str):
    """Get tool efficiency analysis (for thesis research)."""
    calculator = get_metrics_calculator()
    analysis = await calculator.get_tool_efficiency_analysis(experiment_id)
    return analysis


@router.get("/experiments/{experiment_id}/context-analysis")
async def get_context_analysis(experiment_id: str):
    """Get context window impact analysis (for thesis research)."""
    calculator = get_metrics_calculator()
    analysis = await calculator.get_context_window_analysis(experiment_id)
    return analysis


# ==================== Export ====================

@router.get("/experiments/{experiment_id}/export")
async def export_experiment(experiment_id: str, format: str = "json"):
    """Export experiment data. Supports 'json' and 'csv' formats."""
    storage = get_storage()

    experiment = await storage.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if format == "json":
        data = await storage.export_results_to_dict(experiment_id)
        return JSONResponse(content=data)

    elif format == "csv":
        # Get all results for CSV export
        results = await storage.get_results_for_experiment(experiment_id, limit=10000)
        test_results = await storage.get_test_results_for_experiment(experiment_id)

        # Build test result lookup
        test_by_result_id = {tr.result_id: tr for tr in test_results}

        # Build rows for CSV
        rows = []
        for r in results:
            tr = test_by_result_id.get(r.id)
            row = {
                "instance_id": r.instance_id,
                "provider": r.provider,
                "model": r.model,
                "tool_set": r.tool_set,
                "success": r.success,
                "error_message": r.error_message or "",
                "prompt_tokens": r.prompt_tokens or 0,
                "completion_tokens": r.completion_tokens or 0,
                "total_tokens": r.total_tokens or 0,
                "context_size_tokens": r.context_size_tokens or 0,
                "response_time_seconds": r.response_time_seconds or 0,
                "tool_call_count": r.tool_call_count,
                "successful_tool_calls": r.successful_tool_calls,
                "has_patch": bool(r.generated_patch),
                "patch_applied": tr.patch_applied if tr else None,
                "resolved": tr.resolved if tr else None,
                "fail_to_pass_total": tr.fail_to_pass_total if tr else None,
                "fail_to_pass_passed": tr.fail_to_pass_passed if tr else None,
                "pass_to_pass_total": tr.pass_to_pass_total if tr else None,
                "pass_to_pass_passed": tr.pass_to_pass_passed if tr else None,
            }
            rows.append(row)

        # Create DataFrame and export to CSV
        df = pd.DataFrame(rows)

        # Write to buffer
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        csv_content = buffer.getvalue()

        # Return as downloadable file
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=experiment_{experiment_id}.csv"
            }
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}. Use 'json' or 'csv'.")
