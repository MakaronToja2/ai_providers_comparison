"""
Dashboard routes for web UI.
"""
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..benchmark.storage import Storage
from ..benchmark.metrics import MetricsCalculator
from ..benchmark.runner import ExperimentRunner


router = APIRouter(prefix="/dashboard", tags=["dashboard"])

# Setup templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global instances
_storage: Optional[Storage] = None
_metrics: Optional[MetricsCalculator] = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage


def get_metrics() -> MetricsCalculator:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCalculator(get_storage())
    return _metrics


@router.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Dashboard home page."""
    storage = get_storage()
    await storage.initialize()

    experiments = await storage.list_experiments(limit=10)

    # Calculate quick stats
    total_experiments = len(experiments)
    running_count = sum(1 for e in experiments if e.status.value == "running")
    completed_count = sum(1 for e in experiments if e.status.value == "completed")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "experiments": experiments,
        "total_experiments": total_experiments,
        "running_count": running_count,
        "completed_count": completed_count,
    })


@router.get("/experiments", response_class=HTMLResponse)
async def experiments_list(request: Request):
    """Experiments list page."""
    storage = get_storage()
    await storage.initialize()

    experiments = await storage.list_experiments(limit=100)

    return templates.TemplateResponse("experiments.html", {
        "request": request,
        "experiments": experiments,
    })


@router.get("/experiments/{experiment_id}", response_class=HTMLResponse)
async def experiment_detail(request: Request, experiment_id: str):
    """Single experiment detail page."""
    storage = get_storage()
    metrics_calc = get_metrics()
    await storage.initialize()

    experiment = await storage.get_experiment(experiment_id)
    if not experiment:
        return HTMLResponse(content="Experiment not found", status_code=404)

    # Get metrics and stats
    metrics = await storage.get_metrics(experiment_id)
    if not metrics:
        metrics = await metrics_calc.calculate_metrics(experiment_id)

    results = await storage.get_results_for_experiment(experiment_id, limit=50)
    test_results = await storage.get_test_results_for_experiment(experiment_id)

    # Get detailed stats
    stats = await metrics_calc.get_detailed_stats(experiment_id)

    return templates.TemplateResponse("experiment_detail.html", {
        "request": request,
        "experiment": experiment,
        "metrics": metrics,
        "results": results,
        "test_results": test_results,
        "stats": stats,
    })


@router.get("/compare", response_class=HTMLResponse)
async def compare_view(request: Request, experiment_id: Optional[str] = None):
    """Provider comparison page."""
    storage = get_storage()
    metrics_calc = get_metrics()
    await storage.initialize()

    experiments = await storage.list_experiments(limit=100)

    comparison_data = None
    tool_analysis = None
    context_analysis = None
    selected_experiment = None

    if experiment_id:
        selected_experiment = await storage.get_experiment(experiment_id)
        if selected_experiment:
            comparison_data = await metrics_calc.compare_providers(experiment_id)
            tool_analysis = await metrics_calc.get_tool_efficiency_analysis(experiment_id)
            context_analysis = await metrics_calc.get_context_window_analysis(experiment_id)

    return templates.TemplateResponse("compare.html", {
        "request": request,
        "experiments": experiments,
        "selected_experiment": selected_experiment,
        "comparison_data": comparison_data,
        "tool_analysis": tool_analysis,
        "context_analysis": context_analysis,
    })


@router.get("/compare-experiments", response_class=HTMLResponse)
async def compare_experiments_view(request: Request, ids: Optional[str] = None):
    """Cross-experiment comparison page."""
    storage = get_storage()
    metrics_calc = get_metrics()
    await storage.initialize()

    experiments = await storage.list_experiments(limit=100)

    comparison_data = None
    selected_experiments = []

    if ids:
        experiment_ids = [id.strip() for id in ids.split(",") if id.strip()]
        if len(experiment_ids) >= 2:
            comparison_data = await metrics_calc.compare_experiments(experiment_ids)
            selected_experiments = experiment_ids

    return templates.TemplateResponse("compare_experiments.html", {
        "request": request,
        "experiments": experiments,
        "selected_experiments": selected_experiments,
        "comparison_data": comparison_data,
    })
