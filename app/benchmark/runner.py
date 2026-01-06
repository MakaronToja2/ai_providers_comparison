"""
Benchmark runner for executing SWE-bench experiments.
Handles batch processing with concurrency, resume capability, and progress tracking.
"""
import asyncio
import os
import re
import time
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from ..core.provider_factory import ProviderFactory
from ..models.requests import AIProvider
from ..core.tools import tool_registry
from ..models.requests import ChatMessage
from ..models.benchmark import (
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    ExperimentResult,
    ResultStatus,
    ToolCallRecord,
    ProgressUpdate,
)
from ..utils.swe_bench_loader import SWEBenchLoader
from ..utils.repo_context import RepoContext
from .storage import Storage
from .rate_limiter import RateLimiterManager
from .patch_extractor import patch_extractor


@dataclass
class WorkItem:
    """A single unit of work to process."""
    instance_id: str
    provider: str
    model: str
    tool_set_name: str
    enabled_tools: List[str]
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: Optional[str]
    patch: Optional[str]
    fail_to_pass: List[str]
    pass_to_pass: List[str]
    split: str = "dev"


class ExperimentRunner:
    """Runs benchmark experiments with concurrency and resume support."""

    def __init__(
        self,
        storage: Storage,
        rate_limiter_manager: Optional[RateLimiterManager] = None,
        swe_bench_loader: Optional[SWEBenchLoader] = None,
    ):
        self.storage = storage
        self.rate_limiters = rate_limiter_manager or RateLimiterManager()
        self.swe_bench_loader = swe_bench_loader or SWEBenchLoader()
        self._running_experiments: Set[str] = set()
        self._stop_signals: Set[str] = set()

    async def start_experiment(
        self,
        experiment_id: str,
        max_concurrent: int = 3,
    ) -> AsyncGenerator[ProgressUpdate, None]:
        """
        Start or resume an experiment with progress streaming.

        Yields ProgressUpdate for each completed work item.
        """
        experiment = await self.storage.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        if experiment_id in self._running_experiments:
            raise ValueError(f"Experiment {experiment_id} is already running")

        self._running_experiments.add(experiment_id)
        self._stop_signals.discard(experiment_id)

        try:
            # Update experiment status
            experiment.status = ExperimentStatus.RUNNING
            experiment.started_at = experiment.started_at or datetime.utcnow()
            await self.storage.update_experiment(experiment)

            # Get work items (excluding already completed)
            work_items = await self._get_pending_work(experiment)
            total_count = len(work_items)

            if total_count == 0:
                experiment.status = ExperimentStatus.COMPLETED
                experiment.completed_at = datetime.utcnow()
                await self.storage.update_experiment(experiment)
                return

            # Update total count
            experiment.total_instances = total_count + experiment.completed_instances
            await self.storage.update_experiment(experiment)

            # Process with concurrency limit
            semaphore = asyncio.Semaphore(max_concurrent)
            completed_count = experiment.completed_instances

            async def process_with_limit(work: WorkItem) -> Optional[ProgressUpdate]:
                if experiment_id in self._stop_signals:
                    return None

                async with semaphore:
                    if experiment_id in self._stop_signals:
                        return None

                    # Rate limit
                    await self.rate_limiters.acquire(work.provider)

                    # Process the work item
                    result = await self._process_work_item(experiment_id, work)

                    return ProgressUpdate(
                        experiment_id=experiment_id,
                        instance_id=work.instance_id,
                        provider=work.provider,
                        status="completed" if result.success else "failed",
                        success=result.success,
                        error=result.error_message,
                        completed_count=0,  # Will be updated
                        total_count=total_count,
                    )

            # Create tasks
            tasks = [asyncio.create_task(process_with_limit(w)) for w in work_items]

            # Yield progress as tasks complete
            for coro in asyncio.as_completed(tasks):
                update = await coro
                if update:
                    completed_count += 1
                    update.completed_count = completed_count

                    # Update experiment progress
                    experiment.completed_instances = completed_count
                    await self.storage.update_experiment(experiment)

                    yield update

                if experiment_id in self._stop_signals:
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    break

            # Final status update
            if experiment_id in self._stop_signals:
                experiment.status = ExperimentStatus.PAUSED
            else:
                experiment.status = ExperimentStatus.COMPLETED
                experiment.completed_at = datetime.utcnow()

            await self.storage.update_experiment(experiment)

        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            await self.storage.update_experiment(experiment)
            raise
        finally:
            self._running_experiments.discard(experiment_id)
            self._stop_signals.discard(experiment_id)

    async def stop_experiment(self, experiment_id: str) -> None:
        """Signal an experiment to stop."""
        if experiment_id in self._running_experiments:
            self._stop_signals.add(experiment_id)

    def is_running(self, experiment_id: str) -> bool:
        """Check if an experiment is currently running."""
        return experiment_id in self._running_experiments

    async def _get_pending_work(self, experiment: Experiment) -> List[WorkItem]:
        """Get work items that haven't been completed yet."""
        config = experiment.config

        # Get completed result keys
        completed_keys = await self.storage.get_completed_result_keys(experiment.id)

        # Load SWE-bench instances
        instances = await self._load_instances(config)

        # Build work items
        work_items = []
        for instance in instances:
            for provider in config.providers:
                model = config.models.get(provider, self._get_default_model(provider))

                for tool_set in config.tool_sets:
                    key = (instance.instance_id, provider, model, tool_set.name)

                    if key not in completed_keys:
                        work_items.append(WorkItem(
                            instance_id=instance.instance_id,
                            provider=provider,
                            model=model,
                            tool_set_name=tool_set.name,
                            enabled_tools=tool_set.enabled_tools,
                            repo=instance.repo,
                            base_commit=instance.base_commit,
                            problem_statement=instance.problem_statement,
                            hints_text=instance.hints_text,
                            patch=instance.patch,
                            fail_to_pass=instance.FAIL_TO_PASS or [],
                            pass_to_pass=instance.PASS_TO_PASS or [],
                            split=config.split or "dev",
                        ))

        return work_items

    async def _load_instances(self, config: ExperimentConfig) -> List[dict]:
        """Load SWE-bench instances based on config filters."""
        split = config.split or "dev"

        # Load the specified split
        await asyncio.get_event_loop().run_in_executor(
            None, self.swe_bench_loader.load_dataset, split
        )

        # Get all instances
        all_instance_ids = await asyncio.get_event_loop().run_in_executor(
            None, self.swe_bench_loader.list_instances, split, 1000
        )

        instances = []
        for instance_id in all_instance_ids:
            instance = await asyncio.get_event_loop().run_in_executor(
                None, self.swe_bench_loader.get_instance, instance_id, split
            )
            if instance:
                # Apply filters
                if config.instance_ids and instance_id not in config.instance_ids:
                    continue
                if config.repos and instance.repo not in config.repos:
                    continue

                instances.append(instance)

                if config.limit and len(instances) >= config.limit:
                    break

        return instances

    async def _process_work_item(
        self,
        experiment_id: str,
        work: WorkItem
    ) -> ExperimentResult:
        """Process a single work item and return the result."""
        result = ExperimentResult(
            experiment_id=experiment_id,
            instance_id=work.instance_id,
            provider=work.provider,
            model=work.model,
            tool_set=work.tool_set_name,
            split=work.split,
            started_at=datetime.utcnow(),
        )

        try:
            # Get provider
            provider = ProviderFactory.create_provider(AIProvider(work.provider), work.model)

            # Configure tools
            original_tool_states = {}
            for tool_name, tool in tool_registry._tools.items():
                original_tool_states[tool_name] = tool_registry._enabled_tools.get(tool_name, False)
                tool_registry._enabled_tools[tool_name] = tool_name in work.enabled_tools

            try:
                # Clone repository and work in its context
                async with RepoContext(work.repo, work.base_commit) as repo_ctx:
                    # Build prompt
                    messages = self._build_analysis_messages(work)
                    result.context_size_chars = sum(len(m.content) for m in messages)

                    # Generate response
                    # Note: 8000 tokens gives models more room to explore AND produce a final diff
                    # Previously 4000 was too low - models got truncated mid-response
                    start_time = time.time()
                    response = await provider.generate_response(
                        messages=messages,
                        temperature=0.3,
                        max_tokens=8000,
                    )
                    result.response_time_seconds = time.time() - start_time

                    # Extract results
                    result.response_content = response.content
                    result.success = response.success

                    if response.usage:
                        result.prompt_tokens = response.usage.prompt_tokens
                        result.completion_tokens = response.usage.completion_tokens
                        result.total_tokens = response.usage.total_tokens
                        result.context_size_tokens = response.usage.prompt_tokens

                    # Process tool calls
                    if response.tool_calls:
                        for tc in response.tool_calls:
                            result.tool_calls.append(ToolCallRecord(
                                name=tc.name,
                                parameters=tc.parameters,
                                result=tc.result,
                                success=tc.success,
                                error=tc.error,
                            ))
                        result.tool_call_count = len(response.tool_calls)
                        result.successful_tool_calls = sum(1 for tc in response.tool_calls if tc.success)

                    # Classify the result status
                    cannot_solve_info = self._check_cannot_solve(response.content)

                    if cannot_solve_info:
                        # Model honestly admitted it cannot solve - this is better than hallucination
                        result.success = False
                        result.result_status = ResultStatus.FAILURE_EXPLICIT
                        result.error_message = f"Model admitted failure: {cannot_solve_info['reason']} - {cannot_solve_info['explanation']}"
                    else:
                        # Try to extract patch from response
                        result.generated_patch = self._extract_patch(response.content)

                        if result.generated_patch:
                            # Successfully extracted a patch
                            result.success = True
                            result.result_status = ResultStatus.SUCCESS_PATCH_GENERATED
                            if not response.success:
                                result.error_message = response.error_message
                        else:
                            # Model responded but produced no valid output
                            result.success = False
                            result.result_status = ResultStatus.HALLUCINATION_FORMAT_ERROR
                            result.error_message = "No valid patch generated (missing ```diff``` block or <<<CANNOT_SOLVE>>> marker)"

            finally:
                # Restore original tool states
                for tool_name, state in original_tool_states.items():
                    tool_registry._enabled_tools[tool_name] = state

        except Exception as e:
            result.success = False
            result.result_status = ResultStatus.API_ERROR
            result.error_message = str(e)

        result.completed_at = datetime.utcnow()

        # Save result
        await self.storage.save_result(result)

        return result

    # Standardized failure markers for when model cannot solve
    CANNOT_SOLVE_MARKER = "<<<CANNOT_SOLVE>>>"
    CANNOT_SOLVE_REASONS = [
        "insufficient_context",      # Can't find relevant code
        "too_complex",               # Problem is too complex to solve
        "unclear_requirements",      # Bug report is unclear
        "missing_dependencies",      # Would need external info/deps
    ]

    def _build_analysis_messages(self, work: WorkItem) -> List[ChatMessage]:
        """Build the prompt messages for analyzing an instance."""
        system_prompt = """You are an expert software engineer tasked with fixing a bug from a GitHub issue.

IMPORTANT: Your response MUST end with either:
1. A complete patch in unified diff format, OR
2. A standardized failure response if you truly cannot solve it

You have access to tools to explore the codebase:
- read_file: Read file contents (use file_path parameter)
- search_code: Search for patterns using regex (use pattern parameter)
- list_directory: List directory contents (use directory_path parameter)

Workflow:
1. Use tools to find the relevant code (2-5 tool calls should be enough)
2. Analyze the bug and determine the fix
3. OUTPUT A COMPLETE PATCH in unified diff format

Your response MUST contain a patch formatted as:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```

IF AND ONLY IF you genuinely cannot produce a patch after exploring the code, respond with EXACTLY:
<<<CANNOT_SOLVE>>>
reason: [insufficient_context|too_complex|unclear_requirements|missing_dependencies]
explanation: [brief explanation of why you cannot solve this]

DO NOT use <<<CANNOT_SOLVE>>> if you can make any reasonable attempt at a patch. A partial or uncertain patch is better than giving up. Only use this if you truly cannot determine what changes to make."""

        user_message = f"""## Repository: {work.repo}

## Bug Report
{work.problem_statement}
"""

        if work.hints_text:
            user_message += f"""
## Hints
{work.hints_text}
"""

        user_message += """
## Instructions
1. Use the tools to locate the relevant code (be efficient - don't over-explore)
2. Identify the root cause of the bug
3. Generate a fix as a unified diff patch

Remember: Your response MUST end with either a ```diff``` code block OR <<<CANNOT_SOLVE>>> marker. No other response format is acceptable."""

        return [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message),
        ]

    def _extract_patch(self, content: Optional[str]) -> Optional[str]:
        """Extract a patch from the response content using the robust extractor."""
        if not content:
            return None

        # Use the robust patch extractor
        patch = patch_extractor.extract(content)
        if patch:
            return patch

        return None

    def _check_cannot_solve(self, content: Optional[str]) -> Optional[Dict[str, str]]:
        """Check if the response contains a <<<CANNOT_SOLVE>>> marker.

        Returns:
            Dict with 'reason' and 'explanation' if marker found, None otherwise.
        """
        if not content:
            return None

        if self.CANNOT_SOLVE_MARKER not in content:
            return None

        # Found the marker - try to extract reason and explanation
        result = {
            "reason": "unknown",
            "explanation": "Model indicated it cannot solve this problem"
        }

        # Try to extract reason
        reason_match = re.search(r'reason:\s*(\w+)', content, re.IGNORECASE)
        if reason_match:
            reason = reason_match.group(1).lower()
            if reason in self.CANNOT_SOLVE_REASONS:
                result["reason"] = reason

        # Try to extract explanation
        explanation_match = re.search(r'explanation:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if explanation_match:
            result["explanation"] = explanation_match.group(1).strip()[:200]  # Limit length

        return result

    def _get_default_model(self, provider: str) -> str:
        """Get default model for a provider from environment or defaults."""
        env_defaults = {
            "openai": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini"),
            "anthropic": os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-haiku-4-5-20251001"),
            "google": os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-2.5-flash-lite"),
        }
        return env_defaults.get(provider, "unknown")


# Global runner instance
_runner: Optional[ExperimentRunner] = None


def get_runner(storage: Optional[Storage] = None) -> ExperimentRunner:
    """Get or create the global runner instance."""
    global _runner
    if _runner is None:
        _runner = ExperimentRunner(storage or Storage())
    return _runner
