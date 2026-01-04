"""
Benchmark runner for executing SWE-bench experiments.
Handles batch processing with concurrency, resume capability, and progress tracking.
"""
import asyncio
import time
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Set, Tuple
from dataclasses import dataclass

from ..core.provider_factory import provider_factory
from ..core.tools import tool_registry
from ..models.requests import ChatMessage
from ..models.benchmark import (
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    ExperimentResult,
    ToolCallRecord,
    ProgressUpdate,
)
from ..utils.swe_bench_loader import SWEBenchLoader
from ..utils.repo_context import RepoContext
from .storage import Storage
from .rate_limiter import RateLimiterManager


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
                    key = (instance["instance_id"], provider, model, tool_set.name)

                    if key not in completed_keys:
                        work_items.append(WorkItem(
                            instance_id=instance["instance_id"],
                            provider=provider,
                            model=model,
                            tool_set_name=tool_set.name,
                            enabled_tools=tool_set.enabled_tools,
                            repo=instance["repo"],
                            base_commit=instance["base_commit"],
                            problem_statement=instance["problem_statement"],
                            hints_text=instance.get("hints_text"),
                            patch=instance.get("patch"),
                            fail_to_pass=instance.get("FAIL_TO_PASS", []),
                            pass_to_pass=instance.get("PASS_TO_PASS", []),
                        ))

        return work_items

    async def _load_instances(self, config: ExperimentConfig) -> List[dict]:
        """Load SWE-bench instances based on config filters."""
        # Load the dev split (or could be configurable)
        await asyncio.get_event_loop().run_in_executor(
            None, self.swe_bench_loader.load_dataset, "dev"
        )

        # Get all instances
        all_instance_ids = await asyncio.get_event_loop().run_in_executor(
            None, self.swe_bench_loader.list_instances, "dev", 1000
        )

        instances = []
        for instance_id in all_instance_ids:
            instance = await asyncio.get_event_loop().run_in_executor(
                None, self.swe_bench_loader.get_instance, instance_id, "dev"
            )
            if instance:
                # Apply filters
                if config.instance_ids and instance_id not in config.instance_ids:
                    continue
                if config.repos and instance["repo"] not in config.repos:
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
            started_at=datetime.utcnow(),
        )

        try:
            # Get provider
            provider = provider_factory.get_provider(work.provider, work.model)

            # Configure tools
            original_tool_states = {}
            for tool_name, tool in tool_registry._tools.items():
                original_tool_states[tool_name] = tool_registry._enabled_tools.get(tool_name, False)
                tool_registry._enabled_tools[tool_name] = tool_name in work.enabled_tools

            try:
                # Build prompt
                messages = self._build_analysis_messages(work)
                result.context_size_chars = sum(len(m.content) for m in messages)

                # Generate response
                start_time = time.time()
                response = await provider.generate_response(
                    messages=messages,
                    temperature=0.3,
                    max_tokens=4000,
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

                # Extract patch from response (simple heuristic)
                result.generated_patch = self._extract_patch(response.content)

                if not response.success:
                    result.error_message = response.error_message

            finally:
                # Restore original tool states
                for tool_name, state in original_tool_states.items():
                    tool_registry._enabled_tools[tool_name] = state

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        result.completed_at = datetime.utcnow()

        # Save result
        await self.storage.save_result(result)

        return result

    def _build_analysis_messages(self, work: WorkItem) -> List[ChatMessage]:
        """Build the prompt messages for analyzing an instance."""
        system_prompt = """You are an expert software engineer analyzing a bug report from a GitHub issue.
Your task is to understand the problem and propose a solution in the form of a code patch.

You have access to the following tools to explore the codebase:
- read_file: Read the contents of a file
- search_code: Search for code patterns using regex
- list_directory: List files in a directory

Use these tools to understand the codebase structure and find relevant code before proposing a fix.

When you have enough information, provide a patch in unified diff format that fixes the issue.
Format your patch inside a code block with ```diff ... ``` markers."""

        user_message = f"""## Repository
{work.repo}

## Problem Statement
{work.problem_statement}
"""

        if work.hints_text:
            user_message += f"""
## Hints
{work.hints_text}
"""

        user_message += """
Please analyze this issue and provide a fix. Start by exploring the codebase to understand the relevant code, then provide a patch."""

        return [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message),
        ]

    def _extract_patch(self, content: Optional[str]) -> Optional[str]:
        """Extract a patch from the response content."""
        if not content:
            return None

        # Look for diff code blocks
        import re
        diff_pattern = r'```(?:diff)?\s*\n(.*?)```'
        matches = re.findall(diff_pattern, content, re.DOTALL)

        if matches:
            # Return the first diff-like block
            for match in matches:
                if match.strip().startswith(('+', '-', '@@', 'diff', '---', '+++')):
                    return match.strip()
                if 'diff' in match.lower() or '@@' in match:
                    return match.strip()

        return None

    def _get_default_model(self, provider: str) -> str:
        """Get default model for a provider."""
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "google": "gemini-2.5-flash-lite",
        }
        return defaults.get(provider, "unknown")


# Global runner instance
_runner: Optional[ExperimentRunner] = None


def get_runner(storage: Optional[Storage] = None) -> ExperimentRunner:
    """Get or create the global runner instance."""
    global _runner
    if _runner is None:
        _runner = ExperimentRunner(storage or Storage())
    return _runner
