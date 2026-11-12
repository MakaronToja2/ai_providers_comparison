"""
Benchmark runner for executing SWE-bench experiments.
Handles batch processing with concurrency, resume capability, and progress tracking.
"""
import asyncio
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set
from dataclasses import dataclass

from ..core.provider_factory import ProviderFactory
from ..models.requests import AIProvider
from ..core.base_provider import PATCH_APPROVED_MARKER, PATCH_REVISED_MARKER
from ..core.edit_format import EDIT_FORMAT_INSTRUCTIONS, parse_edit_blocks
from ..core.tools import tool_registry
from ..core.trajectory import TrajectoryRecorder
from ..models.requests import ChatMessage
from ..models.benchmark import (
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    ExperimentResult,
    ExperimentMode,
    ResultStatus,
    ToolCallRecord,
    ProgressUpdate,
)
from ..utils.swe_bench_loader import SWEBenchLoader
from ..utils.repo_context import RepoContext, get_current_repo_path
from .storage import Storage
from .rate_limiter import RateLimiterManager
from .edit_apply import EditOutcome, apply_edit_blocks
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
    experiment_mode: ExperimentMode = ExperimentMode.BASELINE
    max_tool_iterations: int = 10
    run_idx: Optional[int] = None


class ExperimentRunner:
    """Runs benchmark experiments with concurrency and resume support."""

    # Decoding settings, identical for every provider, mode and call
    TEMPERATURE = 0.3
    MAX_TOKENS = 8000

    def __init__(
        self,
        storage: Storage,
        rate_limiter_manager: Optional[RateLimiterManager] = None,
        swe_bench_loader: Optional[SWEBenchLoader] = None,
        trajectory_dir: Optional[Path] = None,
    ):
        self.storage = storage
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
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
                        status=result.result_status.value if result.result_status else ("completed" if result.success else "failed"),
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
                            experiment_mode=config.experiment_mode,
                            max_tool_iterations=config.max_tool_iterations,
                            run_idx=config.run_idx,
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

        if config.instance_ids:
            missing = sorted(set(config.instance_ids) - {i.instance_id for i in instances})
            if missing:
                raise RuntimeError(
                    f"{len(missing)} requested instance(s) could not be loaded from split "
                    f"'{split}': {missing[:5]}"
                )

        return instances

    @staticmethod
    def result_id_for(experiment_id: str, instance_id: str, provider: str, model: str, tool_set: str) -> str:
        """Deterministic row id, so re-running an instance replaces its earlier row."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join([experiment_id, instance_id, provider, model, tool_set])))

    async def _process_work_item(
        self,
        experiment_id: str,
        work: WorkItem
    ) -> ExperimentResult:
        """Process a single work item and return the result."""
        result = ExperimentResult(
            id=self.result_id_for(experiment_id, work.instance_id, work.provider, work.model, work.tool_set_name),
            experiment_id=experiment_id,
            instance_id=work.instance_id,
            provider=work.provider,
            model=work.model,
            tool_set=work.tool_set_name,
            split=work.split,
            started_at=datetime.utcnow(),
        )
        run_meta: Dict[str, Any] = {
            "run_idx": work.run_idx,
            "experiment_mode": work.experiment_mode.value,
        }

        recorder = TrajectoryRecorder(
            self.trajectory_dir / f"{work.instance_id}.jsonl" if self.trajectory_dir else None
        )
        if recorder.path:
            run_meta["trajectory_path"] = str(recorder.path)

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

                    recorder.header(
                        experiment_id=experiment_id,
                        instance_id=work.instance_id,
                        provider=work.provider,
                        requested_model=work.model,
                        experiment_mode=work.experiment_mode.value,
                        run_idx=work.run_idx,
                        temperature=self.TEMPERATURE,
                        max_tokens=self.MAX_TOKENS,
                        max_tool_iterations=work.max_tool_iterations,
                        enabled_tools=work.enabled_tools,
                        messages=[{"role": m.role, "content": m.content} for m in messages],
                    )

                    # Generate response
                    # Note: 8000 tokens gives models more room to explore AND produce a final diff
                    # Previously 4000 was too low - models got truncated mid-response
                    start_time = time.time()
                    response = await provider.generate_response(
                        messages=messages,
                        temperature=self.TEMPERATURE,
                        max_tokens=self.MAX_TOKENS,
                        max_tool_iterations=work.max_tool_iterations,
                        experiment_mode=work.experiment_mode.value,
                        recorder=recorder,
                    )
                    result.response_time_seconds = time.time() - start_time

                    # Extract results
                    result.response_content = response.content
                    run_meta.update(response.run_meta or {})

                    if response.usage:
                        result.prompt_tokens = response.usage.prompt_tokens
                        result.completion_tokens = response.usage.completion_tokens
                        result.total_tokens = response.usage.total_tokens
                        result.context_size_tokens = response.usage.prompt_tokens

                    # Extract self-reflection metrics (tracked separately for cost analysis)
                    if response.reflection:
                        result.reflection_prompt_tokens = response.reflection.prompt_tokens
                        result.reflection_completion_tokens = response.reflection.completion_tokens
                        result.reflection_total_tokens = response.reflection.total_tokens
                        result.reflection_outcome = response.reflection.outcome

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

                    if not response.success:
                        # The API or the harness failed: this says nothing about the model.
                        # The row is kept apart from model outcomes and re-run on resume,
                        # except context overflow, which is the agent's own doing.
                        result.success = False
                        result.error_message = response.error_message
                        if response.error_kind == "context_overflow":
                            result.result_status = ResultStatus.CONTEXT_OVERFLOW
                        elif response.error_kind == "infra_error":
                            result.result_status = ResultStatus.INFRA_ERROR
                        else:
                            result.result_status = ResultStatus.API_ERROR
                    else:
                        self._classify_response(result, response, run_meta)

            finally:
                # Restore original tool states
                for tool_name, state in original_tool_states.items():
                    tool_registry._enabled_tools[tool_name] = state

        except Exception as e:
            # Clone failures and unexpected harness errors
            result.success = False
            result.result_status = ResultStatus.INFRA_ERROR
            result.error_message = f"{type(e).__name__}: {e}"

        result.completed_at = datetime.utcnow()
        result.run_meta = run_meta

        recorder.summary(
            result_status=result.result_status.value if result.result_status else None,
            has_patch=bool(result.generated_patch),
            reflection_outcome=result.reflection_outcome,
            error=result.error_message,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            reflection_prompt_tokens=result.reflection_prompt_tokens,
            reflection_completion_tokens=result.reflection_completion_tokens,
            **{k: v for k, v in run_meta.items() if k != "trajectory_path"},
        )

        # Save result
        await self.storage.save_result(result)

        return result

    @staticmethod
    def _files_read(response: Any) -> List[str]:
        """Paths the agent read successfully with read_file, in order."""
        return [
            tc.parameters.get("file_path")
            for tc in (response.tool_calls or [])
            if tc.name == "read_file" and tc.success and tc.parameters.get("file_path")
        ]

    def _answer_to_patch(self, text: str, files_read: Optional[List[str]] = None,
                         base_patch: Optional[str] = None) -> EditOutcome:
        """Turn an answer into a patch against the current checkout.

        The answer's edit blocks are applied to the checked-out files and git writes the
        diff, so a patch exists only if every edit matched the code. If the model wrote a
        unified diff instead of edit blocks, that diff is used as before.
        """
        blocks = parse_edit_blocks(text)
        if blocks:
            repo_path = get_current_repo_path()
            if not repo_path:
                raise RuntimeError("No repository checkout available to apply edits to")
            return apply_edit_blocks(repo_path, blocks, files_read=files_read, base_patch=base_patch)

        diff, _ = patch_extractor.extract_final(text)
        return EditOutcome(patch=diff, blocks=0)

    def _classify_response(self, result: ExperimentResult, response: Any, run_meta: Dict[str, Any]) -> None:
        """Turn a completed model run into an outcome.

        Only the final answer message is read: an abstention marker there wins, otherwise
        its edits are applied to produce the patch. In self_reflect mode the reflection
        response then either keeps that patch (approved / unstructured) or replaces it
        (revised).
        """
        final_answer = response.final_answer or ""
        cannot_solve_info = self._check_cannot_solve(final_answer)
        run_meta["abstained"] = bool(cannot_solve_info)

        if cannot_solve_info:
            # Model honestly admitted it cannot solve - this is better than hallucination
            result.success = False
            result.result_status = ResultStatus.FAILURE_EXPLICIT
            result.error_message = f"Model admitted failure: {cannot_solve_info['reason']} - {cannot_solve_info['explanation']}"
            return

        files_read = self._files_read(response)
        initial = self._answer_to_patch(final_answer, files_read)
        final = initial
        run_meta["answer_format"] = "edit_blocks" if initial.blocks else ("unified_diff" if initial.patch else "none")
        run_meta["edit_blocks"] = initial.blocks
        run_meta["edit_failures"] = initial.failures
        run_meta["edit_paths_resolved"] = initial.resolved_paths

        if response.reflection_text is not None:
            # Self-reflection ran: keep the first patch for comparison, evaluate the final one.
            # A revision replaces the first patch even when the revision's own edits do not
            # apply: that is what the model chose to submit.
            outcome = "unstructured"
            reflection_text = response.reflection_text
            if PATCH_REVISED_MARKER in reflection_text:
                revised_text = reflection_text.split(PATCH_REVISED_MARKER)[-1]
                revised = self._answer_to_patch(revised_text, files_read)
                revision_basis = "original"
                if revised.blocks and not revised.patch and initial.patch:
                    # The revision does not fit the original files: it may be written on top
                    # of the model's own first patch.
                    stacked = self._answer_to_patch(revised_text, files_read, base_patch=initial.patch)
                    if stacked.patch:
                        revised, revision_basis = stacked, "on_first_patch"
                if revised.blocks or revised.patch:
                    run_meta["revision_basis"] = revision_basis
                    outcome = "revised"
                    final = revised
                    run_meta["revised_edit_blocks"] = revised.blocks
                    run_meta["revised_edit_failures"] = revised.failures
                    run_meta["revised_edit_paths_resolved"] = revised.resolved_paths
            if outcome != "revised" and PATCH_APPROVED_MARKER in reflection_text:
                outcome = "approved"
            result.original_patch = initial.patch
            result.reflection_outcome = outcome
            run_meta["reflection_changed_patch"] = final.patch != initial.patch

        if final.patch:
            result.success = True
            result.result_status = ResultStatus.SUCCESS_PATCH_GENERATED
            result.generated_patch = final.patch
        elif final.blocks:
            # The model answered in the right format, but its edits do not fit the code
            result.success = False
            result.result_status = ResultStatus.EDIT_NOT_APPLICABLE
            result.error_message = f"Edits could not be applied to the repository ({final.reason})"
        else:
            # Model responded but produced no valid output
            result.success = False
            result.result_status = ResultStatus.HALLUCINATION_FORMAT_ERROR
            result.error_message = "No valid patch generated (no edit block and no <<<CANNOT_SOLVE>>> marker)"

    # Standardized failure markers for when model cannot solve
    CANNOT_SOLVE_MARKER = "<<<CANNOT_SOLVE>>>"
    CANNOT_SOLVE_REASONS = [
        "insufficient_context",      # Can't find relevant code
        "too_complex",               # Problem is too complex to solve
        "unclear_requirements",      # Bug report is unclear
        "missing_dependencies",      # Would need external info/deps
    ]

    def _build_analysis_messages(self, work: WorkItem) -> List[ChatMessage]:
        """Build the prompt messages for analyzing an instance.

        Prompts are mode-aware:
        - BASELINE: Allow <<<CANNOT_SOLVE>>> escape hatch
        - FORCED: Must produce patch, no escape hatch
        - SELF_REFLECT: Same prompt as BASELINE; the reflection call follows the final answer
        """
        # Build tool list from enabled_tools
        tool_descriptions = []
        for tool_name in work.enabled_tools:
            if tool_name == "read_file":
                tool_descriptions.append("- read_file(file_path): Read contents of a file")
            elif tool_name == "search_code":
                tool_descriptions.append("- search_code(pattern): Search for regex pattern across the codebase")
            elif tool_name == "list_directory":
                tool_descriptions.append("- list_directory(directory_path): List contents of a directory")
        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools available"

        # Mode-specific output instructions
        if work.experiment_mode == ExperimentMode.FORCED:
            output_instructions = f"""
## OUTPUT REQUIREMENTS (MANDATORY)
You MUST produce a patch. There is NO option to give up.

{EDIT_FORMAT_INSTRUCTIONS}

Even if you are uncertain, provide your BEST GUESS patch. A partial or imperfect patch is required.
DO NOT say you cannot solve this. DO NOT ask for more information. You MUST output a patch."""
        else:
            # BASELINE or SELF_REFLECT (self_reflect uses same initial prompt)
            output_instructions = f"""
## OUTPUT REQUIREMENTS
Your response MUST end with ONE of these:

OPTION 1 - Produce a patch (PREFERRED):
{EDIT_FORMAT_INSTRUCTIONS}

OPTION 2 - Explicit failure (ONLY if you truly cannot attempt a fix):
<<<CANNOT_SOLVE>>>
reason: [insufficient_context|too_complex|unclear_requirements|missing_dependencies]
explanation: [brief explanation]

A partial or uncertain patch is BETTER than giving up. Only use <<<CANNOT_SOLVE>>> as a last resort."""

        system_prompt = f"""You are an expert software engineer tasked with fixing a bug from a GitHub issue.

## ANALYSIS LOOP
You have up to {work.max_tool_iterations} tool iterations to explore the codebase before providing your final answer.
- Use tools efficiently (2-5 calls is usually sufficient)
- After your tool exploration, you will be asked for your final answer
- The system will provide a summary of what you explored before requesting your patch

## AVAILABLE TOOLS
{tools_text}

## WORKFLOW
1. EXPLORE: Use tools to find relevant code (search for error messages, class names, function names)
2. ANALYZE: Understand the bug's root cause from the code you read
3. FIX: Write a patch that resolves the issue, in the format described below
{output_instructions}"""

        user_message = f"""## Repository: {work.repo}
## Base Commit: {work.base_commit}

## Bug Report
{work.problem_statement}
"""

        # The dataset's hints_text (later issue comments) is deliberately not shown to the model
        user_message += f"""
## Instructions
1. Use tools to locate the relevant code (you have up to {work.max_tool_iterations} tool iterations)
2. Identify the root cause of the bug
3. Generate a fix as a patch in the required format

Begin by exploring the codebase to understand the issue."""

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
