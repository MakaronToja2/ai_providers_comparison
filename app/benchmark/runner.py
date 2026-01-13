"""
Benchmark runner for executing SWE-bench experiments.
Handles batch processing with concurrency, resume capability, and progress tracking.
"""
import asyncio
import os
import re
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple
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
    ExperimentMode,
    ResultStatus,
    ToolCallRecord,
    ProgressUpdate,
)
from ..utils.swe_bench_loader import SWEBenchLoader
from ..utils.repo_context import RepoContext
from .storage import Storage
from .rate_limiter import RateLimiterManager
from .patch_extractor import patch_extractor
from .evaluator import TestEvaluator


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
                            experiment_mode=config.experiment_mode,
                            max_tool_iterations=config.max_tool_iterations,
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
                        max_tool_iterations=work.max_tool_iterations,
                        experiment_mode=work.experiment_mode.value,
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

                            # TEST-GUIDED SELF-REFLECTION for self_reflect mode
                            if work.experiment_mode == ExperimentMode.SELF_REFLECT:
                                try:
                                    print(f"[{work.provider}] Running inline evaluation for test-guided reflection...")

                                    # Run inline evaluation
                                    test_results = await self._run_inline_evaluation(
                                        result, work, repo_ctx
                                    )

                                    if test_results and not test_results.get("resolved", False):
                                        # Tests failed - ask model to reflect and revise
                                        print(f"[{work.provider}] Tests failed, requesting revision...")

                                        revised_content, reflection_metrics = await provider.do_reflection_with_test_results(
                                            original_content=response.content,
                                            conversation_history=response.conversation_history or [],  # Use FULL context with file contents
                                            test_results=test_results,
                                            temperature=0.3,
                                            max_tokens=4000,
                                        )

                                        # Update result with reflection metrics
                                        if reflection_metrics:
                                            result.reflection_prompt_tokens = reflection_metrics.prompt_tokens
                                            result.reflection_completion_tokens = reflection_metrics.completion_tokens
                                            result.reflection_total_tokens = reflection_metrics.total_tokens
                                            result.reflection_outcome = reflection_metrics.outcome

                                        # Try to extract revised patch
                                        revised_patch = self._extract_patch(revised_content)
                                        if revised_patch:
                                            print(f"[{work.provider}] Got revised patch from reflection")
                                            # Store original patch for comparison
                                            result.original_patch = result.generated_patch
                                            result.generated_patch = revised_patch
                                            result.response_content = revised_content
                                        else:
                                            print(f"[{work.provider}] No revised patch extracted, keeping original")

                                    elif test_results and test_results.get("resolved", False):
                                        # Tests passed on first try!
                                        print(f"[{work.provider}] Tests passed on first patch!")
                                        result.reflection_outcome = "tests_passed"

                                except Exception as e:
                                    print(f"[{work.provider}] Inline evaluation failed: {e}")
                                    result.reflection_outcome = "eval_failed"

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
        """Build the prompt messages for analyzing an instance.

        Prompts are mode-aware:
        - BASELINE: Allow <<<CANNOT_SOLVE>>> escape hatch
        - FORCED: Must produce patch, no escape hatch
        - SELF_REFLECT: Same as FORCED initially, reflection happens in post-processing
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
            output_instructions = """
## OUTPUT REQUIREMENTS (MANDATORY)
You MUST produce a patch. There is NO option to give up.

Your response MUST contain a unified diff patch:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```

Even if you are uncertain, provide your BEST GUESS patch. A partial or imperfect patch is required.
DO NOT say you cannot solve this. DO NOT ask for more information. You MUST output a diff."""
        else:
            # BASELINE or SELF_REFLECT (self_reflect uses same initial prompt)
            output_instructions = """
## OUTPUT REQUIREMENTS
Your response MUST end with ONE of these:

OPTION 1 - Produce a patch (PREFERRED):
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```

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
3. FIX: Generate a unified diff patch that resolves the issue
{output_instructions}"""

        user_message = f"""## Repository: {work.repo}
## Base Commit: {work.base_commit}

## Bug Report
{work.problem_statement}
"""

        if work.hints_text:
            user_message += f"""
## Hints
{work.hints_text}
"""

        user_message += f"""
## Instructions
1. Use tools to locate the relevant code (you have up to {work.max_tool_iterations} tool iterations)
2. Identify the root cause of the bug
3. Generate a fix as a unified diff patch

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

    async def _run_inline_evaluation(
        self,
        result: ExperimentResult,
        work: WorkItem,
        repo_ctx: RepoContext,
    ) -> Optional[Dict[str, Any]]:
        """Run inline test evaluation for test-guided self-reflection.

        This applies the patch and runs tests within the existing RepoContext,
        returning test results for the model to reflect on.

        Args:
            result: The experiment result with the generated patch
            work: The work item with instance details
            repo_ctx: The repository context (already cloned)

        Returns:
            Dict with test results or None if evaluation failed:
            {
                "resolved": bool,
                "fail_to_pass_passed": int,
                "fail_to_pass_total": int,
                "pass_to_pass_passed": int,
                "pass_to_pass_total": int,
                "test_output": str,
                "patch_applied": bool,
            }
        """
        import subprocess
        import tempfile

        if not result.generated_patch:
            return None

        test_results = {
            "resolved": False,
            "fail_to_pass_passed": 0,
            "fail_to_pass_total": len(work.fail_to_pass) if work.fail_to_pass else 0,
            "pass_to_pass_passed": 0,
            "pass_to_pass_total": len(work.pass_to_pass) if work.pass_to_pass else 0,
            "test_output": "",
            "patch_applied": False,
        }

        try:
            # Write patch to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(result.generated_patch)
                patch_file = f.name

            try:
                # Apply patch
                apply_result = subprocess.run(
                    ['git', 'apply', '--check', patch_file],
                    cwd=repo_ctx.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if apply_result.returncode != 0:
                    test_results["test_output"] = f"Patch failed to apply:\n{apply_result.stderr}"
                    return test_results

                # Actually apply the patch
                subprocess.run(
                    ['git', 'apply', patch_file],
                    cwd=repo_ctx.repo_path,
                    capture_output=True,
                    timeout=30
                )
                test_results["patch_applied"] = True

                # Run fail_to_pass tests (these should pass after fix)
                if work.fail_to_pass:
                    f2p_output, f2p_passed = await self._run_pytest(
                        repo_ctx.repo_path, work.fail_to_pass, timeout=120
                    )
                    test_results["fail_to_pass_passed"] = f2p_passed
                    test_results["test_output"] += f"\n=== FAIL_TO_PASS Tests ===\n{f2p_output}"

                # Run pass_to_pass tests (regression tests - should still pass)
                if work.pass_to_pass:
                    p2p_output, p2p_passed = await self._run_pytest(
                        repo_ctx.repo_path, work.pass_to_pass, timeout=120
                    )
                    test_results["pass_to_pass_passed"] = p2p_passed
                    test_results["test_output"] += f"\n=== PASS_TO_PASS Tests ===\n{p2p_output}"

                # Check if resolved (all tests pass)
                f2p_ok = test_results["fail_to_pass_passed"] == test_results["fail_to_pass_total"]
                p2p_ok = test_results["pass_to_pass_passed"] == test_results["pass_to_pass_total"]
                test_results["resolved"] = f2p_ok and p2p_ok

                print(f"[Inline Eval] F2P: {test_results['fail_to_pass_passed']}/{test_results['fail_to_pass_total']}, "
                      f"P2P: {test_results['pass_to_pass_passed']}/{test_results['pass_to_pass_total']}, "
                      f"Resolved: {test_results['resolved']}")

            finally:
                # Clean up patch file
                os.unlink(patch_file)

                # Revert patch for clean state
                subprocess.run(
                    ['git', 'checkout', '.'],
                    cwd=repo_ctx.repo_path,
                    capture_output=True,
                    timeout=30
                )

        except Exception as e:
            test_results["test_output"] = f"Evaluation error: {str(e)}"
            print(f"[Inline Eval] Error: {e}")

        return test_results

    async def _run_pytest(
        self, repo_path: str, tests: List[str], timeout: int = 120
    ) -> Tuple[str, int]:
        """Run pytest on a list of tests and return output and pass count.

        Args:
            repo_path: Path to the repository
            tests: List of test identifiers
            timeout: Timeout in seconds

        Returns:
            Tuple of (output_string, number_of_tests_passed)
        """
        import subprocess

        if not tests:
            return "", 0

        try:
            # Run pytest
            result = subprocess.run(
                ['python', '-m', 'pytest', '-xvs'] + tests,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, 'PYTHONPATH': repo_path}
            )

            output = result.stdout + result.stderr

            # Count passed tests (rough heuristic)
            passed = output.count(' PASSED') + output.count(' passed')

            return output[-2000:] if len(output) > 2000 else output, passed

        except subprocess.TimeoutExpired:
            return "Tests timed out", 0
        except Exception as e:
            return f"Test execution error: {e}", 0

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
