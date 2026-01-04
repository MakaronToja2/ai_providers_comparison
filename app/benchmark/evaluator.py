"""
Test evaluator for verifying generated patches.
Applies patches and runs tests to determine if issues are resolved.
"""
import asyncio
import os
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from ..models.benchmark import TestResult, ExperimentResult
from ..utils.swe_bench_loader import SWEBenchLoader
from .storage import Storage


class TestEvaluator:
    """Evaluates generated patches by running tests."""

    def __init__(
        self,
        storage: Storage,
        swe_bench_loader: Optional[SWEBenchLoader] = None,
        repo_cache_dir: Optional[str] = None,
    ):
        self.storage = storage
        self.swe_bench_loader = swe_bench_loader or SWEBenchLoader()
        self.repo_cache_dir = Path(repo_cache_dir or tempfile.gettempdir()) / "swe_bench_repos"
        self.repo_cache_dir.mkdir(parents=True, exist_ok=True)

    async def evaluate_result(
        self,
        result: ExperimentResult,
        timeout: int = 300,
    ) -> TestResult:
        """
        Evaluate a single result by applying patch and running tests.

        Args:
            result: The experiment result containing the generated patch
            timeout: Maximum time in seconds for test execution

        Returns:
            TestResult with evaluation outcome
        """
        test_result = TestResult(
            result_id=result.id,
            experiment_id=result.experiment_id,
            instance_id=result.instance_id,
        )

        if not result.generated_patch:
            test_result.patch_error = "No patch generated"
            await self.storage.save_test_result(test_result)
            return test_result

        try:
            # Get instance data
            instance = await asyncio.get_event_loop().run_in_executor(
                None,
                self.swe_bench_loader.get_instance,
                result.instance_id,
                "dev"
            )

            if not instance:
                test_result.patch_error = f"Instance {result.instance_id} not found"
                await self.storage.save_test_result(test_result)
                return test_result

            # Get test lists
            fail_to_pass = instance.get("FAIL_TO_PASS", [])
            pass_to_pass = instance.get("PASS_TO_PASS", [])

            test_result.fail_to_pass_total = len(fail_to_pass)
            test_result.pass_to_pass_total = len(pass_to_pass)

            # Clone/checkout repository
            repo_path = await self._get_repo(
                instance["repo"],
                instance["base_commit"]
            )

            if not repo_path:
                test_result.patch_error = "Failed to clone repository"
                await self.storage.save_test_result(test_result)
                return test_result

            try:
                # Apply patch
                apply_success, apply_error = await self._apply_patch(
                    repo_path,
                    result.generated_patch
                )

                if not apply_success:
                    test_result.patch_applied = False
                    test_result.patch_error = apply_error
                    await self.storage.save_test_result(test_result)
                    return test_result

                test_result.patch_applied = True

                # Run tests
                start_time = datetime.utcnow()

                f2p_passed, f2p_output = await self._run_tests(
                    repo_path,
                    fail_to_pass,
                    timeout
                )
                test_result.fail_to_pass_passed = f2p_passed

                p2p_passed, p2p_output = await self._run_tests(
                    repo_path,
                    pass_to_pass,
                    timeout
                )
                test_result.pass_to_pass_passed = p2p_passed

                test_result.execution_time_seconds = (
                    datetime.utcnow() - start_time
                ).total_seconds()

                test_result.test_output = f"FAIL_TO_PASS:\n{f2p_output}\n\nPASS_TO_PASS:\n{p2p_output}"

                # Determine if resolved
                test_result.resolved = (
                    f2p_passed == test_result.fail_to_pass_total and
                    p2p_passed == test_result.pass_to_pass_total
                )

            finally:
                # Reset repository
                await self._reset_repo(repo_path)

        except Exception as e:
            test_result.patch_error = str(e)

        test_result.executed_at = datetime.utcnow()
        await self.storage.save_test_result(test_result)
        return test_result

    async def evaluate_experiment(
        self,
        experiment_id: str,
        max_concurrent: int = 2,
    ) -> List[TestResult]:
        """Evaluate all results for an experiment."""
        results = await self.storage.get_results_for_experiment(
            experiment_id,
            success_only=True,
            limit=10000
        )

        # Filter to results with patches
        results_with_patches = [r for r in results if r.generated_patch]

        if not results_with_patches:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_with_limit(result: ExperimentResult) -> TestResult:
            async with semaphore:
                return await self.evaluate_result(result)

        tasks = [evaluate_with_limit(r) for r in results_with_patches]
        test_results = await asyncio.gather(*tasks)

        return test_results

    async def _get_repo(self, repo: str, commit: str) -> Optional[Path]:
        """Get or clone a repository at a specific commit."""
        # Create cache key from repo and commit
        cache_key = f"{repo.replace('/', '_')}_{commit[:8]}"
        repo_path = self.repo_cache_dir / cache_key

        if repo_path.exists():
            # Verify it's at the right commit
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True
                    )
                )
                if result.returncode == 0 and result.stdout.strip().startswith(commit[:8]):
                    return repo_path
            except Exception:
                pass

            # Clean up and re-clone
            shutil.rmtree(repo_path, ignore_errors=True)

        # Clone repository
        try:
            clone_url = f"https://github.com/{repo}.git"

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["git", "clone", "--depth", "1", clone_url, str(repo_path)],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
            )

            if result.returncode != 0:
                return None

            # Fetch the specific commit
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["git", "fetch", "--depth", "1", "origin", commit],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            )

            # Checkout the commit
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["git", "checkout", commit],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            )

            if result.returncode != 0:
                # Try with FETCH_HEAD
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["git", "checkout", "FETCH_HEAD"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                )

            return repo_path if repo_path.exists() else None

        except Exception as e:
            print(f"Failed to clone {repo}: {e}")
            return None

    async def _apply_patch(
        self,
        repo_path: Path,
        patch: str
    ) -> Tuple[bool, Optional[str]]:
        """Apply a patch to the repository."""
        try:
            # Write patch to temp file
            patch_file = repo_path / ".generated_patch.diff"
            patch_file.write_text(patch)

            # Try applying with git apply
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["git", "apply", "--check", str(patch_file)],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            )

            if result.returncode != 0:
                # Try with patch command as fallback
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["patch", "-p1", "--dry-run", "-i", str(patch_file)],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                )

                if result.returncode != 0:
                    return False, f"Patch cannot be applied: {result.stderr}"

                # Apply for real with patch
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["patch", "-p1", "-i", str(patch_file)],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                )
            else:
                # Apply for real with git
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["git", "apply", str(patch_file)],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                )

            patch_file.unlink(missing_ok=True)

            if result.returncode != 0:
                return False, f"Patch application failed: {result.stderr}"

            return True, None

        except Exception as e:
            return False, str(e)

    async def _run_tests(
        self,
        repo_path: Path,
        test_names: List[str],
        timeout: int = 300
    ) -> Tuple[int, str]:
        """Run specific tests and return count of passed tests."""
        if not test_names:
            return 0, "No tests to run"

        passed = 0
        output_lines = []

        for test_name in test_names:
            try:
                # Try running with pytest
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda tn=test_name: subprocess.run(
                        ["python", "-m", "pytest", tn, "-v", "--tb=short"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=timeout // len(test_names) if test_names else timeout
                    )
                )

                if result.returncode == 0:
                    passed += 1
                    output_lines.append(f"PASS: {test_name}")
                else:
                    output_lines.append(f"FAIL: {test_name}")
                    if result.stderr:
                        output_lines.append(result.stderr[:500])

            except subprocess.TimeoutExpired:
                output_lines.append(f"TIMEOUT: {test_name}")
            except Exception as e:
                output_lines.append(f"ERROR: {test_name} - {str(e)}")

        return passed, "\n".join(output_lines)

    async def _reset_repo(self, repo_path: Path) -> None:
        """Reset repository to clean state."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["git", "checkout", "."],
                    cwd=repo_path,
                    capture_output=True,
                    timeout=30
                )
            )
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["git", "clean", "-fd"],
                    cwd=repo_path,
                    capture_output=True,
                    timeout=30
                )
            )
        except Exception:
            pass

    async def cleanup_cache(self) -> None:
        """Clean up cached repositories."""
        if self.repo_cache_dir.exists():
            shutil.rmtree(self.repo_cache_dir, ignore_errors=True)
            self.repo_cache_dir.mkdir(parents=True, exist_ok=True)
