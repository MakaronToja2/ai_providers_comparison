"""
Integration with official SWE-bench evaluation harness.
Uses Docker containers for accurate, reproducible evaluation.
"""
import asyncio
import glob
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple

from ..models.benchmark import ExperimentResult, TestResult
from .storage import Storage


class SWEBenchHarness:
    """
    Wrapper around the official SWE-bench evaluation harness.

    This uses Docker containers to run tests in the correct environment,
    providing accurate resolve rates.
    """

    def __init__(
        self,
        storage: Storage,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "dev",  # dev (23 instances) or test (300 instances)
        max_workers: int = 4,
        output_dir: Optional[str] = None,
    ):
        self.storage = storage
        self.dataset_name = dataset_name
        self.split = split
        self.max_workers = max_workers
        self.output_dir = Path(output_dir or "swebench_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def export_predictions(
        self,
        experiment_id: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Export experiment results to SWE-bench predictions format.

        Creates a JSONL file with format:
        {"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}
        """
        results = await self.storage.get_results_for_experiment(
            experiment_id,
            success_only=True,
            limit=10000
        )

        # Filter to results with patches
        results_with_patches = [r for r in results if r.generated_patch]

        if not results_with_patches:
            raise ValueError(f"No results with patches found for experiment {experiment_id}")

        # Create output path
        if output_path is None:
            output_path = self.output_dir / f"predictions_{experiment_id}.jsonl"

        # Write predictions in SWE-bench format
        with open(output_path, 'w') as f:
            for result in results_with_patches:
                prediction = {
                    "instance_id": result.instance_id,
                    "model_name_or_path": f"{result.provider}/{result.model}",
                    "model_patch": result.generated_patch,
                }
                f.write(json.dumps(prediction) + '\n')

        print(f"[SWE-bench] Exported {len(results_with_patches)} predictions to {output_path}")
        return output_path

    def _find_completed_evaluations(self, experiment_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Scan all existing log directories for completed evaluations for this experiment.
        Returns dict mapping "provider/instance_id" to result data.
        """
        completed = {}
        logs_base = Path("logs/run_evaluation")

        if not logs_base.exists():
            return completed

        # Find all run directories for this experiment
        exp_prefix = f"exp_{experiment_id[:8]}_"
        for run_dir in logs_base.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith(exp_prefix):
                continue

            # Scan provider directories
            for provider_dir in run_dir.iterdir():
                if not provider_dir.is_dir():
                    continue

                provider_model = provider_dir.name

                for instance_dir in provider_dir.iterdir():
                    if not instance_dir.is_dir():
                        continue

                    log_file = instance_dir / "run_instance.log"
                    if log_file.exists():
                        result_data = self._parse_instance_log(log_file)
                        if result_data:
                            # Key format: "provider__model/instance_id"
                            key = f"{provider_model}/{instance_dir.name}"
                            # Keep the most recent result if duplicate
                            if key not in completed:
                                completed[key] = result_data

        return completed

    async def run_evaluation_with_progress(
        self,
        predictions_path: Path,
        run_id: str,
        experiment_id: str,
        poll_interval: int = 30,
        progress_every: int = 10,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Run SWE-bench evaluation with incremental progress saving.

        Instead of waiting for full completion, this:
        1. Starts SWE-bench as background process
        2. Polls log directory for completed evaluations
        3. Saves results to DB as they complete
        4. Reports progress every N evaluations

        If resume=True, skips predictions that already have completed evaluations.
        """
        print(f"[SWE-bench] Starting evaluation with run_id: {run_id}")
        print(f"[SWE-bench] Dataset: {self.dataset_name}")
        print(f"[SWE-bench] Split: {self.split}")
        print(f"[SWE-bench] Predictions: {predictions_path}")
        print(f"[SWE-bench] Max workers: {self.max_workers}")
        print(f"[SWE-bench] Progress reporting every {progress_every} evaluations")
        print(f"[SWE-bench] Resume mode: {resume}")

        # Check Docker is available
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                check=True,
                timeout=30
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Docker is not available. SWE-bench evaluation requires Docker.\n"
                "Please install Docker and ensure the daemon is running."
            ) from e

        # Load all predictions
        with open(predictions_path) as f:
            all_predictions = [json.loads(line) for line in f]
        total_original = len(all_predictions)
        print(f"[SWE-bench] Total predictions in file: {total_original}")

        # Check for already completed evaluations if resume mode
        already_completed = {}
        db_evaluated_keys: Set[str] = set()

        if resume:
            # First, check database for already-evaluated predictions (provider+model+instance)
            db_evaluated_keys = await self.storage.get_evaluated_prediction_keys(experiment_id)
            if db_evaluated_keys:
                print(f"[SWE-bench] Found {len(db_evaluated_keys)} already evaluated in database")

            # Also check log files for completed evaluations (may not be in DB yet)
            already_completed = self._find_completed_evaluations(experiment_id)
            if already_completed:
                print(f"[SWE-bench] Found {len(already_completed)} completed evaluations in logs")

            # Combine both sources - filter predictions that are either in DB or in logs
            remaining_predictions = []
            skipped_from_db = 0
            skipped_from_logs = 0

            for pred in all_predictions:
                instance_id = pred.get("instance_id", "")
                # Build key to match: "provider__model/instance_id"
                model_path = pred.get("model_name_or_path", "unknown")
                provider_model = model_path.replace("/", "__")
                key = f"{provider_model}/{instance_id}"

                # Skip if already in database (by provider+model+instance key)
                if key in db_evaluated_keys:
                    skipped_from_db += 1
                    continue

                # Skip if already in logs
                if key in already_completed:
                    skipped_from_logs += 1
                    continue

                remaining_predictions.append(pred)

            print(f"[SWE-bench] Skipped {skipped_from_db} from DB, {skipped_from_logs} from logs")

            if not remaining_predictions:
                print(f"[SWE-bench] All predictions already evaluated! Nothing to do.")
                # Import any log results not yet in DB
                if already_completed:
                    await self._import_log_results_to_db(experiment_id, already_completed)
                total_evaluated = len(db_evaluated_keys) + skipped_from_logs
                resolved = await self.storage.count_test_results(experiment_id, resolved_only=True)
                return {
                    "run_id": run_id,
                    "total": total_evaluated,
                    "resolved": resolved,
                    "skipped": total_evaluated,
                    "summary": {"resolved": resolved, "total": total_evaluated}
                }

            print(f"[SWE-bench] Remaining to evaluate: {len(remaining_predictions)}")

            # Write filtered predictions to a new file
            filtered_path = predictions_path.parent / f"predictions_{experiment_id[:8]}_remaining.jsonl"
            with open(filtered_path, 'w') as f:
                for pred in remaining_predictions:
                    f.write(json.dumps(pred) + "\n")
            predictions_path = filtered_path
            print(f"[SWE-bench] Filtered predictions written to: {filtered_path}")

            # Import log results not yet in database
            if already_completed:
                await self._import_log_results_to_db(experiment_id, already_completed)

        # Count predictions to evaluate
        with open(predictions_path) as f:
            total_predictions = sum(1 for _ in f)
        print(f"[SWE-bench] Predictions to evaluate this run: {total_predictions}")

        if total_predictions == 0:
            print(f"[SWE-bench] No predictions remaining to evaluate!")
            return {
                "run_id": run_id,
                "total": len(already_completed),
                "resolved": sum(1 for r in already_completed.values() if r.get("resolved", False)),
                "skipped": len(already_completed),
                "summary": {"resolved": sum(1 for r in already_completed.values() if r.get("resolved", False)),
                           "total": len(already_completed)}
            }

        # Run the official SWE-bench harness as background process
        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", self.dataset_name,
            "--split", self.split,
            "--predictions_path", str(predictions_path),
            "--max_workers", str(self.max_workers),
            "--run_id", run_id,
        ]

        print(f"[SWE-bench] Running: {' '.join(cmd)}")

        # Start process without waiting
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Get our results for matching
        our_results = await self.storage.get_results_for_experiment(
            experiment_id,
            success_only=True,
            limit=10000
        )
        results_by_key = {}
        for r in our_results:
            key = f"{r.provider}__{r.model}/{r.instance_id}"
            results_by_key[key] = r

        # Poll for results
        logs_dir = Path("logs/run_evaluation") / run_id
        saved_instances: Set[str] = set()
        last_progress_count = 0

        while process.poll() is None:
            await asyncio.sleep(poll_interval)

            # Check for completed log files
            completed = self._scan_completed_logs(logs_dir, results_by_key)

            # Save new results
            new_count = 0
            for instance_key, result_data in completed.items():
                if instance_key not in saved_instances:
                    our_result = results_by_key.get(instance_key)
                    if our_result:
                        await self._save_instance_result(
                            experiment_id, our_result, result_data
                        )
                        saved_instances.add(instance_key)
                        new_count += 1

            # Report progress
            total_saved = len(saved_instances)
            if total_saved >= last_progress_count + progress_every or new_count > 0:
                resolved = sum(1 for k in saved_instances
                             if completed.get(k, {}).get("resolved", False))
                print(f"[SWE-bench] Progress: {total_saved}/{total_predictions} evaluated, "
                      f"{resolved} resolved ({resolved*100//max(total_saved,1)}%)")
                last_progress_count = total_saved

        # Process finished - get final results
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"[SWE-bench] Process finished with errors: {stderr[-500:]}")
        else:
            print(f"[SWE-bench] Process completed successfully")

        # Final scan for any remaining results
        completed = self._scan_completed_logs(logs_dir, results_by_key)
        for instance_key, result_data in completed.items():
            if instance_key not in saved_instances:
                our_result = results_by_key.get(instance_key)
                if our_result:
                    await self._save_instance_result(
                        experiment_id, our_result, result_data
                    )
                    saved_instances.add(instance_key)

        # Final summary
        total_saved = len(saved_instances)
        resolved = sum(1 for k in saved_instances
                     if completed.get(k, {}).get("resolved", False))
        print(f"[SWE-bench] FINAL: {total_saved}/{total_predictions} evaluated, "
              f"{resolved} resolved ({resolved*100//max(total_saved,1)}%)")

        return {
            "run_id": run_id,
            "total": total_saved,
            "resolved": resolved,
            "summary": {"resolved": resolved, "total": total_saved}
        }

    def _scan_completed_logs(
        self,
        logs_dir: Path,
        results_by_key: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Scan log directory for completed evaluations."""
        completed = {}

        if not logs_dir.exists():
            return completed

        # Scan each provider/model directory
        for provider_dir in logs_dir.iterdir():
            if not provider_dir.is_dir():
                continue

            provider_model = provider_dir.name  # e.g., "openai__gpt-4.1-mini"

            for instance_dir in provider_dir.iterdir():
                if not instance_dir.is_dir():
                    continue

                instance_id = instance_dir.name
                log_file = instance_dir / "run_instance.log"

                if log_file.exists() and log_file.stat().st_size > 0:
                    instance_key = f"{provider_model}/{instance_id}"
                    result_data = self._parse_instance_log(log_file)
                    if result_data:
                        completed[instance_key] = result_data

        return completed

    def _parse_instance_log(self, log_file: Path) -> Optional[Dict[str, Any]]:
        """Parse a single instance log file to extract results."""
        try:
            content = log_file.read_text()

            result = {
                "patch_applied": False,
                "resolved": False,
                "log_file": str(log_file),
            }

            # Check if patch was applied
            if "Applied patch successfully" in content or ">>>>> Applied Patch" in content:
                result["patch_applied"] = True

            # Check for test results - look for resolution status
            # SWE-bench outputs "Result for <instance>: resolved: True/False"
            if "resolved: True" in content:
                result["resolved"] = True
            elif ">>>>> All Tests Passed" in content:
                result["resolved"] = True

            # Extract test output snippet (last 1000 chars)
            result["test_output"] = content[-1000:] if len(content) > 1000 else content

            return result

        except Exception as e:
            print(f"[SWE-bench] Error parsing log {log_file}: {e}")
            return None

    async def _save_instance_result(
        self,
        experiment_id: str,
        our_result: ExperimentResult,
        result_data: Dict[str, Any],
    ):
        """Save a single instance result to the database (skip if exists)."""
        # Check if result already exists BY RESULT_ID (unique per provider+model+instance)
        # Each instance can have multiple predictions from different providers
        existing = await self.storage.get_test_result_by_result_id(our_result.id)
        if existing:
            return  # Already saved for this specific prediction, skip

        test_result = TestResult(
            result_id=our_result.id,
            experiment_id=experiment_id,
            instance_id=our_result.instance_id,
            patch_applied=result_data.get("patch_applied", False),
            resolved=result_data.get("resolved", False),
            test_output=result_data.get("test_output", ""),
            executed_at=datetime.utcnow(),
        )
        await self.storage.save_test_result(test_result)

    async def _import_log_results_to_db(
        self,
        experiment_id: str,
        log_results: Dict[str, Dict[str, Any]],
    ) -> Tuple[int, int]:
        """Import results from log files into database (skip existing)."""
        # Get our results for matching
        our_results = await self.storage.get_results_for_experiment(
            experiment_id,
            success_only=True,
            limit=10000
        )
        results_by_key = {}
        for r in our_results:
            key = f"{r.provider}__{r.model}/{r.instance_id}"
            results_by_key[key] = r

        # Get already-evaluated keys (provider+model+instance) to avoid duplicates
        db_evaluated_keys = await self.storage.get_evaluated_prediction_keys(experiment_id)

        imported_count = 0
        resolved_count = 0
        for key, result_data in log_results.items():
            # Skip if already in database (by full key)
            if key in db_evaluated_keys:
                continue

            our_result = results_by_key.get(key)
            if our_result:
                await self._save_instance_result(experiment_id, our_result, result_data)
                imported_count += 1
                if result_data.get("resolved", False):
                    resolved_count += 1

        if imported_count > 0:
            print(f"[SWE-bench] Imported {imported_count} new results from logs ({resolved_count} resolved)")

        return imported_count, resolved_count

    def run_evaluation(
        self,
        predictions_path: Path,
        run_id: str,
        timeout: int = 7200,  # Increased to 2 hours
    ) -> Dict[str, Any]:
        """
        Run official SWE-bench evaluation using Docker (legacy sync method).

        Requires Docker to be installed and running.
        """
        print(f"[SWE-bench] Starting evaluation with run_id: {run_id}")
        print(f"[SWE-bench] Dataset: {self.dataset_name}")
        print(f"[SWE-bench] Split: {self.split}")
        print(f"[SWE-bench] Predictions: {predictions_path}")
        print(f"[SWE-bench] Max workers: {self.max_workers}")

        # Check Docker is available
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                check=True,
                timeout=30
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Docker is not available. SWE-bench evaluation requires Docker.\n"
                "Please install Docker and ensure the daemon is running."
            ) from e

        # Run the official SWE-bench harness
        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", self.dataset_name,
            "--split", self.split,
            "--predictions_path", str(predictions_path),
            "--max_workers", str(self.max_workers),
            "--run_id", run_id,
        ]

        print(f"[SWE-bench] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            print(f"[SWE-bench] stdout: {result.stdout[-2000:]}")
            if result.stderr:
                print(f"[SWE-bench] stderr: {result.stderr[-1000:]}")

            if result.returncode != 0:
                raise RuntimeError(f"SWE-bench evaluation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SWE-bench evaluation timed out after {timeout}s")

        # Parse results
        return self._parse_results(run_id)

    def _parse_results(self, run_id: str) -> Dict[str, Any]:
        """Parse SWE-bench evaluation results."""
        results = {
            "run_id": run_id,
            "instances": [],
            "summary": {},
        }

        # SWE-bench creates results as {model_name}.{run_id}.json in current dir
        # Find any matching file
        import glob
        pattern = f"*.{run_id}.json"
        matching_files = glob.glob(pattern)

        if matching_files:
            results_file = Path(matching_files[0])
            print(f"[SWE-bench] Found results file: {results_file}")
            with open(results_file) as f:
                data = json.load(f)
                results["summary"] = data

                # Convert to instance results format
                for instance_id in data.get("resolved_ids", []):
                    results["instances"].append({
                        "instance_id": instance_id,
                        "resolved": True,
                        "patch_applied": True,
                    })
                for instance_id in data.get("unresolved_ids", []):
                    results["instances"].append({
                        "instance_id": instance_id,
                        "resolved": False,
                        "patch_applied": True,
                    })
                for instance_id in data.get("error_ids", []):
                    results["instances"].append({
                        "instance_id": instance_id,
                        "resolved": False,
                        "patch_applied": False,
                        "error": True,
                    })
        else:
            # Try legacy location
            results_dir = Path("evaluation_results") / run_id
            results_file = results_dir / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results["summary"] = json.load(f)

            instance_results = results_dir / "instance_results.jsonl"
            if instance_results.exists():
                with open(instance_results) as f:
                    for line in f:
                        if line.strip():
                            results["instances"].append(json.loads(line))

        print(f"[SWE-bench] Parsed {len(results['instances'])} instance results")
        return results

    async def import_results(
        self,
        experiment_id: str,
        swebench_results: Dict[str, Any],
    ) -> List[TestResult]:
        """
        Import SWE-bench results back into our storage.
        """
        test_results = []

        # Get our results to match with SWE-bench results
        our_results = await self.storage.get_results_for_experiment(
            experiment_id,
            success_only=True,
            limit=10000
        )
        results_by_instance = {r.instance_id: r for r in our_results}

        for instance_result in swebench_results.get("instances", []):
            instance_id = instance_result.get("instance_id")
            our_result = results_by_instance.get(instance_id)

            if not our_result:
                continue

            test_result = TestResult(
                result_id=our_result.id,
                experiment_id=experiment_id,
                instance_id=instance_id,
                patch_applied=instance_result.get("patch_applied", False),
                resolved=instance_result.get("resolved", False),
                test_output=json.dumps(instance_result),
                executed_at=datetime.utcnow(),
            )

            await self.storage.save_test_result(test_result)
            test_results.append(test_result)

        print(f"[SWE-bench] Imported {len(test_results)} test results")
        return test_results

    async def evaluate_experiment(
        self,
        experiment_id: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full evaluation pipeline:
        1. Export predictions
        2. Run SWE-bench harness
        3. Import results
        """
        if run_id is None:
            run_id = f"exp_{experiment_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Export
        predictions_path = await self.export_predictions(experiment_id)

        # Evaluate
        results = self.run_evaluation(predictions_path, run_id)

        # Import
        await self.import_results(experiment_id, results)

        return results


# Convenience function for API usage
async def run_swebench_evaluation(
    experiment_id: str,
    storage: Optional[Storage] = None,
    split: str = "dev",
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Run SWE-bench evaluation for an experiment."""
    storage = storage or Storage()
    harness = SWEBenchHarness(storage, split=split, max_workers=max_workers)
    return await harness.evaluate_experiment(experiment_id)
