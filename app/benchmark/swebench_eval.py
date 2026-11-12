"""
Helpers for evaluating patches with the official SWE-bench harness and reading its
per-instance output. Everything here is pure (files in, values out); the driver is
scripts/evaluate.py.

A patch ends in exactly one of four states:
- resolved:      report.json says the FAIL_TO_PASS and PASS_TO_PASS tests all pass;
- tests_failed:  the patch applied but the tests did not all pass;
- apply_failed:  the harness could not apply the patch;
- eval_error:    the harness did not get as far as a verdict (missing image, Docker
                 error, timeout). This is never a verdict on the patch: it is retried.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# File names and log markers of swebench 4.1.0 (swebench/harness/constants)
LOG_DIR = Path("logs/run_evaluation")
REPORT_FILE = "report.json"
INSTANCE_LOG_FILE = "run_instance.log"
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
TIMEOUT_MARKERS = ("Test timed out after", ">>>>> Tests Timed Out")

RESOLVED = "resolved"
TESTS_FAILED = "tests_failed"
APPLY_FAILED = "apply_failed"
EVAL_ERROR = "eval_error"
TERMINAL_VERDICTS = frozenset({RESOLVED, TESTS_FAILED, APPLY_FAILED})


@dataclass
class Verdict:
    status: str
    # tests_failed: f2p_failed | p2p_regression | both | no_test_results | timeout
    # apply_failed: hunk_failed | malformed | missing_file | other
    # eval_error:   no_output | image | timeout | other
    kind: Optional[str] = None
    f2p_passed: Optional[int] = None
    f2p_total: Optional[int] = None
    p2p_passed: Optional[int] = None
    p2p_total: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def instance_log_dir(eval_cwd: Path, run_id: str, model_name: str, instance_id: str) -> Path:
    """Directory the harness writes for one prediction."""
    return eval_cwd / LOG_DIR / run_id / model_name.replace("/", "__") / instance_id


def _apply_failure_kind(log_text: str) -> str:
    tail = log_text[log_text.find(APPLY_PATCH_FAIL):]
    lowered = tail.lower()
    if "malformed patch" in lowered or "only garbage" in lowered or "corrupt patch" in lowered:
        return "malformed"
    if "no such file" in lowered or "can't find file to patch" in lowered:
        return "missing_file"
    if "hunk" in lowered and "failed" in lowered:
        return "hunk_failed"
    return "other"


def classify_instance_dir(directory: Path, instance_id: str) -> Verdict:
    """Read the harness output for one prediction and return its verdict."""
    report_path = directory / REPORT_FILE
    log_path = directory / INSTANCE_LOG_FILE

    if report_path.exists():
        report = json.loads(report_path.read_text()).get(instance_id) or {}
        tests = report.get("tests_status") or {}
        f2p = tests.get("FAIL_TO_PASS") or {}
        p2p = tests.get("PASS_TO_PASS") or {}
        counts = dict(
            f2p_passed=len(f2p.get("success") or []),
            f2p_total=len(f2p.get("success") or []) + len(f2p.get("failure") or []),
            p2p_passed=len(p2p.get("success") or []),
            p2p_total=len(p2p.get("success") or []) + len(p2p.get("failure") or []),
        ) if tests else {}

        if report.get("resolved"):
            return Verdict(RESOLVED, **counts)
        if not tests:
            return Verdict(TESTS_FAILED, "no_test_results")
        f2p_failed = bool(f2p.get("failure"))
        p2p_failed = bool(p2p.get("failure"))
        if f2p_failed and p2p_failed:
            kind = "both"
        elif f2p_failed:
            kind = "f2p_failed"
        elif p2p_failed:
            kind = "p2p_regression"
        else:
            kind = "no_test_results"
        return Verdict(TESTS_FAILED, kind, **counts)

    if not log_path.exists() or log_path.stat().st_size == 0:
        return Verdict(EVAL_ERROR, "no_output")

    log_text = log_path.read_text(errors="replace")
    if APPLY_PATCH_FAIL in log_text:
        return Verdict(APPLY_FAILED, _apply_failure_kind(log_text))
    if any(marker in log_text for marker in TIMEOUT_MARKERS):
        return Verdict(EVAL_ERROR, "timeout")
    if "BuildImageError" in log_text or "No such image" in log_text or "pull access denied" in log_text:
        return Verdict(EVAL_ERROR, "image")
    return Verdict(EVAL_ERROR, "other")


def settle_timeouts(verdicts: Dict[str, Verdict]) -> Dict[str, Verdict]:
    """After the retry budget: a patch whose tests still time out is unresolved."""
    return {
        instance_id: Verdict(TESTS_FAILED, "timeout") if v.status == EVAL_ERROR and v.kind == "timeout" else v
        for instance_id, v in verdicts.items()
    }


def build_predictions(rows: Iterable[Dict[str, Any]], model_name: str, kind: str) -> List[Dict[str, str]]:
    """Predictions for one job.

    kind "final":   every row with a patch (generated_patch).
    kind "initial": self-reflection rows whose first patch differs from the final one
                    (identical patches share the final verdict and are not evaluated twice).
    """
    predictions = []
    for row in rows:
        final_patch = row.get("generated_patch")
        initial_patch = row.get("original_patch")
        if kind == "final":
            patch = final_patch
        elif kind == "initial":
            patch = initial_patch if initial_patch and initial_patch != final_patch else None
        else:
            raise ValueError(f"Unknown prediction kind '{kind}'")
        if patch:
            predictions.append({
                "instance_id": row["instance_id"],
                "model_name_or_path": model_name,
                "model_patch": patch,
            })

    instance_ids = [p["instance_id"] for p in predictions]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError(f"Duplicate instance ids in predictions for {model_name}")
    return predictions


def harness_command(
    python: str,
    dataset: str,
    split: str,
    predictions_path: str,
    run_id: str,
    instance_ids: List[str],
    max_workers: int,
    timeout: int,
    report_dir: str,
) -> List[str]:
    """Command line for swebench.harness.run_evaluation (flags of release 4.1.0).

    Prebuilt images are used (--namespace swebench) and instance images are kept
    (--cache_level instance), so every image is pulled once for the whole study.
    """
    return [
        python, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset,
        "--split", split,
        "--predictions_path", predictions_path,
        "--run_id", run_id,
        "--max_workers", str(max_workers),
        "--timeout", str(timeout),
        "--namespace", "swebench",
        "--cache_level", "instance",
        "--clean", "False",
        "--report_dir", report_dir,
        "--instance_ids", *instance_ids,
    ]
