#!/usr/bin/env python3
"""
Build the tidy results table: one row per (model, regime, repetition, instance).

Sources: the per-provider result databases (what the model did) and the verdict files
written by scripts/evaluate.py (what the SWE-bench harness decided). Nothing is
re-computed about the model's answer here.

Usage:
    .venv/bin/python analysis/build_table.py [--tag rerun-2026-10] [--subset full]
Output: analysis/out/<tag>__<subset>__tidy.csv
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmark.pricing import billed_cost_usd, cost_usd  # noqa: E402
from app.benchmark.swebench_eval import classify_instance_dir, instance_log_dir  # noqa: E402

PROVIDERS = ["anthropic", "openai", "google"]
MODEL_LABEL = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gpt-4.1-mini": "GPT-4.1-mini",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
}
# What happened to an instance, in the order of the pipeline
OUTCOMES = [
    "abstained",            # explicit <<<CANNOT_SOLVE>>>
    "no_answer",            # no edit and no abstention in the final answer
    "edit_not_applicable",  # edits given, but they do not match the code
    "context_overflow",
    "apply_failed",         # patch produced, harness could not apply it
    "tests_failed",         # patch applied, tests not all passing
    "resolved",
    "not_evaluated",        # patch produced, no harness verdict yet
]
STATUS_TO_OUTCOME = {
    "failure_explicit": "abstained",
    "hallucination_format_error": "no_answer",
    "edit_not_applicable": "edit_not_applicable",
    "context_overflow": "context_overflow",
}


def load_settled(eval_dir: Path, job: str, kind: str) -> dict:
    """Verdict file written by scripts/evaluate.py at the end of a pass (used only to know
    which persistent test timeouts were settled as failed tests)."""
    path = eval_dir / "verdicts" / f"{job}__{kind}.json"
    return json.loads(path.read_text())["verdicts"] if path.exists() else {}


def patch_outcome(eval_dir: Path, job: str, kind: str, instance_id: str, patch: str, settled: dict) -> tuple[str, str | None]:
    """Harness verdict for one stored patch, read from the harness' own output.

    The output only counts if it belongs to exactly this patch (the harness keeps the
    evaluated diff next to its report)."""
    run_id = f"{job}__{kind}"
    directory = instance_log_dir(eval_dir, run_id, run_id, instance_id)
    evaluated = directory / "patch.diff"
    if evaluated.exists() and evaluated.read_text() != patch:
        return "not_evaluated", None
    verdict = classify_instance_dir(directory, instance_id)
    if verdict.status == "eval_error":
        settled_verdict = settled.get(instance_id)
        if verdict.kind == "timeout" and settled_verdict and settled_verdict["status"] == "tests_failed":
            return "tests_failed", "timeout"
        return "not_evaluated", None
    return verdict.status, verdict.kind


def build(run_dir: Path, subset: str) -> pd.DataFrame:
    eval_dir = run_dir / "eval"
    records = []
    for provider in PROVIDERS:
        db_path = run_dir / f"results_{provider}.db"
        if not db_path.exists():
            continue
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        for experiment in connection.execute("SELECT id, name, config FROM experiments"):
            if not experiment["name"].endswith(f"__{subset}"):
                continue
            config = json.loads(experiment["config"])
            job = experiment["name"]
            settled_final = load_settled(eval_dir, job, "final")
            settled_initial = load_settled(eval_dir, job, "initial")

            for row in connection.execute("SELECT * FROM experiment_results WHERE experiment_id = ?", (experiment["id"],)):
                status = row["result_status"]
                if status in ("api_error", "infra_error", None):
                    continue  # not a model outcome; the job is incomplete until these are re-run
                meta = json.loads(row["run_meta"] or "{}")
                instance_id = row["instance_id"]
                model = row["model"]

                # Final patch
                if row["generated_patch"]:
                    outcome, failure_kind = patch_outcome(eval_dir, job, "final", instance_id, row["generated_patch"], settled_final)
                else:
                    outcome, failure_kind = STATUS_TO_OUTCOME[status], None

                # First patch of the self-reflection regime (before the reflection call)
                initial_has_patch = initial_outcome = None
                if config["experiment_mode"] == "self_reflect" and row["reflection_outcome"] not in (None, "skipped"):
                    initial_patch = row["original_patch"]
                    initial_has_patch = bool(initial_patch)
                    if not initial_patch:
                        initial_outcome = "edit_not_applicable"
                    elif initial_patch == row["generated_patch"]:
                        initial_outcome = outcome
                    else:
                        initial_outcome, _ = patch_outcome(eval_dir, job, "initial", instance_id, initial_patch, settled_initial)

                prompt = (row["prompt_tokens"] or 0) + (row["reflection_prompt_tokens"] or 0)
                completion = (row["completion_tokens"] or 0) + (row["reflection_completion_tokens"] or 0)
                cached = meta.get("cached_input_tokens", 0) or 0
                records.append({
                    "provider": provider,
                    "model": model,
                    "model_label": MODEL_LABEL.get(model, model),
                    "mode": config["experiment_mode"],
                    "rep": config.get("run_idx"),
                    "instance_id": instance_id,
                    "repo": instance_id.rsplit("-", 1)[0],
                    "outcome": outcome,
                    "failure_kind": failure_kind,
                    "has_patch": bool(row["generated_patch"]),
                    "resolved": outcome == "resolved",
                    "evaluated": outcome != "not_evaluated",
                    "abstained": bool(meta.get("abstained")),
                    "reflection_outcome": row["reflection_outcome"],
                    "initial_has_patch": initial_has_patch,
                    "initial_outcome": initial_outcome,
                    "initial_resolved": (initial_outcome == "resolved") if initial_outcome is not None else None,
                    "loop_prompt_tokens": row["prompt_tokens"] or 0,
                    "loop_completion_tokens": row["completion_tokens"] or 0,
                    "reflection_prompt_tokens": row["reflection_prompt_tokens"] or 0,
                    "reflection_completion_tokens": row["reflection_completion_tokens"] or 0,
                    "input_tokens": prompt,
                    "output_tokens": completion,
                    "cached_input_tokens": cached,
                    "cost_list_usd": cost_usd(model, prompt, completion),
                    "cost_billed_usd": billed_cost_usd(model, prompt, completion, cached),
                    "iterations": meta.get("iterations"),
                    "termination_reason": meta.get("termination_reason"),
                    "cap_hit": bool(meta.get("cap_hit")),
                    "fallback_used": bool(meta.get("fallback_used")),
                    "tool_calls": row["tool_call_count"] or 0,
                    "api_retries": meta.get("api_retry_count", 0),
                    "answer_format": meta.get("answer_format"),
                    "path_resolved": bool(meta.get("edit_paths_resolved") or meta.get("revised_edit_paths_resolved")),
                    "edit_failure_reasons": ",".join(sorted({f["reason"] for f in meta.get("edit_failures") or []})),
                    "api_model_ids": ",".join(meta.get("api_model_ids") or []),
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "response_seconds": row["response_time_seconds"],
                    "harness": config.get("harness_commit"),
                    "job": job,
                })
        connection.close()
    return pd.DataFrame.from_records(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="rerun-2026-10")
    parser.add_argument("--subset", default="full")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    args = parser.parse_args()

    table = build(args.runs_dir / args.tag, args.subset)
    if table.empty:
        print("no rows found")
        return 1
    out = PROJECT_ROOT / "analysis" / "out" / f"{args.tag}__{args.subset}__tidy.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    jobs = table.groupby(["model_label", "mode", "rep"]).agg(
        instances=("instance_id", "nunique"), patches=("has_patch", "sum"), resolved=("resolved", "sum"),
        not_evaluated=("outcome", lambda s: (s == "not_evaluated").sum()),
    )
    print(jobs.to_string())
    print(f"\n{len(table)} rows -> {out.relative_to(PROJECT_ROOT)}")
    duplicated = table.duplicated(["model", "mode", "rep", "instance_id"]).sum()
    if duplicated:
        print(f"WARNING: {duplicated} duplicated (model, mode, rep, instance) rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
