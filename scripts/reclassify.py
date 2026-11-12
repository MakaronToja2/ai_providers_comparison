#!/usr/bin/env python3
"""
Re-read stored model answers with the current parser and edit-application rules.

No model is called. For every finished row the script takes the final answer (and the
reflection response, if any) from the trajectory log, runs it through the same
classification code the runner uses (in a fresh checkout of the instance), and compares
the outcome with what is stored. With --apply, rows whose outcome changes are updated and
the change is noted in the row's run_meta (`reclassified`).

Use: after a parser or edit-application fix, to bring jobs that were run before the fix
to the same rules without sampling the model again.

Rows it will not touch: self-reflection rows where the stored run made no reflection call
but the re-read first answer does contain a patch (the reflection step would have run and
cannot be reconstructed). They are reported as needing a real re-run.

Usage:
    .venv/bin/python scripts/reclassify.py --provider anthropic [--mode baseline] [--run-idx 1]          # report only
    .venv/bin/python scripts/reclassify.py --provider anthropic --mode baseline --run-idx 1 --apply
"""
import argparse
import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
os.environ.setdefault("TOOL_DEBUG", "false")

TERMINAL = ("success_patch_generated", "failure_explicit", "hallucination_format_error", "edit_not_applicable")


def read_trajectory(path: str) -> dict:
    """Final answer, reflection text and files read, as the run produced them."""
    records = [json.loads(line) for line in open(path)]
    calls = [r for r in records if r["type"] == "api_call" and not r.get("error")]
    answer_calls = [c for c in calls if c["phase"] != "reflection"]
    fallback = [c for c in answer_calls if c["phase"] == "fallback" and (c.get("assistant_text") or "")]
    loop_final = [c for c in answer_calls if c["phase"] == "loop" and not c.get("tool_calls")]
    if fallback:
        final_answer = fallback[-1]["assistant_text"]
    elif loop_final:
        final_answer = loop_final[-1].get("assistant_text") or ""
    else:
        final_answer = ""
    reflection = [c for c in calls if c["phase"] == "reflection"]
    tool_calls = [
        SimpleNamespace(name=r["name"], success=r["success"], parameters=r.get("arguments") or {})
        for r in records if r["type"] == "tool_result"
    ]
    return {
        "final_answer": final_answer,
        "reflection_text": reflection[-1].get("assistant_text") if reflection else None,
        "tool_calls": tool_calls,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "google"])
    parser.add_argument("--mode", nargs="*", default=None)
    parser.add_argument("--run-idx", nargs="*", type=int, default=None)
    parser.add_argument("--subset", default="full")
    parser.add_argument("--tag", default="rerun-2026-10")
    parser.add_argument("--apply", action="store_true", help="Write the changed outcomes to the results database")
    args = parser.parse_args()

    selection = json.loads((PROJECT_ROOT / "data" / "instances_seed20261007.json").read_text())
    os.environ["SWEBENCH_REVISION_LITE"] = selection["dataset_revision"]

    from app.benchmark.runner import ExperimentRunner
    from app.core.base_provider import BaseAIProvider
    from app.models.benchmark import ExperimentResult
    from app.utils.repo_context import RepoContext
    from app.utils.swe_bench_loader import SWEBenchLoader
    from scripts.run_experiment import harness_commit

    loader = SWEBenchLoader()
    loader.load_dataset("lite")
    runner = ExperimentRunner(storage=None, swe_bench_loader=loader)
    db_path = PROJECT_ROOT / "runs" / args.tag / f"results_{args.provider}.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row

    changes = Counter()
    needs_rerun = []
    updates = []
    for experiment in connection.execute("SELECT id, name, config FROM experiments ORDER BY name").fetchall():
        config = json.loads(experiment["config"])
        if not experiment["name"].endswith(f"__{args.subset}"):
            continue
        if args.mode and config["experiment_mode"] not in args.mode:
            continue
        if args.run_idx and config.get("run_idx") not in args.run_idx:
            continue
        rows = connection.execute("SELECT * FROM experiment_results WHERE experiment_id = ?", (experiment["id"],)).fetchall()
        for row in rows:
            if row["result_status"] not in TERMINAL:
                continue
            meta = json.loads(row["run_meta"] or "{}")
            trajectory = meta.get("trajectory_path")
            if not trajectory or not Path(trajectory).exists():
                continue
            stored = read_trajectory(trajectory)
            is_reflect = config["experiment_mode"] == "self_reflect"
            if is_reflect and stored["reflection_text"] is None and \
                    BaseAIProvider._has_patch(stored["final_answer"]) and "<<<CANNOT_SOLVE>>>" not in stored["final_answer"]:
                if row["reflection_outcome"] in (None, "skipped"):
                    needs_rerun.append((experiment["name"], row["instance_id"]))
                    continue

            instance = loader.get_instance(row["instance_id"], "lite")
            result = ExperimentResult(experiment_id=experiment["id"], instance_id=row["instance_id"], provider=args.provider,
                                      model=row["model"], tool_set=row["tool_set"])
            new_meta: dict = {}
            async with RepoContext(instance.repo, instance.base_commit):
                runner._classify_response(result, SimpleNamespace(**stored), new_meta)

            new_status = result.result_status.value
            same = (new_status == row["result_status"] and result.generated_patch == row["generated_patch"]
                    and result.original_patch == row["original_patch"])
            if same:
                continue
            changes[(experiment["name"].split("__", 1)[1], row["result_status"], new_status)] += 1
            merged = {**meta, **new_meta, "reclassified": {
                "from_status": row["result_status"], "to_status": new_status, "harness": harness_commit(),
            }}
            updates.append((result.generated_patch, result.original_patch, new_status, 1 if result.success else 0,
                            result.error_message, result.reflection_outcome if is_reflect else row["reflection_outcome"],
                            json.dumps(merged), row["id"]))

    print(f"{args.provider}: {sum(changes.values())} row(s) would change" if not args.apply else f"{args.provider}: {sum(changes.values())} row(s) changed")
    for (job, old, new), n in sorted(changes.items()):
        print(f"  {n:3d}  {job}: {old} -> {new}")
    if needs_rerun:
        print(f"  {len(needs_rerun)} self-reflection row(s) cannot be re-read (no reflection call was made): {needs_rerun[:5]}")
    if args.apply and updates:
        connection.executemany(
            "UPDATE experiment_results SET generated_patch = ?, original_patch = ?, result_status = ?, success = ?, "
            "error_message = ?, reflection_outcome = ?, run_meta = ? WHERE id = ?", updates)
        connection.commit()
    connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
