#!/usr/bin/env python3
"""
Run one job of the study: one model, one prompting regime, one repetition, over the
smoke or the full instance set. Thin wrapper over app.benchmark.runner.ExperimentRunner.

The job is resume-safe: re-running the same command continues where it stopped and
re-runs only instances whose row records an API or harness error. The exit code is 0
only when every instance has a model outcome.

Usage:
    .venv/bin/python scripts/run_experiment.py --provider openai --mode baseline \\
        --subset smoke --run-idx 1
"""
import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # settings read .env from the working directory

# Tool calls are logged to the trajectory files, not printed
os.environ.setdefault("TOOL_DEBUG", "false")

DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemini-2.5-flash-lite",
}
DEFAULT_TAG = "rerun-2026-10"
DEFAULT_INSTANCES = PROJECT_ROOT / "data" / "instances_seed20261007.json"
COST_CHECKPOINT = 10  # print a projected job cost after this many instances


def job_name(tag: str, provider: str, mode: str, run_idx: int, subset: str) -> str:
    return f"{tag}__{provider}__{mode}__rep{run_idx}__{subset}"


def job_experiment_id(tag: str, provider: str, model: str, mode: str, run_idx: int, subset: str) -> str:
    """Deterministic experiment id, so the same command always addresses the same job."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join([tag, provider, model, mode, f"rep{run_idx}", subset])))


def harness_commit() -> str:
    """Commit hash of the harness, with a digest of uncommitted changes if there are any."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-c", "core.fileMode=false", *args], cwd=PROJECT_ROOT,
                              capture_output=True, text=True).stdout
    head = git("rev-parse", "HEAD").strip()
    diff = git("diff", "HEAD", "--", "app", "scripts")
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "app", "scripts")
    if diff or untracked.strip():
        digest = hashlib.sha256()
        digest.update(diff.encode())
        for name in sorted(untracked.split()):
            digest.update(name.encode())
            digest.update((PROJECT_ROOT / name).read_bytes())
        return f"{head}+dirty.{digest.hexdigest()[:12]}"
    return head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", required=True, choices=sorted(DEFAULT_MODELS))
    parser.add_argument("--model", default=None, help="Model id (default: the study's model for the provider)")
    parser.add_argument("--mode", required=True, choices=["baseline", "forced", "self_reflect"])
    parser.add_argument("--subset", required=True, choices=["smoke", "full"])
    parser.add_argument("--run-idx", type=int, required=True, help="Repetition index, starting at 1")
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Name of the run directory under runs/")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N instances of the set; the job is then named <subset><N> (e.g. smoke5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do all setup (tools, dataset, repository mirrors), print the job, and stop before any API call or database write")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    model = args.model or DEFAULT_MODELS[args.provider]
    selection = json.loads(args.instances.read_text())
    instance_ids = selection["smoke_instance_ids"] if args.subset == "smoke" else selection["instance_ids"]
    if args.limit:
        instance_ids = instance_ids[:args.limit]
        args.subset = f"{args.subset}{args.limit}"

    run_dir = (args.runs_dir / args.tag).resolve()
    db_path = run_dir / f"results_{args.provider}.db"
    if "archive" in db_path.parts:
        raise SystemExit("Refusing to write into the archive directory")
    name = job_name(args.tag, args.provider, args.mode, args.run_idx, args.subset)
    trajectory_dir = run_dir / "trajectories" / args.provider / args.mode / f"rep{args.run_idx}" / args.subset
    run_dir.mkdir(parents=True, exist_ok=True)

    # The dataset revision the instances were drawn from is the one the agent sees
    os.environ["SWEBENCH_REVISION_LITE"] = selection["dataset_revision"]

    from app.benchmark.pricing import billed_cost_usd, cost_usd
    from app.benchmark.runner import ExperimentRunner
    from app.benchmark.storage import Storage
    from app.core.tool_manager import initialize_tools
    from app.core.tools import tool_registry
    from app.models.benchmark import (
        Experiment, ExperimentConfig, ExperimentMode, ResultStatus, TERMINAL_RESULT_STATUSES,
    )
    from app.tools.code_search import ripgrep_binary
    from app.utils.repo_context import ensure_mirror
    from app.utils.swe_bench_loader import SWEBenchLoader

    if not ripgrep_binary():
        raise SystemExit("ripgrep (rg) not found: search_code would silently fall back to grep. Install it or set RG_PATH.")

    initialize_tools()
    enabled = sorted(t.name for t in tool_registry.get_enabled_tools())
    if enabled != ["list_directory", "read_file", "search_code"]:
        raise SystemExit(f"Expected the three read-only tools to be enabled, got {enabled}")

    # Dataset at the pinned revision, and local mirrors of the repositories (created once,
    # shared by all jobs)
    loader = SWEBenchLoader()
    loader.load_dataset("lite")
    instances = [loader.get_instance(i, "lite") for i in instance_ids]
    if any(instance is None for instance in instances):
        raise SystemExit("Some instances could not be loaded from the dataset")
    for repo in sorted({instance.repo for instance in instances}):
        print(f"[setup] mirror ready: {repo} -> {await asyncio.to_thread(ensure_mirror, repo)}", flush=True)

    experiment_id = job_experiment_id(args.tag, args.provider, model, args.mode, args.run_idx, args.subset)
    if args.dry_run:
        print(f"[dry-run] job {name}")
        print(f"[dry-run] model {model} | mode {args.mode} | {len(instance_ids)} instances | max {args.max_concurrent} concurrent")
        print(f"[dry-run] dataset revision {selection['dataset_revision']} | harness {harness_commit()}")
        print(f"[dry-run] results -> {db_path.relative_to(PROJECT_ROOT)} | trajectories -> {trajectory_dir.relative_to(PROJECT_ROOT)}")
        print("[dry-run] setup complete; no API call made, nothing written to the results database")
        return 0

    storage = Storage(str(db_path))
    experiment = await storage.get_experiment(experiment_id)
    if experiment is None:
        config = ExperimentConfig(
            name=name,
            description=f"{args.provider} {model}, mode {args.mode}, repetition {args.run_idx}, {args.subset} set",
            providers=[args.provider],
            models={args.provider: model},
            split="lite",
            experiment_mode=ExperimentMode(args.mode),
            instance_ids=instance_ids,
            max_concurrent=args.max_concurrent,
            run_tag=args.tag,
            run_idx=args.run_idx,
            instances_file=str(args.instances.resolve().relative_to(PROJECT_ROOT)),
            instances_sha256=selection["instance_ids_sha256"],
            dataset_revision=selection["dataset_revision"],
            harness_commit=harness_commit(),
        )
        experiment = await storage.create_experiment(Experiment(id=experiment_id, name=name, description=config.description, config=config))
    elif experiment.config.instance_ids != instance_ids:
        raise SystemExit(f"Job {name} already exists with a different instance list")

    freeze_path = run_dir / "pip-freeze.txt"
    if not freeze_path.exists():
        freeze_path.write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)

    print(f"[job] {name}")
    print(f"[job] model {model} | {len(instance_ids)} instances | db {db_path.relative_to(PROJECT_ROOT)}")

    started = datetime.now(timezone.utc)
    runner = ExperimentRunner(storage, swe_bench_loader=loader, trajectory_dir=trajectory_dir)
    done = 0
    job_cost = 0.0
    async for update in runner.start_experiment(experiment_id, max_concurrent=args.max_concurrent):
        done += 1
        result_id = ExperimentRunner.result_id_for(experiment_id, update.instance_id, args.provider, model, "all_tools")
        row = await storage.get_result(result_id)
        instance_cost = 0.0
        if row is not None:
            instance_cost = cost_usd(model, row.prompt_tokens, row.completion_tokens) + \
                cost_usd(model, row.reflection_prompt_tokens, row.reflection_completion_tokens)
        job_cost += instance_cost
        print(f"[{update.completed_count}/{update.total_count}] {update.instance_id:40s} {update.status:28s} "
              f"${instance_cost:.4f} (job ${job_cost:.3f})", flush=True)
        if done == COST_CHECKPOINT and update.total_count > COST_CHECKPOINT:
            projected = job_cost / done * update.total_count
            print(f"[cost] after {done} instances: ${job_cost:.3f} spent this invocation, "
                  f"projected ${projected:.2f} for the {update.total_count} instances in this invocation", flush=True)

    # Final state of the job, over all invocations
    rows = await storage.get_results_for_experiment(experiment_id, limit=100000)
    statuses = Counter(r.result_status.value if r.result_status else "none" for r in rows)
    terminal = {s.value for s in TERMINAL_RESULT_STATUSES}
    incomplete = len(instance_ids) - sum(n for status, n in statuses.items() if status in terminal)
    def tokens(r):
        """Input and output tokens of an instance, main loop and reflection together."""
        return ((r.prompt_tokens or 0) + (r.reflection_prompt_tokens or 0),
                (r.completion_tokens or 0) + (r.reflection_completion_tokens or 0))

    total_cost = sum(cost_usd(model, *tokens(r)) for r in rows)
    cached_tokens = sum((r.run_meta or {}).get("cached_input_tokens", 0) for r in rows)
    billed_cost = sum(billed_cost_usd(model, *tokens(r), (r.run_meta or {}).get("cached_input_tokens", 0)) for r in rows)
    model_ids = sorted({m for r in rows for m in (r.run_meta or {}).get("api_model_ids", [])})

    manifest = {
        "job": name,
        "experiment_id": experiment_id,
        "provider": args.provider,
        "model": model,
        "mode": args.mode,
        "run_idx": args.run_idx,
        "subset": args.subset,
        "instances": len(instance_ids),
        "status_counts": dict(statuses),
        "incomplete": incomplete,
        "api_model_ids": model_ids,
        "input_tokens": sum(tokens(r)[0] for r in rows),
        "output_tokens": sum(tokens(r)[1] for r in rows),
        "cached_input_tokens": cached_tokens,
        "cost_usd_list_price": round(total_cost, 6),
        "cost_usd_as_billed_estimate": round(billed_cost, 6),
        "last_invocation_started": started.isoformat(timespec="seconds"),
        "last_invocation_finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "harness_commit": experiment.config.harness_commit,
        "dataset_revision": experiment.config.dataset_revision,
        "instances_sha256": experiment.config.instances_sha256,
    }
    jobs_dir = run_dir / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    (jobs_dir / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[job] outcomes: {dict(statuses)}")
    print(f"[job] model ids returned by the API: {model_ids}")
    print(f"[job] cost at list price: ${total_cost:.3f} | as billed (cached input discounted): ${billed_cost:.3f}")
    if incomplete:
        print(f"[job] {incomplete} instance(s) have no model outcome yet (API or harness errors). Re-run the same command to resume.")
        return 2
    print("[job] complete")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))
