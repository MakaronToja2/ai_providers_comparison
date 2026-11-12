#!/usr/bin/env python3
"""
Evaluate the patches of finished jobs with the official SWE-bench harness (Docker).

For every selected job this writes one predictions file per patch kind ("final", and
for the self-reflection regime also "initial"), runs the harness, reads the verdict of
every patch from the harness' own report files, and re-runs anything that ended without
a verdict (missing image, Docker error, timeout). It is idempotent: patches that already
have a verdict are not evaluated again. The exit code is 0 only when every patch has one.

Usage:
    .venv/bin/python scripts/evaluate.py --subset smoke --gold          # validate the harness
    .venv/bin/python scripts/evaluate.py --subset smoke --provider openai
    .venv/bin/python scripts/evaluate.py --subset full                  # all finished jobs

Low-disk mode: --batch-size N pulls the images of N instances, evaluates every pending
patch for those instances across all selected jobs (and the reference patches with
--with-gold), and with --remove-images deletes those images before the next batch.
    .venv/bin/python scripts/evaluate.py --subset full --with-gold --batch-size 15 --remove-images
"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmark.swebench_eval import (  # noqa: E402
    EVAL_ERROR, TERMINAL_VERDICTS, Verdict, build_predictions, classify_instance_dir,
    harness_command, instance_log_dir, settle_timeouts,
)

DEFAULT_TAG = "rerun-2026-10"
DEFAULT_INSTANCES = PROJECT_ROOT / "data" / "instances_seed20261007.json"
PROVIDERS = ["openai", "anthropic", "google"]
DATASET = "SWE-bench/SWE-bench_Lite"
SPLIT = "test"


def export_dataset(selection: dict, instance_ids: List[str], path: Path) -> None:
    """Write the instances at the pinned dataset revision to a local file for the harness,
    so evaluation uses exactly the revision the instances were drawn from."""
    if path.exists():
        return
    from datasets import load_dataset
    dataset = load_dataset(DATASET, split=SPLIT, revision=selection["dataset_revision"])
    wanted = set(instance_ids)
    rows = [dict(row) for row in dataset if row["instance_id"] in wanted]
    if len(rows) != len(wanted):
        raise SystemExit(f"Dataset export: expected {len(wanted)} instances, found {len(rows)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, default=str))


def load_jobs(run_dir: Path, providers: List[str], modes: Optional[List[str]],
              run_indices: Optional[List[int]], subset: str) -> List[dict]:
    """Finished or partly finished jobs from the per-provider result databases."""
    jobs = []
    for provider in providers:
        db_path = run_dir / f"results_{provider}.db"
        if not db_path.exists():
            continue
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        for experiment in connection.execute("SELECT id, name, config FROM experiments ORDER BY name"):
            config = json.loads(experiment["config"])
            if not experiment["name"].endswith(f"__{subset}"):
                continue
            if modes and config["experiment_mode"] not in modes:
                continue
            if run_indices and config.get("run_idx") not in run_indices:
                continue
            rows = [dict(r) for r in connection.execute(
                "SELECT instance_id, generated_patch, original_patch, result_status "
                "FROM experiment_results WHERE experiment_id = ?", (experiment["id"],))]
            jobs.append({"name": experiment["name"], "mode": config["experiment_mode"], "rows": rows})
        connection.close()
    return jobs


def evaluate_predictions(
    *, eval_cwd: Path, run_id: str, model_name: str, predictions_path: str, instance_ids: List[str],
    dataset_path: Path, args: argparse.Namespace,
) -> Dict[str, Verdict]:
    """Run the harness until every prediction has a verdict or the attempts are used up."""

    def read_verdicts() -> Dict[str, Verdict]:
        return {
            instance_id: classify_instance_dir(instance_log_dir(eval_cwd, run_id, model_name, instance_id), instance_id)
            for instance_id in instance_ids
        }

    verdicts = read_verdicts()
    for attempt in range(1, args.max_attempts + 1):
        pending = [i for i, v in verdicts.items() if v.status not in TERMINAL_VERDICTS]
        if not pending:
            break
        print(f"[eval] {run_id}: attempt {attempt}/{args.max_attempts}, {len(pending)} patch(es) to evaluate", flush=True)
        command = harness_command(
            python=str(args.swebench_python), dataset=str(dataset_path), split=SPLIT,
            predictions_path=predictions_path, run_id=run_id, instance_ids=pending,
            max_workers=args.max_workers, timeout=args.timeout, report_dir="reports",
        )
        log_path = eval_cwd / "harness_logs" / f"{run_id}.attempt{attempt}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log_file:
            completed = subprocess.run(command, cwd=eval_cwd, stdout=log_file, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            print(f"[eval] {run_id}: harness exited with code {completed.returncode} (see {log_path})", flush=True)
        verdicts = read_verdicts()

    return verdicts


def image_name(instance_id: str) -> str:
    """Prebuilt evaluation image of an instance (naming of swebench 4.1.0 with --namespace swebench)."""
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest".lower()


def local_images() -> set:
    output = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True).stdout
    return set(output.split())


def ensure_images(instance_ids: List[str], attempts: int = 4) -> List[str]:
    """Pull the images that are not present yet. Returns the instances whose image is still missing."""
    present = local_images()
    missing = []
    for instance_id in instance_ids:
        image = image_name(instance_id)
        if image in present:
            continue
        for attempt in range(1, attempts + 1):
            result = subprocess.run(["docker", "pull", "--quiet", image], capture_output=True, text=True)
            if result.returncode == 0:
                break
            message = ((result.stderr or result.stdout).strip().splitlines() or ["unknown error"])[-1]
            print(f"[eval] pull attempt {attempt}/{attempts} failed for {image}: {message}", flush=True)
            time.sleep(min(300, 30 * attempt))
        else:
            missing.append(instance_id)
    return missing


def remove_images(instance_ids: List[str]) -> None:
    present = local_images()
    images = [image_name(i) for i in instance_ids if image_name(i) in present]
    if images:
        subprocess.run(["docker", "rmi", *images], capture_output=True, text=True)


def write_verdicts(eval_cwd: Path, run_id: str, meta: dict, verdicts: Dict[str, Verdict]) -> Counter:
    verdicts = settle_timeouts(verdicts)
    counts = Counter(v.status for v in verdicts.values())
    path = eval_cwd / "verdicts" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        **meta,
        "run_id": run_id,
        "counts": dict(counts),
        "verdicts": {instance_id: v.to_dict() for instance_id, v in sorted(verdicts.items())},
    }, indent=2) + "\n")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", required=True, choices=["smoke", "full"])
    parser.add_argument("--provider", nargs="*", default=PROVIDERS, choices=PROVIDERS)
    parser.add_argument("--mode", nargs="*", default=None, choices=["baseline", "forced", "self_reflect"])
    parser.add_argument("--run-idx", nargs="*", type=int, default=None)
    parser.add_argument("--gold", action="store_true", help="Evaluate the reference patches instead of model patches")
    parser.add_argument("--with-gold", action="store_true", help="Evaluate the reference patches as well as the model patches")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Work through the instances in batches of this size, pulling each batch's images first (0: no batching)")
    parser.add_argument("--remove-images", action="store_true", help="With --batch-size: remove a batch's images when it is done")
    parser.add_argument("--limit", type=int, default=None,
                        help="Jobs run with --limit N: the first N instances of the set (job suffix <subset><N>)")
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--parallel-jobs", type=int, default=1,
                        help="Harness runs started side by side (each with --max-workers containers); "
                             "every job has its own run id, containers and report files")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds allowed for one instance's tests")
    parser.add_argument("--swebench-python", type=Path, default=PROJECT_ROOT / ".venv-eval" / "bin" / "python")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selection = json.loads(args.instances.read_text())
    subset_ids = selection["smoke_instance_ids"] if args.subset == "smoke" else selection["instance_ids"]
    if args.limit:
        subset_ids = subset_ids[:args.limit]
        args.subset = f"{args.subset}{args.limit}"

    run_dir = (args.runs_dir / args.tag).resolve()
    eval_cwd = run_dir / "eval"
    eval_cwd.mkdir(parents=True, exist_ok=True)
    dataset_path = eval_cwd / f"dataset_{args.subset}.json"
    export_dataset(selection, subset_ids, dataset_path)

    def build_targets() -> list:
        """What to evaluate: (run id, model name, predictions path, instance ids, meta).

        Read afresh from the result databases each time, so patches written by agent jobs
        that are still running are picked up by the batches that follow.
        """
        targets = []
        if args.gold or args.with_gold:
            targets.append((f"gold__{args.subset}", "gold", "gold", list(subset_ids),
                            {"job": "gold", "kind": "gold", "subset": args.subset}))
        if not args.gold:
            for job in load_jobs(run_dir, args.provider, args.mode, args.run_idx, args.subset):
                kinds = ["final", "initial"] if job["mode"] == "self_reflect" else ["final"]
                for kind in kinds:
                    run_id = f"{job['name']}__{kind}"
                    predictions = build_predictions(job["rows"], model_name=run_id, kind=kind)
                    if not predictions:
                        continue
                    predictions_path = eval_cwd / "predictions" / f"{run_id}.jsonl"
                    predictions_path.parent.mkdir(parents=True, exist_ok=True)
                    predictions_path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
                    targets.append((run_id, run_id, str(predictions_path), [p["instance_id"] for p in predictions],
                                    {"job": job["name"], "kind": kind, "subset": args.subset}))
        return targets

    if not build_targets():
        print("[eval] nothing to evaluate")
        return 1

    def drop_stale(run_id: str, model_name: str, predictions_path: str) -> None:
        """Remove harness output that belongs to an earlier version of a patch (a row can
        change when stored answers are re-read after a parser fix), so it is evaluated again."""
        if predictions_path == "gold":
            return
        for line in Path(predictions_path).read_text().splitlines():
            prediction = json.loads(line)
            directory = instance_log_dir(eval_cwd, run_id, model_name, prediction["instance_id"])
            evaluated = directory / "patch.diff"
            if evaluated.exists() and evaluated.read_text() != prediction["model_patch"]:
                shutil.rmtree(directory, ignore_errors=True)

    def pending(run_id: str, model_name: str, instance_ids: List[str]) -> List[str]:
        return [i for i in instance_ids
                if classify_instance_dir(instance_log_dir(eval_cwd, run_id, model_name, i), i).status not in TERMINAL_VERDICTS]

    batches = ([subset_ids[i:i + args.batch_size] for i in range(0, len(subset_ids), args.batch_size)]
               if args.batch_size else [list(subset_ids)])
    for number, batch in enumerate(batches, 1):
        in_batch = set(batch)
        current = build_targets()
        for run_id, model_name, path, _, _ in current:
            drop_stale(run_id, model_name, path)
        work = [(run_id, model_name, path, [i for i in pending(run_id, model_name, ids) if i in in_batch])
                for run_id, model_name, path, ids, _ in current]
        work = [w for w in work if w[3]]
        if not work:
            continue
        needed = sorted({i for w in work for i in w[3]})
        if args.batch_size:
            print(f"[eval] batch {number}/{len(batches)}: {len(needed)} instance(s), {sum(len(w[3]) for w in work)} patch(es)", flush=True)
            missing = ensure_images(needed)
            if missing:
                print(f"[eval] images still missing after pulls: {missing}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.parallel_jobs)) as pool:
            futures = [pool.submit(evaluate_predictions, eval_cwd=eval_cwd, run_id=run_id, model_name=model_name,
                                   predictions_path=path, instance_ids=ids, dataset_path=dataset_path, args=args)
                       for run_id, model_name, path, ids in work]
            for future in futures:
                future.result()
        if args.batch_size and args.remove_images:
            remove_images(needed)

    targets = build_targets()
    for run_id, model_name, path, _, _ in targets:
        drop_stale(run_id, model_name, path)
    unresolved_errors = 0
    gold_failures: List[str] = []
    for run_id, model_name, _, ids, meta in targets:
        verdicts = {i: classify_instance_dir(instance_log_dir(eval_cwd, run_id, model_name, i), i) for i in ids}
        counts = write_verdicts(eval_cwd, run_id, meta, verdicts)
        unresolved_errors += counts.get(EVAL_ERROR, 0)
        if model_name == "gold":
            print(f"[eval] gold patches on the {args.subset} set: {dict(counts)}", flush=True)
            gold_failures = sorted(i for i, v in settle_timeouts(verdicts).items() if v.status != "resolved")
        else:
            print(f"[eval] {run_id}: {len(ids)} patch(es) -> {dict(counts)}", flush=True)

    if gold_failures:
        print(f"[eval] gold patches NOT resolved: {gold_failures}")
        return 1
    if unresolved_errors:
        print(f"[eval] {unresolved_errors} patch(es) still without a verdict. Re-run the same command.")
        return 2
    print("[eval] every patch has a verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
