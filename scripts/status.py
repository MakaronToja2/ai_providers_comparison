#!/usr/bin/env python3
"""
Progress of the study's runs: agent jobs, Docker evaluation and spend, read from the
result databases, the harness logs and the process table. It changes nothing.

Usage:
    .venv/bin/python scripts/status.py              # one snapshot
    .venv/bin/python scripts/status.py --watch 5    # refresh every 5 seconds (Ctrl-C to leave)
    .venv/bin/python scripts/status.py --all        # include the small smoke jobs
"""
import argparse
import json
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmark.pricing import billed_cost_usd, cost_usd  # noqa: E402
from app.benchmark.swebench_eval import classify_instance_dir, instance_log_dir  # noqa: E402

PROVIDERS = ["openai", "google", "anthropic"]
MODES = ["baseline", "forced", "self_reflect"]
MODE_SHORT = {"baseline": "baseline", "forced": "forced", "self_reflect": "reflect"}
TERMINAL = {"success_patch_generated", "failure_explicit", "hallucination_format_error", "edit_not_applicable", "context_overflow"}
BAR_WIDTH = 24


def bar(done: int, total: int) -> str:
    filled = round(BAR_WIDTH * done / total) if total else 0
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def running_processes() -> dict:
    """Agent jobs and evaluations currently running, from the process table."""
    output = subprocess.run(["ps", "-eo", "etimes,args"], capture_output=True, text=True).stdout
    jobs, evaluating, queues = {}, [], []
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        elapsed, args = parts
        if "scripts/run_experiment.py" in args and "ps -eo" not in args:
            get = lambda flag: (re.search(rf"--{flag} (\S+)", args) or [None, None])[1]
            subset = (get("subset") or "") + (get("limit") or "")
            jobs[(get("provider"), get("mode"), get("run-idx"), subset)] = int(elapsed)
        elif "swebench.harness.run_evaluation" in args:
            match = re.search(r"--run_id (\S+)", args)
            if match:
                evaluating.append(match.group(1))
        elif "scripts/run_queue.sh" in args and "grep" not in args:
            match = re.search(r"run_queue\.sh (\S+) (\S+) (.+?)(?: --|$)", args)
            if match:
                queues.append(f"{match.group(1)} {match.group(2)} reps {match.group(3)}")
    return {"jobs": jobs, "evaluating": evaluating, "queues": sorted(set(queues))}


def load_jobs(run_dir: Path) -> list:
    jobs = []
    for provider in PROVIDERS:
        db_path = run_dir / f"results_{provider}.db"
        if not db_path.exists():
            continue
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            experiments = connection.execute("SELECT id, name, config FROM experiments").fetchall()
            for experiment in experiments:
                config = json.loads(experiment["config"])
                rows = connection.execute(
                    "SELECT instance_id, model, result_status, generated_patch, original_patch, prompt_tokens, completion_tokens, "
                    "reflection_prompt_tokens, reflection_completion_tokens, run_meta, completed_at "
                    "FROM experiment_results WHERE experiment_id = ?", (experiment["id"],)).fetchall()
                jobs.append({"name": experiment["name"], "provider": provider, "config": config, "rows": [dict(r) for r in rows]})
        finally:
            connection.close()
    return jobs


def job_line(job: dict, processes: dict, eval_cwd: Path) -> tuple:
    config, rows = job["config"], job["rows"]
    name = job["name"]
    subset = name.rsplit("__", 1)[-1]
    total = len(config.get("instance_ids") or [])
    model = config["models"][job["provider"]]
    done = [r for r in rows if r["result_status"] in TERMINAL]
    errors = len(rows) - len(done)
    patches = sum(1 for r in done if r["generated_patch"])
    abstained = sum(1 for r in done if r["result_status"] == "failure_explicit")

    cost_list = cost_billed = 0.0
    for r in rows:
        prompt = (r["prompt_tokens"] or 0) + (r["reflection_prompt_tokens"] or 0)
        completion = (r["completion_tokens"] or 0) + (r["reflection_completion_tokens"] or 0)
        cached = json.loads(r["run_meta"] or "{}").get("cached_input_tokens", 0) or 0
        cost_list += cost_usd(model, prompt, completion)
        cost_billed += billed_cost_usd(model, prompt, completion, cached)

    key = (job["provider"], config["experiment_mode"], str(config.get("run_idx")), subset)
    elapsed = processes["jobs"].get(key)
    if len(done) >= total and not errors:
        state = "done"
    elif elapsed is not None:
        times = sorted(datetime.fromisoformat(r["completed_at"]) for r in done if r["completed_at"])[-12:]
        state = "running"
        if len(times) >= 4 and (times[-1] - times[0]).total_seconds() > 0:
            rate = (len(times) - 1) / (times[-1] - times[0]).total_seconds()
            state = f"running, ~{max(0, (total - len(done)) / rate / 60):.0f} min left"
    else:
        state = "stopped (incomplete)" if rows else "not started"

    # Docker evaluation of this job's final patches
    run_id = f"{name}__final"
    verdicts = {"resolved": 0, "tests_failed": 0, "apply_failed": 0, "pending": 0}
    for r in done:
        if r["generated_patch"]:
            status = classify_instance_dir(instance_log_dir(eval_cwd, run_id, run_id, r["instance_id"]), r["instance_id"]).status
            verdicts[status if status in verdicts else "pending"] += 1
    evaluated = patches - verdicts["pending"]
    if not patches:
        evaluation = "no patches yet"
    elif evaluated == 0:
        evaluation = "evaluating" if run_id in processes["evaluating"] else "not evaluated"
    else:
        evaluation = f"{evaluated}/{patches} evaluated: {verdicts['resolved']} resolved, {verdicts['tests_failed']} tests failed" + \
            (f", {verdicts['apply_failed']} apply failed" if verdicts["apply_failed"] else "") + \
            (" (running)" if run_id in processes["evaluating"] else "")

    line = (f"  {MODE_SHORT[config['experiment_mode']]:9s} rep {config.get('run_idx')}  {bar(len(done), total)} {len(done):3d}/{total:<3d} "
            f"patch {patches:3d} | no patch {len(done) - patches:3d} (of which abstained {abstained})" + (f"  ERRORS {errors}" if errors else "") +
            f"  ${cost_list:6.2f} list / ${cost_billed:6.2f} billed  [{state}]\n"
            f"{'':16s}evaluation: {evaluation}")
    return line, cost_list, cost_billed, len(done), total


def snapshot(args: argparse.Namespace) -> str:
    run_dir = args.runs_dir / args.tag
    processes = running_processes()
    jobs = load_jobs(run_dir)
    lines = [f"Run {args.tag}   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   (list price / as billed with cache discount)"]

    for provider in PROVIDERS:
        mine = [j for j in jobs if j["provider"] == provider]
        spend_list = spend_billed = 0.0
        body = []
        # Every job counts towards the spend; only the 100-instance jobs are listed unless --all
        for job in sorted(mine, key=lambda j: (j["config"].get("run_idx") or 0, MODES.index(j["config"]["experiment_mode"]), j["name"])):
            line, cost_list, cost_billed, _, _ = job_line(job, processes, run_dir / "eval")
            spend_list += cost_list
            spend_billed += cost_billed
            full = job["name"].endswith("__full")
            if full or args.all:
                body.append(line + ("" if full else f"   ({job['name'].rsplit('__', 1)[-1]})"))
        model = mine[0]["config"]["models"][provider] if mine else "-"
        lines.append(f"\n{provider.upper()}  {model}   spent in this run: ${spend_list:.2f} list / ${spend_billed:.2f} billed")
        lines.extend(body or ["  no 100-instance jobs yet" + ("" if args.all else " (use --all to see smoke jobs)")])

    gold = [f for f in (run_dir / "eval" / "verdicts").glob("gold__*.json")] if (run_dir / "eval" / "verdicts").exists() else []
    for path in sorted(gold):
        counts = json.loads(path.read_text())["counts"]
        lines.append(f"\nReference patches ({path.stem.split('__')[1]}): {counts}")
    lines.append("\nActive: " + (", ".join(
        [f"queue {q}" for q in processes["queues"]] + [f"evaluating {e}" for e in processes["evaluating"]]) or "nothing running"))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="rerun-2026-10")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--watch", type=float, default=0, metavar="SECONDS", help="Refresh continuously")
    parser.add_argument("--all", action="store_true", help="Include smoke jobs")
    args = parser.parse_args()

    if not args.watch:
        print(snapshot(args))
        return 0
    try:
        while True:
            text = snapshot(args)
            sys.stdout.write("\033[2J\033[H" + text + f"\n\n(refreshing every {args.watch:g}s, Ctrl-C to leave)\n")
            sys.stdout.flush()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
