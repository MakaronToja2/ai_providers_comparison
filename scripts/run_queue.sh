#!/usr/bin/env bash
# Run one model's jobs one after another: for each repetition, the three regimes.
# Each job is resume-safe; a job that ends with API or harness errors is retried, and the
# queue stops if it still has errors after the retries.
#
# QUEUE_BUDGET_USD (optional): stop before starting a job once this provider's spend in the
# run (list price, all jobs in its results database) has reached that amount.
#
# Usage: scripts/run_queue.sh <provider> <subset> "<repetitions>" [extra run_experiment.py args]
#   scripts/run_queue.sh google smoke "1" --limit 5
#   QUEUE_BUDGET_USD=96 scripts/run_queue.sh anthropic full "1 2 3"
set -u
cd "$(dirname "$0")/.."
provider=$1; subset=$2; reps=$3; shift 3

spend() {
  .venv/bin/python - "$provider" <<'PY'
import json, sqlite3, sys
from pathlib import Path
sys.path.insert(0, ".")
from app.benchmark.pricing import cost_usd
db = Path("runs/rerun-2026-10") / f"results_{sys.argv[1]}.db"
total = 0.0
if db.exists():
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for model, p, o, rp, ro in c.execute("select model, prompt_tokens, completion_tokens, reflection_prompt_tokens, reflection_completion_tokens from experiment_results"):
        total += cost_usd(model, (p or 0) + (rp or 0), (o or 0) + (ro or 0))
print(f"{total:.2f}")
PY
}

for rep in $reps; do
  for mode in baseline forced self_reflect; do
    if [ -n "${QUEUE_BUDGET_USD:-}" ]; then
      spent=$(spend)
      if awk "BEGIN{exit !($spent >= $QUEUE_BUDGET_USD)}"; then
        echo "[queue] stopping before $provider $mode rep$rep: spend \$$spent has reached the budget \$$QUEUE_BUDGET_USD"
        exit 3
      fi
      echo "[queue] $provider spend so far \$$spent (budget \$$QUEUE_BUDGET_USD)"
    fi
    for attempt in 1 2 3; do
      .venv/bin/python scripts/run_experiment.py --provider "$provider" --mode "$mode" --subset "$subset" --run-idx "$rep" "$@" 2>&1 \
        | grep --line-buffered -vE 'Warning|warn|HF_TOKEN|unauth|httpx|^\[setup\]|Database tables'
      status=${PIPESTATUS[0]}
      [ "$status" -eq 0 ] && break
      echo "[queue] $provider $mode rep$rep ended with status $status (attempt $attempt/3)"
      sleep 60
    done
    if [ "$status" -ne 0 ]; then
      echo "[queue] stopping: $provider $mode rep$rep still incomplete after 3 attempts"
      exit 1
    fi
  done
done
echo "[queue] $provider: all jobs complete for repetitions: $reps"
