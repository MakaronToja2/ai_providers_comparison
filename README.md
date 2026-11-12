# Prompting regimes for low-cost LLM bug-fixing agents on SWE-bench Lite

Replication repository for

> J. Wloch. *A Controlled Comparison of Prompting Regimes for Low-Cost LLM Bug-Fixing Agents on SWE-bench Lite.*
> MDIS 2026, Communications in Computer and Information Science, Springer (to appear).

The study runs one minimal, read-only bug-fixing agent (three tools: `read_file`, `search_code`,
`list_directory`; at most ten tool iterations; no code execution) with three low-cost models,
Claude Haiku 4.5, GPT-4.1-mini and Gemini 2.5 Flash-Lite, under three prompting regimes:

- **baseline**: the model may abstain with an explicit marker;
- **forced**: the abstention option is removed;
- **self_reflect**: one extra call in which the model reviews its own patch, without any test feedback.

Every configuration was run three times on the same 100 SWE-bench Lite instances (a seeded sample,
stratified by repository), and every patch was evaluated with the official SWE-bench harness
(package version 4.1.0, prebuilt Docker images). This repository contains the harness, the instance
sample, the prompts, the per-run trajectories and results, the evaluation verdicts and the analysis
scripts that produce every number in the paper.

## Layout

| Path | Contents |
|---|---|
| `app/` | The harness: provider clients, agent loop, tools, answer format and edit application, runner. The FastAPI app and dashboard of the earlier version are still here but are not used by the scripts below. |
| `scripts/` | Command-line entry points: instance selection, API preflight, running jobs, pulling evaluation images, evaluation, re-reading stored answers, progress. |
| `analysis/` | Scripts that turn the stored runs into the paper's tables and statistics; `analysis/out/` holds their output (tidy table, report, CSV tables). |
| `data/instances_seed20261007.json` | The sample: dataset revision, seed, method, per-repository allocation and the instance ids. |
| `runs/rerun-2026-10/` | All runs reported in the paper (see below). |
| `runs/stage*/` | Pilot runs and jobs that were discarded during the study; kept for transparency, not used in the paper's results. |
| `tests/` | Unit and integration tests of the harness (`pytest`). |

### `runs/rerun-2026-10/`

- `results_{anthropic,openai,google}.db`: SQLite databases with one row per run (model, regime, repetition, instance): outcome, answer, patches and token counts (cost is computed from the token counts by the analysis).
- `trajectories/<provider>/<regime>/rep<k>/<subset>/<instance>.jsonl`: the full conversation of every run, including tool calls and results.
- `jobs/`: metadata of every job; `queue_logs/`: console logs of the job queues; `pip-freeze.txt`: exact package versions.
- `eval/`: the SWE-bench evaluation: `predictions/` (the patches submitted), `verdicts/` (one verdict per patch, including the reference patches in `gold__*.json`), `harness_logs/`, `dataset_full.json`, and `logs/` with the harness's per-instance `report.json`, `patch.diff`, `eval.sh` and `run_instance.log`.

The raw test output of each evaluation (`test_output.txt`, about 2 GB in total) is not included;
`scripts/evaluate.py` regenerates it.

`smoke` (also `smoke5`) is a small instance set, disjoint from the sample, that was used to check the
pipeline; `full` is the 100-instance sample.

## Setup

Two Python environments are used: one for the agent, analysis and tests, and one for the
official evaluation harness, whose version is pinned.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-rerun.txt
python3 -m venv .venv-eval && .venv-eval/bin/pip install -r requirements-eval.txt
cp .env.example .env   # then fill in OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
```

Evaluation needs Docker. `requirements.txt` is the environment of the earlier version of the code
and is not needed for the reruns.

## Reproducing the analysis from the stored runs

No API key or Docker is needed for this:

```bash
.venv/bin/python analysis/build_table.py     # tidy table from the databases and verdicts
.venv/bin/python analysis/report.py          # statistics (bootstrap intervals, paired tests)
.venv/bin/python analysis/tolerances.py      # how often each answer-format tolerance was needed
.venv/bin/python analysis/answer_kinds.py    # what answers without an edit look like
.venv/bin/python analysis/leaderboard_compare.py   # published Lite systems on the same instances (downloads the public leaderboard data)
```

Outputs go to `analysis/out/`. `analysis/make_paper_inputs.py` and `analysis/check_claims.py`
write and test the manuscript's tables and statements; the manuscript sources are not part of this
repository.

## Running the study again

```bash
.venv/bin/python scripts/select_instances.py --seed 20261007 --out data/instances_seed20261007.json
.venv/bin/python scripts/preflight.py                      # one tiny call per model
.venv/bin/python scripts/run_experiment.py --provider anthropic --mode baseline --subset full --run-idx 1
QUEUE_BUDGET_USD=96 scripts/run_queue.sh anthropic full "1 2 3"   # all regimes and repetitions of one model
.venv-eval/bin/python scripts/pull_images.py --subset full
.venv/bin/python scripts/evaluate.py --subset full --with-gold
```

`--mode` is one of `baseline`, `forced`, `self_reflect`; `--provider` one of `anthropic`, `openai`,
`google`. Jobs are resume-safe, and the evaluation is idempotent. Model outputs are not deterministic,
so a new run will not reproduce the stored trajectories exactly. Without `--revision`,
`select_instances.py` uses the current dataset head; the revision used in the study is recorded in
`data/instances_seed20261007.json`.

## Earlier version

The tag `thesis-2026-01` marks the code and results of the January 2026 version of this study. An
audit found defects in that harness that invalidated its results (among them, tool calls replayed to
the models as plain text, and a reflection step that reported failing tests that had never been run);
the harness was corrected and all experiments were run again. No result of the January runs is used
in the paper.

## License

MIT, see `LICENSE`. The trajectories and evaluation logs contain excerpts of the open-source
repositories in SWE-bench, which remain under their own licenses.
