#!/usr/bin/env python3
"""
How often each tolerance of the answer parser was needed.

The harness accepts four slips of form (see app/core/edit_format.py and
app/benchmark/edit_apply.py). Two of them are logged with each run (a path that had to
be resolved through the files the agent read; a revision applied on top of the first
patch). The other two (marker lines without their keyword, a missing separator line) are
measured here by parsing every stored answer twice: with the parser as used, and with
that tolerance switched off. An answer "needed" a tolerance if switching it off loses or
changes at least one edit block.

No model is called and no stored result is changed.

Usage:
    .venv/bin/python analysis/tolerances.py [--tag rerun-2026-10] [--subset full]
Output: analysis/out/<tag>__<subset>__tolerances.csv
"""
import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core import edit_format  # noqa: E402

PROVIDERS = ["anthropic", "openai", "google"]
MODEL_LABEL = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gpt-4.1-mini": "GPT-4.1-mini",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
}
STRICT_SEARCH = re.compile(r"^<{7} SEARCH\s*$")
STRICT_REPLACE = re.compile(r"^>{7} REPLACE\s*$")


@contextmanager
def parser_with(**replacements):
    """Temporarily replace module-level parts of the parser (in this process only)."""
    saved = {name: getattr(edit_format, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(edit_format, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(edit_format, name, value)


def blocks(text: str) -> list:
    return [(b.path, b.search, b.replace) for b in edit_format.parse_edit_blocks(text)]


def needs(text: str) -> dict:
    """Which tolerances this answer depends on."""
    normal = blocks(text)
    if not normal:
        return {"blocks": 0, "markers": False, "divider": False}
    with parser_with(_SEARCH_LINE=STRICT_SEARCH, _REPLACE_LINE=STRICT_REPLACE):
        strict_markers = blocks(text)
    with parser_with(_split_without_divider=lambda *args, **kwargs: None):
        strict_divider = blocks(text)
    return {"blocks": len(normal), "markers": strict_markers != normal, "divider": strict_divider != normal}


def read_answers(path: Path, stored_answer: str | None = None) -> tuple[str, str]:
    """(final answer, reflection response) from a trajectory log, as scripts/reclassify.py reads them.

    Three logs lost their last lines when the machine went down on 10 October (the runs
    themselves had finished and are complete in the database). A log without its closing
    record is not trusted: the answer stored in the database is used instead.
    """
    records = []
    for line in open(path):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break
    if not records or records[-1].get("type") != "summary":
        if stored_answer is None:
            raise ValueError(f"incomplete trajectory and no stored answer: {path}")
        return stored_answer, ""
    calls = [r for r in records if r["type"] == "api_call" and not r.get("error")]
    answer_calls = [c for c in calls if c["phase"] != "reflection"]
    fallback = [c for c in answer_calls if c["phase"] == "fallback" and (c.get("assistant_text") or "")]
    loop_final = [c for c in answer_calls if c["phase"] == "loop" and not c.get("tool_calls")]
    final = fallback[-1]["assistant_text"] if fallback else (loop_final[-1].get("assistant_text") or "") if loop_final else ""
    reflection = [c for c in calls if c["phase"] == "reflection"]
    return final, (reflection[-1].get("assistant_text") or "") if reflection else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="rerun-2026-10")
    parser.add_argument("--subset", default="full")
    args = parser.parse_args()
    run_dir = PROJECT_ROOT / "runs" / args.tag

    rows, variants, missing = [], Counter(), 0
    for provider in PROVIDERS:
        db_path = run_dir / f"results_{provider}.db"
        if not db_path.exists():
            continue
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        for experiment in connection.execute("SELECT id, name, config FROM experiments"):
            if not experiment["name"].endswith(f"__{args.subset}"):
                continue
            config = json.loads(experiment["config"])
            for row in connection.execute(
                    "SELECT instance_id, model, result_status, generated_patch, run_meta, response_content FROM experiment_results WHERE experiment_id = ?",
                    (experiment["id"],)):
                if row["result_status"] in ("api_error", "infra_error", None):
                    continue
                meta = json.loads(row["run_meta"] or "{}")
                trajectory = meta.get("trajectory_path")
                path = (PROJECT_ROOT / trajectory) if trajectory and not Path(trajectory).is_absolute() else Path(trajectory or "")
                if not trajectory or not path.exists():
                    missing += 1
                    continue
                final, reflection = read_answers(path, stored_answer=row["response_content"] or "")
                first, second = needs(final), needs(reflection)
                for text in (final, reflection):
                    for line in text.split("\n"):
                        if (edit_format._SEARCH_LINE.match(line) and not STRICT_SEARCH.match(line)) or \
                                (edit_format._REPLACE_LINE.match(line) and not STRICT_REPLACE.match(line)):
                            variants[(MODEL_LABEL.get(row["model"], row["model"]), line.strip())] += 1
                rows.append({
                    "model": MODEL_LABEL.get(row["model"], row["model"]),
                    "mode": config["experiment_mode"], "rep": config.get("run_idx"), "instance_id": row["instance_id"],
                    "has_patch": bool(row["generated_patch"]),
                    "answer_has_edits": first["blocks"] > 0,
                    "markers": first["markers"] or second["markers"],
                    "divider": first["divider"] or second["divider"],
                    "path_resolved": bool(meta.get("edit_paths_resolved") or meta.get("revised_edit_paths_resolved")),
                    "revision_on_first_patch": meta.get("revision_basis") == "on_first_patch",
                })
        connection.close()

    table = pd.DataFrame(rows)
    if table.empty:
        print("no rows")
        return 1
    table["any"] = table[["markers", "divider", "path_resolved", "revision_on_first_patch"]].any(axis=1)
    summary = []
    for model, frame in table.groupby("model", sort=False):
        with_patch = frame[frame["has_patch"]]
        summary.append({
            "model": model, "runs": len(frame), "answers_with_edits": int(frame["answer_has_edits"].sum()), "patches": len(with_patch),
            **{f"{key}_runs": int(frame[key].sum()) for key in ("markers", "divider", "path_resolved", "revision_on_first_patch", "any")},
            **{f"{key}_patches": int(with_patch[key].sum()) for key in ("markers", "divider", "path_resolved", "revision_on_first_patch", "any")},
        })
    summary = pd.DataFrame(summary)
    out = PROJECT_ROOT / "analysis" / "out" / f"{args.tag}__{args.subset}__tolerances.csv"
    summary.to_csv(out, index=False)
    print(summary.T.to_string())
    print("\nMarker lines other than '<<<<<<< SEARCH' / '>>>>>>> REPLACE' (model, line, count):")
    for (model, line), count in sorted(variants.items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f"  {model:24s} {line!r:28s} {count}")
    if missing:
        print(f"\nWARNING: {missing} row(s) without a readable trajectory were skipped")
    print(f"\n-> {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
