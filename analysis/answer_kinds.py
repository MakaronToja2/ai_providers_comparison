#!/usr/bin/env python3
"""
What the answers look like that contain neither an edit nor an abstention.

Every run that ended as "no usable answer" is put into exactly one class, from the
stored final answer and the stop reason the API reported for the last call:

  empty            the final answer has no text
  truncated        the last call stopped at the output-token limit
  edit_incomplete  an edit was begun (an opening marker is present) but no edit is complete
  no_edit          text without any edit marker (prose, or code shown without the format)

No model is called and no stored result is changed.

Usage:
    .venv/bin/python analysis/answer_kinds.py [--tag rerun-2026-10] [--subset full]
Output: analysis/out/<tag>__<subset>__answer_kinds.csv
"""
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.tolerances import MODEL_LABEL, PROVIDERS, read_answers  # noqa: E402

KINDS = ["empty", "truncated", "edit_incomplete", "no_edit"]
LIMIT_STOPS = {"length", "max_tokens"}
OPENING = re.compile(r"^<{5,9}", re.M)


def classify(final: str, last_stop: str) -> str:
    if not final.strip():
        return "empty"
    if last_stop.lower() in LIMIT_STOPS:
        return "truncated"
    return "edit_incomplete" if OPENING.search(final) else "no_edit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="rerun-2026-10")
    parser.add_argument("--subset", default="full")
    args = parser.parse_args()
    run_dir = PROJECT_ROOT / "runs" / args.tag

    rows = []
    for provider in PROVIDERS:
        db_path = run_dir / f"results_{provider}.db"
        if not db_path.exists():
            continue
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        query = ("SELECT e.name, r.model, r.instance_id, r.run_meta, r.response_content FROM experiment_results r "
                 "JOIN experiments e ON e.id = r.experiment_id WHERE r.result_status = 'hallucination_format_error'")
        for name, model, instance_id, meta, stored_answer in connection.execute(query):
            if not name.endswith(f"__{args.subset}"):
                continue
            meta = json.loads(meta or "{}")
            trajectory = Path(meta["trajectory_path"])
            final, _ = read_answers(trajectory if trajectory.is_absolute() else PROJECT_ROOT / trajectory, stored_answer=stored_answer or "")
            last_stop = str((meta.get("stop_reasons") or [""])[-1])
            rows.append({"model": MODEL_LABEL.get(model, model), "job": name, "instance_id": instance_id,
                         "kind": classify(final, last_stop), "last_stop": last_stop,
                         "termination": meta.get("termination_reason")})
        connection.close()

    table = pd.DataFrame(rows)
    if table.empty:
        print("no rows")
        return 1
    summary = table.groupby("model")["kind"].value_counts().unstack(fill_value=0).reindex(columns=KINDS, fill_value=0)
    summary["total"] = summary.sum(axis=1)
    summary["unexpected_tool_call"] = table[table["last_stop"] == "UNEXPECTED_TOOL_CALL"].groupby("model").size().reindex(summary.index, fill_value=0)
    out = PROJECT_ROOT / "analysis" / "out" / f"{args.tag}__{args.subset}__answer_kinds.csv"
    summary.reset_index().to_csv(out, index=False)
    print(summary.to_string())
    print("\nStop reason of the last call, by model:")
    print(table.groupby("model")["last_stop"].value_counts().unstack(fill_value=0).to_string())
    print(f"\n-> {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
