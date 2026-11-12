#!/usr/bin/env python3
"""
Score published SWE-bench Lite systems on the study's 100 instances.

The official leaderboard repository (github.com/swe-bench/experiments) stores, for every
Lite submission, the list of instance ids it resolved. From those lists this script
computes each system's resolve rate on all 300 Lite instances and on our 100-instance
sample. That gives (a) a like-for-like reference for our numbers and (b) a check of how
representative the sample is: if the sample were easier or harder than Lite as a whole,
published systems would score systematically higher or lower on it.

No model is called. The submissions' own results are taken as published.

Usage:
    .venv/bin/python analysis/leaderboard_compare.py
Output: analysis/out/leaderboard_lite_on_sample.csv (+ a short summary on stdout)
"""
import json
import re
import statistics
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTANCES = PROJECT_ROOT / "data" / "instances_seed20261007.json"
API = "https://api.github.com/repos/swe-bench/experiments/contents/evaluation/lite?per_page=100"
RAW = "https://raw.githubusercontent.com/swe-bench/experiments/main/evaluation/lite/{name}/{path}"
LITE_SIZE = 300


def fetch(url: str) -> str | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "leaderboard-compare"}), timeout=30) as response:
            return response.read().decode("utf-8", "replace")
    except Exception:
        return None


def display_name(metadata: str | None, fallback: str) -> str:
    match = re.search(r"^\s*name:\s*(.+)$", metadata or "", re.MULTILINE)
    return match.group(1).strip().strip("'\"") if match else fallback


def main() -> int:
    sample = set(json.loads(INSTANCES.read_text())["instance_ids"])
    listing = fetch(API)
    if listing is None:
        print("could not list the leaderboard repository")
        return 1
    names = sorted(entry["name"] for entry in json.loads(listing) if entry["type"] == "dir")

    rows = []
    for name in names:
        results = fetch(RAW.format(name=name, path="results/results.json"))
        if results is None:
            continue
        resolved = set(json.loads(results).get("resolved", []))
        on_sample = len(resolved & sample)
        rows.append({
            "submission": name,
            "date": f"{name[:4]}-{name[4:6]}-{name[6:8]}",
            "system": display_name(fetch(RAW.format(name=name, path="metadata.yaml")), name),
            "resolved_lite_300": len(resolved),
            "pct_lite_300": round(100 * len(resolved) / LITE_SIZE, 2),
            "resolved_sample_100": on_sample,
            "pct_sample_100": round(100 * on_sample / len(sample), 2),
        })

    import pandas as pd
    table = pd.DataFrame(rows)
    out = PROJECT_ROOT / "analysis" / "out" / "leaderboard_lite_on_sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    differences = (table["pct_sample_100"] - table["pct_lite_300"]).tolist()
    print(f"{len(table)} Lite submissions with published per-instance results (retrieved from swe-bench/experiments)")
    print(f"sample minus full Lite, percentage points: mean {statistics.mean(differences):+.2f}, median {statistics.median(differences):+.2f}, "
          f"min {min(differences):+.2f}, max {max(differences):+.2f}, SD {statistics.stdev(differences):.2f}")
    print(f"correlation between the two rates across submissions: {table['pct_sample_100'].corr(table['pct_lite_300']):.3f}")
    print(f"written: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
