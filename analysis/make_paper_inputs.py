#!/usr/bin/env python3
"""
Write the manuscript's tables and number macros from the analysis output, so that no
figure in the paper is typed by hand.

Run after analysis/build_table.py and analysis/report.py:
    .venv/bin/python analysis/make_paper_inputs.py [--tag rerun-2026-10] [--subset full]
Output: paper/mdis/generated/{numbers,tab_*}.tex
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

from app.benchmark.swebench_eval import classify_instance_dir, instance_log_dir  # noqa: E402

MODELS = ["Claude Haiku 4.5", "GPT-4.1-mini", "Gemini 2.5 Flash-Lite"]
MODEL_SHORT = {"Claude Haiku 4.5": "Claude Haiku 4.5", "GPT-4.1-mini": "GPT-4.1-mini", "Gemini 2.5 Flash-Lite": "Gemini 2.5 FL"}
MODEL_ABBR = {"Claude Haiku 4.5": "Claude", "GPT-4.1-mini": "GPT", "Gemini 2.5 Flash-Lite": "Gemini"}
MODES = ["baseline", "forced", "self_reflect"]
MODE_LABEL = {"baseline": "Baseline", "forced": "Forced", "self_reflect": "Reflection"}
EXTERNAL = [  # submissions shown in the comparison table (directory names in swe-bench/experiments, Lite)
    "20240402_sweagent_gpt4", "20240621_autocoderover-v20240620",
    "20241028_agentless-1.5_gpt4o", "20241025_OpenHands-CodeAct-2.1-sonnet-20241022",
    "20250526_sweagent_claude-4-sonnet-20250514", "20250625_ExpeRepair-v1_claude-4-sonnet-20250514",
]
EXTERNAL_LABEL = {
    "20240402_rag_claude3opus": "Retrieval only (no agent) + Claude 3 Opus",
    "20240402_sweagent_gpt4": "SWE-agent + GPT-4",
    "20240630_agentless_gpt4o": "Agentless + GPT-4o",
    "20240621_autocoderover-v20240620": "AutoCodeRover + GPT-4o",
    "20241028_agentless-1.5_gpt4o": "Agentless 1.5 + GPT-4o",
    "20241025_OpenHands-CodeAct-2.1-sonnet-20241022": "OpenHands CodeAct + Claude 3.5 Sonnet",
    "20250526_sweagent_claude-4-sonnet-20250514": "SWE-agent + Claude 4 Sonnet",
    "20250625_ExpeRepair-v1_claude-4-sonnet-20250514": "ExpeRepair + Claude 4 Sonnet",
}


EXTERNAL_KEY = {
    "20240402_rag_claude3opus": "rag", "20240402_sweagent_gpt4": "sweagentGptFour", "20240630_agentless_gpt4o": "agentless",
    "20240621_autocoderover-v20240620": "autocoderover", "20241028_agentless-1.5_gpt4o": "agentlessNew",
    "20241025_OpenHands-CodeAct-2.1-sonnet-20241022": "openhands",
    "20250526_sweagent_claude-4-sonnet-20250514": "sweagentClaudeFour", "20250625_ExpeRepair-v1_claude-4-sonnet-20250514": "experepair",
}


def pm(mean: float, sd: float) -> str:
    return f"{mean:.1f}" if pd.isna(sd) else f"{mean:.1f} $\\pm$ {sd:.1f}"


def pct_ci(value: float, lo: float, hi: float) -> str:
    return "--" if pd.isna(value) else f"{100 * value:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def p_value(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["_m"] = frame["model"].map(MODELS.index)
    frame["_r"] = frame["mode"].map(MODES.index)
    return frame.sort_values(["_m", "_r"])


def grouped_rows(frame: pd.DataFrame, cells) -> str:
    """Table body with the model name once per group of regimes."""
    lines, previous = [], None
    for _, row in ordered(frame).iterrows():
        if previous is not None and row["model"] != previous:
            lines.append("\\addlinespace")
        name = MODEL_SHORT[row["model"]] if row["model"] != previous else ""
        lines.append(" & ".join([name, MODE_LABEL[row["mode"]], *cells(row)]) + " \\\\")
        previous = row["model"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="rerun-2026-10")
    parser.add_argument("--subset", default="full")
    args = parser.parse_args()

    out = PROJECT_ROOT / "analysis" / "out"
    prefix = f"{args.tag}__{args.subset}"
    gen = PROJECT_ROOT / "paper" / "mdis" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(out / f"{prefix}__{name}.csv")
    tidy, main_t, outcomes, paired, cost, reflection, protocol = (
        read(n) for n in ("tidy", "main", "outcomes", "paired", "cost", "reflection", "protocol"))
    n_instances = tidy["instance_id"].nunique()

    # ---- Table: patches, resolved, success, precision
    body = grouped_rows(main_t, lambda r: [
        pm(r["patches_mean"], r["patches_sd"]), pm(r["resolved_mean"], r["resolved_sd"]),
        pct_ci(r["success"], r["success_lo"], r["success_hi"]), pct_ci(r["precision"], r["precision_lo"], r["precision_hi"])])
    (gen / "tab_main.tex").write_text(
        "\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n & & & & Success \\% & Precision \\% \\\\\n"
        "Model & Regime & Patches & Resolved & [95\\% CI] & [95\\% CI] \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Table: outcomes per repetition
    # A patch that the evaluation harness could not apply is shown with the patches that failed
    # tests (the caption says so); the count is cited from the facts
    outcomes = outcomes.copy()
    if "apply_failed" in outcomes.columns:
        outcomes["tests_failed"] = outcomes["tests_failed"] + outcomes["apply_failed"]
        outcomes = outcomes.drop(columns="apply_failed")
    columns = [("abstained", ("", "Abstained")), ("no_answer", ("No usable", "answer")), ("edit_not_applicable", ("Edit not", "applicable")),
               ("tests_failed", ("Tests", "failed")), ("resolved", ("", "Resolved"))]
    columns = [(key, label) for key, label in columns if key in outcomes.columns and outcomes[key].sum() > 0]
    body = grouped_rows(outcomes, lambda r: [f"{r[key]:.1f}" for key, _ in columns])
    (gen / "tab_outcomes.tex").write_text(
        "\\begin{tabular}{@{}ll" + "r" * len(columns) + "@{}}\n\\toprule\n & & " + " & ".join(label[0] for _, label in columns)
        + " \\\\\nModel & Regime & " + " & ".join(label[1] for _, label in columns)
        + " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Table: paired comparisons, regimes within a model and models within a regime
    def paired_rows(family: str, group_label, comparison_label) -> list:
        lines, previous = [], None
        for _, r in paired[paired["family"] == family].iterrows():
            name = group_label(r["within"]) if r["within"] != previous else ""
            lines.append(f"{name} & {comparison_label(r['comparison'])} & {100 * r['diff']:+.1f} [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] & "
                         f"{int(r['only_first'])} / {int(r['only_second'])} & {p_value(r['p'])} & {p_value(r['p_holm'])} \\\\")
            previous = r["within"]
        return [re.sub(r"(?<![\w$])-(?=\d)", "$-$", line) for line in lines]   # typeset minus signs of numbers

    def abbreviate(comparison: str) -> str:
        for full, short in MODEL_ABBR.items():
            comparison = comparison.replace(full, short)
        return comparison.replace("Self-reflection", "Reflection").replace(" vs ", " -- ")

    header = ("\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n & First -- second & $\\Delta$ success [95\\% CI] & Discordant & $p$ & $p_{\\text{Holm}}$ \\\\\n\\midrule\n")
    (gen / "tab_paired.tex").write_text(
        header + "\\multicolumn{6}{l}{\\emph{Regimes, within each model}} \\\\\n"
        + "\n".join(paired_rows("regimes", lambda m: MODEL_SHORT[m], abbreviate))
        + "\n\\midrule\n\\multicolumn{6}{l}{\\emph{Models, within each regime}} \\\\\n"
        + "\n".join(paired_rows("models", lambda m: m.replace("Self-reflection", "Reflection"), abbreviate))
        + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Table: reflection, first versus final patch
    lines = []
    for model in MODELS:
        r = reflection[reflection["model"] == model]
        if r.empty:
            continue
        r = r.iloc[0]
        lines.append(f"{MODEL_SHORT[model]} & {int(r['calls'])} & {int(r['approved'])} & {int(r['revised'])} & {int(r['no_verdict'])} & "
                     f"{int(r['patches_before'])} / {int(r['patches_after'])} & {int(r['resolved_before'])} / {int(r['resolved_after'])} & "
                     f"{int(r['gained'])} / {int(r['lost'])} & {p_value(r['p'])} \\\\")
    (gen / "tab_reflection.tex").write_text(
        "\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n & & Ap- & Re- & No & Patches & Resolved & Gained & \\\\\n"
        "Model & Calls & proved & vised & verdict & first / final & first / final & / lost & $p$ \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Table: tokens and cost
    money = lambda list_price, billed, digits: "--" if pd.isna(list_price) else f"{list_price:.{digits}f} ({billed:.{digits}f})"
    body = grouped_rows(cost, lambda r: [
        f"{r['input_median'] / 1000:.1f} [{r['input_q1'] / 1000:.1f}--{r['input_q3'] / 1000:.1f}]", f"{r['input_sd'] / 1000:.1f}",
        f"{r['output_median'] / 1000:.1f}", f"{100 * r['cached_share']:.0f}",
        money(r["cost_list_mean"], r["cost_billed_mean"], 3), money(r["cost_list_per_resolved"], r["cost_billed_per_resolved"], 2)])
    (gen / "tab_cost.tex").write_text(
        "\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n & & \\multicolumn{3}{c}{Tokens per instance (thousands)} & Input & \\$ per & \\$ per \\\\\n"
        "Model & Regime & input, median [IQR] & input, SD & output, median & cached \\% & instance & resolved \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Table: published systems on the same instances
    board = pd.read_csv(out / "leaderboard_lite_on_sample.csv")
    lines = []
    for name in EXTERNAL:
        r = board[board["submission"] == name]
        if r.empty:
            continue
        r = r.iloc[0]
        lines.append(f"{EXTERNAL_LABEL[name]} & {r['date'][:4]} & {r['pct_lite_300']:.1f} & {r['pct_sample_100']:.1f} \\\\")
    lines.append("\\midrule")
    for model in MODELS:
        r = main_t[(main_t["model"] == model) & (main_t["mode"] == "baseline")]
        if not r.empty:
            lines.append(f"This study (baseline): {MODEL_SHORT[model]} & 2026 & -- & {100 * r.iloc[0]['success']:.1f} \\\\")
    (gen / "tab_external.tex").write_text(
        "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nSystem & Year & Lite (300) \\% & Our sample \\% \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Number macros
    differences = board["pct_sample_100"] - board["pct_lite_300"]
    selection = json.loads((PROJECT_ROOT / "data" / "instances_seed20261007.json").read_text())
    eval_dir = PROJECT_ROOT / "runs" / args.tag / "eval"
    gold = [classify_instance_dir(instance_log_dir(eval_dir, f"gold__{args.subset}", "gold", i), i).status for i in selection["instance_ids"]]
    gold_text = f"{gold.count('resolved')} of {len(gold)}" + ("" if all(g in ("resolved", "tests_failed", "apply_failed") for g in gold) else " [EVALUATION INCOMPLETE]")
    days = sorted({str(d)[:10] for d in pd.concat([tidy["started_at"], tidy["completed_at"]]).dropna()})
    fmt_day = lambda d: f"{int(d[8:10])}~October 2026" if d[5:7] == "10" else d
    run_dates = fmt_day(days[0]) if len(days) == 1 else f"{fmt_day(days[0])} and {fmt_day(days[-1])}" if len(days) == 2 else f"{fmt_day(days[0])} to {fmt_day(days[-1])}"
    macros = {
        "nInstances": str(n_instances),
        "nLeaderboard": str(len(board)),
        "sampleOffset": f"{differences.mean():.1f}",
        "sampleOffsetSD": f"{differences.std():.1f}",
        "sampleCorr": f"{board['pct_sample_100'].corr(board['pct_lite_300']):.2f}",
        "goldResolved": gold_text,
        "nExternal": str(int(board["submission"].isin(EXTERNAL).sum())),
        "leaderboardLast": pd.Timestamp(board["date"].max()).strftime("%B %Y"),
        "runDates": run_dates,
    }
    (gen / "numbers.tex").write_text(
        "% Written by analysis/make_paper_inputs.py. Do not edit by hand.\n"
        + "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in macros.items()))

    # ---- Facts: every number the text cites, as \N{key}
    facts = json.loads((out / f"{prefix}__facts.json").read_text())
    tolerances_path = out / f"{prefix}__tolerances.csv"
    if tolerances_path.exists():
        key = {"Claude Haiku 4.5": "c", "GPT-4.1-mini": "g", "Gemini 2.5 Flash-Lite": "f"}
        for _, r in pd.read_csv(tolerances_path).iterrows():
            k = f"t.{key[r['model']]}"
            facts[f"{k}.patches"] = int(r["patches"])
            for name, column in (("markers", "markers"), ("divider", "divider"), ("path", "path_resolved"),
                                 ("stacked", "revision_on_first_patch"), ("any", "any")):
                facts[f"{k}.{name}"] = int(r[f"{column}_patches"])
                facts[f"{k}.{name}Share"] = f"{100 * r[f'{column}_patches'] / r['patches']:.0f}" if r["patches"] else "--"
    kinds_path = out / f"{prefix}__answer_kinds.csv"
    if kinds_path.exists():
        key = {"Claude Haiku 4.5": "c", "GPT-4.1-mini": "g", "Gemini 2.5 Flash-Lite": "f"}
        for _, r in pd.read_csv(kinds_path).iterrows():
            k = f"na.{key[r['model']]}"
            for name, column in (("total", "total"), ("empty", "empty"), ("truncated", "truncated"), ("editIncomplete", "edit_incomplete"),
                                 ("noEdit", "no_edit"), ("unexpectedToolCall", "unexpected_tool_call")):
                facts[f"{k}.{name}"] = int(r[column])
            runs = int(facts[f"{key[r['model']]}.runs"])
            facts[f"{k}.per100"] = f"{100 * r['total'] / runs:.1f}"
            facts[f"{k}.emptyPer100"] = f"{100 * r['empty'] / runs:.1f}"
            # If every empty final answer had instead been a patch as often correct as the model's other patches
            patches, resolved = int(facts[f"{key[r['model']]}.patchesTotal"]), int(facts[f"{key[r['model']]}.resolvedTotal"])
            facts[f"{k}.emptyBound"] = f"{100 * r['empty'] / runs * resolved / patches:.1f}" if patches else "--"
            # The same for every final-answer call that ended as an attempted tool call (with or without text)
            facts[f"{k}.unexpectedPer100"] = f"{100 * r['unexpected_tool_call'] / runs:.1f}"
            facts[f"{k}.unexpectedBound"] = f"{100 * r['unexpected_tool_call'] / runs * resolved / patches:.1f}" if patches else "--"
    # Figures derived from the tables above, so that the text does not carry typed arithmetic
    for m in "cgf":
        total = int(facts.get(f"{m}.kind.total", 0))
        if total:
            facts[f"{m}.kind.targetShare"] = f"{100 * (int(facts[f'{m}.kind.f2p_failed']) + int(facts[f'{m}.kind.both'])) / total:.0f}"
            facts[f"{m}.kind.regressionOnly"] = int(facts[f"{m}.kind.p2p_regression"])
    kind_total = sum(int(facts.get(f"{m}.kind.total", 0)) for m in "cgf")
    if kind_total:
        facts["kind.total"] = kind_total
        facts["kind.targetShare"] = f"{100 * sum(int(facts[f'{m}.kind.f2p_failed']) + int(facts[f'{m}.kind.both']) for m in 'cgf') / kind_total:.0f}"
        facts["kind.regressionOnly"] = sum(int(facts[f"{m}.kind.p2p_regression"]) for m in "cgf")
    facts["applyFailedTotal"] = sum(int(facts.get(f"{m}.applyFailedTotal", 0)) for m in "cgf")
    regimes, between = paired[paired["family"] == "regimes"], paired[paired["family"] == "models"]
    model_key_for_q = {"Claude Haiku 4.5": "c", "GPT-4.1-mini": "g", "Gemini 2.5 Flash-Lite": "f"}
    facts["p.regimes.n"] = len(regimes)
    facts["p.regimes.minP"] = f"{regimes['p'].min():.3f}"
    facts["p.regimes.minHolm"] = f"{regimes['p_holm'].min():.3f}"
    facts["p.regimes.maxAbsDiff"] = f"{100 * regimes['diff'].abs().max():.1f}"
    facts["p.regimes.widestBound"] = f"{100 * max(regimes['lo'].abs().max(), regimes['hi'].abs().max()):.1f}"
    facts["p.regimes.unadjustedBelow05"] = int((regimes["p"] < 0.05).sum())
    model_key = {"Claude Haiku 4.5": "c", "GPT-4.1-mini": "g", "Gemini 2.5 Flash-Lite": "f"}
    pair_key = {"Baseline vs Forced": "bf", "Baseline vs Self-reflection": "br", "Forced vs Self-reflection": "fr"}
    for model, k in model_key.items():
        own = regimes[regimes["within"] == model]
        facts[f"p.{k}.regimes.maxAbsDiff"] = f"{100 * own['diff'].abs().max():.1f}"
        for _, r in own.iterrows():
            pk = f"p.{k}.{pair_key[r['comparison']]}"
            facts[f"{pk}.absDiff"] = f"{100 * abs(r['diff']):.1f}"
            facts[f"{pk}.absLo"] = f"{100 * min(abs(r['lo']), abs(r['hi'])):.1f}"
            facts[f"{pk}.absHi"] = f"{100 * max(abs(r['lo']), abs(r['hi'])):.1f}"
            facts[f"{pk}.excludesZero"] = int(r["lo"] > 0 or r["hi"] < 0)
    others = regimes[regimes["within"] != "Claude Haiku 4.5"]
    facts["p.gf.regimes.maxAbsDiff"] = f"{100 * others['diff'].abs().max():.1f}"
    facts["p.models.n"] = len(between)
    facts["p.models.maxHolm"] = "<0.001" if between["p_holm"].max() < 0.001 else f"{between['p_holm'].max():.3f}"
    facts["p.models.minDiff"] = f"{100 * between['diff'].min():.1f}"
    facts["p.models.maxDiff"] = f"{100 * between['diff'].max():.1f}"
    facts["q.minP"] = min(str(facts[f"q.{m}.p"]) for m in "cgf" if f"q.{m}.p" in facts)
    # Cochran's Q is computed once per model: Holm's correction across the three tests
    cochran = read("cochran").sort_values("p").reset_index(drop=True)
    running = 0.0
    for rank, row in cochran.iterrows():
        running = max(running, min(1.0, row["p"] * (len(cochran) - rank)))
        facts[f"q.{model_key_for_q[row['model']]}.pHolm"] = f"{running:.3f}"
    key = {"Claude Haiku 4.5": "c", "GPT-4.1-mini": "g", "Gemini 2.5 Flash-Lite": "f"}
    for model, k in key.items():
        rows = cost[cost["model"] == model].set_index("mode")
        if {"baseline", "self_reflect"} <= set(rows.index):
            facts[f"{k}.r.costIncrease"] = f"{100 * (rows.loc['self_reflect', 'cost_list_mean'] / rows.loc['baseline', 'cost_list_mean'] - 1):.0f}"
        list_total, billed_total = float(facts[f"{k}.costListTotal"]), float(facts[f"{k}.costBilledTotal"])
        facts[f"{k}.billedDiscount"] = f"{100 * (1 - billed_total / list_total):.0f}" if list_total else "--"
    # Reflection: runs in which the revision created a patch where the first edit did not apply, and the reverse
    calls = tidy[(tidy["mode"] == "self_reflect") & tidy["initial_outcome"].notna()]
    for model, k in key.items():
        g = calls[calls["model_label"] == model]
        first, final = g["initial_has_patch"].astype(bool), g["has_patch"].astype(bool)
        facts[f"r.{k}.patchGained"] = int((~first & final).sum())
        facts[f"r.{k}.patchLost"] = int((first & ~final).sum())
        facts[f"r.{k}.patchGainedResolved"] = int((~first & final & g["resolved"].astype(bool)).sum())
        facts[f"r.{k}.firstNotApplicable"] = int((g["initial_outcome"] == "edit_not_applicable").sum())
    for name, short in EXTERNAL_KEY.items():
        r = board[board["submission"] == name]
        if not r.empty:
            facts[f"ext.{short}.lite"] = f"{r.iloc[0]['pct_lite_300']:.1f}"
            facts[f"ext.{short}.sample"] = f"{r.iloc[0]['pct_sample_100']:.1f}"
    facts["ext.maxSample"] = f"{board['pct_sample_100'].max():.1f}"
    facts["ext.maxLite"] = f"{board['pct_lite_300'].max():.1f}"

    # Integrity of the run data, read from the result databases themselves
    errors, per_job, reclassified = 0, [], 0
    for provider in ("anthropic", "openai", "google"):
        db_path = PROJECT_ROOT / "runs" / args.tag / f"results_{provider}.db"
        if not db_path.exists():
            continue
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        for experiment_id, name in connection.execute("SELECT id, name FROM experiments"):
            if not name.endswith(f"__{args.subset}"):
                continue
            rows = connection.execute("SELECT result_status, run_meta FROM experiment_results WHERE experiment_id = ?", (experiment_id,)).fetchall()
            statuses = [row[0] for row in rows]
            # outcomes re-read from the stored answer under the final parsing rules (scripts/reclassify.py)
            reclassified += sum(bool(json.loads(row[1] or "{}").get("reclassified")) for row in rows)
            errors += sum(status in ("api_error", "infra_error", None) for status in statuses)
            per_job.append(sum(status not in ("api_error", "infra_error", None) for status in statuses))
        connection.close()
    facts["integrity.jobs"] = len(per_job)
    facts["integrity.reclassified"] = reclassified
    facts["pr.retriesTotal"] = sum(int(facts.get(f"pr.{m}.retries", 0)) for m in "cgf")
    facts["pr.unifiedDiffTotal"] = sum(int(facts.get(f"pr.{m}.unifiedDiff", 0)) for m in "cgf")
    facts["integrity.errorRows"] = errors
    facts["integrity.rowsPerJobMin"] = min(per_job) if per_job else 0
    facts["integrity.rowsPerJobMax"] = max(per_job) if per_job else 0

    def tex(value) -> str:
        # usable in text and in math mode; whole numbers from 1,000 get a thousands separator
        if str(value).isdigit() and int(value) >= 1000:
            return f"{int(value):,}".replace(",", "{,}")
        text = str(value).replace("_", "\\_").replace("%", "\\%").replace("<", "\\ensuremath{<}")
        return "\\ensuremath{-}" + text[1:] if text.startswith("-") and text[1:2].isdigit() else text
    (gen / "facts.tex").write_text(
        "% Written by analysis/make_paper_inputs.py from the analysis output. Do not edit by hand.\n"
        + "".join(f"\\expandafter\\def\\csname fact@{name}\\endcsname{{{tex(value)}}}\n" for name, value in sorted(facts.items())))

    incomplete = int((tidy["outcome"] == "not_evaluated").sum())
    reps = tidy.groupby(["model_label", "mode"])["rep"].nunique()
    print(f"tables and numbers written to {gen.relative_to(PROJECT_ROOT)} ({len(facts)} facts)")
    print(f"jobs: {facts['integrity.jobs']} | rows per job: {facts['integrity.rowsPerJobMin']}-{facts['integrity.rowsPerJobMax']} | error rows left: {errors}")
    print(f"patches without a verdict: {incomplete} | repetitions per cell: min {reps.min()}, max {reps.max()} | reference patches: {gold_text} | dates: {run_dates}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
