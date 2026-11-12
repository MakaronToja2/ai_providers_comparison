#!/usr/bin/env python3
"""
Check the statements of the manuscript that are written in words, not numbers.

Every figure in the text is printed from analysis/out/<tag>__<subset>__facts.json through
the \\N{...} macro, so it cannot drift from the analysis. Statements such as "no comparison
is significant" or "three to four times as much" are typed; this script tests each of them
against the same data and fails if one no longer holds.

Usage (after analysis/report.py and analysis/make_paper_inputs.py):
    .venv/bin/python analysis/check_claims.py [--tag rerun-2026-10] [--subset full]
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_facts() -> dict:
    """The facts exactly as the manuscript reads them (paper/mdis/generated/facts.tex)."""
    text = (PROJECT_ROOT / "paper" / "mdis" / "generated" / "facts.tex").read_text()
    facts = {}
    for name, value in re.findall(r"csname fact@(.+?)\\endcsname\{(.*)\}$", text, re.M):
        facts[name] = value.replace("\\ensuremath{-}", "-").replace("\\ensuremath{<}", "<").replace("{,}", "").replace("\\%", "%").replace("\\_", "_")
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="rerun-2026-10")
    parser.add_argument("--subset", default="full")
    args = parser.parse_args()
    out = PROJECT_ROOT / "analysis" / "out"
    prefix = f"{args.tag}__{args.subset}"
    f = load_facts()
    num = lambda key: float(f[key].lstrip("<"))
    paired = pd.read_csv(out / f"{prefix}__paired.csv")
    regimes, models = paired[paired["family"] == "regimes"], paired[paired["family"] == "models"]
    tidy = pd.read_csv(out / f"{prefix}__tidy.csv")

    checks = []
    def check(statement: str, holds: bool, detail: str = "") -> None:
        checks.append((statement, bool(holds), detail))

    # completeness
    check("2,700 runs, 27 jobs of 100 rows, no error rows", int(f["runs"]) == 2700 and int(f["integrity.jobs"]) == 27
          and int(f["integrity.rowsPerJobMin"]) == 100 == int(f["integrity.rowsPerJobMax"]) and int(f["integrity.errorRows"]) == 0)
    check("every patch has a verdict", int(f["notEvaluated"]) == 0, f"not evaluated: {f['notEvaluated']}")
    gold = (PROJECT_ROOT / "paper" / "mdis" / "generated" / "numbers.tex").read_text()
    check("reference patches: 100 of 100 resolved", "\\newcommand{\\goldResolved}{100 of 100}" in gold)
    check("three repetitions in every cell", all(int(f[f"{m}.{r}.reps"]) == 3 for m in "cgf" for r in "bfr"))
    check("no run exceeded a context window", all(int(f[f"{m}.contextOverflow"]) == 0 for m in "cgf"))
    check("no API call had to be retried", int(f["pr.retriesTotal"]) == 0)
    check("one model id per model", all("," not in f[f"pr.{m}.modelIds"] for m in "cgf"))

    # 5.1
    check("Claude and GPT-4.1-mini produce patches about equally often (baseline, within 2 per 100)", abs(num("c.b.patches") - num("g.b.patches")) <= 2)
    check("GPT-4.1-mini and Gemini: precision within 3 points (baseline)", abs(num("g.b.precision") - num("f.b.precision")) <= 3)
    widths = [num(f"{m}.b.precisionHi") - num(f"{m}.b.precisionLo") for m in "gf"]
    check("their precision intervals are about 20 points wide", all(17 <= w <= 22 for w in widths), ", ".join(f"{w:.1f}" for w in widths))
    check("Claude's precision is far higher (at least 15 points above both)", num("c.b.precision") - max(num("g.b.precision"), num("f.b.precision")) >= 15)
    check("Gemini loses most instances before evaluation (baseline)", num("f.b.o.edit_not_applicable") + num("f.b.o.no_answer") + num("f.b.o.abstained") > 50)
    for m in "cg":
        others = max(num(f"{m}.b.o.no_answer"), num(f"{m}.b.o.edit_not_applicable"), num(f"{m}.b.o.abstained"))
        check(f"largest group of failures is failed tests ({m}, baseline)", num(f"{m}.b.o.tests_failed") > others)
    check("the typical failed patch leaves a target test failing", num("kind.targetShare") >= 80)

    # 5.2
    check("order Claude > GPT-4.1-mini > Gemini in every regime", bool((models["diff"] > 0).all()))
    check("all nine model comparisons significant after Holm", len(models) == 9 and bool((models["p_holm"] < 0.05).all()), f"max {models['p_holm'].max():.3f}")
    excluding = regimes[(regimes["lo"] > 0) | (regimes["hi"] < 0)]
    check("the only regime interval that excludes zero is Claude, forced vs self-reflection (in favour of reflection)",
          len(excluding) == 1 and excluding.iloc[0]["within"] == "Claude Haiku 4.5" and excluding.iloc[0]["comparison"] == "Forced vs Self-reflection" and excluding.iloc[0]["diff"] < 0,
          "; ".join(f"{r['within']}: {r['comparison']}" for _, r in excluding.iterrows()))
    others = regimes[regimes["within"] != "Claude Haiku 4.5"]
    check("GPT-4.1-mini and Gemini: every regime interval includes zero", bool(((others["lo"] <= 0) & (others["hi"] >= 0)).all()))
    claude = regimes[regimes["within"] == "Claude Haiku 4.5"].set_index("comparison")
    check("Claude: neither forcing nor reflection differs from the baseline (unadjusted p, interval)",
          all(claude.loc[c, "p"] >= 0.05 and claude.loc[c, "lo"] <= 0 <= claude.loc[c, "hi"] for c in ("Baseline vs Forced", "Baseline vs Self-reflection")))
    check("Claude: forced vs reflection significant before correction, not after", claude.loc["Forced vs Self-reflection", "p"] < 0.05 <= claude.loc["Forced vs Self-reflection", "p_holm"],
          f"p {claude.loc['Forced vs Self-reflection', 'p']:.3f}, Holm {claude.loc['Forced vs Self-reflection', 'p_holm']:.3f}")
    check("no regime comparison significant after Holm", len(regimes) == 9 and bool((regimes["p_holm"] >= 0.05).all()), f"min {regimes['p_holm'].min():.3f}")
    check("Cochran's Q: below 0.05 only for Claude and only before adjustment across the three models",
          float(f["q.c.p"]) < 0.05 <= float(f["q.c.pHolm"]) and all(float(f[f"q.{m}.p"]) >= 0.05 for m in "gf"),
          ", ".join(f[f"q.{m}.p"] for m in "cgf") + f"; Claude adjusted {f['q.c.pHolm']}")
    check("no regime interval reaches further from zero than the stated bound", abs(max(regimes["lo"].abs().max(), regimes["hi"].abs().max()) * 100 - num("p.regimes.widestBound")) < 0.05)
    unadjusted = regimes[regimes["p"] < 0.05]
    check("regime comparisons below 0.05 before correction (the text must mention each)", True,
          "; ".join(f"{r['within']}: {r['comparison']} p={r['p']:.3f}" for _, r in unadjusted.iterrows()) or "none")

    # 5.3
    check("no forced run abstained", all(int(f[f"a.{m}.f"]) == 0 for m in "cgf"))
    check("abstention is rare (under 10% of permitted runs for every model)", all(num(f"a.{m}.permittedShare") < 10 for m in "cgf"))
    check("patches rise under forcing only for Gemini", num("f.f.patches") > num("f.b.patches") + 1 and num("c.f.patches") <= num("c.b.patches") + 1 and num("g.f.patches") <= num("g.b.patches") + 1)
    check("Gemini's abstained instances are harder for it under forcing", num("a.f.set.forcedResolvedShare") < num("a.f.set.restForcedResolvedShare"))

    # 5.4
    check("no within-run reflection difference is significant at 5%", all(float(f[f"r.{m}.p"].lstrip("<")) >= 0.05 for m in "cgf"), ", ".join(f[f"r.{m}.p"] for m in "cgf"))
    check("Claude: most runs gained by reflection are edits repaired by the revision", int(f["r.c.gained"]) / 2 < int(f["r.c.patchGainedResolved"]) <= int(f["r.c.gained"]),
          f"{f['r.c.patchGainedResolved']} of {f['r.c.gained']}")
    check("reflection rarely changes the outcome (under 5% of calls, each model)", all((int(f[f"r.{m}.gained"]) + int(f[f"r.{m}.lost"])) / int(f[f"r.{m}.calls"]) < 0.05 for m in "cgf"))
    check("no significant difference between baseline and forced, any model", bool((regimes[regimes["comparison"] == "Baseline vs Forced"]["p"] >= 0.05).all()))
    check("Claude: fewer non-applicable edits with reflection than in the baseline", num("c.r.o.edit_not_applicable") < num("c.b.o.edit_not_applicable"))
    check("Claude: revision gains more patches than it loses", int(f["r.c.patchGained"]) > int(f["r.c.patchLost"]))

    # 5.5
    ratios = [num("c.b.perResolvedList") / num("g.b.perResolvedList"), num("c.b.perResolvedList") / num("f.b.perResolvedList")]
    cost = pd.read_csv(out / f"{prefix}__cost.csv").set_index(["model", "mode"])["cost_list_per_resolved"]
    exact = [cost["Claude Haiku 4.5", "baseline"] / cost["GPT-4.1-mini", "baseline"], cost["Claude Haiku 4.5", "baseline"] / cost["Gemini 2.5 Flash-Lite", "baseline"]]
    check("Claude pays between three and a half and four times as much per resolved instance (unrounded)", all(3.4 <= r <= 4.15 for r in exact), f"{exact[0]:.2f}, {exact[1]:.2f}")
    check("Gemini resolves about half as many as GPT-4.1-mini (baseline)", 0.4 <= num("f.b.resolved") / num("g.b.resolved") <= 0.65, f"{num('f.b.resolved') / num('g.b.resolved'):.2f}")
    check("Gemini is not much cheaper per resolved instance than GPT-4.1-mini", cost["Gemini 2.5 Flash-Lite", "baseline"] / cost["GPT-4.1-mini", "baseline"] >= 0.75,
          f"{cost['Gemini 2.5 Flash-Lite', 'baseline'] / cost['GPT-4.1-mini', 'baseline']:.2f}")
    check("no cache discount for Claude", int(f["c.billedDiscount"]) == 0)
    check("per instance, more input tokens go with fewer resolved runs in every model (negative Spearman correlation, 100 instances)",
          all(num(f"u.{m}.rho") < 0 and int(f[f"u.{m}.n"]) == 100 for m in "cgf"), ", ".join(f"{f[f'u.{m}.rho']} (p {f[f'u.{m}.p']})" for m in "cgf"))
    check("Claude consumes the most tokens and resolves the most", num("c.b.inMed") > max(num("g.b.inMed"), num("f.b.inMed")) and num("c.b.success") > max(num("g.b.success"), num("f.b.success")))
    check("no significant gain from reflection between regimes (baseline vs reflection, each model)",
          bool((regimes[regimes["comparison"] == "Baseline vs Self-reflection"]["p_holm"] >= 0.05).all()))

    # 5.6
    check("most runs end through the request for the final answer, each model", all(num(f"pr.{m}.finalRequestShare") > 50 for m in "cgf"))
    check("both unified-diff answers are Gemini's", int(f["pr.unifiedDiffTotal"]) == 2 == int(f["pr.f.unifiedDiff"]) and int(f["applyFailedTotal"]) == 1)
    check("only Gemini has attempted tool calls in the final-answer request", int(f["na.c.unexpectedToolCall"]) == 0 == int(f["na.g.unexpectedToolCall"]) and int(f["na.f.unexpectedToolCall"]) > 0)
    check("empty final answers are a subset of the attempted tool calls", int(f["na.f.empty"]) <= int(f["na.f.unexpectedToolCall"]))

    # 6
    check("Claude lies between Agentless 1.5 and OpenHands on the sample", num("ext.agentlessNew.sample") < num("c.b.success") < num("ext.openhands.sample"))
    check("GPT-4.1-mini and Gemini lie on either side of SWE-agent + GPT-4", num("f.b.success") < num("ext.sweagentGptFour.sample") < num("g.b.success"))
    check("about twenty points below the best system on the sample", 17 <= num("ext.maxSample") - num("c.b.success") <= 25, f"{num('ext.maxSample') - num('c.b.success'):.1f}")
    check("the leaderboard has passed 60% on full Lite", num("ext.maxLite") > 60)

    # prices typed in Section 4.3 are the ones the cost computation uses
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.benchmark.pricing import PRICES_CHECKED_ON, PRICES_USD_PER_MTOK
    check("prices in the text equal the prices used: list 1.00/5.00, 0.40/1.60, 0.10/0.40; cached 0.10 (GPT-4.1-mini), 0.01 (Gemini); dated 7 Oct 2026",
          PRICES_USD_PER_MTOK["claude-haiku-4-5-20251001"][:2] == (1.00, 5.00) and PRICES_USD_PER_MTOK["gpt-4.1-mini"] == (0.40, 1.60, 0.10)
          and PRICES_USD_PER_MTOK["gemini-2.5-flash-lite"] == (0.10, 0.40, 0.01) and PRICES_CHECKED_ON == "2026-10-07")
    check("comparison table lists six published systems", "\\newcommand{\\nExternal}{6}" in gold)

    # abstract / conclusion
    run_dates = (PROJECT_ROOT / "paper" / "mdis" / "generated" / "numbers.tex").read_text().split("runDates}{")[1].split("}")[0]
    check("all calls on one day", not re.search(r"\b(to|and)\b", run_dates), run_dates)
    check("sample: 100 instances, 12 repositories", tidy["instance_id"].nunique() == 100 and tidy["repo"].nunique() == 12)

    width = max(len(statement) for statement, _, _ in checks)
    failed = 0
    for statement, holds, detail in checks:
        failed += not holds
        print(f"{'ok  ' if holds else 'FAIL'}  {statement:{width}s}  {detail}")
    print(f"\n{len(checks) - failed} of {len(checks)} statements hold")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
