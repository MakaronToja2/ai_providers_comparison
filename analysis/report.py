#!/usr/bin/env python3
"""
Statistics for the paper, computed from the tidy table (analysis/build_table.py).

Unit of analysis: the instance. Repetitions of the same instance are not independent
observations, so intervals come from a bootstrap that resamples instances (keeping all
repetitions of a drawn instance together), and paired tests are run on one value per
instance (resolved in the majority of repetitions).

Usage:
    .venv/bin/python analysis/report.py [--tag rerun-2026-10] [--subset full]
Output: analysis/out/<tag>__<subset>__report.md and CSV tables next to it.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ["baseline", "forced", "self_reflect"]
MODE_LABEL = {"baseline": "Baseline", "forced": "Forced", "self_reflect": "Self-reflection"}
OUTCOME_ORDER = ["resolved", "tests_failed", "apply_failed", "edit_not_applicable", "no_answer",
                 "abstained", "context_overflow", "not_evaluated"]
BOOTSTRAP_DRAWS = 10_000
SEED = 20261007
# Short keys for the fact file read by analysis/make_paper_inputs.py
MODEL_KEY = {"Claude Haiku 4.5": "c", "GPT-4.1-mini": "g", "Gemini 2.5 Flash-Lite": "f"}
MODE_KEY = {"baseline": "b", "forced": "f", "self_reflect": "r"}


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_k(tokens: float) -> str:
    return f"{tokens / 1000:.1f}"


def pivot(cell: pd.DataFrame, column: str, instances: list) -> np.ndarray:
    """Instances x repetitions matrix of a 0/1 column (NaN where a repetition is missing)."""
    wide = cell.pivot(index="instance_id", columns="rep", values=column).reindex(instances)
    return wide.to_numpy(dtype=float)


def bootstrap_indices(n_instances: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_instances, size=(BOOTSTRAP_DRAWS, n_instances))


def ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high)


def majority(matrix: np.ndarray) -> np.ndarray:
    """One value per instance: true in more than half of the available repetitions."""
    return (np.nansum(matrix, axis=1) > np.sum(~np.isnan(matrix), axis=1) / 2).astype(int)


def fmt_pct(x: float) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{100 * x:.1f}"


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """(only a, only b, exact p) for two paired 0/1 vectors."""
    only_a, only_b = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    both, neither = int(np.sum((a == 1) & (b == 1))), int(np.sum((a == 0) & (b == 0)))
    p = float(mcnemar([[both, only_a], [only_b, neither]], exact=True).pvalue) if only_a + only_b else 1.0
    return only_a, only_b, p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="rerun-2026-10")
    parser.add_argument("--subset", default="full")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "analysis" / "out"
    prefix = f"{args.tag}__{args.subset}"
    table = pd.read_csv(out_dir / f"{prefix}__tidy.csv")
    instances = sorted(table["instance_id"].unique())
    n = len(instances)
    models = [m for m in ["Claude Haiku 4.5", "GPT-4.1-mini", "Gemini 2.5 Flash-Lite"] if m in set(table["model_label"])]
    rng = np.random.default_rng(SEED)
    draws = bootstrap_indices(n, rng)
    lines: list[str] = []
    facts: dict[str, str] = {}

    def emit(text: str = "") -> None:
        lines.append(text)

    def fact(key: str, value) -> None:
        """A number the manuscript may cite, already formatted for print."""
        assert key not in facts, key
        facts[key] = str(value)

    emit(f"# Results: {args.tag}, {args.subset} set ({n} instances)")
    not_evaluated = int((table["outcome"] == "not_evaluated").sum())
    if not_evaluated:
        emit(f"\n**Incomplete: {not_evaluated} patches have no harness verdict yet; they count as unresolved below.**")

    cells: dict[tuple[str, str], dict] = {}
    for model, mode in itertools.product(models, MODES):
        cell = table[(table["model_label"] == model) & (table["mode"] == mode)]
        if cell.empty:
            continue
        cells[(model, mode)] = {
            "resolved": pivot(cell, "resolved", instances),
            "patch": pivot(cell, "has_patch", instances),
            "reps": sorted(cell["rep"].unique()),
            "frame": cell,
        }

    # ---------------------------------------------------------------- 1. main table
    emit("\n## 1. Patches, resolved, success and precision per model and regime")
    emit("\nPer repetition counts are out of the instances run; mean ± SD over repetitions. Success = resolved / instances. "
         "Precision = resolved / patches. 95% intervals: bootstrap over instances.\n")
    emit("| Model | Regime | Reps | Patches | Resolved | Success % [95% CI] | Precision % [95% CI] |")
    emit("|---|---|---|---|---|---|---|")
    main_rows = []
    for (model, mode), c in cells.items():
        r, p = c["resolved"], c["patch"]
        per_rep_patches, per_rep_resolved = np.nansum(p, axis=0), np.nansum(r, axis=0)
        sd = (lambda v: f" ± {np.std(v, ddof=1):.1f}" if len(v) > 1 else "")
        success = np.nanmean(r)
        precision = np.nansum(r) / np.nansum(p) if np.nansum(p) else np.nan
        boot_success = np.nanmean(r[draws], axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            boot_precision = np.nansum(r[draws], axis=(1, 2)) / np.nansum(p[draws], axis=(1, 2))
        s_lo, s_hi = ci(boot_success)
        p_lo, p_hi = ci(boot_precision)
        emit(f"| {model} | {MODE_LABEL[mode]} | {len(c['reps'])} | {per_rep_patches.mean():.1f}{sd(per_rep_patches)} | "
             f"{per_rep_resolved.mean():.1f}{sd(per_rep_resolved)} | {fmt_pct(success)} [{fmt_pct(s_lo)}, {fmt_pct(s_hi)}] | "
             f"{fmt_pct(precision)} [{fmt_pct(p_lo)}, {fmt_pct(p_hi)}] |")
        k = f"{MODEL_KEY[model]}.{MODE_KEY[mode]}"
        fact(f"{k}.reps", len(c["reps"]))
        fact(f"{k}.patches", f"{per_rep_patches.mean():.1f}")
        fact(f"{k}.resolved", f"{per_rep_resolved.mean():.1f}")
        fact(f"{k}.patchesRange", f"{int(per_rep_patches.min())}--{int(per_rep_patches.max())}")
        fact(f"{k}.resolvedRange", f"{int(per_rep_resolved.min())}--{int(per_rep_resolved.max())}")
        fact(f"{k}.success", fmt_pct(success)); fact(f"{k}.successLo", fmt_pct(s_lo)); fact(f"{k}.successHi", fmt_pct(s_hi))
        fact(f"{k}.precision", fmt_pct(precision)); fact(f"{k}.precisionLo", fmt_pct(p_lo)); fact(f"{k}.precisionHi", fmt_pct(p_hi))
        main_rows.append({"model": model, "mode": mode, "reps": len(c["reps"]), "patches_mean": per_rep_patches.mean(),
                          "patches_sd": np.std(per_rep_patches, ddof=1) if len(per_rep_patches) > 1 else np.nan,
                          "resolved_mean": per_rep_resolved.mean(),
                          "resolved_sd": np.std(per_rep_resolved, ddof=1) if len(per_rep_resolved) > 1 else np.nan,
                          "success": success, "success_lo": s_lo, "success_hi": s_hi,
                          "precision": precision, "precision_lo": p_lo, "precision_hi": p_hi})
    pd.DataFrame(main_rows).to_csv(out_dir / f"{prefix}__main.csv", index=False)

    # ---------------------------------------------------------------- 2. outcomes
    emit("\n## 2. What happened to each instance (mean count per repetition)\n")
    present = [o for o in OUTCOME_ORDER if o in set(table["outcome"])]
    emit("| Model | Regime | " + " | ".join(present) + " |")
    emit("|---|---|" + "---|" * len(present))
    outcome_rows = []
    for (model, mode), c in cells.items():
        counts = c["frame"].groupby("rep")["outcome"].value_counts().unstack(fill_value=0).reindex(columns=present, fill_value=0)
        means = counts.mean()
        emit(f"| {model} | {MODE_LABEL[mode]} | " + " | ".join(f"{means[o]:.1f}" for o in present) + " |")
        outcome_rows.append({"model": model, "mode": mode, **{o: means[o] for o in present}})
        for o in present:
            fact(f"{MODEL_KEY[model]}.{MODE_KEY[mode]}.o.{o}", f"{means[o]:.1f}")
    pd.DataFrame(outcome_rows).to_csv(out_dir / f"{prefix}__outcomes.csv", index=False)
    kinds = table[table["outcome"] == "tests_failed"].groupby(["model_label", "mode"])["failure_kind"].value_counts().unstack(fill_value=0)
    if not kinds.empty:
        emit("\nPatches that applied but failed tests, by kind (all repetitions): f2p_failed = the target tests still fail; "
             "p2p_regression = a previously passing test broke; both.\n")
        emit(kinds.to_markdown())
        for model in models:
            total = table[(table["model_label"] == model) & (table["outcome"] == "tests_failed")]["failure_kind"].value_counts()
            for kind in ("f2p_failed", "p2p_regression", "both", "timeout"):
                fact(f"{MODEL_KEY[model]}.kind.{kind}", int(total.get(kind, 0)))
            fact(f"{MODEL_KEY[model]}.kind.total", int(total.sum()))
    reasons = table[table["outcome"] == "edit_not_applicable"].groupby("model_label")["edit_failure_reasons"].value_counts().unstack(fill_value=0)
    if not reasons.empty:
        emit("\nEdits that could not be applied, by reason (final answer; all repetitions):\n")
        emit(reasons.to_markdown())
    for model in models:
        m = table[table["model_label"] == model]
        fact(f"{MODEL_KEY[model]}.runs", len(m))
        fact(f"{MODEL_KEY[model]}.patchesTotal", int(m["has_patch"].sum()))
        fact(f"{MODEL_KEY[model]}.resolvedTotal", int(m["resolved"].sum()))
        fact(f"{MODEL_KEY[model]}.applyFailedTotal", int((m["outcome"] == "apply_failed").sum()))
        fact(f"{MODEL_KEY[model]}.contextOverflow", int((m["outcome"] == "context_overflow").sum()))
        # instances resolved in every / in at least one of the baseline repetitions
        if (model, "baseline") in cells:
            r = cells[(model, "baseline")]["resolved"]
            fact(f"{MODEL_KEY[model]}.b.resolvedAllReps", int(np.sum(np.nansum(r, axis=1) == np.sum(~np.isnan(r), axis=1))))
            fact(f"{MODEL_KEY[model]}.b.resolvedAnyRep", int(np.sum(np.nansum(r, axis=1) > 0)))
    fact("runs", len(table))
    fact("patchesTotal", int(table["has_patch"].sum()))
    fact("notEvaluated", int((table["outcome"] == "not_evaluated").sum()))

    # ---------------------------------------------------------------- 3. paired comparisons
    emit("\n## 3. Paired comparisons (same instances)")
    emit("\nDifference in success (first minus second) with a bootstrap 95% interval over instances; McNemar's exact test on "
         "one value per instance (resolved in the majority of repetitions). 'only first' / 'only second' are the discordant instances.\n")
    emit("| Comparison | Within | Δ success, points [95% CI] | only first | only second | McNemar p | Holm p |")
    emit("|---|---|---|---|---|---|---|")
    comparisons = []
    for model in models:
        for a, b in itertools.combinations(MODES, 2):
            if (model, a) in cells and (model, b) in cells:
                comparisons.append((f"{MODE_LABEL[a]} vs {MODE_LABEL[b]}", model, cells[(model, a)], cells[(model, b)], "regimes"))
    for mode in MODES:
        for a, b in itertools.combinations(models, 2):
            if (a, mode) in cells and (b, mode) in cells:
                comparisons.append((f"{a} vs {b}", MODE_LABEL[mode], cells[(a, mode)], cells[(b, mode)], "models"))
    results = []
    for label, within, ca, cb, family in comparisons:
        ra, rb = ca["resolved"], cb["resolved"]
        diff = np.nanmean(ra) - np.nanmean(rb)
        boot = np.nanmean(ra[draws], axis=(1, 2)) - np.nanmean(rb[draws], axis=(1, 2))
        lo, hi = ci(boot)
        only_a, only_b, p = mcnemar_exact(majority(ra), majority(rb))
        results.append([label, within, diff, lo, hi, only_a, only_b, p, family])
    for family in ("regimes", "models"):
        idx = [i for i, r in enumerate(results) if r[8] == family]
        if idx:
            adjusted = multipletests([results[i][7] for i in idx], method="holm")[1]
            for i, adj in zip(idx, adjusted):
                results[i].append(float(adj))
    label_key = {MODE_LABEL[m]: MODE_KEY[m] for m in MODES} | MODEL_KEY
    for label, within, diff, lo, hi, only_a, only_b, p, family, holm in results:
        emit(f"| {label} | {within} | {100 * diff:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {only_a} | {only_b} | {p:.3f} | {holm:.3f} |")
        first, second = label.split(" vs ")
        k = f"p.{label_key[within]}.{label_key[first]}{label_key[second]}"   # e.g. p.c.bf (regimes within Claude), p.b.cg (models under baseline)
        fact(f"{k}.diff", f"{100 * diff:+.1f}"); fact(f"{k}.lo", f"{100 * lo:+.1f}"); fact(f"{k}.hi", f"{100 * hi:+.1f}")
        fact(f"{k}.onlyFirst", only_a); fact(f"{k}.onlySecond", only_b); fact(f"{k}.p", fmt_p(p)); fact(f"{k}.pHolm", fmt_p(holm))
    pd.DataFrame(results, columns=["comparison", "within", "diff", "lo", "hi", "only_first", "only_second", "p", "family", "p_holm"]) \
        .to_csv(out_dir / f"{prefix}__paired.csv", index=False)
    emit("\nHolm correction is applied separately to the regime comparisons and to the model comparisons.")

    emit("\nCochran's Q across the three regimes, per model (majority outcome per instance):\n")
    cochran_rows = []
    for model in models:
        if all((model, mode) in cells for mode in MODES):
            x = np.column_stack([majority(cells[(model, mode)]["resolved"]) for mode in MODES])
            if len(np.unique(x.sum(axis=1))) > 1 or x.any():
                q = cochrans_q(x)
                emit(f"- {model}: Q = {q.statistic:.2f}, df = {q.df}, p = {q.pvalue:.3f}")
                cochran_rows.append({"model": model, "Q": q.statistic, "df": q.df, "p": q.pvalue})
                fact(f"q.{MODEL_KEY[model]}.Q", f"{q.statistic:.2f}"); fact(f"q.{MODEL_KEY[model]}.p", fmt_p(q.pvalue))
    pd.DataFrame(cochran_rows).to_csv(out_dir / f"{prefix}__cochran.csv", index=False)

    # ---------------------------------------------------------------- 4. reflection
    emit("\n## 4. Self-reflection: first patch versus final patch (same runs)\n")
    reflect = table[(table["mode"] == "self_reflect") & table["initial_outcome"].notna()]
    if reflect.empty:
        emit("No self-reflection runs with a reflection call yet.")
    else:
        emit("| Model | Reflection calls | approved | revised | no verdict | resolved before | resolved after | gained | lost | McNemar p |")
        emit("|---|---|---|---|---|---|---|---|---|---|")
        reflection_rows = []
        for model in models:
            m = reflect[reflect["model_label"] == model]
            if m.empty:
                continue
            before, after = m["initial_resolved"].astype(int).to_numpy(), m["resolved"].astype(int).to_numpy()
            lost, gained, p = mcnemar_exact(before, after)
            oc = m["reflection_outcome"].value_counts()
            emit(f"| {model} | {len(m)} | {oc.get('approved', 0)} | {oc.get('revised', 0)} | {oc.get('unstructured', 0)} | "
                 f"{before.sum()} | {after.sum()} | {gained} | {lost} | {p:.3f} |")
            revised = m[m["reflection_outcome"] == "revised"]
            k = f"r.{MODEL_KEY[model]}"
            fact(f"{k}.calls", len(m)); fact(f"{k}.approved", int(oc.get("approved", 0))); fact(f"{k}.revised", int(oc.get("revised", 0)))
            fact(f"{k}.noVerdict", int(oc.get("unstructured", 0)))
            fact(f"{k}.revisedShare", f"{100 * oc.get('revised', 0) / len(m):.0f}")
            fact(f"{k}.resolvedBefore", int(before.sum())); fact(f"{k}.resolvedAfter", int(after.sum()))
            fact(f"{k}.gained", gained); fact(f"{k}.lost", lost); fact(f"{k}.p", fmt_p(p))
            fact(f"{k}.patchesBefore", int(m["initial_has_patch"].astype(bool).sum())); fact(f"{k}.patchesAfter", int(m["has_patch"].sum()))
            fact(f"{k}.revisedNoPatch", int((~revised["has_patch"]).sum()))
            fact(f"{k}.revisedResolvedBefore", int(revised["initial_resolved"].astype(int).sum()))
            fact(f"{k}.revisedResolvedAfter", int(revised["resolved"].astype(int).sum()))
            reflection_rows.append({
                "model": model, "calls": len(m), "approved": int(oc.get("approved", 0)), "revised": int(oc.get("revised", 0)),
                "no_verdict": int(oc.get("unstructured", 0)), "resolved_before": int(before.sum()), "resolved_after": int(after.sum()),
                "gained": gained, "lost": lost, "p": p,
                "patches_before": int(m["initial_has_patch"].astype(bool).sum()), "patches_after": int(m["has_patch"].sum()),
                "revised_without_patch": int((~revised["has_patch"]).sum()),
            })
        pd.DataFrame(reflection_rows).to_csv(out_dir / f"{prefix}__reflection.csv", index=False)
        emit("\nCounts pool the repetitions. 'gained' = unresolved first patch, resolved final patch; 'lost' = the reverse.")
        changed = reflect[reflect["reflection_outcome"] == "revised"].groupby("model_label").apply(
            lambda g: pd.Series({"revised": len(g), "revision produced no patch": int((~g["has_patch"]).sum())}), include_groups=False)
        if not changed.empty:
            emit("\n" + changed.to_markdown())

    # ---------------------------------------------------------------- 5. abstention and forcing
    emit("\n## 5. Abstention and forcing\n")
    abstain = table.groupby(["model_label", "mode"])["abstained"].sum().unstack(fill_value=0).reindex(columns=[m for m in MODES if m in set(table['mode'])])
    emit("Explicit abstentions (all repetitions together):\n")
    emit(abstain.to_markdown())
    abstain.to_csv(out_dir / f"{prefix}__abstention.csv")
    for model in models:
        for mode in MODES:
            cell = table[(table["model_label"] == model) & (table["mode"] == mode)]
            if len(cell):
                fact(f"a.{MODEL_KEY[model]}.{MODE_KEY[mode]}", int(cell["abstained"].sum()))
                fact(f"a.{MODEL_KEY[model]}.{MODE_KEY[mode]}.runs", len(cell))
        permitted = table[(table["model_label"] == model) & (table["mode"] != "forced")]
        if len(permitted):
            fact(f"a.{MODEL_KEY[model]}.permitted", int(permitted["abstained"].sum()))
            fact(f"a.{MODEL_KEY[model]}.permittedRuns", len(permitted))
            fact(f"a.{MODEL_KEY[model]}.permittedShare", f"{100 * permitted['abstained'].mean():.1f}")
            fact(f"a.{MODEL_KEY[model]}.permittedInstances", int(permitted.loc[permitted["abstained"], "instance_id"].nunique()))
    emit("\nInstances on which a model abstained at least once where abstention was permitted (baseline and self-reflection regimes), "
         "and what happened to them elsewhere:\n")
    for model in models:
        mine = table[table["model_label"] == model]
        permitted = mine[mine["mode"] != "forced"]
        declined = set(permitted.loc[permitted["abstained"], "instance_id"])
        k = f"a.{MODEL_KEY[model]}.set"
        fact(f"{k}.instances", len(declined))
        if not declined:
            emit(f"- {model}: never abstained.")
            continue
        forced = mine[mine["mode"] == "forced"]
        on, off = forced[forced["instance_id"].isin(declined)], forced[~forced["instance_id"].isin(declined)]
        others = table[(table["model_label"] != model) & table["instance_id"].isin(declined)]
        same_other_runs = permitted[permitted["instance_id"].isin(declined) & ~permitted["abstained"]]
        always = permitted[permitted["instance_id"].isin(declined)].groupby("instance_id")["abstained"].all().sum()
        fact(f"{k}.alwaysAbstained", int(always))
        fact(f"{k}.forcedRuns", len(on)); fact(f"{k}.forcedPatches", int(on["has_patch"].sum())); fact(f"{k}.forcedResolved", int(on["resolved"].sum()))
        fact(f"{k}.forcedResolvedInstances", int(on.loc[on["resolved"], "instance_id"].nunique()))
        fact(f"{k}.forcedResolvedShare", fmt_pct(on["resolved"].mean()) if len(on) else "--")
        fact(f"{k}.restForcedRuns", len(off)); fact(f"{k}.restForcedResolved", int(off["resolved"].sum()))
        fact(f"{k}.restForcedResolvedShare", fmt_pct(off["resolved"].mean()) if len(off) else "--")
        fact(f"{k}.sameModelOtherRuns", len(same_other_runs)); fact(f"{k}.sameModelOtherResolved", int(same_other_runs["resolved"].sum()))
        fact(f"{k}.otherModelsResolvedInstances", int(others.loc[others["resolved"], "instance_id"].nunique()))
        emit(f"- {model}: {len(declined)} instance(s), {int(always)} of them abstained on in every permitted run. Under forcing: "
             f"{int(on['has_patch'].sum())} patches and {int(on['resolved'].sum())} resolved in {len(on)} runs "
             f"({fmt_pct(on['resolved'].mean()) if len(on) else 'n/a'}%), against {fmt_pct(off['resolved'].mean()) if len(off) else 'n/a'}% on its other instances. "
             f"Its own non-abstaining permitted runs on these instances: {int(same_other_runs['resolved'].sum())} resolved of {len(same_other_runs)}. "
             f"Resolved at least once by another model: {int(others.loc[others['resolved'], 'instance_id'].nunique())} of {len(declined)} instances.")

    # ---------------------------------------------------------------- 6. tokens and cost
    emit("\n## 6. Tokens and cost per instance\n")
    emit("Input and output tokens include the reflection call. Cost at list price ignores cache discounts; 'as billed' prices the "
         "input tokens the provider reports as cached at its cached rate.\n")
    emit("| Model | Regime | Input tokens median [IQR] | SD | Output median | Cached input % | $/instance list | $/instance billed | $/resolved list | $/resolved billed |")
    emit("|---|---|---|---|---|---|---|---|---|---|")
    cost_rows = []
    for (model, mode), c in cells.items():
        f = c["frame"]
        q1, med, q3 = f["input_tokens"].quantile([0.25, 0.5, 0.75])
        resolved = f["resolved"].sum()
        per_resolved = (lambda total: f"{total / resolved:.3f}" if resolved else "n/a")
        cached_share = f["cached_input_tokens"].sum() / f["input_tokens"].sum() if f["input_tokens"].sum() else 0
        emit(f"| {model} | {MODE_LABEL[mode]} | {med:,.0f} [{q1:,.0f}–{q3:,.0f}] | {f['input_tokens'].std():,.0f} | {f['output_tokens'].median():,.0f} | "
             f"{100 * cached_share:.0f} | {f['cost_list_usd'].mean():.4f} | {f['cost_billed_usd'].mean():.4f} | "
             f"{per_resolved(f['cost_list_usd'].sum())} | {per_resolved(f['cost_billed_usd'].sum())} |")
        k = f"{MODEL_KEY[model]}.{MODE_KEY[mode]}"
        fact(f"{k}.inMed", fmt_k(med)); fact(f"{k}.inQ1", fmt_k(q1)); fact(f"{k}.inQ3", fmt_k(q3)); fact(f"{k}.inSD", fmt_k(f["input_tokens"].std()))
        fact(f"{k}.inMean", fmt_k(f["input_tokens"].mean())); fact(f"{k}.outMed", fmt_k(f["output_tokens"].median()))
        fact(f"{k}.cached", f"{100 * cached_share:.0f}")
        fact(f"{k}.costList", f"{f['cost_list_usd'].mean():.3f}"); fact(f"{k}.costBilled", f"{f['cost_billed_usd'].mean():.3f}")
        fact(f"{k}.perResolvedList", f"{f['cost_list_usd'].sum() / resolved:.2f}" if resolved else "--")
        fact(f"{k}.perResolvedBilled", f"{f['cost_billed_usd'].sum() / resolved:.2f}" if resolved else "--")
        cost_rows.append({"model": model, "mode": mode, "input_median": med, "input_q1": q1, "input_q3": q3, "input_sd": f["input_tokens"].std(),
                          "output_median": f["output_tokens"].median(), "cached_share": cached_share,
                          "cost_list_mean": f["cost_list_usd"].mean(), "cost_billed_mean": f["cost_billed_usd"].mean(),
                          "cost_list_per_resolved": f["cost_list_usd"].sum() / resolved if resolved else np.nan,
                          "cost_billed_per_resolved": f["cost_billed_usd"].sum() / resolved if resolved else np.nan})
    pd.DataFrame(cost_rows).to_csv(out_dir / f"{prefix}__cost.csv", index=False)
    refl = table[(table["mode"] == "self_reflect") & (table["reflection_prompt_tokens"] > 0)]
    if not refl.empty:
        emit("\nReflection call alone (runs where it happened), mean tokens: " + "; ".join(
            f"{model}: {g['reflection_prompt_tokens'].mean():,.0f} in / {g['reflection_completion_tokens'].mean():,.0f} out"
            for model, g in refl.groupby("model_label")))
        for model, g in refl.groupby("model_label"):
            fact(f"r.{MODEL_KEY[model]}.tokensIn", fmt_k(g["reflection_prompt_tokens"].mean()))
            fact(f"r.{MODEL_KEY[model]}.tokensOut", fmt_k(g["reflection_completion_tokens"].mean()))
            price = g["cost_list_usd"].sum() and (g.apply(lambda row: row["cost_list_usd"] * (row["reflection_prompt_tokens"] + row["reflection_completion_tokens"])
                                                  / max(1, row["input_tokens"] + row["output_tokens"]), axis=1).mean())
            fact(f"r.{MODEL_KEY[model]}.tokenShare", f"{100 * (g['reflection_prompt_tokens'] + g['reflection_completion_tokens']).sum() / (g['input_tokens'] + g['output_tokens']).sum():.0f}")
    for model in models:
        m = table[table["model_label"] == model]
        fact(f"{MODEL_KEY[model]}.costListTotal", f"{m['cost_list_usd'].sum():.2f}"); fact(f"{MODEL_KEY[model]}.costBilledTotal", f"{m['cost_billed_usd'].sum():.2f}")
        fact(f"{MODEL_KEY[model]}.cachedAll", f"{100 * m['cached_input_tokens'].sum() / m['input_tokens'].sum():.0f}")
    fact("costListTotal", f"{table['cost_list_usd'].sum():.2f}"); fact("costBilledTotal", f"{table['cost_billed_usd'].sum():.2f}")

    emit("\nDo instances on which the agent uses more tokens get resolved more often? Within each model, baseline regime, one "
         "observation per instance (the repetitions of an instance are not independent): Spearman's rank correlation between the "
         "mean input tokens of the instance's runs and the number of those runs that were resolved.\n")
    for model in models:
        f = table[(table["model_label"] == model) & (table["mode"] == "baseline")]
        per_instance = f.groupby("instance_id").agg(tokens=("input_tokens", "mean"), solved=("resolved", "sum"), runs=("resolved", "size"))
        if len(per_instance) >= 10 and per_instance["solved"].nunique() > 1:
            rho = stats.spearmanr(per_instance["tokens"], per_instance["solved"])
            mostly = per_instance["solved"] * 2 > per_instance["runs"]
            emit(f"- {model}: rho = {rho.statistic:+.2f}, p = {rho.pvalue:.3f}, n = {len(per_instance)} instances; median of the per-instance "
                 f"mean: {per_instance.loc[mostly, 'tokens'].median():,.0f} where most runs were resolved (n={int(mostly.sum())}), "
                 f"{per_instance.loc[~mostly, 'tokens'].median():,.0f} elsewhere")
            k = f"u.{MODEL_KEY[model]}"
            fact(f"{k}.rho", f"{rho.statistic:+.2f}"); fact(f"{k}.p", fmt_p(rho.pvalue)); fact(f"{k}.n", len(per_instance))
            fact(f"{k}.majorityMed", fmt_k(per_instance.loc[mostly, "tokens"].median()))
            fact(f"{k}.otherMed", fmt_k(per_instance.loc[~mostly, "tokens"].median()))

    # ---------------------------------------------------------------- 7. protocol
    emit("\n## 7. Protocol descriptives\n")
    emit("| Model | Regime | Runs | Mean tool calls | Used all 10 iterations % | Forced final-answer call % | Unified-diff answers | Path resolved | Retried API calls |")
    emit("|---|---|---|---|---|---|---|---|---|")
    protocol_rows = []
    for (model, mode), c in cells.items():
        f = c["frame"]
        emit(f"| {model} | {MODE_LABEL[mode]} | {len(f)} | {f['tool_calls'].mean():.1f} | {100 * f['cap_hit'].mean():.0f} | "
             f"{100 * f['fallback_used'].mean():.0f} | {int((f['answer_format'] == 'unified_diff').sum())} | {int(f['path_resolved'].sum())} | {int(f['api_retries'].sum())} |")
        protocol_rows.append({"model": model, "mode": mode, "runs": len(f), "tool_calls_mean": f["tool_calls"].mean(),
                              "cap_hit_share": f["cap_hit"].mean(), "forced_final_share": f["fallback_used"].mean(),
                              "unified_diff_answers": int((f["answer_format"] == "unified_diff").sum()),
                              "path_resolved": int(f["path_resolved"].sum()), "api_retries": int(f["api_retries"].sum()),
                              "patches": int(f["has_patch"].sum())})
    pd.DataFrame(protocol_rows).to_csv(out_dir / f"{prefix}__protocol.csv", index=False)
    for model in models:
        m = table[table["model_label"] == model]
        k = f"pr.{MODEL_KEY[model]}"
        fact(f"{k}.toolCalls", f"{m['tool_calls'].mean():.1f}")
        fact(f"{k}.capShare", f"{100 * m['cap_hit'].mean():.0f}")
        fact(f"{k}.finalRequestShare", f"{100 * m['termination_reason'].isin(['cap_hit', 'fallback_used']).mean():.0f}")
        fact(f"{k}.stoppedShare", f"{100 * (m['termination_reason'] == 'model_stopped').mean():.0f}")
        fact(f"{k}.iterationsMed", f"{m['iterations'].median():.0f}")
        fact(f"{k}.retries", int(m["api_retries"].sum()))
        fact(f"{k}.unifiedDiff", int((m["answer_format"] == "unified_diff").sum()))
        fact(f"{k}.modelIds", ", ".join(sorted(set(",".join(m["api_model_ids"].dropna()).split(",")) - {""})))
        fact(f"{k}.responseSecondsMed", f"{m['response_seconds'].median():.0f}")
    emit("\nIteration counts (all regimes):\n")
    emit(table.groupby("model_label")["iterations"].value_counts().unstack(fill_value=0).to_markdown())
    emit("\nAPI calls were made between " + str(table["started_at"].min())[:16] + " and " + str(table["completed_at"].max())[:16] + " UTC.")
    emit("Model ids returned by the APIs: " + "; ".join(f"{m}: {', '.join(sorted(set(','.join(g.dropna()).split(',')) - {''}))}"
                                                        for m, g in table.groupby("model_label")["api_model_ids"]))
    emit("Harness states: " + ", ".join(sorted(str(h) for h in table["harness"].dropna().unique())))
    emit("\nInstances per repository: " + ", ".join(f"{repo} {count}" for repo, count in
                                                     table.drop_duplicates("instance_id")["repo"].value_counts().items()))

    import json
    (out_dir / f"{prefix}__facts.json").write_text(json.dumps(facts, indent=1) + "\n")
    report = "\n".join(lines) + "\n"
    (out_dir / f"{prefix}__report.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
