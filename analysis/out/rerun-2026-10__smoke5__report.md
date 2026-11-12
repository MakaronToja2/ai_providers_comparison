# Results: rerun-2026-10, smoke5 set (5 instances)

## 1. Patches, resolved, success and precision per model and regime

Per repetition counts are out of the instances run; mean ± SD over repetitions. Success = resolved / instances. Precision = resolved / patches. 95% intervals: bootstrap over instances.

| Model | Regime | Reps | Patches | Resolved | Success % [95% CI] | Precision % [95% CI] |
|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | Baseline | 1 | 5.0 | 3.0 | 60.0 [20.0, 100.0] | 60.0 [20.0, 100.0] |
| Claude Haiku 4.5 | Forced | 1 | 5.0 | 3.0 | 60.0 [20.0, 100.0] | 60.0 [20.0, 100.0] |
| Claude Haiku 4.5 | Self-reflection | 1 | 4.0 | 2.0 | 40.0 [0.0, 80.0] | 50.0 [0.0, 100.0] |
| Gemini 2.5 Flash-Lite | Baseline | 1 | 3.0 | 2.0 | 40.0 [0.0, 80.0] | 66.7 [0.0, 100.0] |
| Gemini 2.5 Flash-Lite | Forced | 1 | 2.0 | 1.0 | 20.0 [0.0, 60.0] | 50.0 [0.0, 100.0] |
| Gemini 2.5 Flash-Lite | Self-reflection | 1 | 2.0 | 2.0 | 40.0 [0.0, 80.0] | 100.0 [100.0, 100.0] |

## 2. What happened to each instance (mean count per repetition)

| Model | Regime | resolved | tests_failed | edit_not_applicable | no_answer |
|---|---|---|---|---|---|
| Claude Haiku 4.5 | Baseline | 3.0 | 2.0 | 0.0 | 0.0 |
| Claude Haiku 4.5 | Forced | 3.0 | 2.0 | 0.0 | 0.0 |
| Claude Haiku 4.5 | Self-reflection | 2.0 | 2.0 | 1.0 | 0.0 |
| Gemini 2.5 Flash-Lite | Baseline | 2.0 | 1.0 | 1.0 | 1.0 |
| Gemini 2.5 Flash-Lite | Forced | 1.0 | 1.0 | 2.0 | 1.0 |
| Gemini 2.5 Flash-Lite | Self-reflection | 2.0 | 0.0 | 1.0 | 2.0 |

Patches that applied but failed tests, by kind (all repetitions): f2p_failed = the target tests still fail; p2p_regression = a previously passing test broke; both.

|                                       |   f2p_failed |
|:--------------------------------------|-------------:|
| ('Claude Haiku 4.5', 'baseline')      |            2 |
| ('Claude Haiku 4.5', 'forced')        |            2 |
| ('Claude Haiku 4.5', 'self_reflect')  |            2 |
| ('Gemini 2.5 Flash-Lite', 'baseline') |            1 |
| ('Gemini 2.5 Flash-Lite', 'forced')   |            1 |

Edits that could not be applied, by reason (final answer; all repetitions):

| model_label           |   search_text_not_found |
|:----------------------|------------------------:|
| Claude Haiku 4.5      |                       1 |
| Gemini 2.5 Flash-Lite |                       4 |

## 3. Paired comparisons (same instances)

Difference in success (first minus second) with a bootstrap 95% interval over instances; McNemar's exact test on one value per instance (resolved in the majority of repetitions). 'only first' / 'only second' are the discordant instances.

| Comparison | Within | Δ success, points [95% CI] | only first | only second | McNemar p | Holm p |
|---|---|---|---|---|---|---|
| Baseline vs Forced | Claude Haiku 4.5 | +0.0 [+0.0, +0.0] | 0 | 0 | 1.000 | 1.000 |
| Baseline vs Self-reflection | Claude Haiku 4.5 | +20.0 [+0.0, +60.0] | 1 | 0 | 1.000 | 1.000 |
| Forced vs Self-reflection | Claude Haiku 4.5 | +20.0 [+0.0, +60.0] | 1 | 0 | 1.000 | 1.000 |
| Baseline vs Forced | Gemini 2.5 Flash-Lite | +20.0 [+0.0, +60.0] | 1 | 0 | 1.000 | 1.000 |
| Baseline vs Self-reflection | Gemini 2.5 Flash-Lite | +0.0 [+0.0, +0.0] | 0 | 0 | 1.000 | 1.000 |
| Forced vs Self-reflection | Gemini 2.5 Flash-Lite | -20.0 [-60.0, +0.0] | 0 | 1 | 1.000 | 1.000 |
| Claude Haiku 4.5 vs Gemini 2.5 Flash-Lite | Baseline | +20.0 [+0.0, +60.0] | 1 | 0 | 1.000 | 1.000 |
| Claude Haiku 4.5 vs Gemini 2.5 Flash-Lite | Forced | +40.0 [+0.0, +80.0] | 2 | 0 | 0.500 | 1.000 |
| Claude Haiku 4.5 vs Gemini 2.5 Flash-Lite | Self-reflection | +0.0 [+0.0, +0.0] | 0 | 0 | 1.000 | 1.000 |

Holm correction is applied separately to the regime comparisons and to the model comparisons.

Cochran's Q across the three regimes, per model (majority outcome per instance):

- Claude Haiku 4.5: Q = 2.00, df = 2, p = 0.368
- Gemini 2.5 Flash-Lite: Q = 2.00, df = 2, p = 0.368

## 4. Self-reflection: first patch versus final patch (same runs)

| Model | Reflection calls | approved | revised | no verdict | resolved before | resolved after | gained | lost | McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | 5 | 3 | 1 | 1 | 2 | 2 | 0 | 0 | 1.000 |
| Gemini 2.5 Flash-Lite | 3 | 2 | 1 | 0 | 2 | 2 | 0 | 0 | 1.000 |

Counts pool the repetitions. 'gained' = unresolved first patch, resolved final patch; 'lost' = the reverse.

| model_label           |   revised |   revision produced no patch |
|:----------------------|----------:|-----------------------------:|
| Claude Haiku 4.5      |         1 |                            0 |
| Gemini 2.5 Flash-Lite |         1 |                            1 |

## 5. Abstention and forcing

Explicit abstentions (all repetitions together):

| model_label           |   baseline |   forced |   self_reflect |
|:----------------------|-----------:|---------:|---------------:|
| Claude Haiku 4.5      |          0 |        0 |              0 |
| Gemini 2.5 Flash-Lite |          0 |        0 |              0 |

## 6. Tokens and cost per instance

Input and output tokens include the reflection call. Cost at list price ignores cache discounts; 'as billed' prices the input tokens the provider reports as cached at its cached rate.

| Model | Regime | Input tokens median [IQR] | SD | Output median | Cached input % | $/instance list | $/instance billed | $/resolved list | $/resolved billed |
|---|---|---|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | Baseline | 59,961 [54,430–95,546] | 36,639 | 2,405 | 0 | 0.0825 | 0.0825 | 0.138 | 0.138 |
| Claude Haiku 4.5 | Forced | 67,725 [54,914–89,690] | 36,443 | 2,312 | 0 | 0.0829 | 0.0829 | 0.138 | 0.138 |
| Claude Haiku 4.5 | Self-reflection | 79,726 [56,789–112,457] | 35,257 | 2,804 | 0 | 0.0974 | 0.0974 | 0.244 | 0.244 |
| Gemini 2.5 Flash-Lite | Baseline | 69,276 [13,978–73,806] | 40,345 | 1,572 | 49 | 0.0064 | 0.0041 | 0.016 | 0.010 |
| Gemini 2.5 Flash-Lite | Forced | 46,967 [18,002–71,164] | 31,443 | 1,016 | 36 | 0.0051 | 0.0037 | 0.026 | 0.019 |
| Gemini 2.5 Flash-Lite | Self-reflection | 18,576 [18,507–97,599] | 49,099 | 8,191 | 44 | 0.0075 | 0.0056 | 0.019 | 0.014 |

Reflection call alone (runs where it happened), mean tokens: Claude Haiku 4.5: 12,600 in / 639 out; Gemini 2.5 Flash-Lite: 9,142 in / 1,411 out

Do runs that use more tokens resolve more? Within each model, baseline regime: input tokens of resolved vs unresolved runs (Mann–Whitney U; rank-biserial r > 0 means resolved runs used more).


## 7. Protocol descriptives

| Model | Regime | Runs | Mean tool calls | Used all 10 iterations % | Forced final-answer call % | Unified-diff answers | Path resolved | Retried API calls |
|---|---|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | Baseline | 5 | 9.8 | 40 | 40 | 0 | 0 | 0 |
| Claude Haiku 4.5 | Forced | 5 | 9.6 | 40 | 80 | 0 | 0 | 0 |
| Claude Haiku 4.5 | Self-reflection | 5 | 10.0 | 40 | 40 | 0 | 0 | 0 |
| Gemini 2.5 Flash-Lite | Baseline | 5 | 6.2 | 40 | 100 | 0 | 2 | 0 |
| Gemini 2.5 Flash-Lite | Forced | 5 | 6.2 | 40 | 100 | 0 | 0 | 0 |
| Gemini 2.5 Flash-Lite | Self-reflection | 5 | 5.2 | 40 | 80 | 0 | 1 | 0 |

Iteration counts (all regimes):

| model_label           |   2 |   4 |   5 |   6 |   7 |   8 |   9 |   10 |
|:----------------------|----:|----:|----:|----:|----:|----:|----:|-----:|
| Claude Haiku 4.5      |   0 |   1 |   0 |   1 |   1 |   1 |   2 |    9 |
| Gemini 2.5 Flash-Lite |   4 |   1 |   2 |   0 |   1 |   1 |   0 |    6 |

API calls were made between 2026-10-10T08:10 and 2026-10-10T08:34 UTC.
Model ids returned by the APIs: Claude Haiku 4.5: claude-haiku-4-5-20251001; Gemini 2.5 Flash-Lite: gemini-2.5-flash-lite
Harness states: df830ca3b811f3ce273099cbcc89666d5a22f7c9+dirty.3ec5f30592d7, df830ca3b811f3ce273099cbcc89666d5a22f7c9+dirty.42c0b5f1d82f

Instances per repository: django__django 2, matplotlib__matplotlib 1, pytest-dev__pytest 1, scikit-learn__scikit-learn 1
