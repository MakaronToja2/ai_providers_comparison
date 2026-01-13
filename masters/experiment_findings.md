# SWE-Bench Provider Comparison - Experiment Findings

## Overview
Comparison of AI providers (OpenAI, Anthropic, Google) on SWE-bench bug-fixing tasks using an agentic tool loop (ReAct pattern).

---

## Key Findings

### Initial Results (Second23Instances) - Before Fixes
| Provider   | Success Rate |
|------------|-------------|
| OpenAI     | 91.3%       |
| Anthropic  | 21.7%       |
| Google     | 8.7%        |

**Overall patch rate: 39.1%** with many format errors (HALLUCINATION_FORMAT_ERROR)

### After Fixes (Firth23Instances)
| Provider   | Success Rate | Change      |
|------------|-------------|-------------|
| OpenAI     | 100%        | +8.7 pp     |
| Anthropic  | 40%         | +18.3 pp    |
| Google     | 40%         | +31.3 pp    |

---

## FINAL EXPERIMENT RESULTS (SWE-bench Lite - 100 instances)

### Summary Table

| Experiment | Total Patches | Evaluated | Resolved | Rate |
|------------|---------------|-----------|----------|------|
| Exp1: CANNOT_SOLVE allowed | 228/300 | 228 | 79 | **34.6%** |
| Exp2: Forced patch | 216/300 | 212 | 55 | **25.9%** |
| Exp3: Self-reflect | 217/300 | 217 | 28 | **12.9%** |

### Results by Provider

#### Experiment 1: Baseline (CANNOT_SOLVE allowed)
| Provider | Patches | Resolved | Rate |
|----------|---------|----------|------|
| Anthropic (claude-haiku-4-5) | 84/100 | 39 | **46.4%** |
| OpenAI (gpt-4.1-mini) | 100/100 | 31 | **31.0%** |
| Google (gemini-2.5-flash-lite) | 44/100 | 9 | **20.5%** |

#### Experiment 2: Forced Patch Generation
| Provider | Patches | Resolved | Rate |
|----------|---------|----------|------|
| OpenAI (gpt-4.1-mini) | 100/100 | 30 | **30.0%** |
| Anthropic (claude-haiku-4-5) | 76/100 | 22 | **28.9%** |
| Google (gemini-2.5-flash-lite) | 40/100 | 3 | **7.5%** |

#### Experiment 3: Self-Reflect with Test Results
| Provider | Patches | Resolved | Rate |
|----------|---------|----------|------|
| Anthropic (claude-haiku-4-5) | 79/100 | 15 | **19.0%** |
| Google (gemini-2.5-flash-lite) | 39/100 | 4 | **10.3%** |
| OpenAI (gpt-4.1-mini) | 99/100 | 9 | **9.1%** |

---

## Key Observations

### 1. Anthropic Performs Best Overall
- In Experiment 1, Anthropic achieved **46.4%** resolve rate - significantly higher than OpenAI (31%) and Google (20.5%)
- Anthropic is more conservative about generating patches (84/100 vs OpenAI's 100/100), but when it does generate them, they are higher quality
- This suggests Anthropic's "honesty" about uncertainty actually correlates with better patch quality

### 2. Forcing Patches Hurts Quality
- Experiment 2 (forced) had **WORSE** results than Experiment 1 (25.9% vs 34.6%)
- Google dropped dramatically from 20.5% to 7.5% when forced
- Hypothesis: When models are forced to produce patches they're uncertain about, quality decreases
- Exception: OpenAI maintained ~30% in both experiments (already at 100% patch rate)

### 3. Self-Reflection Experiment Had Issues
- Experiment 3 performed worst at **12.9%** resolve rate
- **IMPORTANT CAVEAT**: This experiment may have been misconfigured
- The self-reflection phase may not have received sufficient context after test results
- All providers performed worse than their Experiment 1 baselines
- This needs investigation - the implementation may have introduced bugs

### 4. Google Struggles Most
- Consistently lowest resolve rates across all experiments
- Patch generation rate also lowest (40-44% vs 76-100% for others)
- May need different prompting strategy or has fundamental limitations on this task

### 5. Patch Generation vs Quality Trade-off
- OpenAI: Always generates patches (100%), moderate quality (30-31%)
- Anthropic: Selective about patches (76-84%), but highest quality when it does (28-46%)
- Google: Low patch rate AND low quality - double disadvantage

---

## Bugs Discovered and Fixed

### Bug 1: Content Lost During Tool Iterations
**Location:** `app/core/base_provider.py`

**Problem:** During the agentic tool loop, when a model made tool calls, any text content in that response was being discarded. Only the final iteration's content was kept.

**Impact:** Models that produced partial analysis/reasoning during tool calls lost all that context. This particularly affected Anthropic which often emits thinking alongside tool calls.

**Fix:** Accumulate ALL content parts throughout the loop:
```python
all_content_parts = []  # Accumulate ALL content throughout iterations

# In the loop:
if content:
    all_content_parts.append(content)

# After loop:
final_content = "\n\n".join(all_content_parts) if all_content_parts else ""
```

### Bug 2: max_tokens Too Low (Truncation)
**Location:** `app/benchmark/runner.py`

**Problem:** `max_tokens` was set to 4000, which was insufficient for models to explore code AND produce a complete diff patch.

**Impact:** Models were truncated mid-response, cutting off patches before completion.

**Fix:** Increased to 8000 tokens:
```python
response = await provider.generate_response(
    messages=messages,
    temperature=0.3,
    max_tokens=8000,  # Was 4000
)
```

### Bug 3: Models Stuck in Exploration Mode
**Location:** `app/core/base_provider.py`

**Problem:** Models would use tools to explore the codebase, then stop making tool calls, but never produce a patch. The "force final answer" logic only triggered at `max_tool_iterations` (10), but most failures stopped at 3-8 iterations.

**Impact:** Models explored correctly but silently failed to produce output.

**Fix:** Force final answer whenever:
1. Tools were available (so this was an agentic task)
2. Either tool calls were made OR some content was produced
3. No valid patch (````diff`) or explicit failure (`<<<CANNOT_SOLVE>>>`) was found

```python
should_force_answer = (
    not has_patch and
    not has_cannot_solve and
    tools_spec is not None and
    (all_tool_calls or final_content)
)

if should_force_answer:
    # Add message demanding final answer
    # Call WITHOUT tools to force text response
    raw_response = await self._make_api_call(..., tools=None)
```

---

## Provider-Specific Observations

### OpenAI (gpt-4.1-mini / gpt-4o-mini)
- Highest success rate (91-100%)
- Consistently follows output format instructions
- Produces patches even when uncertain
- Sometimes needs forcing but always succeeds when forced

### Anthropic (claude-haiku-4-5)
- Medium success rate after fixes (40%)
- Produces reasoning text alongside tool calls
- `content_len` during forcing: ~300 chars (has thinking)
- May need prompt tuning for better format compliance

### Google (gemini-2.5-flash-lite)
- Medium success rate after fixes (40%)
- Produces NO text during tool iterations (`content_len=0`)
- Tool schema conversion required (JSON Schema -> Google Schema)
- Occasionally has tool initialization failures
- The forcing mechanism is ESSENTIAL for Google

---

## Result Classification System

### ResultStatus Enum
```python
class ResultStatus(str, Enum):
    SUCCESS_PATCH_GENERATED = "success_patch_generated"      # Valid diff patch produced
    FAILURE_EXPLICIT = "failure_explicit"                    # Model used <<<CANNOT_SOLVE>>>
    HALLUCINATION_FORMAT_ERROR = "hallucination_format_error"  # No valid output format
    API_ERROR = "api_error"                                  # Provider API failed
```

### Expected Output Format
Models must produce either:
1. A unified diff patch in ```diff``` code block
2. `<<<CANNOT_SOLVE>>>` marker with reason and explanation

---

## Architecture Notes

### Agentic Tool Loop (ReAct Pattern)
1. Model receives problem + tool definitions
2. Model can call tools: `read_file`, `search_code`, `list_directory`
3. Tool results are fed back as user messages
4. Loop continues until: no tool calls OR max iterations (10)
5. If no valid output, force final answer without tools

### Tools Available
- `read_file(file_path)` - Read file contents
- `search_code(pattern)` - Regex search across codebase
- `list_directory(directory_path)` - List directory contents

---

## Prompts Used (by Experiment Mode)

### Common System Prompt Structure
All experiments use this base structure:
```
You are an expert software engineer tasked with fixing a bug from a GitHub issue.

## ANALYSIS LOOP
You have up to {max_tool_iterations} tool iterations to explore the codebase.
- Use tools efficiently (2-5 calls is usually sufficient)
- After exploration, you will be asked for your final answer
- The system will provide a summary of what you explored

## AVAILABLE TOOLS
- read_file(file_path): Read contents of a file
- search_code(pattern): Search for regex pattern across codebase
- list_directory(directory_path): List contents of a directory

## WORKFLOW
1. EXPLORE: Use tools to find relevant code
2. ANALYZE: Understand the bug's root cause
3. FIX: Generate a unified diff patch
```

### Experiment 1: BASELINE Mode
**Output Instructions:**
```
Your response MUST end with ONE of:

OPTION 1 - Produce a patch (PREFERRED):
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
...
```

OPTION 2 - Explicit failure (ONLY if you truly cannot attempt a fix):
<<<CANNOT_SOLVE>>>
reason: [insufficient_context|too_complex|unclear_requirements|missing_dependencies]
explanation: [brief explanation]

A partial or uncertain patch is BETTER than giving up.
```

**Forcing Prompt (when no patch produced):**
```
## Exploration Summary
[List of files read, searches made, directories listed]

FINAL ANSWER REQUIRED
You MUST output ONE of:
1. A ```diff``` patch with your fix (PREFERRED), OR
2. <<<CANNOT_SOLVE>>> with reason (only if you truly cannot attempt a fix)

Do not ask for more information. Make your best attempt NOW.
```

### Experiment 2: FORCED Mode
**Output Instructions:**
```
## OUTPUT REQUIREMENTS (MANDATORY)
You MUST produce a patch. There is NO option to give up.

Your response MUST contain a unified diff patch.
Even if you are uncertain, provide your BEST GUESS patch.
DO NOT say you cannot solve this. You MUST output a diff.
```

**Forcing Prompt (when no patch produced):**
```
## Exploration Summary
[List of files read, searches made, directories listed]

FINAL ANSWER REQUIRED - NO ESCAPE OPTION

Based on your exploration, you MUST now generate a patch.
- DO NOT say you cannot solve this
- DO NOT ask for more information
- Provide your BEST GUESS even if uncertain

Output a unified diff patch NOW.
```

### Experiment 3: SELF_REFLECT Mode
Uses same initial prompts as BASELINE, but adds a reflection phase after patch generation.

**Initial Phase:** Same as BASELINE (allows `<<<CANNOT_SOLVE>>>`)

**Reflection Phase (after patch is generated):**
```
## SELF-REFLECTION PHASE

You just generated the following patch:
[The patch content]

Now critically review your own patch:

1. **Correctness**: Does the patch actually fix the reported bug?
2. **Completeness**: Are there any edge cases or related code that also needs changes?
3. **Side Effects**: Could this change break anything else?
4. **Syntax**: Is the diff format correct? Are line numbers plausible?

After your review, you MUST output ONE of:

**If the patch is good:**
<<<PATCH_APPROVED>>>
[Optional brief explanation of why it's correct]

**If you want to revise:**
<<<PATCH_REVISED>>>
```diff
[Your revised patch here]
```
[Brief explanation of what you changed and why]

Do NOT simply repeat the same patch. Either approve it or make meaningful improvements.
```

**Output Markers:**
- `<<<PATCH_APPROVED>>>` - Model reviewed and approved its own patch
- `<<<PATCH_REVISED>>>` - Model revised its patch after reflection

**Metrics to Track:**
- % of patches that were revised vs approved
- Did revision improve patch quality?
- Additional token cost of reflection phase

---

## Recommendations for Further Investigation

1. **Prompt Engineering**: Test different system prompts to improve Anthropic/Google format compliance

2. **Temperature Tuning**: Current 0.3 may be suboptimal for some providers

3. **Tool Call Patterns**: Analyze which tools each provider uses and whether certain patterns correlate with success

4. **Error Analysis**: Deep-dive into the 60% failures for Anthropic/Google to understand common failure modes

5. **Larger Dataset**: Run on full lite split (300 instances) for statistically significant results

---

## Experiment Configuration

- **Dataset**: SWE-bench dev split (23 instances for testing)
- **Temperature**: 0.3
- **Max Tokens**: 8000
- **Max Tool Iterations**: 10
- **Concurrency**: 3 (to avoid rate limits)

---

## Thesis Experiment Plan

### Experiment 1: "Baseline with Escape Hatch"
**Status:** COMPLETED

**Configuration:**
- Allows `<<<CANNOT_SOLVE>>>` option
- Models can choose to not produce a patch if uncertain

**Results:**
- Total: 228/300 patches generated, 79 resolved (34.6%)
- Anthropic: 84 patches, 39 resolved (46.4%) - BEST
- OpenAI: 100 patches, 31 resolved (31.0%)
- Google: 44 patches, 9 resolved (20.5%)

**Findings:**
- Hypothesis CONFIRMED: OpenAI produces patches 100% of the time, while Anthropic (84%) and Google (44%) are more conservative
- SURPRISING: Anthropic's conservatism correlates with HIGHER quality - 46.4% vs 31% for OpenAI
- Being selective about when to produce patches appears to be a better strategy

---

### Experiment 2: "Forced Patch Generation"
**Status:** COMPLETED

**Configuration:**
- Remove `<<<CANNOT_SOLVE>>>` option entirely
- Models MUST produce a diff patch, no exceptions
- More aggressive forcing prompt

**Results:**
- Total: 216/300 patches generated, 55 resolved (25.9%)
- OpenAI: 100 patches, 30 resolved (30.0%)
- Anthropic: 76 patches, 22 resolved (28.9%)
- Google: 40 patches, 3 resolved (7.5%)

**Findings:**
- Hypothesis PARTIALLY CONFIRMED: Conservative models do produce worse patches when forced
- UNEXPECTED: Overall resolve rate DROPPED from 34.6% to 25.9%
- Google suffered most - dropped from 20.5% to 7.5%
- OpenAI maintained performance (~30%) since it was already generating 100% patches
- Anthropic dropped from 46.4% to 28.9% - significant quality loss when forced
- Forcing patches is COUNTERPRODUCTIVE - quality suffers

---

### Experiment 3: "Self-Reflection Strategy"
**Status:** COMPLETED (with caveats)

**Configuration:**
- Phase 1: Model generates initial patch
- Phase 2: Model receives test results and can revise
- Phase 3: Model critiques and optionally revises patch

**Results:**
- Total: 217/300 patches generated, 28 resolved (12.9%)
- Anthropic: 79 patches, 15 resolved (19.0%)
- Google: 39 patches, 4 resolved (10.3%)
- OpenAI: 99 patches, 9 resolved (9.1%)

**CRITICAL ISSUES:**
- This experiment performed WORST - significantly below baseline
- All providers performed worse than their Experiment 1 results
- **Possible cause**: Implementation issue - self-reflection phase may not have received proper context
- The test results feedback may not have been integrated correctly into the reflection prompt
- Needs investigation before drawing conclusions about self-reflection strategy

**Questions Remaining:**
- Was the implementation correct?
- Did the model receive adequate context during reflection?
- Should be re-run with verified implementation

**Future Extension - Cross-Model Review:**
In a future version, we can extend this to cross-model reflection:
1. Model A generates patch (no self-reflection)
2. Model A reviews its own patch
3. Model B reviews Model A's patch
4. Model C reviews Model A's patch
5. Compare: Which model is the best "code reviewer"?

This would answer: *"Which model works best as a patch reviewer/checker?"* - valuable for PR review automation use cases.

---

### Evaluation Methodology

**How We Plan to Evaluate "Best":**

1. **Patch Generation Metrics (Experiments 1 & 2)**
   - Patch Rate: % of instances where a valid diff was produced
   - Format Compliance: % that follow the exact output format
   - Escape Hatch Usage: % that used `<<<CANNOT_SOLVE>>>`

2. **Patch Quality Metrics (SWE-bench Harness)**
   - Resolve Rate: % of patches that pass ALL test cases
   - Partial Success: % that pass SOME test cases
   - Regression Rate: % that break existing tests
   - Exact Match: % that match the gold patch exactly

3. **Cost-Effectiveness Metrics**
   - Tokens per patch (prompt + completion)
   - Cost per successful patch (USD)
   - Time per patch (seconds)

4. **Self-Reflection Metrics (Experiment 3)**
   - Delta in resolve rate (before vs after reflection)
   - Token overhead for reflection step
   - Cases where reflection helped vs hurt

**Evaluation Pipeline:**
```
Generated Patches --> SWE-bench Harness --> Test Results --> Analysis
                          |
                          v
                   Docker containers
                   Run actual tests
                   Compare to gold patch
```

---

### Additional Experiment Ideas

| Experiment | Variable | Question | Priority |
|------------|----------|----------|----------|
| Temperature Study | 0.1 vs 0.3 vs 0.7 | Does creativity help or hurt? | Medium |
| Tools vs No Tools | Enable/disable agentic loop | How much do tools help? | High |
| Model Tiers | GPT-4 vs GPT-4o-mini, Claude Sonnet vs Haiku | Cost vs quality tradeoff | High |
| Prompt Styles | Direct vs Chain-of-thought | Does explicit reasoning help? | Medium |
| Context Size | With/without hints | How much context matters? | Low |
| Token Budget | 4000 vs 8000 vs 16000 | Does more space help? | Medium |

---

## Dataset Splits Available

| Split | Dataset | Instances | Use Case |
|-------|---------|-----------|----------|
| dev | SWE-bench_Lite | 23 | Quick testing, debugging |
| lite | SWE-bench_Lite | 300 | Main benchmark (thesis) |
| test | SWE-bench | 2294 | Full benchmark (if needed) |
| verified | SWE-bench_Verified | 500 | Human-verified subset |

---

## Key Observations So Far

1. **OpenAI follows instructions better**: Even without forcing, OpenAI produces patches consistently. This suggests better instruction-following capability.

2. **Anthropic/Google are more "honest"**: These models prefer to admit uncertainty rather than hallucinate. This is arguably better behavior, but harder to evaluate.

3. **Forcing is essential for comparison**: Without forcing, we can't compare patch quality across providers since Anthropic/Google often don't produce patches.

4. **Tool usage patterns differ**: Need to analyze if providers use tools differently and if that correlates with success.

---

*Last updated: 2026-01-12 (Final experiment results added)*
