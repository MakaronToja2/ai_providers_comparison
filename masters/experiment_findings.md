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

### Experiment 1: "Baseline with Escape Hatch" (Current)
**Status:** Running on 300 instances (lite split)

**Configuration:**
- Allows `<<<CANNOT_SOLVE>>>` option
- Models can choose to not produce a patch if uncertain

**Hypothesis:** OpenAI will confidently produce patches while Anthropic/Google will be more conservative and use the escape hatch more often.

**Preliminary Finding:** Confirmed - OpenAI produces patches 91-100% of time, Anthropic/Google only 40% even with forcing mechanism.

**Branch:** Current main branch (checkpointed before Experiment 2)

---

### Experiment 2: "Forced Patch Generation"
**Status:** Planned

**Configuration:**
- Remove `<<<CANNOT_SOLVE>>>` option entirely
- Models MUST produce a diff patch, no exceptions
- More aggressive forcing prompt

**Hypothesis:** When forced to always produce patches:
1. Conservative models (Anthropic/Google) may produce lower quality patches
2. Patch generation rate will increase to 80-90%
3. Quality/accuracy will vary by provider

**Questions to Answer:**
- Do conservative models produce worse patches when forced?
- Is OpenAI's confidence justified by higher accuracy?
- What's the quality vs quantity tradeoff?

---

### Experiment 3: "Patch Evaluation with SWE-bench Harness"
**Status:** Planned (after Experiments 1 & 2)

**Configuration:**
- Run official SWE-bench evaluation harness on generated patches
- Test if patches actually fix the bugs (pass test cases)

**Metrics:**
- **Resolve Rate**: % of patches that pass all tests
- **Partial Success**: % of patches that pass some tests
- **Regression Rate**: % of patches that break existing tests

**Questions to Answer:**
- Which provider produces patches that actually work?
- Is high patch generation rate correlated with accuracy?
- Do forced patches have lower resolve rate?

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

*Last updated: 2026-01-06*
