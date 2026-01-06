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

*Last updated: 2026-01-06*
