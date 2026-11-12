from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import asyncio
import json
import random
import time

import httpx

from ..models.requests import ChatMessage, LLMRequest, ToolCallRequest, ToolResultMessage
from ..models.responses import LLMResponse, TokenUsage, ToolCall, ReflectionMetrics
from .edit_format import EDIT_FORMAT_INSTRUCTIONS, first_block_start, has_edit_block
from .tools import tool_registry
from .trajectory import TrajectoryRecorder, utc_now_iso
from ..config.settings import get_settings

CANNOT_SOLVE_MARKER = "<<<CANNOT_SOLVE>>>"
PATCH_APPROVED_MARKER = "<<<PATCH_APPROVED>>>"
PATCH_REVISED_MARKER = "<<<PATCH_REVISED>>>"

# Substrings the three providers use when a request exceeds the context window.
_CONTEXT_OVERFLOW_HINTS = (
    "context_length_exceeded",
    "maximum context length",
    "prompt is too long",
    "exceeds the maximum number of tokens",
    "input token count",
)


@dataclass
class ParsedResponse:
    """One model response in provider-neutral form."""
    text: str = ""
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    usage: Optional[TokenUsage] = None
    stop_reason: Optional[str] = None
    api_model_id: Optional[str] = None
    response_id: Optional[str] = None
    raw_usage: Optional[Dict[str, Any]] = None
    # Input tokens the provider reports as served from its prompt cache (billed at a lower rate)
    cached_tokens: int = 0
    native: Any = None


class ProviderCallError(Exception):
    """An API call failed for good: retries exhausted, a non-retryable error, or context overflow."""

    def __init__(self, kind: str, message: str, attempts: int = 0):
        super().__init__(message)
        self.kind = kind  # "api_error" or "context_overflow"
        self.attempts = attempts


@dataclass
class _RunStats:
    """Per-instance counters. Kept off the provider object, which is shared across tasks."""
    api_call_count: int = 0
    api_retry_count: int = 0
    max_input_tokens: int = 0
    cached_input_tokens: int = 0
    api_model_ids: List[str] = field(default_factory=list)
    stop_reasons: List[str] = field(default_factory=list)


class BaseAIProvider(ABC):
    """Abstract base class for all AI providers"""

    # API call policy (SDK-level retries are switched off so every attempt is logged here)
    max_api_attempts = 6
    retry_base_seconds = 2.0
    retry_max_seconds = 60.0
    request_timeout_seconds = 600.0

    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.config = kwargs
        self._validate_api_key()
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider"""
        pass
    
    @abstractmethod
    def _validate_api_key(self) -> None:
        """Validate the API key"""
        pass
    
    @abstractmethod
    async def _make_api_call(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        **kwargs
    ) -> Any:
        """Make the actual API call to the provider.

        tool_choice is "auto" or "none". With "none" the tools stay declared (the history
        contains native tool turns) but the model must answer in text.
        """
        pass

    @abstractmethod
    def _parse_response(self, raw_response: Any) -> ParsedResponse:
        """Parse the provider's response into standardized format"""
        pass

    def _is_retryable(self, exc: Exception) -> bool:
        """Whether a failed API call should be retried. Providers extend this."""
        return isinstance(exc, (httpx.TransportError, asyncio.TimeoutError))

    def _is_context_overflow(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(hint in message for hint in _CONTEXT_OVERFLOW_HINTS)

    def _retry_delay(self, attempt: int) -> float:
        """Jittered exponential backoff before the next attempt."""
        ceiling = min(self.retry_max_seconds, self.retry_base_seconds * (2 ** (attempt - 1)))
        return ceiling * random.uniform(0.5, 1.0)

    async def _call_model(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: str,
        phase: str,
        iteration: int,
        recorder: TrajectoryRecorder,
        stats: _RunStats,
        **kwargs
    ) -> ParsedResponse:
        """One logical model call: retries transient failures, logs every attempt."""
        for attempt in range(1, self.max_api_attempts + 1):
            ts_start = utc_now_iso()
            try:
                raw_response = await self._make_api_call(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs
                )
                parsed = self._parse_response(raw_response)
            except Exception as exc:
                overflow = self._is_context_overflow(exc)
                retryable = not overflow and self._is_retryable(exc)
                recorder.api_call(
                    phase=phase, iteration=iteration, attempt=attempt, ts_start=ts_start,
                    requested_model=self.model, tool_choice=tool_choice,
                    error=f"{type(exc).__name__}: {exc}", retryable=retryable,
                )
                if overflow:
                    raise ProviderCallError("context_overflow", str(exc), attempt) from exc
                if not retryable or attempt == self.max_api_attempts:
                    raise ProviderCallError("api_error", f"{type(exc).__name__}: {exc}", attempt) from exc
                stats.api_retry_count += 1
                delay = self._retry_delay(attempt)
                print(f"[{self.provider_name}] API call failed (attempt {attempt}/{self.max_api_attempts}), retrying in {delay:.1f}s: {exc}")
                await asyncio.sleep(delay)
                continue

            stats.api_call_count += 1
            stats.cached_input_tokens += parsed.cached_tokens or 0
            if parsed.usage:
                stats.max_input_tokens = max(stats.max_input_tokens, parsed.usage.prompt_tokens)
            if parsed.api_model_id and parsed.api_model_id not in stats.api_model_ids:
                stats.api_model_ids.append(parsed.api_model_id)
            if parsed.stop_reason:
                stats.stop_reasons.append(parsed.stop_reason)
            recorder.api_call(
                phase=phase, iteration=iteration, attempt=attempt, ts_start=ts_start,
                requested_model=self.model, api_model_id=parsed.api_model_id,
                response_id=parsed.response_id, stop_reason=parsed.stop_reason,
                tool_choice=tool_choice, usage=parsed.raw_usage,
                assistant_text=parsed.text,
                tool_calls=[tc.model_dump() for tc in parsed.tool_calls],
            )
            return parsed

        raise ProviderCallError("api_error", "retry loop ended without a response", self.max_api_attempts)

    def _coerce_arguments(self, tool: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce argument values to the types the tool schema declares.

        Providers do not all return JSON integers as integers (Gemini returns 500.0 for 500),
        and models sometimes quote numbers. Values that cannot be coerced are left as sent.
        """
        properties = (tool.parameters_schema or {}).get("properties", {})
        coerced = dict(parameters)
        for name, value in parameters.items():
            declared = properties.get(name, {}).get("type")
            if declared == "integer" and not isinstance(value, bool):
                if isinstance(value, float) and value.is_integer():
                    coerced[name] = int(value)
                elif isinstance(value, str) and value.strip().lstrip("+-").isdigit():
                    coerced[name] = int(value.strip())
            elif declared == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
                coerced[name] = value.strip().lower() == "true"
        return coerced

    async def _execute_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> ToolCall:
        """Execute a tool call and return the result"""
        settings = get_settings()
        
        # Get the tool from registry
        tool = tool_registry.get_tool(tool_name)
        if not tool:
            return ToolCall(
                name=tool_name,
                parameters=parameters,
                success=False,
                error=f"Tool '{tool_name}' not found or not enabled"
            )
        
        parameters = self._coerce_arguments(tool, parameters)

        try:
            # Execute the tool
            result = await tool.execute(**parameters)
            
            # Log tool call if debugging is enabled
            if settings.tool_debug:
                print(f"Tool call: {tool_name} with {parameters}")
                print(f"Tool result: {result.model_dump()}")
            
            return ToolCall(
                name=tool_name,
                parameters=parameters,
                result=result.model_dump(),
                success=result.success,
                error=result.error,
                rendered=tool.render_result(result.data) if result.success else None,
            )
            
        except Exception as e:
            return ToolCall(
                name=tool_name,
                parameters=parameters,
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )

    def _build_exploration_summary(self, tool_calls: List[ToolCall]) -> str:
        """Build a summary of what was explored during tool iterations.

        This summary is provided to the model before forcing a final answer,
        giving it context about what information it gathered.
        """
        if not tool_calls:
            return "You have not explored any files yet."

        # Group tool calls by type
        files_read = []
        searches_made = []
        directories_listed = []

        for tc in tool_calls:
            if tc.name == "read_file":
                file_path = tc.parameters.get("file_path", "unknown")
                if tc.success:
                    files_read.append(file_path)
                else:
                    files_read.append(f"{file_path} (failed: {tc.error})")
            elif tc.name == "search_code":
                pattern = tc.parameters.get("pattern", "unknown")
                if tc.success and tc.result:
                    match_count = tc.result.get("data", {}).get("total_matches", 0)
                    searches_made.append(f"'{pattern}' ({match_count} matches)")
                else:
                    searches_made.append(f"'{pattern}' (no results)")
            elif tc.name == "list_directory":
                dir_path = tc.parameters.get("directory_path", "unknown")
                directories_listed.append(dir_path)

        # Build summary text
        summary_parts = ["## Exploration Summary"]

        if files_read:
            summary_parts.append(f"\n**Files Read ({len(files_read)}):**")
            for f in files_read[:10]:  # Limit to first 10
                summary_parts.append(f"- {f}")
            if len(files_read) > 10:
                summary_parts.append(f"- ... and {len(files_read) - 10} more")

        if searches_made:
            summary_parts.append(f"\n**Searches Made ({len(searches_made)}):**")
            for s in searches_made[:5]:  # Limit to first 5
                summary_parts.append(f"- {s}")

        if directories_listed:
            summary_parts.append(f"\n**Directories Listed ({len(directories_listed)}):**")
            for d in directories_listed[:5]:
                summary_parts.append(f"- {d}")

        summary_parts.append(f"\n**Total Tool Calls:** {len(tool_calls)}")

        return "\n".join(summary_parts)

    @staticmethod
    def _has_diff(text: str) -> bool:
        return "```diff" in text or "--- a/" in text or "+++ b/" in text

    @classmethod
    def _has_patch(cls, text: str) -> bool:
        """Whether an answer contains a patch: edit blocks (the required format) or,
        if the model wrote one anyway, a unified diff."""
        return has_edit_block(text) or cls._has_diff(text)

    @staticmethod
    def _tool_result_text(tc: ToolCall) -> str:
        """Text of one tool result as the model sees it (identical for all providers).

        Successful results are the tool's own plain-text rendering, so source code reaches
        the model with real line ends and quotes rather than as an escaped JSON string.
        """
        if tc.success:
            if tc.rendered is not None:
                return tc.rendered
            return json.dumps(tc.result, indent=2) if tc.result else "Success"
        return f"Error: {tc.error}"

    @staticmethod
    def _add_usage(total: TokenUsage, usage: Optional[TokenUsage]) -> None:
        if usage:
            total.prompt_tokens += usage.prompt_tokens
            total.completion_tokens += usage.completion_tokens
            total.total_tokens += usage.total_tokens

    async def _do_self_reflection(
        self,
        final_answer: str,
        conversation: List[ChatMessage],
        temperature: float,
        max_tokens: int,
        tools_spec: Optional[List[Dict[str, Any]]],
        recorder: TrajectoryRecorder,
        stats: _RunStats,
        **kwargs
    ) -> tuple[str, ReflectionMetrics]:
        """Single-shot self-reflection: the model reviews its own patch and approves or revises it.

        The model gets the full conversation (including everything it read) and no new
        information: no tests are run and no tools can be called. Tokens are tracked
        separately from the main loop. An API failure here propagates, so the instance is
        recorded as an API error and re-run rather than silently keeping the first patch.
        """
        patch_start = first_block_start(final_answer)
        if patch_start == -1:
            patch_start = final_answer.find("```diff")
        if patch_start == -1:
            patch_start = final_answer.find("--- a/")
        if patch_start == -1:
            patch_start = 0

        reflection_prompt = f"""## SELF-REFLECTION PHASE

You just generated the following patch:

{final_answer[patch_start:]}

Now critically review your own patch:

1. **Correctness**: Does the patch actually fix the reported bug?
2. **Completeness**: Are there any edge cases or related code that also needs changes?
3. **Side Effects**: Could this change break anything else?
4. **Format**: Does every SEARCH part copy the existing code exactly?

After your review, you MUST output ONE of:

**If the patch is good:**
<<<PATCH_APPROVED>>>
[Optional brief explanation of why it's correct]

**If you want to revise:**
<<<PATCH_REVISED>>>
[Your complete revised patch as edit blocks, in the same format as before]
[Brief explanation of what you changed and why]

Do NOT simply repeat the same patch. Either approve it or make meaningful improvements."""

        conversation.append(ChatMessage(role="user", content=reflection_prompt))

        parsed = await self._call_model(
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools_spec,
            tool_choice="none",
            phase="reflection",
            iteration=0,
            recorder=recorder,
            stats=stats,
            **kwargs
        )
        conversation.append(ChatMessage(role="assistant", content=parsed.text, native=parsed.native))

        # Provisional outcome from the markers; the runner confirms "revised" only when a
        # valid diff follows the marker.
        if PATCH_REVISED_MARKER in parsed.text:
            outcome = "revised"
        elif PATCH_APPROVED_MARKER in parsed.text:
            outcome = "approved"
        else:
            outcome = "unstructured"
        print(f"[{self.provider_name}] Self-reflect: {outcome}")

        usage = parsed.usage
        return parsed.text, ReflectionMetrics(
            outcome=outcome,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

    async def generate_response(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_tools: bool = True,
        max_tool_iterations: int = 10,
        experiment_mode: str = "baseline",  # "baseline", "forced", or "self_reflect"
        recorder: Optional[TrajectoryRecorder] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response with agentic tool loop and standardized output"""
        start_time = time.time()
        all_tool_calls = []
        total_usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        conversation = list(messages)  # Copy messages for the loop
        recorder = recorder or TrajectoryRecorder()
        stats = _RunStats()
        iteration = 0

        def run_meta(**extra: Any) -> Dict[str, Any]:
            return {
                "iterations": iteration,
                "api_call_count": stats.api_call_count,
                "api_retry_count": stats.api_retry_count,
                "api_model_ids": list(stats.api_model_ids),
                "max_input_tokens": stats.max_input_tokens,
                "cached_input_tokens": stats.cached_input_tokens,
                "stop_reasons": list(stats.stop_reasons),
                **extra,
            }

        try:
            # Get available tools if enabled
            tools_spec = None
            if use_tools:
                tool_definitions = tool_registry.get_tool_definitions()
                if tool_definitions:
                    tools_spec = [tool.model_dump() for tool in tool_definitions]

            all_content_parts = []  # Accumulate ALL content throughout iterations
            final_answer = ""  # Text of the message in which the model stopped calling tools
            cap_hit = False

            while iteration < max_tool_iterations:
                iteration += 1

                parsed = await self._call_model(
                    messages=conversation,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools_spec,
                    tool_choice="auto",
                    phase="loop",
                    iteration=iteration,
                    recorder=recorder,
                    stats=stats,
                    **kwargs
                )
                self._add_usage(total_usage, parsed.usage)

                if parsed.text:
                    all_content_parts.append(parsed.text)

                # The model's own turn always goes into the history, tool calls included
                conversation.append(ChatMessage(
                    role="assistant",
                    content=parsed.text,
                    tool_calls=parsed.tool_calls or None,
                    native=parsed.native,
                ))

                # If no tool calls, this message is the model's answer
                if not parsed.tool_calls:
                    final_answer = parsed.text
                    break

                # Execute tool calls and return each result as a native tool-result turn
                tool_results = []
                for call in parsed.tool_calls:
                    if call.parse_error:
                        executed_call = ToolCall(
                            name=call.name, parameters={}, success=False,
                            error=f"Invalid tool arguments: {call.parse_error}",
                        )
                    else:
                        executed_call = await self._execute_tool_call(call.name, call.arguments)
                    all_tool_calls.append(executed_call)
                    result_text = self._tool_result_text(executed_call)
                    tool_results.append(ToolResultMessage(
                        tool_call_id=call.id,
                        name=call.name,
                        content=result_text,
                        is_error=not executed_call.success,
                    ))
                    recorder.tool_result(
                        iteration=iteration, tool_call_id=call.id, name=call.name,
                        arguments=call.arguments, success=executed_call.success,
                        error=executed_call.error, content=result_text,
                    )
                conversation.append(ChatMessage(role="tool", tool_results=tool_results))
            else:
                # Iteration cap reached while the model was still calling tools
                cap_hit = max_tool_iterations > 0

            # A final answer must contain a patch or the explicit failure marker. Only the
            # final answer message counts: drafts written during exploration do not.
            has_patch = self._has_patch(final_answer)
            has_cannot_solve = CANNOT_SOLVE_MARKER in final_answer

            # If there is no valid final answer, make one more call that demands it.
            # Tools stay declared (the history holds native tool turns) but cannot be called.
            fallback_used = tools_spec is not None and not has_patch and not has_cannot_solve

            if fallback_used:
                # Log that we're forcing a final answer
                print(f"[{self.provider_name}] Forcing final answer: iterations={iteration}, tool_calls={len(all_tool_calls)}, cap_hit={cap_hit}, mode={experiment_mode}")

                # Build detailed exploration summary
                exploration_summary = self._build_exploration_summary(all_tool_calls)

                # Mode-aware forcing prompt
                if experiment_mode == "forced":
                    # FORCED mode: No escape hatch, must produce patch
                    forcing_prompt = f"""{exploration_summary}

FINAL ANSWER REQUIRED - NO ESCAPE OPTION

Based on your exploration, you MUST now generate a patch.
- DO NOT say you cannot solve this
- DO NOT ask for more information
- Provide your BEST GUESS even if uncertain

Output your patch NOW.

{EDIT_FORMAT_INSTRUCTIONS}"""
                else:
                    # BASELINE or SELF_REFLECT: Allow escape hatch
                    forcing_prompt = f"""{exploration_summary}

FINAL ANSWER REQUIRED

Based on your exploration, provide your final answer.
You MUST output ONE of:

1. A patch with your fix (PREFERRED), OR
2. <<<CANNOT_SOLVE>>> with reason and explanation (only if you truly cannot attempt a fix)

Do not ask for more information. Make your best attempt at a patch NOW.

{EDIT_FORMAT_INSTRUCTIONS}"""

                # Add forcing message
                conversation.append(ChatMessage(role="user", content=forcing_prompt))

                parsed = await self._call_model(
                    messages=conversation,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools_spec,
                    tool_choice="none",
                    phase="fallback",
                    iteration=iteration,
                    recorder=recorder,
                    stats=stats,
                    **kwargs
                )
                self._add_usage(total_usage, parsed.usage)
                conversation.append(ChatMessage(role="assistant", content=parsed.text, native=parsed.native))
                if parsed.text:
                    all_content_parts.append(parsed.text)
                    final_answer = parsed.text

            if cap_hit:
                termination_reason = "cap_hit"
            elif fallback_used:
                termination_reason = "fallback_used"
            else:
                termination_reason = "model_stopped"

            # Self-reflection phase for SELF_REFLECT mode: only when there is a patch to review
            reflection_metrics = None
            reflection_text = None
            if experiment_mode == "self_reflect":
                if self._has_patch(final_answer) and CANNOT_SOLVE_MARKER not in final_answer:
                    reflection_text, reflection_metrics = await self._do_self_reflection(
                        final_answer=final_answer,
                        conversation=conversation,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools_spec=tools_spec,
                        recorder=recorder,
                        stats=stats,
                        **kwargs
                    )
                else:
                    reflection_metrics = ReflectionMetrics(outcome="skipped")

            response_time = time.time() - start_time

            # Combine all accumulated content (full transcript text, kept for the record)
            content = "\n\n".join(all_content_parts) if all_content_parts else ""
            if reflection_text:
                content = content + "\n\n---\n## SELF-REFLECTION RESULT\n" + reflection_text

            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                content=content,
                usage=total_usage if total_usage.total_tokens > 0 else None,
                reflection=reflection_metrics,  # Separate tracking for self-reflection
                response_time=response_time,
                success=True,
                raw_response=None,  # Per-call records go to the trajectory log
                tool_calls=all_tool_calls if all_tool_calls else None,
                conversation_history=self._serialise_history(conversation),
                final_answer=final_answer,
                reflection_text=reflection_text,
                run_meta=run_meta(
                    termination_reason=termination_reason,
                    fallback_used=fallback_used,
                    cap_hit=cap_hit,
                ),
            )

        except ProviderCallError as e:
            # Not a model outcome: the runner records it separately and re-runs the instance
            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                content="",
                usage=total_usage if total_usage.total_tokens > 0 else None,
                response_time=time.time() - start_time,
                success=False,
                error_kind=e.kind,
                error_message=str(e),
                tool_calls=all_tool_calls if all_tool_calls else None,
                run_meta=run_meta(),
            )
        except Exception as e:
            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                content="",
                response_time=time.time() - start_time,
                success=False,
                error_kind="infra_error",
                error_message=f"{type(e).__name__}: {e}",
                tool_calls=all_tool_calls if all_tool_calls else None,
                run_meta=run_meta(),
            )

    @staticmethod
    def _serialise_history(conversation: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Conversation as plain dicts (tool turns included, provider objects dropped)."""
        return [msg.model_dump(exclude_none=True) for msg in conversation]

    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible"""
        try:
            test_messages = [ChatMessage(role="user", content="Hi")]
            response = await self.generate_response(
                messages=test_messages,
                max_tokens=10
            )
            print(f"Health check for {self.provider_name}: success={response.success}, content='{response.content[:50]}...', error={response.error_message}")
            return response.success
        except Exception as e:
            print(f"Health check exception for {self.provider_name}: {str(e)}")
            return False