from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import time
import asyncio
import json
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models.requests import ChatMessage, LLMRequest
from ..models.responses import LLMResponse, TokenUsage, ToolCall, ReflectionMetrics
from .tools import tool_registry
from ..config.settings import get_settings


class BaseAIProvider(ABC):
    """Abstract base class for all AI providers"""
    
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
        **kwargs
    ) -> Dict[str, Any]:
        """Make the actual API call to the provider"""
        pass
    
    @abstractmethod
    def _parse_response(self, raw_response: Dict[str, Any]) -> tuple[str, Optional[TokenUsage], Optional[List[Dict[str, Any]]]]:
        """Parse the provider's response into standardized format"""
        pass
    
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
                error=result.error
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

    async def _do_self_reflection(
        self,
        final_content: str,
        conversation: List[ChatMessage],
        temperature: float,
        max_tokens: int,
        test_results: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> tuple[str, Optional[ReflectionMetrics]]:
        """Perform self-reflection on generated patch.

        This implements Experiment 3: Test-Guided Self-Reflection Strategy
        1. Check if we have a patch
        2. If test_results provided: reflect based on actual test feedback
        3. Otherwise: self-review the patch (fallback)
        4. Ask model to revise if needed
        5. Return the final content and reflection metrics (tokens tracked separately)

        Args:
            test_results: Optional dict with keys:
                - resolved: bool - whether all tests passed
                - fail_to_pass_passed: int - count of fixed tests
                - fail_to_pass_total: int - total tests that should pass
                - pass_to_pass_passed: int - count of non-regression tests passing
                - pass_to_pass_total: int - total non-regression tests
                - test_output: str - actual test output/errors
        """
        # Check if we have a patch to reflect on
        has_patch = "```diff" in final_content or "--- a/" in final_content

        if not has_patch:
            # No patch to reflect on, skip reflection
            print(f"[{self.provider_name}] Self-reflect: No patch found, skipping reflection")
            return final_content, ReflectionMetrics(outcome="skipped")

        print(f"[{self.provider_name}] Self-reflect: Starting reflection phase")

        # Extract the patch from the content
        patch_start = final_content.find("```diff")
        if patch_start == -1:
            patch_start = final_content.find("--- a/")

        # Build reflection prompt based on whether we have test results
        if test_results and not test_results.get("resolved", False):
            # TEST-GUIDED REFLECTION: We have test results showing failures
            f2p_passed = test_results.get("fail_to_pass_passed", 0)
            f2p_total = test_results.get("fail_to_pass_total", 0)
            p2p_passed = test_results.get("pass_to_pass_passed", 0)
            p2p_total = test_results.get("pass_to_pass_total", 0)
            test_output = test_results.get("test_output", "No output available")

            # Truncate test output if too long
            if len(test_output) > 4000:
                test_output = test_output[:2000] + "\n\n... [truncated] ...\n\n" + test_output[-2000:]

            reflection_prompt = f"""## TEST-GUIDED REFLECTION PHASE

Your patch was applied and tested. Here are the results:

**Test Results:**
- Tests that should now pass (fail_to_pass): {f2p_passed}/{f2p_total}
- Tests that should still pass (pass_to_pass): {p2p_passed}/{p2p_total}
- **Overall: {'PASSED' if test_results.get('resolved') else 'FAILED'}**

**Test Output:**
```
{test_output}
```

**Your Original Patch:**
{final_content[patch_start:]}

Based on the ACTUAL test failures above, analyze what went wrong and generate a REVISED patch.

You MUST output:
<<<PATCH_REVISED>>>
```diff
[Your revised patch that addresses the test failures]
```
[Brief explanation of what you changed based on the test output]

Focus on fixing the specific errors shown in the test output."""

            print(f"[{self.provider_name}] Self-reflect: Using test-guided reflection (tests failed: {f2p_passed}/{f2p_total})")

        elif test_results and test_results.get("resolved", False):
            # Tests passed! Just note it and approve
            print(f"[{self.provider_name}] Self-reflect: Tests passed, auto-approving patch")
            return final_content + "\n\n---\n## TEST RESULT: PASSED\nAll tests passed.", ReflectionMetrics(outcome="tests_passed")

        else:
            # SELF-REVIEW FALLBACK: No test results, do traditional self-reflection
            reflection_prompt = f"""## SELF-REFLECTION PHASE

You just generated the following patch:

{final_content[patch_start:]}

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

Do NOT simply repeat the same patch. Either approve it or make meaningful improvements."""

        # Add reflection request to conversation
        reflection_conversation = list(conversation)
        reflection_conversation.append(ChatMessage(role="user", content=reflection_prompt))

        try:
            # Make reflection call WITHOUT tools
            raw_response = await self._make_api_call(
                messages=reflection_conversation,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=None,
                **kwargs
            )

            reflection_content, usage, _ = self._parse_response(raw_response)

            # Build reflection metrics (tokens tracked SEPARATELY from base generation)
            reflection_metrics = ReflectionMetrics(
                outcome="unknown",
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            )

            if reflection_content:
                # Check if patch was revised
                if "<<<PATCH_REVISED>>>" in reflection_content:
                    print(f"[{self.provider_name}] Self-reflect: Patch was revised")
                    reflection_metrics.outcome = "revised"
                    final_content = final_content + "\n\n---\n## SELF-REFLECTION RESULT\n" + reflection_content
                elif "<<<PATCH_APPROVED>>>" in reflection_content:
                    print(f"[{self.provider_name}] Self-reflect: Patch was approved")
                    reflection_metrics.outcome = "approved"
                    final_content = final_content + "\n\n---\n## SELF-REFLECTION RESULT\n" + reflection_content
                else:
                    # Model didn't follow format, append anyway
                    print(f"[{self.provider_name}] Self-reflect: Response didn't follow format")
                    reflection_metrics.outcome = "unstructured"
                    final_content = final_content + "\n\n---\n## SELF-REFLECTION (unstructured)\n" + reflection_content

            return final_content, reflection_metrics

        except Exception as e:
            print(f"[{self.provider_name}] Self-reflect: Failed - {str(e)}")
            return final_content, ReflectionMetrics(outcome="failed")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_response(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_tools: bool = True,
        max_tool_iterations: int = 10,
        experiment_mode: str = "baseline",  # "baseline", "forced", or "self_reflect"
        **kwargs
    ) -> LLMResponse:
        """Generate response with agentic tool loop and standardized output"""
        start_time = time.time()
        all_tool_calls = []
        total_usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        conversation = list(messages)  # Copy messages for the loop

        try:
            # Get available tools if enabled
            tools_spec = None
            if use_tools:
                tool_definitions = tool_registry.get_tool_definitions()
                if tool_definitions:
                    tools_spec = [tool.model_dump() for tool in tool_definitions]

            all_content_parts = []  # Accumulate ALL content throughout iterations
            iteration = 0

            while iteration < max_tool_iterations:
                iteration += 1

                raw_response = await self._make_api_call(
                    messages=conversation,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools_spec,
                    **kwargs
                )

                content, usage, tool_calls_data = self._parse_response(raw_response)

                # Accumulate usage
                if usage:
                    total_usage.prompt_tokens += usage.prompt_tokens
                    total_usage.completion_tokens += usage.completion_tokens
                    total_usage.total_tokens += usage.total_tokens

                # Always accumulate content (fixes bug where content was lost during tool iterations)
                if content:
                    all_content_parts.append(content)

                # If no tool calls, we're done
                if not tool_calls_data:
                    break

                # Execute tool calls
                tool_results = []
                for tool_call_data in tool_calls_data:
                    tool_name = tool_call_data.get("name")
                    parameters = tool_call_data.get("parameters", {})

                    executed_call = await self._execute_tool_call(tool_name, parameters)
                    all_tool_calls.append(executed_call)
                    tool_results.append(executed_call)

                # Add assistant message with tool calls to conversation
                assistant_content = content or f"Calling tools: {', '.join(tc.name for tc in tool_results)}"
                conversation.append(ChatMessage(role="assistant", content=assistant_content))

                # Add tool results to conversation
                tool_result_text = self._format_tool_results(tool_results)
                conversation.append(ChatMessage(role="user", content=tool_result_text))

            response_time = time.time() - start_time

            # Combine all accumulated content
            final_content = "\n\n".join(all_content_parts) if all_content_parts else ""

            # Check if we have a valid patch or explicit failure marker in content
            has_patch = "```diff" in final_content or "--- a/" in final_content or "+++ b/" in final_content
            has_cannot_solve = "<<<CANNOT_SOLVE>>>" in final_content

            # If no valid output after tool loop, force one final call WITHOUT tools
            # This handles: max iterations, early loop exit, truncation, tool setup failure, etc.
            # Trigger if: tools were available AND (used tools OR have some content) AND no valid output
            should_force_answer = (
                not has_patch and
                not has_cannot_solve and
                tools_spec is not None and  # Tools were available
                (all_tool_calls or final_content)  # Either used tools or got some content
            )

            if should_force_answer:
                # Log that we're forcing a final answer
                print(f"[{self.provider_name}] Forcing final answer: iterations={iteration}, tool_calls={len(all_tool_calls)}, content_len={len(final_content)}, mode={experiment_mode}")

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

Output a unified diff patch NOW:
```diff
--- a/path/to/file
+++ b/path/to/file
@@ ... @@
...
```"""
                else:
                    # BASELINE or SELF_REFLECT: Allow escape hatch
                    forcing_prompt = f"""{exploration_summary}

FINAL ANSWER REQUIRED

Based on your exploration, provide your final answer.
You MUST output ONE of:

1. A ```diff``` patch with your fix (PREFERRED), OR
2. <<<CANNOT_SOLVE>>> with reason and explanation (only if you truly cannot attempt a fix)

Do not ask for more information. Make your best attempt at a patch NOW."""

                # Add forcing message
                conversation.append(ChatMessage(role="user", content=forcing_prompt))

                # Make final call WITHOUT tools to force a text response
                # Use retry logic to handle transient failures
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        raw_response = await self._make_api_call(
                            messages=conversation,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            tools=None,  # No tools - force text response
                            **kwargs
                        )

                        content, usage, _ = self._parse_response(raw_response)
                        if content:
                            final_content = final_content + "\n\n" + content if final_content else content
                            print(f"[{self.provider_name}] Forcing call succeeded on attempt {attempt + 1}")
                        if usage:
                            total_usage.prompt_tokens += usage.prompt_tokens
                            total_usage.completion_tokens += usage.completion_tokens
                            total_usage.total_tokens += usage.total_tokens
                        break  # Success, exit retry loop
                    except Exception as e:
                        print(f"[{self.provider_name}] Forcing call attempt {attempt + 1}/{max_retries} FAILED: {e}")
                        if attempt < max_retries - 1:
                            # Wait before retry (exponential backoff)
                            wait_time = 2 ** attempt  # 1s, 2s, 4s
                            print(f"[{self.provider_name}] Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            print(f"[{self.provider_name}] All forcing attempts failed, keeping original content")

            # Self-reflection phase for SELF_REFLECT mode
            # NOTE: When test-guided reflection is needed, the runner will call
            # do_reflection_with_test_results() separately after running tests.
            # Internal self-reflection only happens if no test results will be provided.
            reflection_metrics = None
            # Skip internal self-reflection - runner will handle it with test results
            # if experiment_mode == "self_reflect":
            #     final_content, reflection_metrics = await self._do_self_reflection(
            #         final_content=final_content,
            #         conversation=conversation,
            #         temperature=temperature,
            #         max_tokens=max_tokens,
            #         **kwargs
            #     )

            # Convert conversation to serializable format for storage
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in conversation
            ]

            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                content=final_content,
                usage=total_usage if total_usage.total_tokens > 0 else None,
                reflection=reflection_metrics,  # Separate tracking for self-reflection
                response_time=response_time,
                success=True,
                raw_response=None,  # Don't store all raw responses
                tool_calls=all_tool_calls if all_tool_calls else None,
                conversation_history=conversation_history,  # Include full context for reflection
            )

        except Exception as e:
            response_time = time.time() - start_time
            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                content="",
                response_time=response_time,
                success=False,
                error_message=str(e),
                tool_calls=all_tool_calls if all_tool_calls else None
            )

    def _format_tool_results(self, tool_calls: List[ToolCall]) -> str:
        """Format tool call results for sending back to the LLM."""
        results = []
        for tc in tool_calls:
            if tc.success:
                result_str = json.dumps(tc.result, indent=2) if tc.result else "Success"
                results.append(f"## Tool: {tc.name}\n{result_str}")
            else:
                results.append(f"## Tool: {tc.name}\nError: {tc.error}")
        return "Here are the tool results:\n\n" + "\n\n".join(results)

    async def do_reflection_with_test_results(
        self,
        original_content: str,
        conversation_history: List[Dict[str, str]],
        test_results: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> tuple[str, Optional[ReflectionMetrics]]:
        """Public method for runner to call test-guided self-reflection.

        This is called by the runner after:
        1. Initial patch generation
        2. Applying patch and running SWE-bench tests
        3. Tests failed - need model to revise based on actual failures

        Args:
            original_content: The original response content with the patch
            conversation_history: The FULL conversation history including file contents
                                  from tool calls (from LLMResponse.conversation_history)
            test_results: Dict with test results:
                - resolved: bool
                - fail_to_pass_passed/total: int
                - pass_to_pass_passed/total: int
                - test_output: str
            temperature: LLM temperature
            max_tokens: Max tokens for reflection response

        Returns:
            Tuple of (revised_content, reflection_metrics)
        """
        # Convert dict format back to ChatMessage objects
        conversation = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in conversation_history
        ]

        return await self._do_self_reflection(
            final_content=original_content,
            conversation=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            test_results=test_results,
        )

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