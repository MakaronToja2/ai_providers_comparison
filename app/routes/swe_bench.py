import os
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ..models.requests import AIProvider, ChatMessage
from ..models.swe_bench import SWEBenchAnalysisRequest, SWEBenchAnalysisResponse, SWEBenchDatasetInfo
from ..core.provider_factory import ProviderFactory
from ..utils.swe_bench_loader import swe_bench_loader
from ..utils.repo_context import repo_context

router = APIRouter()


@router.get("/swe-bench/dataset/{split}", response_model=SWEBenchDatasetInfo)
async def load_dataset(split: str):
    """Load SWE-bench dataset split and return information"""
    try:
        dataset_info = swe_bench_loader.load_dataset(split)
        return dataset_info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")


@router.get("/swe-bench/instances/{split}")
async def list_instances(split: str = "dev", limit: int = 10, repo: str = None):
    """List instances from SWE-bench dataset with optional filtering"""
    try:
        if repo:
            instances = swe_bench_loader.search_instances(split=split, repo=repo, limit=limit)
        else:
            instances = swe_bench_loader.list_instances(split=split, limit=limit)
        
        return {
            "split": split,
            "instances": instances,
            "count": len(instances),
            "filter": {"repo": repo} if repo else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list instances: {str(e)}")


@router.get("/swe-bench/instance/{instance_id}")
async def get_instance(instance_id: str, split: str = "dev"):
    """Get a specific SWE-bench instance by ID"""
    try:
        instance = swe_bench_loader.get_instance(instance_id, split)
        if not instance:
            raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found in {split} split")
        
        return instance
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve instance: {str(e)}")


@router.post("/swe-bench/analyze", response_model=SWEBenchAnalysisResponse)
async def analyze_instance(request: SWEBenchAnalysisRequest):
    """Analyze a SWE-bench instance with selected AI provider"""
    try:
        # Get the instance
        instance = swe_bench_loader.get_instance(request.instance_id)
        if not instance:
            raise HTTPException(status_code=404, detail=f"Instance {request.instance_id} not found")
        
        # Create provider with default models from environment
        default_models = {
            AIProvider.OPENAI: os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini"),
            AIProvider.ANTHROPIC: os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-haiku-4-5-20251001"),
            AIProvider.GOOGLE: os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-2.5-flash-lite")
        }
        
        model = request.model or default_models.get(request.provider, "default")
        provider = ProviderFactory.create_provider(
            provider_type=request.provider,
            model=model
        )
        
        # Clone repository for analysis
        async with repo_context(instance.repo, instance.base_commit) as repo_ctx:
            # Build analysis prompt with repo context
            prompt = _build_analysis_prompt(instance, request, repo_ctx.working_directory)
            
            # Generate analysis with tools enabled
            messages = [ChatMessage(role="user", content=prompt)]
            llm_response = await provider.generate_response(
                messages=messages,
                max_tokens=2000,
                temperature=0.3,
                use_tools=True  # Enable tool calling
            )
        
        # Create structured analysis summary
        analysis_summary = _create_analysis_summary(instance, request, llm_response)
        
        return SWEBenchAnalysisResponse(
            instance_id=request.instance_id,
            analysis_type=request.analysis_type,
            llm_response=llm_response,
            analysis_summary=analysis_summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def _build_analysis_prompt(instance, request: SWEBenchAnalysisRequest, repo_path: str = None) -> str:
    """Build analysis prompt based on request type and instance data"""
    
    if request.custom_prompt:
        return request.custom_prompt
    
    base_info = f"""
Repository: {instance.repo}
Problem Statement: {instance.problem_statement}
"""
    
    if request.include_patch and instance.patch:
        base_info += f"\nPatch/Diff:\n{instance.patch}\n"
    
    if request.include_tests:
        if instance.FAIL_TO_PASS:
            base_info += f"\nTests that should pass: {instance.FAIL_TO_PASS}\n"
        if instance.PASS_TO_PASS:
            base_info += f"\nTests that should continue passing: {instance.PASS_TO_PASS}\n"
    
    if instance.hints_text:
        base_info += f"\nHints: {instance.hints_text}\n"
    
    if request.analysis_type == "bug_analysis":
        prompt = f"""Please analyze this software bug from a pull request:

{base_info}

The repository has been cloned at commit {instance.base_commit} for your analysis.

You have access to tools to explore the codebase:
- search_code: Search for patterns using regex (e.g., "def function_name", "class ClassName")
- read_file: Read specific files with optional line ranges
- list_directory: Browse directory structure

Use these tools to understand the codebase context and provide a detailed analysis including:
1. What is the bug or issue?
2. What are the root causes?
3. What components/files are affected?
4. How severe is the issue?
5. What would be the approach to fix it?

Start by exploring the affected files mentioned in the patch to understand the full context."""
        
    elif request.analysis_type == "solution_generation":
        prompt = f"""Generate a solution for this software issue.

{base_info}

You must strictly follow these rules:

1. **Analysis:** Briefly explain the root cause and your plan.
2. **Success Output:** If you can fix the issue, provide the code changes in a standard Unified Diff format wrapped in a code block.
   - The diff must start with `diff --git` or `---` / `+++` headers.
   - Use standard diff markers (`+`, `-`, `@@`).
   - Example:
     ```diff
     --- a/file.py
     +++ b/file.py
     @@ -10,1 +10,1 @@
     - old_code
     + new_code
     ```
3. **Failure Output:** If you genuinely cannot solve this, respond with EXACTLY:
<<<CANNOT_SOLVE>>>
reason: [insufficient_context|too_complex|unclear_requirements|missing_dependencies]
explanation: [brief explanation]

DO NOT describe the code changes in plain text without a diff.
A partial or uncertain patch is better than giving up - only use <<<CANNOT_SOLVE>>> if you truly cannot make any attempt.
"""
        
    else:
        prompt = f"""Analyze this software development issue:

{base_info}

Provide your analysis and insights about this issue."""
    
    return prompt


def _create_analysis_summary(instance, request: SWEBenchAnalysisRequest, llm_response) -> Dict[str, Any]:
    """Create structured summary of the analysis"""
    
    summary = {
        "instance_info": {
            "repo": instance.repo,
            "base_commit": instance.base_commit,
            "has_patch": bool(instance.patch),
            "has_tests": bool(instance.FAIL_TO_PASS or instance.PASS_TO_PASS),
            "has_hints": bool(instance.hints_text)
        },
        "analysis_config": {
            "provider": request.provider.value,
            "analysis_type": request.analysis_type,
            "included_patch": request.include_patch,
            "included_tests": request.include_tests,
            "used_custom_prompt": bool(request.custom_prompt)
        },
        "response_metrics": {
            "success": llm_response.success,
            "response_time": llm_response.response_time,
            "content_length": len(llm_response.content) if llm_response.content else 0,
            "token_usage": llm_response.usage.model_dump() if llm_response.usage else None
        }
    }
    
    if not llm_response.success:
        summary["error"] = llm_response.error_message
    
    return summary