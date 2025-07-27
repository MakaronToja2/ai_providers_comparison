from fastapi import APIRouter, HTTPException, Depends
from typing import List
import uuid
import asyncio
from datetime import datetime

from ..models.requests import LLMRequest, ComparisonRequest, AIProvider
from ..models.responses import LLMResponse, ComparisonResponse, HealthResponse, ProviderHealth
from ..config.settings import get_settings, Settings
from .provider_factory import ProviderFactory

router = APIRouter()


@router.post("/generate", response_model=LLMResponse)
async def generate_response(request: LLMRequest):
    """Generate response from a single AI provider"""
    try:
        provider = ProviderFactory.create_provider(
            provider_type=request.provider,
            model=request.model
        )
        
        response = await provider.generate_response(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/compare", response_model=ComparisonResponse)
async def compare_providers(request: ComparisonRequest):
    """Compare responses from multiple AI providers"""
    try:
        request_id = str(uuid.uuid4())
        
        # Create provider instances
        providers = []
        for provider_type in request.providers:
            if provider_type.value not in request.model_mapping:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model mapping missing for provider {provider_type}"
                )
            
            model = request.model_mapping[provider_type.value]
            provider = ProviderFactory.create_provider(
                provider_type=provider_type,
                model=model
            )
            providers.append(provider)
        
        # Generate responses concurrently
        tasks = [
            provider.generate_response(
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            for provider in providers
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in responses
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                # Create error response
                error_response = LLMResponse(
                    provider=request.providers[i].value,
                    model=request.model_mapping[request.providers[i].value],
                    content="",
                    response_time=0.0,
                    success=False,
                    error_message=str(response)
                )
                valid_responses.append(error_response)
            else:
                valid_responses.append(response)
        
        # Calculate summary statistics
        successful_responses = [r for r in valid_responses if r.success]
        summary = {
            "total_providers": len(request.providers),
            "successful_responses": len(successful_responses),
            "failed_responses": len(valid_responses) - len(successful_responses),
            "average_response_time": sum(r.response_time for r in successful_responses) / len(successful_responses) if successful_responses else 0,
            "total_tokens_used": sum(r.usage.total_tokens for r in successful_responses if r.usage) if successful_responses else 0
        }
        
        return ComparisonResponse(
            request_id=request_id,
            responses=valid_responses,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    """Check the health of all available providers"""
    try:
        provider_status = {}
        available_providers = ProviderFactory.get_available_providers()
        
        for provider_type in available_providers:
            try:
                # Use the cheapest models for health_check
                default_models = {
                    AIProvider.OPENAI: "gpt-4o-mini",
                    AIProvider.ANTHROPIC: "claude-3-haiku-latest",
                    AIProvider.GOOGLE: "gemini2.5-flash-lite"
                }
                
                model = default_models.get(provider_type, "default")
                provider = ProviderFactory.create_provider(
                    provider_type=provider_type,
                    model=model
                )
                
                is_healthy = await provider.health_check()
                provider_status[provider_type.value] = ProviderHealth(
                    healthy=is_healthy,
                    error=None
                )
                
            except Exception as e:
                provider_status[provider_type.value] = ProviderHealth(
                    healthy=False,
                    error=str(e)
                )
        
        overall_status = "healthy" if any(status.healthy for status in provider_status.values()) else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            providers=provider_status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/providers")
async def list_providers():
    """List all available providers"""
    available_providers = ProviderFactory.get_available_providers()
    return {
        "providers": [provider.value for provider in available_providers],
        "count": len(available_providers)
    }


@router.get("/debug/test-openai")
async def debug_test_openai():
    """Debug endpoint to test OpenAI provider directly"""
    try:
        from ..models.requests import ChatMessage
        
        provider = ProviderFactory.create_provider(
            provider_type=AIProvider.OPENAI,
            model="gpt-4o-mini"
        )
        
        test_messages = [ChatMessage(role="user", content="Hi")]
        response = await provider.generate_response(
            messages=test_messages,
            max_tokens=10
        )
        
        return {
            "success": response.success,
            "content": response.content,
            "error": response.error_message,
            "response_time": response.response_time
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }