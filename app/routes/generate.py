from fastapi import APIRouter, HTTPException
import uuid
import asyncio

from ..models.requests import LLMRequest, ComparisonRequest
from ..models.responses import LLMResponse, ComparisonResponse
from ..core.provider_factory import ProviderFactory

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