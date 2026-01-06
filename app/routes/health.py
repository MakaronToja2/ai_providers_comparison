import os
from fastapi import APIRouter, HTTPException, Depends

from ..models.requests import AIProvider
from ..models.responses import HealthResponse, ProviderHealth
from ..config.settings import get_settings, Settings
from ..core.provider_factory import ProviderFactory

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    """Check the health of all available providers"""
    try:
        provider_status = {}
        available_providers = ProviderFactory.get_available_providers()
        
        for provider_type in available_providers:
            try:
                # Use the default models from environment for health_check
                default_models = {
                    AIProvider.OPENAI: os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini"),
                    AIProvider.ANTHROPIC: os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-haiku-4-5-20251001"),
                    AIProvider.GOOGLE: os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-2.5-flash-lite")
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