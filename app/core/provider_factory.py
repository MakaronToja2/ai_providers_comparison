from typing import Dict, Type, Optional
from ..models.requests import AIProvider
from ..config.settings import get_settings
from .base_provider import BaseAIProvider
from ..providers import OpenAIProvider, AnthropicProvider, GoogleProvider


class ProviderFactory:
    """Factory for creating AI provider instances"""
    
    _providers: Dict[AIProvider, Type[BaseAIProvider]] = {}
    _instances: Dict[str, BaseAIProvider] = {}
    
    @classmethod
    def register_provider(cls, provider_type: AIProvider, provider_class: Type[BaseAIProvider]):
        """Register a new provider class"""
        cls._providers[provider_type] = provider_class
    
    @classmethod
    def create_provider(
        cls, 
        provider_type: AIProvider, 
        model: str, 
        api_key: Optional[str] = None
    ) -> BaseAIProvider:
        """Create a provider instance"""
        
        if provider_type not in cls._providers:
            raise ValueError(f"Provider {provider_type} not registered")
        
        # Use provided API key or get from settings
        if not api_key:
            settings = get_settings()
            api_key = cls._get_api_key_from_settings(provider_type, settings)
        
        if not api_key:
            raise ValueError(f"No API key found for provider {provider_type}")
        
        # Create cache key
        cache_key = f"{provider_type}_{model}_{hash(api_key)}"
        
        # Return cached instance if exists
        if cache_key in cls._instances:
            return cls._instances[cache_key]
        
        # Create new instance
        provider_class = cls._providers[provider_type]
        instance = provider_class(api_key=api_key, model=model)
        
        # Cache the instance
        cls._instances[cache_key] = instance
        return instance
    
    @classmethod
    def _get_api_key_from_settings(cls, provider_type: AIProvider, settings) -> Optional[str]:
        """Get API key from settings based on provider type"""
        key_mapping = {
            AIProvider.OPENAI: settings.openai_api_key,
            AIProvider.ANTHROPIC: settings.anthropic_api_key,
            AIProvider.GOOGLE: settings.google_api_key,
        }
        return key_mapping.get(provider_type)
    
    @classmethod
    def get_available_providers(cls) -> list[AIProvider]:
        """Get list of registered providers"""
        return list(cls._providers.keys())
    
    @classmethod
    def clear_cache(cls):
        """Clear the provider instance cache"""
        cls._instances.clear()


# Register all providers
ProviderFactory.register_provider(AIProvider.OPENAI, OpenAIProvider)
ProviderFactory.register_provider(AIProvider.ANTHROPIC, AnthropicProvider)
ProviderFactory.register_provider(AIProvider.GOOGLE, GoogleProvider)