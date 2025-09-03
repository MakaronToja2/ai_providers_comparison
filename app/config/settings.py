from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Default model parameters
    default_temperature: float = 0.7
    default_max_tokens: int = 1000
    request_timeout: int = 30
    
    # Rate limiting
    default_rate_limit: int = 10  # requests per minute
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Logging
    log_level: str = "INFO"
    log_api_calls: bool = True
    
    # App settings
    app_name: str = "LLM Comparison API"
    debug: bool = False
    
    # Tool configuration
    enabled_tools: List[str] = ["read_file", "list_directory", "search_code"]
    tool_debug: bool = True  # Log tool calls and results for testing
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()