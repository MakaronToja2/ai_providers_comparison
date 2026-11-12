"""
Token bucket rate limiter for API calls.
"""
import asyncio
import time
from typing import Dict


class TokenBucketRateLimiter:
    """Rate limiter using token bucket algorithm."""

    def __init__(self, requests_per_minute: int = 60):
        self.rate = requests_per_minute / 60.0  # requests per second
        self.max_tokens = requests_per_minute
        self.tokens = float(self.max_tokens)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class RateLimiterManager:
    """Manages rate limiters for multiple providers."""

    # Default rate limits per provider (requests per minute)
    DEFAULT_LIMITS = {
        "openai": 60,
        "anthropic": 60,
        "google": 15,  # More conservative for free tier
    }

    def __init__(self, custom_limits: Dict[str, int] = None):
        self._limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._custom_limits = custom_limits or {}

    def get_limiter(self, provider: str) -> TokenBucketRateLimiter:
        """Get or create rate limiter for a provider."""
        if provider not in self._limiters:
            limit = self._custom_limits.get(
                provider,
                self.DEFAULT_LIMITS.get(provider, 60)
            )
            self._limiters[provider] = TokenBucketRateLimiter(limit)
        return self._limiters[provider]

    async def acquire(self, provider: str):
        """Acquire a token for the specified provider."""
        limiter = self.get_limiter(provider)
        await limiter.acquire()
