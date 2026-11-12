"""
Prices used for cost figures, in USD per million tokens.

The provider APIs return token counts, not money, so cost is computed here from the
logged tokens. Prices are the providers' published standard (non-batch) API prices,
checked on 2026-10-07.

Two figures are reported:
- list price: every input token at the input rate. Comparable across providers.
- as billed (estimate): input tokens the provider reports as served from its prompt
  cache at the cached-input rate. OpenAI and Gemini 2.5 cache automatically; Anthropic
  caches only on request, which the harness never does. The provider's invoice is the
  authority; this estimate is for cross-checking it.
"""
from typing import Dict, Optional

PRICES_CHECKED_ON = "2026-10-07"

# model id -> (input, output, cached input) in $/MTok
PRICES_USD_PER_MTOK: Dict[str, tuple] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00, 0.10),
    "gpt-4.1-mini": (0.40, 1.60, 0.10),
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01),
}


def _prices(model: str) -> tuple:
    """Unknown models raise: a cost must never silently be zero."""
    if model not in PRICES_USD_PER_MTOK:
        raise KeyError(f"No price recorded for model '{model}'")
    return PRICES_USD_PER_MTOK[model]


def cost_usd(model: str, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> float:
    """Cost at list price: no cache discount."""
    input_price, output_price, _ = _prices(model)
    return ((prompt_tokens or 0) * input_price + (completion_tokens or 0) * output_price) / 1_000_000


def billed_cost_usd(model: str, prompt_tokens: Optional[int], completion_tokens: Optional[int],
                    cached_tokens: Optional[int]) -> float:
    """Estimated cost as billed: cached input tokens (a subset of prompt_tokens) at the cached rate."""
    input_price, output_price, cached_price = _prices(model)
    cached = min(cached_tokens or 0, prompt_tokens or 0)
    uncached = (prompt_tokens or 0) - cached
    return (uncached * input_price + cached * cached_price + (completion_tokens or 0) * output_price) / 1_000_000
