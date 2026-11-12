#!/usr/bin/env python3
"""
One tiny call per model: confirms the API key works, the model is served, and shows the
model id the API reports. Costs a fraction of a cent per model.

Usage:
    .venv/bin/python scripts/preflight.py [--provider openai anthropic google]
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
os.environ.setdefault("TOOL_DEBUG", "false")

MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemini-2.5-flash-lite",
}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", nargs="*", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args()

    from app.core.provider_factory import ProviderFactory
    from app.models.requests import AIProvider, ChatMessage

    failures = 0
    for name in args.provider:
        provider = ProviderFactory.create_provider(AIProvider(name), MODELS[name])
        response = await provider.generate_response(
            [ChatMessage(role="user", content="Reply with the single word OK.")],
            temperature=0.3, max_tokens=16, use_tools=False,
        )
        if response.success:
            print(f"{name:10s} ok    requested {MODELS[name]} | API reports {response.run_meta['api_model_ids']} | "
                  f"{response.usage.total_tokens if response.usage else '?'} tokens | reply {response.content!r}")
        else:
            failures += 1
            print(f"{name:10s} FAILED ({response.error_kind}): {response.error_message}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
