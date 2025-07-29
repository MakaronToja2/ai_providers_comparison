from fastapi import APIRouter

from ..models.requests import AIProvider, ChatMessage
from ..core.provider_factory import ProviderFactory

router = APIRouter()


@router.get("/debug/test-openai")
async def debug_test_openai():
    """Debug endpoint to test OpenAI provider directly"""
    try:
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


@router.get("/debug/test-anthropic")
async def debug_test_anthropic():
    """Debug endpoint to test Anthropic provider directly"""
    try:
        provider = ProviderFactory.create_provider(
            provider_type=AIProvider.ANTHROPIC,
            model="claude-3-haiku-20240307"
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


@router.get("/debug/test-google")
async def debug_test_google():
    """Debug endpoint to test Google provider directly"""
    try:
        provider = ProviderFactory.create_provider(
            provider_type=AIProvider.GOOGLE,
            model="gemini-2.5-flash-lite"
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