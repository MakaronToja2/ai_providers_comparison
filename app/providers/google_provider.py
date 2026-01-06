from typing import List, Dict, Any, Optional
import os
import time
import google.generativeai as genai

from ..core.base_provider import BaseAIProvider
from ..models.requests import ChatMessage
from ..models.responses import LLMResponse, TokenUsage

# Default model from environment or fallback
DEFAULT_GOOGLE_MODEL = os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-2.5-flash-lite")


class GoogleProvider(BaseAIProvider):
    """Google Gemini provider with proper tool schema conversion."""

    # Type mapping from JSON Schema to Google Schema
    TYPE_MAP = {
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }

    # Fields not supported by Google Schema
    UNSUPPORTED_FIELDS = {"minimum", "maximum", "default", "examples", "enum", "format", "pattern"}

    def __init__(self, api_key: str, model: str = None, **kwargs):
        model = model or DEFAULT_GOOGLE_MODEL
        super().__init__(api_key, model, **kwargs)
        if api_key:
            genai.configure(api_key=api_key)
        # Initialize client without any tools to avoid conflicts
        self.client = genai.GenerativeModel(model_name=model)

    def _convert_schema_to_google(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively convert JSON Schema to Google Schema format."""
        if not isinstance(schema, dict):
            return schema

        result = {}

        # Convert type to uppercase
        if "type" in schema:
            type_val = schema["type"]
            if isinstance(type_val, str):
                result["type"] = self.TYPE_MAP.get(type_val.lower(), "STRING")
            else:
                result["type"] = "STRING"  # Fallback

        # Copy description
        if "description" in schema:
            result["description"] = str(schema["description"])

        # Handle properties (for object types)
        if "properties" in schema:
            result["properties"] = {}
            for prop_name, prop_def in schema["properties"].items():
                result["properties"][prop_name] = self._convert_schema_to_google(prop_def)

        # Handle required fields
        if "required" in schema:
            result["required"] = schema["required"]

        # Handle array items
        if "items" in schema:
            result["items"] = self._convert_schema_to_google(schema["items"])

        return result
    
    async def _make_api_call(self, messages: List[ChatMessage], temperature: float = 0.7,
                           max_tokens: int = 1000, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        # Convert messages to Gemini format
        conversation_parts = []
        
        for msg in messages:
            if msg.role == "system":
                # Gemini doesn't have system role, prepend to first user message
                if conversation_parts and conversation_parts[0].get("role") == "user":
                    conversation_parts[0]["parts"][0] = f"{msg.content}\n\n{conversation_parts[0]['parts'][0]}"
                else:
                    conversation_parts.insert(0, {
                        "role": "user",
                        "parts": [msg.content]
                    })
            else:
                role = "model" if msg.role == "assistant" else msg.role
                conversation_parts.append({
                    "role": role,
                    "parts": [msg.content]
                })
        
        generation_config = {
            "temperature": temperature,
        }
        generation_config["max_output_tokens"] = max_tokens
        
        # Prepare tools if provided - use proper schema conversion
        tool_dicts = None
        if tools:
            try:
                # Convert each tool to Google's format using recursive conversion
                function_declarations = []
                for tool in tools:
                    google_params = self._convert_schema_to_google(tool["parameters"])
                    function_declarations.append({
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": google_params
                    })

                tool_dicts = {
                    "function_declarations": function_declarations
                }
            except Exception as e:
                print(f"Google tool conversion failed: {str(e)}")
                tool_dicts = None
        
        # Google Generative AI doesn't have native async support yet
        # We'll use the sync method but wrap it
        import asyncio
        
        def _generate_with_tools():
            try:
                if tool_dicts:
                    # Create client with tools in constructor
                    try:
                        client_with_tools = genai.GenerativeModel(
                            model_name=self.model,
                            tools=[tool_dicts]
                        )
                        return client_with_tools.generate_content(
                            conversation_parts,
                            generation_config=generation_config
                        )
                    except Exception as init_error:
                        print(f"[GOOGLE ERROR] Tool initialization failed: {str(init_error)}")
                        print(f"[GOOGLE ERROR] Falling back to NO TOOLS - this will hurt performance!")
                        return self.client.generate_content(
                            conversation_parts,
                            generation_config=generation_config
                        )
                else:
                    return self.client.generate_content(
                        conversation_parts,
                        generation_config=generation_config
                    )
            except Exception as e:
                print(f"Google API call failed: {str(e)}")
                raise
        
        response = await asyncio.get_event_loop().run_in_executor(None, _generate_with_tools)
        return response
    
    def _parse_response(self, raw_response: Any) -> tuple[str, Optional[TokenUsage], Optional[List[Dict[str, Any]]]]:
        # Extract text content and tool calls
        text_content = ""
        tool_calls = []
        
        if hasattr(raw_response, 'candidates') and raw_response.candidates:
            candidate = raw_response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_content += part.text
                    elif hasattr(part, 'function_call'):
                        # Google tool call format - convert protobuf to JSON-serializable dict
                        function_call = part.function_call
                        # Convert MapComposite/RepeatedComposite to regular Python types
                        params = {}
                        if function_call.args:
                            for key, value in function_call.args.items():
                                # Handle different protobuf value types
                                if hasattr(value, 'items'):  # It's a dict-like
                                    params[key] = dict(value)
                                elif hasattr(value, '__iter__') and not isinstance(value, str):
                                    params[key] = list(value)
                                else:
                                    params[key] = value
                        tool_calls.append({
                            "name": function_call.name,
                            "parameters": params
                        })
        
        # Fallback to direct text access if the above doesn't work
        if not text_content and hasattr(raw_response, 'text'):
            text_content = raw_response.text
        
        # Google's usage information - check actual fields
        usage = None
        if hasattr(raw_response, 'usage_metadata'):
            usage_data = raw_response.usage_metadata
            # Debug: log what we actually receive
            prompt_tokens = getattr(usage_data, 'prompt_token_count', None)
            completion_tokens = getattr(usage_data, 'candidates_token_count', None)
            total_tokens = getattr(usage_data, 'total_token_count', None)

            # Log if we're getting empty values
            if prompt_tokens is None and completion_tokens is None:
                print(f"[Google] Warning: usage_metadata exists but fields are missing. Raw: {usage_data}")

            usage = TokenUsage(
                prompt_tokens=prompt_tokens or 0,
                completion_tokens=completion_tokens or 0,
                total_tokens=total_tokens or (prompt_tokens or 0) + (completion_tokens or 0)
            )
        
        return text_content, usage, tool_calls if tool_calls else None
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("Google API key is required")