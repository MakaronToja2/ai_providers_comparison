from typing import List, Dict, Any, Optional
import time
import google.generativeai as genai

from ..core.base_provider import BaseAIProvider
from ..models.requests import ChatMessage
from ..models.responses import LLMResponse, TokenUsage


class GoogleProvider(BaseAIProvider):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite", **kwargs):
        super().__init__(api_key, model, **kwargs)
        if api_key:
            genai.configure(api_key=api_key)
        # Initialize client without any tools to avoid conflicts
        self.client = genai.GenerativeModel(model_name=model)
    
    async def _make_api_call(self, messages: List[ChatMessage], temperature: float = 0.7, 
                           max_tokens: int = 1000, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        # Debug logging
        print(f"Google Provider Debug:")
        print(f"  tools parameter: {tools}")
        print(f"  kwargs keys: {list(kwargs.keys())}")
        print(f"  kwargs: {kwargs}")
        
        # Remove any tools from kwargs to avoid conflicts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'tools'}
        print(f"  filtered_kwargs: {filtered_kwargs}")
        
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
        
        # Prepare tools if provided - use Google's to_tools converter
        google_tools = None
        if tools:
            try:
                import google.generativeai.types.content_types as ct
                
                # Convert to Google's expected format - fix schema types
                function_declarations = []
                for tool in tools:
                    # Convert schema to Google's format (object -> OBJECT)
                    google_params = tool["parameters"].copy()
                    if google_params.get("type") == "object":
                        google_params["type"] = "OBJECT"
                    
                    # Convert property types and remove unsupported fields
                    if "properties" in google_params:
                        for prop_def in google_params["properties"].values():
                            if prop_def.get("type") == "string":
                                prop_def["type"] = "STRING"
                            elif prop_def.get("type") == "integer":
                                prop_def["type"] = "INTEGER"
                            elif prop_def.get("type") == "boolean":
                                prop_def["type"] = "BOOLEAN"
                            elif prop_def.get("type") == "array":
                                prop_def["type"] = "ARRAY"
                            
                            # Remove unsupported fields for Google Schema
                            unsupported_fields = ["minimum", "maximum", "default"]
                            for field in unsupported_fields:
                                prop_def.pop(field, None)
                    
                    function_declarations.append({
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": google_params
                    })
                
                tool_dicts = [{
                    "function_declarations": function_declarations
                }]
                
                print(f"  Tool dicts before conversion: {tool_dicts}")
                google_tools = ct.to_tools(tool_dicts)
                print(f"  Converted tools: {google_tools}")
                
            except Exception as e:
                print(f"  Tool conversion failed: {str(e)}")
                google_tools = None
        
        # Google Generative AI doesn't have native async support yet
        # We'll use the sync method but wrap it
        import asyncio
        
        def _generate_with_tools():
            try:
                if google_tools:
                    # Try initializing client with tools in constructor
                    print(f"  Creating new client with tools in constructor")
                    try:
                        client_with_tools = genai.GenerativeModel(
                            model_name=self.model,
                            tools=tool_dicts[0]  # Pass the dict directly 
                        )
                        print(f"  Client created successfully with tools")
                        return client_with_tools.generate_content(
                            conversation_parts,
                            generation_config=generation_config
                        )
                    except Exception as init_error:
                        print(f"  Client initialization with tools failed: {str(init_error)}")
                        # Fallback: try without tools
                        return self.client.generate_content(
                            conversation_parts,
                            generation_config=generation_config
                        )
                else:
                    print(f"  Calling generate_content without tools")
                    return self.client.generate_content(
                        conversation_parts,
                        generation_config=generation_config
                    )
            except Exception as e:
                print(f"  Google API call failed: {str(e)}")
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
                        # Google tool call format
                        function_call = part.function_call
                        tool_calls.append({
                            "name": function_call.name,
                            "parameters": dict(function_call.args) if function_call.args else {}
                        })
        
        # Fallback to direct text access if the above doesn't work
        if not text_content and hasattr(raw_response, 'text'):
            text_content = raw_response.text
        
        # Google's usage information might be limited
        usage = None
        if hasattr(raw_response, 'usage_metadata'):
            usage_data = raw_response.usage_metadata
            usage = TokenUsage(
                prompt_tokens=getattr(usage_data, 'prompt_token_count', 0),
                completion_tokens=getattr(usage_data, 'candidates_token_count', 0),
                total_tokens=getattr(usage_data, 'total_token_count', 0)
            )
        
        return text_content, usage, tool_calls if tool_calls else None
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    def _validate_api_key(self) -> None:
        if not self.api_key:
            raise ValueError("Google API key is required")