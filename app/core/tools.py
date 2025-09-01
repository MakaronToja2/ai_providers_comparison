"""
Abstract tool interface and registry for AI providers.
Allows modular tool integration with enable/disable functionality.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel


class ToolResult(BaseModel):
    """Result from tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None


class ToolDefinition(BaseModel):
    """Tool definition for AI providers"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema for parameters


class BaseTool(ABC):
    """Abstract base class for all tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for AI"""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON schema for tool parameters"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    def get_definition(self) -> ToolDefinition:
        """Get tool definition for AI providers"""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema
        )


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._enabled_tools: Dict[str, bool] = {}
    
    def register_tool(self, tool: BaseTool, enabled: bool = True):
        """Register a tool in the registry"""
        self._tools[tool.name] = tool
        self._enabled_tools[tool.name] = enabled
    
    def enable_tool(self, tool_name: str):
        """Enable a specific tool"""
        if tool_name in self._tools:
            self._enabled_tools[tool_name] = True
    
    def disable_tool(self, tool_name: str):
        """Disable a specific tool"""
        if tool_name in self._tools:
            self._enabled_tools[tool_name] = False
    
    def get_enabled_tools(self) -> List[BaseTool]:
        """Get list of enabled tools"""
        return [
            tool for name, tool in self._tools.items()
            if self._enabled_tools.get(name, False)
        ]
    
    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Get tool definitions for enabled tools"""
        return [tool.get_definition() for tool in self.get_enabled_tools()]
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name if enabled"""
        if name in self._tools and self._enabled_tools.get(name, False):
            return self._tools[name]
        return None
    
    def list_all_tools(self) -> Dict[str, bool]:
        """List all tools and their enabled status"""
        return self._enabled_tools.copy()


# Global tool registry instance
tool_registry = ToolRegistry()