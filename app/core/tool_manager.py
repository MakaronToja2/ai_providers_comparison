"""
Tool manager for configuring and managing available tools
"""
from typing import List
from .tools import tool_registry
from ..tools.file_reader import FileReaderTool, DirectoryListTool
from ..tools.code_search import CodeSearchTool
from ..config.settings import get_settings


def initialize_tools():
    """Initialize and register all available tools based on configuration"""
    settings = get_settings()
    
    # Available tools
    all_tools = {
        "read_file": FileReaderTool(),
        "list_directory": DirectoryListTool(), 
        "search_code": CodeSearchTool(),
    }
    
    # Register tools based on configuration
    for tool_name, tool_instance in all_tools.items():
        enabled = tool_name in settings.enabled_tools
        tool_registry.register_tool(tool_instance, enabled=enabled)


def get_enabled_tool_names() -> List[str]:
    """Get list of enabled tool names"""
    return [tool.name for tool in tool_registry.get_enabled_tools()]


def configure_tools(enabled_tools: List[str]):
    """Configure which tools are enabled"""
    # Disable all tools first
    for tool_name in tool_registry.list_all_tools().keys():
        tool_registry.disable_tool(tool_name)
    
    # Enable specified tools
    for tool_name in enabled_tools:
        tool_registry.enable_tool(tool_name)