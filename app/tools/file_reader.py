"""
File reading tool for repository browsing
"""
import os
from typing import Dict, Any, Optional, List
from ..core.tools import BaseTool, ToolResult
from ..utils.repo_context import get_current_repo_path


def _resolve_path(relative_path: str) -> str:
    """Resolve a relative path using the current repo context."""
    repo_path = get_current_repo_path()
    if repo_path:
        return os.path.join(repo_path, relative_path)
    return relative_path


class FileReaderTool(BaseTool):
    """Tool for reading files from the repository with line range support"""
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read content of a file from the repository. Supports reading specific line ranges or limiting total lines. Use this to examine source code, configuration files, or documentation."
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to repository root)"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (1-based, optional)",
                    "minimum": 1
                },
                "end_line": {
                    "type": "integer", 
                    "description": "Ending line number (1-based, optional)"
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read from start (default: 500)",
                    "default": 500,
                    "minimum": 1
                }
            },
            "required": ["file_path"]
        }
    
    async def execute(self, file_path: str, start_line: Optional[int] = None,
                     end_line: Optional[int] = None, max_lines: int = 500, **kwargs) -> ToolResult:
        """Read file content with line range support and safety checks"""
        try:
            # Security: ensure we stay within repository bounds
            normalized_path = os.path.normpath(file_path)
            if normalized_path.startswith('..') or os.path.isabs(normalized_path):
                return ToolResult(
                    success=False,
                    data=None,
                    error="File path must be relative to repository root"
                )

            # Resolve to absolute path using repo context
            absolute_path = _resolve_path(normalized_path)

            # Check if file exists
            if not os.path.exists(absolute_path):
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"File not found: {normalized_path}"
                )

            # Check if it's a file (not directory)
            if not os.path.isfile(absolute_path):
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Path is not a file: {normalized_path}"
                )

            # Read file content with line range support
            with open(absolute_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            
            total_lines = len(all_lines)
            
            # Handle line range parameters
            if start_line is not None:
                start_idx = max(0, start_line - 1)  # Convert to 0-based
                if end_line is not None:
                    end_idx = min(total_lines, end_line)  # Convert to 0-based + 1
                    selected_lines = all_lines[start_idx:end_idx]
                else:
                    # Read from start_line to max_lines
                    end_idx = min(total_lines, start_idx + max_lines)
                    selected_lines = all_lines[start_idx:end_idx]
                actual_start = start_line
                actual_end = start_idx + len(selected_lines)
            else:
                # Read from beginning up to max_lines
                selected_lines = all_lines[:max_lines]
                actual_start = 1
                actual_end = len(selected_lines)
            
            # Clean lines (remove trailing newlines)
            content_lines = [line.rstrip('\n') for line in selected_lines]
            content = '\n'.join(content_lines)
            
            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "file_path": normalized_path,
                    "total_lines": total_lines,
                    "lines_read": len(content_lines),
                    "start_line": actual_start,
                    "end_line": actual_end,
                    "truncated": len(selected_lines) >= max_lines and actual_end < total_lines
                }
            )
            
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                data=None,
                error="File is not a text file or has invalid encoding"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Error reading file: {str(e)}"
            )


class DirectoryListTool(BaseTool):
    """Tool for listing directory contents"""
    
    @property
    def name(self) -> str:
        return "list_directory"
    
    @property
    def description(self) -> str:
        return "List files and directories in a given path. Use this to explore repository structure and find relevant files."
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Path to directory to list (relative to repository root, default: '.')"
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (starting with .)"
                },
                "file_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by file extensions (e.g., ['.py', '.js'])"
                }
            },
            "required": []
        }
    
    async def execute(self, directory_path: str = ".", show_hidden: bool = False,
                     file_types: Optional[List[str]] = None, **kwargs) -> ToolResult:
        """List directory contents with filtering"""
        try:
            # Security: ensure we stay within repository bounds
            normalized_path = os.path.normpath(directory_path)
            if normalized_path.startswith('..') or os.path.isabs(normalized_path):
                return ToolResult(
                    success=False,
                    data=None,
                    error="Directory path must be relative to repository root"
                )

            # Resolve to absolute path using repo context
            absolute_path = _resolve_path(normalized_path)

            if not os.path.exists(absolute_path):
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Directory not found: {normalized_path}"
                )

            if not os.path.isdir(absolute_path):
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Path is not a directory: {normalized_path}"
                )

            # List directory contents
            items = []
            for item in os.listdir(absolute_path):
                if not show_hidden and item.startswith('.'):
                    continue

                item_abs_path = os.path.join(absolute_path, item)
                item_rel_path = os.path.join(normalized_path, item)
                is_dir = os.path.isdir(item_abs_path)

                # Filter by file types if specified
                if file_types and not is_dir:
                    file_ext = os.path.splitext(item)[1]
                    if file_ext not in file_types:
                        continue
                
                items.append({
                    "name": item,
                    "type": "directory" if is_dir else "file",
                    "path": item_rel_path,
                    "extension": os.path.splitext(item)[1] if not is_dir else None
                })
            
            # Sort: directories first, then files
            items.sort(key=lambda x: (x["type"] == "file", x["name"]))
            
            return ToolResult(
                success=True,
                data={
                    "directory": normalized_path,
                    "items": items,
                    "total_items": len(items)
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Error listing directory: {str(e)}"
            )