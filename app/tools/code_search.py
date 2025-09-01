"""
Code search tool for finding patterns and symbols in repository
"""
import os
import subprocess
from typing import Dict, Any, Optional
from ..core.tools import BaseTool, ToolResult


class CodeSearchTool(BaseTool):
    """Tool for searching code patterns with regex - completely flexible for AI to use any pattern"""
    
    @property
    def name(self) -> str:
        return "search_code"
    
    @property
    def description(self) -> str:
        return "Search for any code pattern using regex. You can search for function definitions (e.g., 'def function_name'), class definitions (e.g., 'class ClassName'), variable usage, imports, or any custom pattern you need."
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for. Examples: 'def myfunction', 'class MyClass', 'import.*requests', 'myVariable.*=', 'async def.*'"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to search in (e.g., '*.py', '*.js', '*.java')",
                    "default": "*"
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to search in (relative to repo root, default: '.')",
                    "default": "."
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search should be case sensitive",
                    "default": False
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 20,
                    "minimum": 1
                }
            },
            "required": ["pattern"]
        }
    
    async def execute(self, pattern: str, file_pattern: str = "*", directory: str = ".", 
                     case_sensitive: bool = False, max_results: int = 20, **kwargs) -> ToolResult:
        """Search for code patterns in the repository"""
        try:
            # Security: ensure we stay within repository bounds
            normalized_dir = os.path.normpath(directory)
            if normalized_dir.startswith('..') or os.path.isabs(normalized_dir):
                return ToolResult(
                    success=False,
                    data=None,
                    error="Directory path must be relative to repository root"
                )
            
            # Build search command
            try:
                # Try ripgrep first (faster and better)
                cmd = ["rg", "-n"]
                if not case_sensitive:
                    cmd.append("-i")
                
                # Add file type filtering if specified
                if file_pattern != "*":
                    if file_pattern.startswith("*."):
                        ext = file_pattern[2:]
                        cmd.extend(["--type-add", f"custom:*.{ext}", "--type", "custom"])
                    else:
                        cmd.extend(["-g", file_pattern])
                
                cmd.extend([pattern, normalized_dir])
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0 and result.returncode != 1:
                    raise subprocess.CalledProcessError(result.returncode, cmd)
                output = result.stdout
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to grep
                cmd = ["grep", "-rn"]
                if not case_sensitive:
                    cmd.append("-i")
                cmd.append("-E")  # Extended regex
                
                if file_pattern != "*":
                    cmd.extend(["--include", file_pattern])
                
                cmd.extend([pattern, normalized_dir])
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                output = result.stdout
            
            # Parse results
            matches = []
            lines = output.strip().split('\n') if output.strip() else []
            
            for line in lines[:max_results]:
                if ':' in line:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_number = parts[1]
                        content = parts[2]
                        
                        # Make file path relative to repo root
                        rel_path = os.path.relpath(file_path, '.')
                        
                        matches.append({
                            "file_path": rel_path,
                            "line_number": int(line_number) if line_number.isdigit() else 0,
                            "content": content.strip(),
                            "pattern": pattern
                        })
            
            return ToolResult(
                success=True,
                data={
                    "pattern": pattern,
                    "matches": matches,
                    "total_matches": len(matches),
                    "truncated": len(lines) > max_results,
                    "case_sensitive": case_sensitive
                }
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                data=None,
                error="Search operation timed out"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Error searching code: {str(e)}"
            )