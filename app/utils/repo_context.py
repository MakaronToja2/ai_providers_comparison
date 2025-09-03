"""
Repository context manager for SWE-bench analysis
"""
import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from typing import Optional


class RepoContext:
    """Context manager for repository analysis sessions"""
    
    def __init__(self, repo_name: str, commit_hash: str):
        self.repo_name = repo_name
        self.commit_hash = commit_hash
        self.temp_dir: Optional[str] = None
        self.repo_path: Optional[str] = None
        self.original_cwd: Optional[str] = None
    
    async def __aenter__(self):
        """Clone repository and change working directory"""
        try:
            # Store original directory
            self.original_cwd = os.getcwd()
            
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="swe_bench_")
            self.repo_path = os.path.join(self.temp_dir, "repo")
            
            # Build GitHub URL
            github_url = f"https://github.com/{self.repo_name}.git"
            
            # Clone repository
            clone_cmd = ["git", "clone", github_url, self.repo_path]
            result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                raise Exception(f"Failed to clone repository: {result.stderr}")
            
            # Checkout specific commit
            checkout_cmd = ["git", "-C", self.repo_path, "checkout", self.commit_hash]
            result = subprocess.run(checkout_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise Exception(f"Failed to checkout commit {self.commit_hash}: {result.stderr}")
            
            # Change to repository directory for file operations
            os.chdir(self.repo_path)
            
            return self
            
        except Exception as e:
            # Clean up on failure
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            if self.original_cwd:
                os.chdir(self.original_cwd)
            raise e
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up cloned repository"""
        # Restore original directory
        if self.original_cwd:
            os.chdir(self.original_cwd)
        
        # Clean up temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @property
    def working_directory(self) -> str:
        """Get the current working directory (should be repo root)"""
        return self.repo_path or ""


@asynccontextmanager
async def repo_context(repo_name: str, commit_hash: str):
    """Async context manager for repository analysis"""
    context = RepoContext(repo_name, commit_hash)
    async with context:
        yield context