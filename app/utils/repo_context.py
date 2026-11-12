"""
Repository context manager for SWE-bench analysis
"""
import asyncio
import fcntl
import os
import shutil
import subprocess
import tempfile
from contextvars import ContextVar
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional


# Context variable to track current repo path per async task
current_repo_path: ContextVar[Optional[str]] = ContextVar('current_repo_path', default=None)

# One bare mirror per GitHub repository is kept here and every instance is cloned from
# it locally. The checked-out tree is the same as a clone from GitHub; only the source
# of the objects differs.
MIRROR_DIR = Path(os.environ.get(
    "REPO_MIRROR_DIR",
    Path(__file__).resolve().parents[2] / ".cache" / "repo_mirrors",
))

MIRROR_CLONE_TIMEOUT = 3600
LOCAL_CLONE_TIMEOUT = 600
CHECKOUT_TIMEOUT = 300


def get_current_repo_path() -> Optional[str]:
    """Get the current repository path for this async task."""
    return current_repo_path.get()


def _run_git(cmd: list, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def ensure_mirror(repo_name: str) -> Path:
    """Return the local bare mirror of a GitHub repository, creating it on first use.

    A file lock makes this safe when several tasks or processes need the same mirror.
    """
    MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    mirror_path = MIRROR_DIR / (repo_name.replace("/", "__") + ".git")
    lock_path = MIRROR_DIR / (repo_name.replace("/", "__") + ".lock")

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if not (mirror_path / "HEAD").exists():
                tmp_path = mirror_path.with_suffix(".git.tmp")
                shutil.rmtree(tmp_path, ignore_errors=True)
                github_url = f"https://github.com/{repo_name}.git"
                result = _run_git(["git", "clone", "--mirror", github_url, str(tmp_path)], MIRROR_CLONE_TIMEOUT)
                if result.returncode != 0:
                    shutil.rmtree(tmp_path, ignore_errors=True)
                    raise Exception(f"Failed to mirror repository {repo_name}: {result.stderr}")
                tmp_path.rename(mirror_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return mirror_path


def _clone_and_checkout(repo_name: str, commit_hash: str, repo_path: str) -> None:
    mirror_path = ensure_mirror(repo_name)

    # Clone repository (from the local mirror)
    result = _run_git(["git", "clone", "--no-checkout", str(mirror_path), repo_path], LOCAL_CLONE_TIMEOUT)
    if result.returncode != 0:
        raise Exception(f"Failed to clone repository: {result.stderr}")

    # Checkout specific commit
    result = _run_git(["git", "-C", repo_path, "checkout", commit_hash], CHECKOUT_TIMEOUT)
    if result.returncode != 0:
        raise Exception(f"Failed to checkout commit {commit_hash}: {result.stderr}")


class RepoContext:
    """Context manager for repository analysis sessions"""

    def __init__(self, repo_name: str, commit_hash: str):
        self.repo_name = repo_name
        self.commit_hash = commit_hash
        self.temp_dir: Optional[str] = None
        self.repo_path: Optional[str] = None
        self._token = None

    async def __aenter__(self):
        """Clone repository and set context variable"""
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="swe_bench_")
            self.repo_path = os.path.join(self.temp_dir, "repo")

            # Git runs in a worker thread so other instances keep making progress
            await asyncio.to_thread(_clone_and_checkout, self.repo_name, self.commit_hash, self.repo_path)

            # Set context variable for this async task (no global chdir!)
            self._token = current_repo_path.set(self.repo_path)

            return self

        except Exception as e:
            # Clean up on failure
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            raise e

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up cloned repository"""
        # Reset context variable
        if self._token is not None:
            current_repo_path.reset(self._token)

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
