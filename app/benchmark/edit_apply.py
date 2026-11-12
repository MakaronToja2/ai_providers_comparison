"""
Apply the model's edit blocks to a repository checkout and turn them into a unified diff.

An edit applies only if its SEARCH text occurs verbatim in the file; the first occurrence
is replaced. An answer applies only if every one of its edits applies (all or nothing,
like a patch whose hunks must all apply). The diff is produced by git, so it is valid by
construction. The checkout is restored afterwards.

One tolerance: if the file named by an edit does not exist, or no file is named (models
sometimes copy the placeholder path of the format example or run the path into the
preceding sentence), the edit is applied to the one file, among the
files the agent read during the run, that contains the SEARCH text verbatim. If no file
or more than one file qualifies, the edit fails as before. Every such resolution is recorded.

A revision from the self-reflection call is applied to the original files. If its edits do
not match there, it is applied on top of the model's first patch instead (models often
write a revision as a change to their own patch); the final patch is then the combined
diff against the original. Which basis was used is recorded.
"""
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.edit_format import EditBlock

GIT_TIMEOUT = 120


@dataclass
class EditOutcome:
    patch: Optional[str] = None  # unified diff, when every edit applied and changed something
    blocks: int = 0
    failures: List[Dict[str, Any]] = field(default_factory=list)
    resolved_paths: List[Dict[str, Any]] = field(default_factory=list)  # edits whose named file did not exist

    @property
    def reason(self) -> Optional[str]:
        """Why there is no patch, in one word per failed edit."""
        if self.patch:
            return None
        if self.failures:
            return ", ".join(sorted({f["reason"] for f in self.failures}))
        return "no_change"


def _git(repo_path: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "core.fileMode=false", "-C", repo_path, *args],
        capture_output=True, text=True, timeout=GIT_TIMEOUT,
    )


def _restore(repo_path: str) -> None:
    _git(repo_path, "reset", "--hard", "-q")
    _git(repo_path, "clean", "-fdq")


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except (UnicodeDecodeError, OSError):
        return None


def _resolve_by_content(root: str, search: str, files_read: List[str], contents: Dict[str, str]) -> Optional[str]:
    """The single file the agent read that contains the SEARCH text, or None."""
    matches = []
    for relative in dict.fromkeys(os.path.normpath(f) for f in files_read):
        absolute = os.path.realpath(os.path.join(root, relative))
        if not absolute.startswith(root + os.sep) or not os.path.isfile(absolute):
            continue
        text = contents[absolute] if absolute in contents else _read_text(absolute)
        if text is not None and search in text:
            matches.append(absolute)
    return matches[0] if len(matches) == 1 else None


def apply_edit_blocks(repo_path: str, blocks: List[EditBlock], files_read: Optional[List[str]] = None,
                      base_patch: Optional[str] = None) -> EditOutcome:
    """Apply edit blocks in order and return the resulting diff (the checkout is left clean).

    files_read: repository-relative paths the agent read during the run, used only to
    resolve an edit whose named file does not exist.
    base_patch: a diff to apply to the checkout first. The edits are then matched against
    the patched files, and the returned diff is the combined change against the original.
    Used for a revision that was written on top of the model's own first patch.
    """
    outcome = EditOutcome(blocks=len(blocks))
    if not blocks:
        return outcome

    root = os.path.realpath(repo_path)
    if base_patch:
        applied = subprocess.run(
            ["git", "-c", "core.fileMode=false", "-C", repo_path, "apply", "-"],
            input=base_patch, capture_output=True, text=True, timeout=GIT_TIMEOUT,
        )
        if applied.returncode != 0:
            _restore(repo_path)
            outcome.failures.append({"block": 0, "path": "", "reason": "base_patch_failed"})
            return outcome

    try:
        return _apply_to_tree(repo_path, root, blocks, files_read, outcome)
    finally:
        _restore(repo_path)


def _apply_to_tree(repo_path: str, root: str, blocks: List[EditBlock], files_read: Optional[List[str]],
                   outcome: EditOutcome) -> EditOutcome:
    contents: Dict[str, str] = {}  # files as edited so far, written only if all edits apply

    for index, block in enumerate(blocks, 1):
        if block.path:
            relative = os.path.normpath(block.path)
            absolute = os.path.realpath(os.path.join(root, relative))
            if relative.startswith("..") or os.path.isabs(relative) or not absolute.startswith(root + os.sep):
                outcome.failures.append({"block": index, "path": block.path, "reason": "path_outside_repository"})
                continue
        else:
            absolute = ""  # no file named: placed by content below, or the edit fails

        if absolute not in contents and not (absolute and os.path.isfile(absolute)) and block.search.strip() and files_read:
            resolved = _resolve_by_content(root, block.search, files_read, contents)
            if resolved:
                outcome.resolved_paths.append({
                    "block": index, "given": block.path, "resolved": os.path.relpath(resolved, root),
                })
                absolute = resolved

        if not absolute:
            outcome.failures.append({"block": index, "path": "", "reason": "no_file_named"})
            continue

        if absolute not in contents:
            if os.path.isfile(absolute):
                try:
                    with open(absolute, "r", encoding="utf-8", newline="") as handle:
                        contents[absolute] = handle.read()
                except UnicodeDecodeError:
                    outcome.failures.append({"block": index, "path": block.path, "reason": "file_not_text"})
                    continue
            elif block.search.strip() == "":
                contents[absolute] = ""  # a new file: empty SEARCH, REPLACE is the content
            else:
                outcome.failures.append({"block": index, "path": block.path, "reason": "file_not_found"})
                continue

        text = contents[absolute]
        if block.search.strip() == "":
            if text != "":
                outcome.failures.append({"block": index, "path": block.path, "reason": "empty_search"})
                continue
            contents[absolute] = block.replace if block.replace.endswith("\n") else block.replace + "\n"
            continue

        occurrences = text.count(block.search)
        if occurrences == 0:
            outcome.failures.append({"block": index, "path": block.path, "reason": "search_text_not_found"})
            continue
        contents[absolute] = text.replace(block.search, block.replace, 1)

    if outcome.failures:
        return outcome

    for absolute, text in contents.items():
        os.makedirs(os.path.dirname(absolute), exist_ok=True)
        with open(absolute, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
    _git(repo_path, "add", "-A")
    diff = _git(repo_path, "diff", "--cached", "--no-color")
    if diff.returncode == 0 and diff.stdout.strip():
        outcome.patch = diff.stdout

    return outcome
