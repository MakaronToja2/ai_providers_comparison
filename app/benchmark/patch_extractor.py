"""
Patch extraction from LLM responses.
"""
import re
from typing import Optional, List, Tuple


class PatchExtractor:
    """Extracts unified diff patches from LLM response text."""

    # Patterns for finding code blocks
    DIFF_BLOCK_PATTERN = re.compile(r'```(?:diff)?\s*\n(.*?)```', re.DOTALL)
    PATCH_BLOCK_PATTERN = re.compile(r'```(?:patch)?\s*\n(.*?)```', re.DOTALL)

    # Patterns for identifying diff content
    DIFF_HEADER_PATTERN = re.compile(r'^(diff --git|---|\+\+\+|@@)', re.MULTILINE)
    HUNK_PATTERN = re.compile(r'^@@\s*-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s*@@', re.MULTILINE)

    def extract(self, content: str) -> Optional[str]:
        """
        Extract a patch from response content.

        Returns the first valid patch found, or None if no patch detected.
        """
        if not content:
            return None

        # Try explicit diff/patch code blocks first
        for pattern in [self.DIFF_BLOCK_PATTERN, self.PATCH_BLOCK_PATTERN]:
            matches = pattern.findall(content)
            for match in matches:
                if self._is_valid_patch(match):
                    return self._normalize_patch(match)

        # Try any code block that looks like a diff
        generic_pattern = re.compile(r'```\w*\s*\n(.*?)```', re.DOTALL)
        for match in generic_pattern.findall(content):
            if self._is_valid_patch(match):
                return self._normalize_patch(match)

        # Last resort: look for inline diff content
        if self.HUNK_PATTERN.search(content):
            # Try to extract just the diff portion
            lines = content.split('\n')
            patch_lines = []
            in_patch = False

            for line in lines:
                if line.startswith(('diff --git', '---', '+++')):
                    in_patch = True
                    patch_lines.append(line)
                elif in_patch:
                    if line.startswith((' ', '+', '-', '@@', '\\')):
                        patch_lines.append(line)
                    elif line.strip() == '':
                        patch_lines.append(line)
                    else:
                        # End of patch section
                        if patch_lines:
                            patch = '\n'.join(patch_lines)
                            if self._is_valid_patch(patch):
                                return self._normalize_patch(patch)
                        patch_lines = []
                        in_patch = False

            if patch_lines:
                patch = '\n'.join(patch_lines)
                if self._is_valid_patch(patch):
                    return self._normalize_patch(patch)

        return None

    def _is_valid_patch(self, text: str) -> bool:
        """Check if text looks like a valid unified diff patch."""
        text = text.strip()
        if not text:
            return False

        # Must have at least one hunk header or diff header
        has_hunk = bool(self.HUNK_PATTERN.search(text))
        has_header = bool(self.DIFF_HEADER_PATTERN.search(text))

        if not (has_hunk or has_header):
            return False

        # Should have some + or - lines (actual changes)
        lines = text.split('\n')
        has_additions = any(l.startswith('+') and not l.startswith('+++') for l in lines)
        has_deletions = any(l.startswith('-') and not l.startswith('---') for l in lines)

        return has_additions or has_deletions

    def _normalize_patch(self, patch: str) -> str:
        """Normalize a patch by cleaning up formatting and fixing common issues."""
        lines = patch.strip().split('\n')
        normalized = []
        in_hunk = False
        hunk_lines = []  # Collect lines for current hunk to recalculate header
        hunk_header_idx = -1

        for line in lines:
            # Remove trailing whitespace but preserve leading
            line = line.rstrip()

            # Track when we're inside a hunk (after @@ line)
            if line.startswith('@@'):
                # If we had a previous hunk, finalize it
                if hunk_lines and hunk_header_idx >= 0:
                    self._fix_hunk_header(normalized, hunk_header_idx, hunk_lines)
                    normalized.extend(hunk_lines)
                    hunk_lines = []

                in_hunk = True
                hunk_header_idx = len(normalized)
                normalized.append(line)  # Will be fixed later
                continue

            # Normalize --- and +++ headers to have a/ and b/ prefixes
            if line.startswith('--- '):
                in_hunk = False
                # Finalize any pending hunk
                if hunk_lines and hunk_header_idx >= 0:
                    self._fix_hunk_header(normalized, hunk_header_idx, hunk_lines)
                    normalized.extend(hunk_lines)
                    hunk_lines = []
                    hunk_header_idx = -1

                path = line[4:]
                # Add a/ prefix if not present and not /dev/null
                if not path.startswith(('a/', '/dev/null')):
                    path = 'a/' + path
                normalized.append('--- ' + path)
                continue

            if line.startswith('+++ '):
                in_hunk = False
                path = line[4:]
                # Add b/ prefix if not present and not /dev/null
                if not path.startswith(('b/', '/dev/null')):
                    path = 'b/' + path
                normalized.append('+++ ' + path)
                continue

            if line.startswith('diff --git'):
                in_hunk = False
                # Finalize any pending hunk
                if hunk_lines and hunk_header_idx >= 0:
                    self._fix_hunk_header(normalized, hunk_header_idx, hunk_lines)
                    normalized.extend(hunk_lines)
                    hunk_lines = []
                    hunk_header_idx = -1
                normalized.append(line)
                continue

            # Inside a hunk, lines should start with +, -, space, or \
            if in_hunk:
                if line.startswith(('+', '-', ' ', '\\')):
                    hunk_lines.append(line)
                elif line == '':
                    # Empty line in hunk should be a context line (just a space)
                    hunk_lines.append(' ')
                else:
                    # Line without proper prefix - assume it's a context line
                    # This fixes LLMs that forget the leading space
                    hunk_lines.append(' ' + line)
            else:
                normalized.append(line)

        # Finalize last hunk
        if hunk_lines and hunk_header_idx >= 0:
            self._fix_hunk_header(normalized, hunk_header_idx, hunk_lines)
            normalized.extend(hunk_lines)

        # Ensure patch ends with newline (required by git)
        result = '\n'.join(normalized)
        if not result.endswith('\n'):
            result += '\n'

        return result

    def _fix_hunk_header(self, lines: List[str], header_idx: int, hunk_lines: List[str]) -> None:
        """Fix the hunk header at header_idx to match actual line counts."""
        if header_idx < 0 or header_idx >= len(lines):
            return

        old_header = lines[header_idx]
        match = re.match(r'^@@\s*-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s*@@(.*)$', old_header)
        if not match:
            return

        old_start = match.group(1)
        new_start = match.group(2)
        context_after = match.group(3)

        # Count lines in hunk
        old_count = 0
        new_count = 0
        for line in hunk_lines:
            if line.startswith('-'):
                old_count += 1
            elif line.startswith('+'):
                new_count += 1
            elif line.startswith(' ') or line.startswith('\\'):
                old_count += 1
                new_count += 1

        # Build corrected header
        new_header = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{context_after}"
        lines[header_idx] = new_header

    def extract_all(self, content: str) -> List[str]:
        """Extract all patches from content."""
        if not content:
            return []

        patches = []

        # Try code blocks
        for pattern in [self.DIFF_BLOCK_PATTERN, self.PATCH_BLOCK_PATTERN]:
            for match in pattern.findall(content):
                if self._is_valid_patch(match):
                    normalized = self._normalize_patch(match)
                    if normalized not in patches:
                        patches.append(normalized)

        return patches

    def extract_files_changed(self, patch: str) -> List[str]:
        """Extract list of files changed in a patch."""
        if not patch:
            return []

        files = set()

        # Look for diff --git headers
        git_pattern = re.compile(r'diff --git a/(.+?) b/(.+)')
        for match in git_pattern.finditer(patch):
            files.add(match.group(2))  # Use the 'b' path

        # Look for +++ headers
        plus_pattern = re.compile(r'^\+\+\+ (?:b/)?(.+)$', re.MULTILINE)
        for match in plus_pattern.finditer(patch):
            path = match.group(1)
            if path != '/dev/null':
                files.add(path)

        return sorted(files)

    def count_changes(self, patch: str) -> Tuple[int, int]:
        """Count lines added and removed in a patch."""
        if not patch:
            return 0, 0

        additions = 0
        deletions = 0

        for line in patch.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                additions += 1
            elif line.startswith('-') and not line.startswith('---'):
                deletions += 1

        return additions, deletions


# Global instance
patch_extractor = PatchExtractor()
