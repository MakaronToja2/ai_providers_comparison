"""
The answer format of the agent: search-and-replace edit blocks.

The model does not write a unified diff. It names a file, quotes the existing lines and
gives the lines that replace them; the harness applies the edits to the checkout and
produces the diff itself (app/benchmark/edit_apply.py). This module holds the format
instructions shown to the model and the parser. It touches no files.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

SEARCH_MARKER = "<<<<<<< SEARCH"
DIVIDER_MARKER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"

# Shown verbatim in the system prompt, the final-answer prompts and the reflection prompt,
# identically in every regime.
EDIT_FORMAT_INSTRUCTIONS = """Write your patch as one or more edit blocks. Each block names a file and replaces existing lines with new lines:

path/to/file.py
<<<<<<< SEARCH
existing lines, copied exactly from the file
=======
the lines that replace them
>>>>>>> REPLACE

Rules for edit blocks:
- The file path is relative to the repository root and stands on its own line directly above the block.
- The SEARCH part must match the file character for character, including indentation. Copy it from the file contents you read.
- Keep each SEARCH part short but unique in the file. Use several blocks for several changes.
- To add new lines, put a nearby existing line in SEARCH and repeat it in REPLACE together with the new lines.
- Write complete code. Do not use placeholders such as "..." or "rest of the function"."""

# The opening and closing markers are accepted with or without their word: models sometimes
# write a bare "<<<<<<<" as in a merge conflict.
_SEARCH_LINE = re.compile(r"^<{5,9}(?: ?(?:SEARCH|ORIGINAL))?\s*$")
# Exactly seven: reStructuredText underlines of other lengths inside the code must not split a block
_DIVIDER_LINE = re.compile(r"^={7}\s*$")
_REPLACE_LINE = re.compile(r"^>{5,9}(?: ?(?:REPLACE|UPDATED))?\s*$")
_FENCE_LINE = re.compile(r"^\s*```[\w+.-]*\s*$")


@dataclass
class EditBlock:
    path: str
    search: str
    replace: str


def has_edit_block(text: Optional[str]) -> bool:
    """Whether the text contains at least one complete edit block."""
    return bool(parse_edit_blocks(text))


def _clean_path(line: str) -> Optional[str]:
    """A file path from the line above a block, tolerating common markdown decoration."""
    candidate = line.strip()
    candidate = re.sub(r"^(?:#+\s*|[-*]\s+|File:\s*|file:\s*)", "", candidate)
    candidate = candidate.strip("`*\"' ").rstrip(":").strip("`*\"' ")
    if not candidate or " " in candidate:
        return None
    if candidate.startswith(("a/", "b/")) and "/" in candidate[2:]:
        candidate = candidate[2:]
    return candidate if ("/" in candidate or "." in candidate) else None


def _split_without_divider(lines: List[str], close: int, path: Optional[str]):
    """Recover an edit written without the ======= line.

    Some answers close the SEARCH part with the closing marker and then give the new lines
    up to a second closing marker, optionally repeating the path and the opening marker:

        path                      path
        <<<<<<< SEARCH            <<<<<<< SEARCH
        old lines                 old lines
        >>>>>>> REPLACE           >>>>>>> REPLACE
        path                      new lines
        <<<<<<< SEARCH            >>>>>>> REPLACE
        new lines
        >>>>>>> REPLACE

    `close` is the index of the first closing marker. Returns (replace lines, index of the
    second closing marker), or None if what follows is not of this shape (in particular if
    a divider appears before the next closing marker: then the next block is a normal one).
    """
    j = close + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    # an optional repeated path line, then an optional opening marker
    if j < len(lines) and not _SEARCH_LINE.match(lines[j]):
        ahead = j + 1
        while ahead < len(lines) and not lines[ahead].strip():
            ahead += 1
        if ahead < len(lines) and _SEARCH_LINE.match(lines[ahead]) and _clean_path(lines[j]) is not None:
            j = ahead
    if j < len(lines) and _SEARCH_LINE.match(lines[j]):
        j += 1

    replace: List[str] = []
    while j < len(lines):
        if _REPLACE_LINE.match(lines[j]):
            while replace and not replace[0].strip():
                replace.pop(0)
            while replace and not replace[-1].strip():
                replace.pop()
            return replace, j
        if _DIVIDER_LINE.match(lines[j]) or _SEARCH_LINE.match(lines[j]):
            return None
        replace.append(lines[j])
        j += 1
    return None


def parse_edit_blocks(text: Optional[str]) -> List[EditBlock]:
    """Parse all complete edit blocks in a message, in order.

    The path is the nearest non-empty line above the block (a code-fence line in between
    is skipped). A block with no path line of its own reuses the path of the block
    before it, or has an empty path if there is none. Incomplete blocks are ignored.
    """
    if not text:
        return []

    lines = text.split("\n")
    blocks: List[EditBlock] = []
    last_path: Optional[str] = None
    i = 0
    block_end = -1  # index of the last line consumed by a block

    while i < len(lines):
        if not _SEARCH_LINE.match(lines[i]):
            i += 1
            continue

        # Path: look upwards, but not into the previous block
        path = None
        j = i - 1
        while j > block_end:
            if lines[j].strip() and not _FENCE_LINE.match(lines[j]):
                path = _clean_path(lines[j])
                break
            j -= 1
        path = path or last_path

        # SEARCH part
        k = i + 1
        search: List[str] = []
        while k < len(lines) and not _DIVIDER_LINE.match(lines[k]):
            if _SEARCH_LINE.match(lines[k]) or _REPLACE_LINE.match(lines[k]):
                break
            search.append(lines[k])
            k += 1
        if k < len(lines) and _REPLACE_LINE.match(lines[k]):
            # No divider: the closing marker was used in its place (see _split_without_divider)
            split = _split_without_divider(lines, k, path)
            if split is not None:
                replace, m = split
                blocks.append(EditBlock(path=path or "", search="\n".join(search), replace="\n".join(replace)))
                last_path = path or last_path
                block_end = m
                i = m + 1
                continue
        if k >= len(lines) or not _DIVIDER_LINE.match(lines[k]):
            i += 1
            continue

        # REPLACE part
        m = k + 1
        replace: List[str] = []
        while m < len(lines) and not _REPLACE_LINE.match(lines[m]):
            if _SEARCH_LINE.match(lines[m]):
                break
            replace.append(lines[m])
            m += 1
        if m >= len(lines) or not _REPLACE_LINE.match(lines[m]):
            i += 1
            continue

        # A complete block with no recognisable path line is kept with an empty path; the
        # applier can still place it through the files the agent read.
        blocks.append(EditBlock(path=path or "", search="\n".join(search), replace="\n".join(replace)))
        last_path = path or last_path
        block_end = m
        i = m + 1

    return blocks


def first_block_start(text: str) -> int:
    """Character offset where the first edit block (its path line) starts, or -1."""
    lines = text.split("\n")
    offset = 0
    for index, line in enumerate(lines):
        if _SEARCH_LINE.match(line):
            start = offset
            j = index - 1
            back = offset
            while j >= 0:
                back -= len(lines[j]) + 1
                if lines[j].strip() and not _FENCE_LINE.match(lines[j]):
                    if _clean_path(lines[j]):
                        start = back
                    break
                j -= 1
            return start
        offset += len(line) + 1
    return -1
