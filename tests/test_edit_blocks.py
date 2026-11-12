"""Edit blocks: parsing what models write, and applying it to a real git checkout."""
import subprocess

import pytest

from app.benchmark.edit_apply import apply_edit_blocks
from app.core.edit_format import EDIT_FORMAT_INSTRUCTIONS, first_block_start, has_edit_block, parse_edit_blocks

SOURCE = 'def f(stats):\n    errors = stats["error"]\n    return errors\n\n\ndef g():\n    return 1\n'


def block(path, search, replace):
    return f"{path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


# ---------------------------------------------------------------- parsing

def test_plain_block():
    blocks = parse_edit_blocks("Here is the fix.\n\n" + block("pkg/mod.py", "    return errors", "    return errors + 1") + "\n\nDone.")
    assert len(blocks) == 1
    assert (blocks[0].path, blocks[0].search, blocks[0].replace) == ("pkg/mod.py", "    return errors", "    return errors + 1")


def test_the_example_in_the_instructions_parses():
    blocks = parse_edit_blocks(EDIT_FORMAT_INSTRUCTIONS)
    assert len(blocks) == 1 and blocks[0].path == "path/to/file.py"


@pytest.mark.parametrize("wrapped", [
    "```\npkg/mod.py\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n```",            # path inside the fence
    "pkg/mod.py\n```python\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n```",      # path above the fence
    "**`pkg/mod.py`**\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE",                # markdown decoration
    "File: pkg/mod.py\n\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE",              # label and blank line
    "### a/pkg/mod.py:\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE",               # diff-style prefix
])
def test_fences_and_decorated_paths_are_tolerated(wrapped):
    blocks = parse_edit_blocks(wrapped)
    assert [(b.path, b.search, b.replace) for b in blocks] == [("pkg/mod.py", "a", "b")]


@pytest.mark.parametrize("opening,closing", [
    ("<<<<<<<", ">>>>>>> REPLACE"),          # GPT-4.1-mini: bare opening marker
    ("<<<<<<< SEARCH", ">>>>>>>"),
    ("<<<<<<< ORIGINAL", ">>>>>>> UPDATED"),
])
def test_marker_variants_are_accepted(opening, closing):
    blocks = parse_edit_blocks(f"pkg/mod.py\n{opening}\nold\n=======\nnew\n{closing}")
    assert [(b.path, b.search, b.replace) for b in blocks] == [("pkg/mod.py", "old", "new")]


def test_other_angle_bracket_lines_are_not_markers():
    assert parse_edit_blocks("pkg/mod.py\n<<<<<<< HEAD\nold\n=======\nnew\n>>>>>>> branch") == []
    assert parse_edit_blocks("x << 7\n>>> print(1)\n=======") == []


def test_edit_written_without_the_divider_as_two_bracketed_parts():
    """Claude Haiku 4.5: SEARCH part closed with the closing marker, new lines in a second bracket."""
    text = ("Here's the patch:\n\npkg/mod.py\n<<<<<<< SEARCH\n    if alias:\n        return 1\n>>>>>>> REPLACE\n"
            "pkg/mod.py\n<<<<<<< SEARCH\n    return 2\n>>>>>>> REPLACE\n\nThe key change is the order.")
    assert [(b.path, b.search, b.replace) for b in parse_edit_blocks(text)] == [("pkg/mod.py", "    if alias:\n        return 1", "    return 2")]


def test_edit_written_without_the_divider_new_lines_follow_directly():
    text = "pkg/mod.py\n<<<<<<< SEARCH\nold = 1\n>>>>>>> REPLACE\n\nnew = 1\nnew2 = 2\n>>>>>>> REPLACE"
    assert [(b.search, b.replace) for b in parse_edit_blocks(text)] == [("old = 1", "new = 1\nnew2 = 2")]


def test_two_such_edits_in_a_row_and_no_false_recovery():
    pair = "a.py\n<<<<<<< SEARCH\n{}\n>>>>>>> REPLACE\na.py\n<<<<<<< SEARCH\n{}\n>>>>>>> REPLACE\n"
    blocks = parse_edit_blocks(pair.format("one", "1") + pair.format("two", "2"))
    assert [(b.search, b.replace) for b in blocks] == [("one", "1"), ("two", "2")]
    # a lone divider-less block is still not an edit
    assert parse_edit_blocks("a.py\n<<<<<<< SEARCH\nold\n>>>>>>> REPLACE\nThat is all.") == []
    # a divider-less block followed by a normal block: only the normal one counts
    mixed = "a.py\n<<<<<<< SEARCH\njunk\n>>>>>>> REPLACE\n" + block("b.py", "old", "new")
    assert [(b.path, b.search, b.replace) for b in parse_edit_blocks(mixed)] == [("b.py", "old", "new")]


def test_several_blocks_and_a_shared_path():
    text = block("a.py", "one", "1") + "\n\n<<<<<<< SEARCH\ntwo\n=======\n2\n>>>>>>> REPLACE\n\n" + block("b.py", "three", "3")
    assert [(b.path, b.search) for b in parse_edit_blocks(text)] == [("a.py", "one"), ("a.py", "two"), ("b.py", "three")]


def test_multiline_parts_keep_indentation_and_blank_lines():
    blocks = parse_edit_blocks(block("m.py", "def f():\n\n    return 1", "def f():\n\n    x = 2\n    return x"))
    assert blocks[0].search == "def f():\n\n    return 1"
    assert blocks[0].replace == "def f():\n\n    x = 2\n    return x"


def test_docstring_underlines_inside_the_code_do_not_split_a_block():
    search = '    """\n    Examples\n    ========\n    >>> f()\n    """'
    replace = '    """\n    Examples\n    ========\n    >>> f(1)\n    """'
    blocks = parse_edit_blocks(block("sympy/core/x.py", search, replace) + "\n\n" + block("docs/a.rst", "Title\n=====", "New title\n========="))
    assert [(b.search, b.replace) for b in blocks] == [(search, replace), ("Title\n=====", "New title\n=========")]


def test_incomplete_blocks_are_not_patches():
    assert parse_edit_blocks("m.py\n<<<<<<< SEARCH\na\n=======\nb") == []           # never closed
    assert parse_edit_blocks("m.py\n<<<<<<< SEARCH\na\n>>>>>>> REPLACE") == []      # no divider
    assert not has_edit_block("I would change line 3.") and not has_edit_block(None)


def test_block_without_a_usable_path_line_is_kept_and_placed_by_content(repo):
    # Gemini ran the path into the sentence before it: "...You MUST output a patch.pkg/mod.py"
    text = "You MUST output a patch.pkg/mod.py\n<<<<<<< SEARCH\n    return errors\n=======\n    return errors or 0\n>>>>>>> REPLACE"
    blocks = parse_edit_blocks(text)
    assert len(blocks) == 1 and blocks[0].path == ""
    assert apply_edit_blocks(str(repo), blocks).reason == "no_file_named"
    outcome = apply_edit_blocks(str(repo), blocks, files_read=["pkg/mod.py"])
    assert "+    return errors or 0" in outcome.patch
    assert outcome.resolved_paths == [{"block": 1, "given": "", "resolved": "pkg/mod.py"}]


def test_deletion_has_an_empty_replace_part():
    blocks = parse_edit_blocks("m.py\n<<<<<<< SEARCH\nold line\n=======\n>>>>>>> REPLACE")
    assert blocks[0].search == "old line" and blocks[0].replace == ""


def test_first_block_start_points_at_the_path_line():
    text = "Explanation.\n\n" + block("m.py", "a", "b")
    assert text[first_block_start(text):].startswith("m.py\n<<<<<<< SEARCH")
    assert first_block_start("no block here") == -1


# ---------------------------------------------------------------- applying

@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text(SOURCE)
    (root / "other.py").write_text("VALUE = 1\n")
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-C", str(root), *cmd], check=True)
    return root


def applies_cleanly(repo, patch) -> bool:
    """The diff we produce must be one that git itself accepts on a clean tree."""
    return subprocess.run(["git", "-C", str(repo), "apply", "--check", "-"], input=patch, text=True, capture_output=True).returncode == 0


def is_clean(repo) -> bool:
    return subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def test_matching_edit_becomes_a_valid_diff_and_the_tree_is_restored(repo):
    outcome = apply_edit_blocks(str(repo), parse_edit_blocks(block("pkg/mod.py", '    errors = stats["error"]', '    errors = stats.get("error", 0)')))
    assert outcome.failures == [] and outcome.blocks == 1
    assert '-    errors = stats["error"]' in outcome.patch and '+    errors = stats.get("error", 0)' in outcome.patch
    assert outcome.patch.startswith("diff --git a/pkg/mod.py b/pkg/mod.py")
    assert applies_cleanly(repo, outcome.patch)
    assert is_clean(repo) and (repo / "pkg" / "mod.py").read_text() == SOURCE


def test_search_text_must_match_exactly(repo):
    wrong_indent = apply_edit_blocks(str(repo), parse_edit_blocks(block("pkg/mod.py", 'errors = stats["error"]\nreturn errors', "x")))
    assert wrong_indent.patch is None and wrong_indent.reason == "search_text_not_found"
    invented = apply_edit_blocks(str(repo), parse_edit_blocks(block("pkg/mod.py", "    if emit:", "    pass")))
    assert invented.patch is None and invented.failures[0]["reason"] == "search_text_not_found"
    assert is_clean(repo)


def test_all_edits_must_apply_or_none_does(repo):
    text = block("pkg/mod.py", "    return 1", "    return 2") + "\n\n" + block("other.py", "NOT THERE", "x")
    outcome = apply_edit_blocks(str(repo), parse_edit_blocks(text))
    assert outcome.patch is None and [f["block"] for f in outcome.failures] == [2]
    assert is_clean(repo) and (repo / "pkg" / "mod.py").read_text() == SOURCE


def test_several_edits_in_one_file_and_across_files(repo):
    text = (block("pkg/mod.py", "    return errors", "    return errors or 0") + "\n\n" +
            block("pkg/mod.py", "    return 1", "    return 2") + "\n\n" + block("other.py", "VALUE = 1", "VALUE = 2"))
    outcome = apply_edit_blocks(str(repo), parse_edit_blocks(text))
    assert outcome.patch.count("diff --git") == 2
    assert "+    return errors or 0" in outcome.patch and "+    return 2" in outcome.patch and "+VALUE = 2" in outcome.patch
    assert applies_cleanly(repo, outcome.patch)


def test_only_the_first_occurrence_is_replaced(repo):
    (repo / "dup.py").write_text("a = 1\nb = 2\na = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "dup"], check=True)
    outcome = apply_edit_blocks(str(repo), parse_edit_blocks(block("dup.py", "a = 1", "a = 9")))
    assert outcome.patch.count("+a = 9") == 1 and outcome.patch.count("-a = 1") == 1
    assert "@@ -1,3 +1,3 @@" in outcome.patch


def test_missing_file_no_op_and_path_escape(repo):
    assert apply_edit_blocks(str(repo), parse_edit_blocks(block("nope.py", "a", "b"))).reason == "file_not_found"
    assert apply_edit_blocks(str(repo), parse_edit_blocks(block("other.py", "VALUE = 1", "VALUE = 1"))).reason == "no_change"
    assert apply_edit_blocks(str(repo), parse_edit_blocks(block("../outside.py", "a", "b"))).reason == "path_outside_repository"
    assert is_clean(repo)


def test_new_file(repo):
    outcome = apply_edit_blocks(str(repo), parse_edit_blocks("pkg/new.py\n<<<<<<< SEARCH\n=======\nX = 1\n>>>>>>> REPLACE"))
    assert "new file mode" in outcome.patch and "+X = 1" in outcome.patch
    assert applies_cleanly(repo, outcome.patch) and not (repo / "pkg" / "new.py").exists()


def test_nonexistent_path_is_resolved_through_the_files_the_agent_read(repo):
    """A model that copies the example's placeholder path still gets its (correct) edit applied."""
    edit = parse_edit_blocks(block("path/to/file.py", '    errors = stats["error"]', '    errors = stats.get("error")'))
    # without knowing what was read: fails as before
    assert apply_edit_blocks(str(repo), edit).reason == "file_not_found"
    # the agent read pkg/mod.py, which is the only read file containing the SEARCH text
    outcome = apply_edit_blocks(str(repo), edit, files_read=["other.py", "pkg/mod.py", "pkg/mod.py"])
    assert outcome.patch.startswith("diff --git a/pkg/mod.py b/pkg/mod.py")
    assert outcome.resolved_paths == [{"block": 1, "given": "path/to/file.py", "resolved": "pkg/mod.py"}]
    assert applies_cleanly(repo, outcome.patch) and is_clean(repo)


def test_path_resolution_needs_exactly_one_matching_file_that_was_read(repo):
    (repo / "copy.py").write_text(SOURCE)
    edit = parse_edit_blocks(block("path/to/file.py", "    return errors", "    return 0"))
    assert apply_edit_blocks(str(repo), edit, files_read=["pkg/mod.py", "copy.py"]).reason == "file_not_found"   # ambiguous
    assert apply_edit_blocks(str(repo), edit, files_read=["other.py"]).reason == "file_not_found"                # not in a read file
    assert apply_edit_blocks(str(repo), edit, files_read=["../etc/passwd"]).reason == "file_not_found"           # outside the repo


def test_an_existing_path_is_never_redirected(repo):
    edit = parse_edit_blocks(block("other.py", "    return errors", "    return 0"))  # wrong file, but it exists
    outcome = apply_edit_blocks(str(repo), edit, files_read=["pkg/mod.py"])
    assert outcome.patch is None and outcome.reason == "search_text_not_found" and outcome.resolved_paths == []


def test_revision_written_on_top_of_the_first_patch(repo):
    """A revision that edits the model's own first patch applies on that basis, and the
    result is one combined diff against the original file."""
    first = apply_edit_blocks(str(repo), parse_edit_blocks(block("pkg/mod.py", "    return errors", "    return errors or 0")))
    revision = parse_edit_blocks(block("pkg/mod.py", "    return errors or 0", "    return errors if errors is not None else 0"))
    assert apply_edit_blocks(str(repo), revision).reason == "search_text_not_found"      # not in the original
    stacked = apply_edit_blocks(str(repo), revision, base_patch=first.patch)
    assert "-    return errors\n" in stacked.patch and "+    return errors if errors is not None else 0" in stacked.patch
    assert "return errors or 0" not in stacked.patch                                    # the intermediate state is gone
    assert applies_cleanly(repo, stacked.patch) and is_clean(repo)


def test_base_patch_that_does_not_apply_is_reported(repo):
    outcome = apply_edit_blocks(str(repo), parse_edit_blocks(block("pkg/mod.py", "    return 1", "    return 2")),
                                base_patch="--- a/nope.py\n+++ b/nope.py\n@@ -1 +1 @@\n-a\n+b\n")
    assert outcome.patch is None and outcome.reason == "base_patch_failed" and is_clean(repo)
