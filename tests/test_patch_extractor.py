"""Unified-diff fallback: which diff is taken when a model writes a diff instead of edit blocks."""
from app.benchmark.patch_extractor import patch_extractor

A1 = "```diff\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n```"
A2 = "```diff\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 3\n```"
B1 = "```diff\n--- a/b.py\n+++ b/b.py\n@@ -5,1 +5,1 @@\n-y = 1\n+y = 2\n```"


def test_single_block():
    patch, blocks = patch_extractor.extract_final(f"Here is the fix.\n{A1}\nDone.")
    assert blocks == 1 and "+x = 2" in patch and patch.startswith("--- a/a.py")


def test_blocks_on_different_files_are_joined_in_order():
    patch, blocks = patch_extractor.extract_final(f"{A1}\nand\n{B1}")
    assert blocks == 2
    assert patch.index("--- a/a.py") < patch.index("--- a/b.py")
    assert "+x = 2" in patch and "+y = 2" in patch


def test_blocks_on_the_same_file_take_the_last():
    patch, blocks = patch_extractor.extract_final(f"First idea:\n{A1}\nBetter:\n{A2}")
    assert blocks == 2 and "+x = 3" in patch and "+x = 2" not in patch


def test_non_diff_code_blocks_are_ignored():
    patch, blocks = patch_extractor.extract_final(f"```python\nprint('hi')\n```\n{A1}")
    assert blocks == 1 and "+x = 2" in patch


def test_no_patch():
    assert patch_extractor.extract_final("I could not find the bug.") == (None, 0)
    assert patch_extractor.extract_final("") == (None, 0)


def test_inline_diff_without_fence_is_still_found():
    patch, blocks = patch_extractor.extract_final("--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n")
    assert blocks == 1 and "+x = 2" in patch
