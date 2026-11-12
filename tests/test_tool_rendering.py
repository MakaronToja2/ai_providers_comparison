"""Tool results reach the model as plain text, byte-for-byte for source code."""
import pytest

from app.core.base_provider import BaseAIProvider
from app.core.tool_manager import initialize_tools
from app.utils.repo_context import current_repo_path

# Source with everything JSON would escape: quotes, backslashes, tabs, blank lines
SOURCE = (
    'def f(stats):\n'
    '    errors = stats["error"]  # "quoted"\n'
    '\n'
    "    pattern = r'\\d+\\n' + \"a\\\\b\"\n"
    '\treturn errors \\\n'
    '        + 1\n'
)


class Provider(BaseAIProvider):
    provider_name = "check"

    def _validate_api_key(self):
        pass

    async def _make_api_call(self, *args, **kwargs):
        raise NotImplementedError

    def _parse_response(self, raw_response):
        raise NotImplementedError


@pytest.fixture
def provider(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(SOURCE)
    (tmp_path / "README.md").write_text("hello\n")
    initialize_tools()
    token = current_repo_path.set(str(tmp_path))
    yield Provider(api_key="x", model="m")
    current_repo_path.reset(token)


async def shown(provider, name, arguments):
    call = await provider._execute_tool_call(name, arguments)
    assert call.success, call.error
    return provider._tool_result_text(call), call


async def test_read_file_is_shown_exactly_as_on_disk(provider):
    text, call = await shown(provider, "read_file", {"file_path": "pkg/mod.py"})
    header, body = text.split("\n\n", 1)
    assert header == "File: pkg/mod.py\nLines 1-6 of 6"
    assert body == SOURCE.rstrip("\n")
    # every source line can be copied verbatim into a diff
    for line in SOURCE.splitlines():
        assert line in text.split("\n")
    assert '\\"' not in text and "\\n    errors" not in text  # no JSON escaping
    assert call.result["data"]["content"] == body  # the structured result is still stored


async def test_read_file_range_and_truncation_are_stated(provider):
    text, _ = await shown(provider, "read_file", {"file_path": "pkg/mod.py", "start_line": 2, "end_line": 3})
    assert text.startswith("File: pkg/mod.py\nLines 2-3 of 6\n\n")
    text, _ = await shown(provider, "read_file", {"file_path": "pkg/mod.py", "max_lines": 2})
    assert "Lines 1-2 of 6 (truncated: more lines follow)" in text


async def test_search_code_is_grep_style(provider):
    text, _ = await shown(provider, "search_code", {"pattern": "stats\\["})
    lines = text.split("\n")
    assert lines[:4] == ["Search pattern: stats\\[", "Case sensitive: no", "Matches: 1", ""]
    assert lines[4] == 'pkg/mod.py:2: errors = stats["error"]  # "quoted"'


async def test_search_without_matches(provider):
    text, _ = await shown(provider, "search_code", {"pattern": "no_such_symbol_anywhere"})
    assert text.rstrip("\n").endswith("Matches: 0")


async def test_list_directory_is_one_entry_per_line(provider):
    text, _ = await shown(provider, "list_directory", {"directory_path": "."})
    # paths are the tool's own (relative to the listed directory, here the repo root)
    assert text.split("\n") == ["Directory: .", "Items: 2", "", "[dir]  ./pkg/", "[file] ./README.md"]


async def test_errors_are_unchanged(provider):
    call = await provider._execute_tool_call("read_file", {"file_path": "missing.py"})
    assert provider._tool_result_text(call) == "Error: File not found: missing.py"
