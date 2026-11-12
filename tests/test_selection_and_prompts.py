"""Instance selection is deterministic and documented; prompts carry no hints."""
import importlib.util
import json
from pathlib import Path

from app.benchmark.runner import ExperimentRunner, WorkItem
from app.models.benchmark import ExperimentMode

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("select_instances", ROOT / "scripts" / "select_instances.py")
select_instances = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select_instances)

LITE_TEST = {
    "django/django": 114, "sympy/sympy": 77, "matplotlib/matplotlib": 23, "scikit-learn/scikit-learn": 23,
    "pytest-dev/pytest": 17, "sphinx-doc/sphinx": 16, "astropy/astropy": 6, "psf/requests": 6,
    "pylint-dev/pylint": 6, "pydata/xarray": 5, "mwaskom/seaborn": 4, "pallets/flask": 3,
}


def test_allocation_for_lite_test():
    seats = select_instances.allocate(LITE_TEST, 100)
    assert sum(seats.values()) == 100
    assert seats == {
        "django/django": 38, "sympy/sympy": 26, "matplotlib/matplotlib": 8, "scikit-learn/scikit-learn": 8,
        "pytest-dev/pytest": 6, "sphinx-doc/sphinx": 5, "astropy/astropy": 2, "psf/requests": 2,
        "pylint-dev/pylint": 2, "pydata/xarray": 1, "mwaskom/seaborn": 1, "pallets/flask": 1,
    }


def test_draw_is_deterministic_and_seed_dependent():
    ids = {repo: [f"{repo}-{i}" for i in range(size)] for repo, size in LITE_TEST.items()}
    seats = select_instances.allocate(LITE_TEST, 100)
    first = select_instances.draw(ids, seats, 20261007)
    assert first == select_instances.draw(ids, seats, 20261007)
    assert first != select_instances.draw(ids, seats, 1)
    assert len(first) == len(set(first)) == 100


def test_committed_selection_file_is_consistent():
    selection = json.loads((ROOT / "data" / "instances_seed20261007.json").read_text())
    assert selection["seed"] == 20261007 and len(selection["instance_ids"]) == 100
    assert len(set(selection["instance_ids"])) == 100
    assert not set(selection["instance_ids"]) & set(selection["smoke_instance_ids"])
    assert len(selection["smoke_instance_ids"]) == 8
    assert sum(selection["allocation"].values()) == 100 and len(selection["allocation"]) == 12
    assert selection["dataset_revision"] and selection["method"]


def work(mode: ExperimentMode) -> WorkItem:
    return WorkItem(
        instance_id="r__r-1", provider="openai", model="m", tool_set_name="all_tools",
        enabled_tools=["read_file", "search_code", "list_directory"], repo="r/r", base_commit="abc",
        problem_statement="It crashes.", hints_text="SECRET HINT: the fix is in foo.py",
        patch="GOLD PATCH", fail_to_pass=["t1"], pass_to_pass=["t2"], split="lite", experiment_mode=mode,
    )


def prompt(mode: ExperimentMode) -> str:
    return "\n".join(m.content for m in ExperimentRunner(storage=None)._build_analysis_messages(work(mode)))


def test_prompt_contains_issue_but_no_hints_gold_patch_or_tests():
    for mode in ExperimentMode:
        text = prompt(mode)
        assert "It crashes." in text
        assert "SECRET HINT" not in text and "## Hints" not in text
        assert "GOLD PATCH" not in text and "t1" not in text


def test_only_the_forced_prompt_removes_the_abstention_option():
    assert "<<<CANNOT_SOLVE>>>" in prompt(ExperimentMode.BASELINE)
    assert "<<<CANNOT_SOLVE>>>" in prompt(ExperimentMode.SELF_REFLECT)
    assert prompt(ExperimentMode.BASELINE) == prompt(ExperimentMode.SELF_REFLECT)
    forced = prompt(ExperimentMode.FORCED)
    assert "<<<CANNOT_SOLVE>>>" not in forced and "There is NO option to give up" in forced


def test_every_regime_asks_for_edit_blocks_with_the_same_format_text():
    from app.core.edit_format import EDIT_FORMAT_INSTRUCTIONS
    for mode in ExperimentMode:
        text = prompt(mode)
        assert EDIT_FORMAT_INSTRUCTIONS in text
        assert "@@" not in text and "unified diff" not in text and "```diff" not in text
