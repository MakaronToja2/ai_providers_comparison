"""Runner + storage end to end with a scripted provider: outcomes, logging, resume."""
import json
from contextlib import asynccontextmanager

import pytest

from app.benchmark import runner as runner_module
from app.benchmark.runner import ExperimentRunner
from app.benchmark.storage import Storage
from app.core.base_provider import BaseAIProvider, ParsedResponse
from app.core.tool_manager import initialize_tools
from app.models.benchmark import Experiment, ExperimentConfig, ExperimentMode, ResultStatus
from app.models.responses import TokenUsage
from app.models.swe_bench import SWEBenchInstance
from app.utils.repo_context import current_repo_path

DIFF = "a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"
REVISED = "a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 3\n>>>>>>> REPLACE"
BAD_EDIT = "a.py\n<<<<<<< SEARCH\nx = 100\n=======\nx = 2\n>>>>>>> REPLACE"
UNIFIED_DIFF = "```diff\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n```"
BEHAVIOURS = {"inst-patch": "PATCH", "inst-abstain": "ABSTAIN", "inst-noformat": "NOFORMAT", "inst-apierr": "APIERR",
              "inst-badedit": "BADEDIT", "inst-diff": "DIFFANSWER", "inst-stacked": "STACKED"}
STACKED_REVISION = "a.py\n<<<<<<< SEARCH\nx = 2\n=======\nx = 4\n>>>>>>> REPLACE"
CORE = ["inst-patch", "inst-abstain", "inst-noformat", "inst-apierr"]


class FakeLoader:
    def load_dataset(self, split):
        return None

    def list_instances(self, split, limit):
        return list(BEHAVIOURS)

    def get_instance(self, instance_id, split):
        return SWEBenchInstance(
            instance_id=instance_id, repo="fake/repo", base_commit="abc", patch="GOLD", test_patch="T",
            problem_statement=f"behaviour={BEHAVIOURS[instance_id]}", hints_text="HINT",
            FAIL_TO_PASS=["t1"], PASS_TO_PASS=["t2"],
        )


class BehaviourProvider(BaseAIProvider):
    """Answers according to the behaviour named in the bug report."""
    retry_base_seconds = 0.0
    api_is_down = True

    def __init__(self):
        super().__init__(api_key="test", model="fake-model")

    @property
    def provider_name(self):
        return "openai"

    def _validate_api_key(self):
        pass

    async def _make_api_call(self, messages, temperature=0.7, max_tokens=1000, tools=None, tool_choice="auto", **kwargs):
        behaviour = messages[1].content.split("behaviour=")[1].split()[0]
        last = messages[-1].content or ""
        if "SELF-REFLECTION PHASE" in last:
            revision = STACKED_REVISION if behaviour == "STACKED" else REVISED
            text = f"<<<PATCH_REVISED>>>\n{revision}\nBetter value."
        elif behaviour in ("PATCH", "STACKED"):
            text = f"Fix:\n{DIFF}"
        elif behaviour == "ABSTAIN":
            text = "<<<CANNOT_SOLVE>>>\nreason: too_complex\nexplanation: cannot tell"
        elif behaviour == "NOFORMAT":
            text = "I think the bug is somewhere in a.py."
        elif behaviour == "BADEDIT":
            text = f"Fix:\n{BAD_EDIT}"
        elif behaviour == "DIFFANSWER":
            text = f"Fix:\n{UNIFIED_DIFF}"
        elif behaviour == "APIERR":
            if BehaviourProvider.api_is_down:
                raise ValueError("401 invalid api key")
            text = f"Fix:\n{DIFF}"
        return ParsedResponse(text=text, usage=TokenUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
                              stop_reason="stop", api_model_id="fake-model-001")

    def _parse_response(self, raw_response):
        return raw_response


@pytest.fixture
def env(tmp_path, monkeypatch):
    import subprocess
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "a.py").write_text("x = 1\n")
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-C", str(tmp_path / "repo"), *cmd], check=True)

    @asynccontextmanager
    async def fake_repo_context(repo, commit):
        token = current_repo_path.set(str(tmp_path / "repo"))
        try:
            yield None
        finally:
            current_repo_path.reset(token)

    provider = BehaviourProvider()
    BehaviourProvider.api_is_down = True
    monkeypatch.setattr(runner_module, "RepoContext", fake_repo_context)
    monkeypatch.setattr(runner_module.ProviderFactory, "create_provider", classmethod(lambda cls, *a, **k: provider))
    # Storage latches its path and initialisation on the class
    monkeypatch.setattr(Storage, "_absolute_db_path", None)
    monkeypatch.setattr(Storage, "_global_initialized", False)
    initialize_tools()

    storage = Storage(str(tmp_path / "results.db"))
    runner = ExperimentRunner(storage, swe_bench_loader=FakeLoader(), trajectory_dir=tmp_path / "traj")
    return storage, runner, tmp_path


async def make_experiment(storage, mode, ids=None):
    config = ExperimentConfig(name=f"job-{mode.value}", providers=["openai"], models={"openai": "fake-model"},
                              split="lite", experiment_mode=mode, instance_ids=ids or CORE, run_idx=2)
    return await storage.create_experiment(Experiment(name=config.name, config=config))


async def run(runner, experiment_id):
    return [update async for update in runner.start_experiment(experiment_id)]


async def test_outcomes_are_recorded_separately_and_resume_reruns_only_errors(env):
    storage, runner, tmp_path = env
    experiment = await make_experiment(storage, ExperimentMode.BASELINE)

    updates = await run(runner, experiment.id)
    assert len(updates) == 4
    rows = {r.instance_id: r for r in await storage.get_results_for_experiment(experiment.id)}

    assert rows["inst-patch"].result_status == ResultStatus.SUCCESS_PATCH_GENERATED
    patch = rows["inst-patch"].generated_patch
    # the stored patch is git's own diff of the applied edit
    assert patch.startswith("diff --git a/a.py b/a.py") and "-x = 1" in patch and "+x = 2" in patch
    assert rows["inst-patch"].original_patch is None
    assert rows["inst-patch"].run_meta["answer_format"] == "edit_blocks" and rows["inst-patch"].run_meta["edit_blocks"] == 1
    assert rows["inst-abstain"].result_status == ResultStatus.FAILURE_EXPLICIT
    assert rows["inst-noformat"].result_status == ResultStatus.HALLUCINATION_FORMAT_ERROR
    # an API failure is its own status, with the real error kept
    assert rows["inst-apierr"].result_status == ResultStatus.API_ERROR
    assert "401 invalid api key" in rows["inst-apierr"].error_message

    # previously unsaved fields now persist
    meta = rows["inst-patch"].run_meta
    assert rows["inst-patch"].split == "lite"
    assert meta["run_idx"] == 2 and meta["experiment_mode"] == "baseline"
    assert meta["termination_reason"] == "model_stopped" and meta["iterations"] == 1
    assert meta["api_model_ids"] == ["fake-model-001"] and meta["abstained"] is False
    assert rows["inst-abstain"].run_meta["abstained"] is True
    assert rows["inst-noformat"].run_meta["termination_reason"] == "fallback_used"

    # one trajectory per instance, closed with a summary
    trajectory = [json.loads(l) for l in (tmp_path / "traj" / "inst-noformat.jsonl").read_text().splitlines()]
    assert trajectory[0]["type"] == "header" and "HINT" not in json.dumps(trajectory[0])
    assert [l["phase"] for l in trajectory if l["type"] == "api_call"] == ["loop", "fallback"]
    assert trajectory[-1]["type"] == "summary" and trajectory[-1]["result_status"] == "hallucination_format_error"

    # resume: only the API-error instance is pending, and its row is replaced, not duplicated
    assert await storage.get_completed_result_keys(experiment.id) == {
        (i, "openai", "fake-model", "all_tools") for i in ("inst-patch", "inst-abstain", "inst-noformat")}
    BehaviourProvider.api_is_down = False
    updates = await run(runner, experiment.id)
    assert [u.instance_id for u in updates] == ["inst-apierr"]
    rows = await storage.get_results_for_experiment(experiment.id)
    assert len(rows) == 4
    assert {r.instance_id: r.result_status for r in rows}["inst-apierr"] == ResultStatus.SUCCESS_PATCH_GENERATED
    assert (tmp_path / "traj" / "inst-apierr.prev1.jsonl").exists()  # the failed attempt's log is kept

    # a finished job has nothing left to do
    assert await run(runner, experiment.id) == []


async def test_self_reflection_stores_initial_and_final_patch(env):
    storage, runner, _ = env
    experiment = await make_experiment(storage, ExperimentMode.SELF_REFLECT, ids=["inst-patch", "inst-abstain"])
    await run(runner, experiment.id)
    rows = {r.instance_id: r for r in await storage.get_results_for_experiment(experiment.id)}

    patched = rows["inst-patch"]
    assert patched.reflection_outcome == "revised"
    assert "+x = 2" in patched.original_patch and "+x = 3" in patched.generated_patch
    assert patched.run_meta["reflection_changed_patch"] is True
    # reflection tokens are kept apart from the main loop
    assert (patched.total_tokens, patched.reflection_total_tokens) == (110, 110)

    assert rows["inst-abstain"].reflection_outcome == "skipped"
    assert rows["inst-abstain"].result_status == ResultStatus.FAILURE_EXPLICIT


async def test_missing_instance_fails_loudly(env):
    storage, runner, _ = env
    experiment = await make_experiment(storage, ExperimentMode.BASELINE, ids=["inst-patch", "inst-does-not-exist"])
    with pytest.raises(RuntimeError, match="could not be loaded"):
        await run(runner, experiment.id)



async def test_edits_that_do_not_match_the_code_are_their_own_outcome(env):
    storage, runner, tmp_path = env
    experiment = await make_experiment(storage, ExperimentMode.BASELINE, ids=["inst-badedit", "inst-diff"])
    await run(runner, experiment.id)
    rows = {r.instance_id: r for r in await storage.get_results_for_experiment(experiment.id)}

    bad = rows["inst-badedit"]
    assert bad.result_status == ResultStatus.EDIT_NOT_APPLICABLE and bad.generated_patch is None
    assert "search_text_not_found" in bad.error_message
    assert bad.run_meta["edit_failures"] == [{"block": 1, "path": "a.py", "reason": "search_text_not_found"}]
    # terminal: not re-run on resume
    assert ("inst-badedit", "openai", "fake-model", "all_tools") in await storage.get_completed_result_keys(experiment.id)

    # a model that writes a unified diff anyway is still evaluated, and the fact is recorded
    diff = rows["inst-diff"]
    assert diff.result_status == ResultStatus.SUCCESS_PATCH_GENERATED and "+x = 2" in diff.generated_patch
    assert diff.run_meta["answer_format"] == "unified_diff"

    # nothing the runner did is left behind in the checkout
    assert (tmp_path / "repo" / "a.py").read_text() == "x = 1\n"



async def test_revision_on_top_of_the_first_patch_is_applied_and_recorded(env):
    storage, runner, tmp_path = env
    experiment = await make_experiment(storage, ExperimentMode.SELF_REFLECT, ids=["inst-patch", "inst-stacked"])
    await run(runner, experiment.id)
    rows = {r.instance_id: r for r in await storage.get_results_for_experiment(experiment.id)}

    # the revision quotes "x = 2", which exists only after the first patch: combined result is 1 -> 4
    stacked = rows["inst-stacked"]
    assert stacked.result_status == ResultStatus.SUCCESS_PATCH_GENERATED and stacked.reflection_outcome == "revised"
    assert "-x = 1" in stacked.generated_patch and "+x = 4" in stacked.generated_patch and "x = 2" not in stacked.generated_patch
    assert "+x = 2" in stacked.original_patch
    assert stacked.run_meta["revision_basis"] == "on_first_patch"
    # a revision that fits the original file is applied there, as before
    assert rows["inst-patch"].run_meta["revision_basis"] == "original"
    assert (tmp_path / "repo" / "a.py").read_text() == "x = 1\n"
