"""Harness output is classified from real January 2026 log directories."""
import json
from pathlib import Path

import pytest

from app.benchmark.swebench_eval import (
    APPLY_FAILED, EVAL_ERROR, RESOLVED, TESTS_FAILED, Verdict, build_predictions,
    classify_instance_dir, harness_command, instance_log_dir, settle_timeouts,
)

LOGS = Path(__file__).resolve().parents[1] / "logs" / "run_evaluation"


def january(run: str, model: str, instance: str) -> Path:
    path = LOGS / run / model / instance
    if not path.exists():
        pytest.skip(f"January fixture missing: {path}")
    return path


def test_image_not_found_is_an_eval_error_never_unresolved():
    d = january("exp_a98255e4_20260111_200223", "openai__gpt-4.1-mini", "django__django-14580")
    verdict = classify_instance_dir(d, "django__django-14580")
    assert (verdict.status, verdict.kind) == (EVAL_ERROR, "image")


def test_apply_failure():
    d = january("exp_95ce813e_20260113_184759", "google__gemini-2.5-flash-lite", "sqlfluff__sqlfluff-1625")
    assert classify_instance_dir(d, "sqlfluff__sqlfluff-1625").status == APPLY_FAILED


def test_failed_tests_report_both_failure_kinds():
    d = january("exp_a98255e4_20260111_203828", "google__gemini-2.5-flash-lite", "django__django-12308")
    verdict = classify_instance_dir(d, "django__django-12308")
    assert (verdict.status, verdict.kind) == (TESTS_FAILED, "both")
    assert verdict.f2p_total > verdict.f2p_passed


def test_every_january_report_agrees_with_its_own_resolved_flag():
    reports = list(LOGS.glob("exp_3766c4cc_*/*/*/report.json"))
    if not reports:
        pytest.skip("January logs missing")
    for report in reports:
        instance_id = report.parent.name
        expected = json.loads(report.read_text())[instance_id]["resolved"]
        assert (classify_instance_dir(report.parent, instance_id).status == RESOLVED) is expected


def test_synthetic_cases(tmp_path):
    empty = tmp_path / "empty"; empty.mkdir()
    assert classify_instance_dir(empty, "x").kind == "no_output"

    timeout = tmp_path / "timeout"; timeout.mkdir()
    (timeout / "run_instance.log").write_text("...\nTest timed out after 1800 seconds.\n")
    verdict = classify_instance_dir(timeout, "x")
    assert (verdict.status, verdict.kind) == (EVAL_ERROR, "timeout")
    assert settle_timeouts({"x": verdict})["x"] == Verdict(TESTS_FAILED, "timeout")

    regress = tmp_path / "regress"; regress.mkdir()
    (regress / "report.json").write_text(json.dumps({"x": {"resolved": False, "tests_status": {
        "FAIL_TO_PASS": {"success": ["t1"], "failure": []}, "PASS_TO_PASS": {"success": ["t2"], "failure": ["t3"]}}}}))
    assert classify_instance_dir(regress, "x").kind == "p2p_regression"

    # a later successful attempt wins over an earlier error left in the log
    retried = tmp_path / "retried"; retried.mkdir()
    (retried / "run_instance.log").write_text("BuildImageError ... No such image\n")
    (retried / "report.json").write_text(json.dumps({"x": {"resolved": True, "tests_status": {
        "FAIL_TO_PASS": {"success": ["t1"], "failure": []}, "PASS_TO_PASS": {"success": [], "failure": []}}}}))
    assert classify_instance_dir(retried, "x").status == RESOLVED


def test_predictions_final_and_initial():
    rows = [
        {"instance_id": "a", "generated_patch": "P1", "original_patch": None},      # baseline-style row
        {"instance_id": "b", "generated_patch": "P2", "original_patch": "P2"},      # approved: same patch
        {"instance_id": "c", "generated_patch": "P3b", "original_patch": "P3a"},    # revised
        {"instance_id": "d", "generated_patch": None, "original_patch": None},      # no patch
    ]
    final = build_predictions(rows, "job__final", "final")
    assert [(p["instance_id"], p["model_patch"]) for p in final] == [("a", "P1"), ("b", "P2"), ("c", "P3b")]
    assert {p["model_name_or_path"] for p in final} == {"job__final"}
    initial = build_predictions(rows, "job__initial", "initial")
    assert [(p["instance_id"], p["model_patch"]) for p in initial] == [("c", "P3a")]  # identical patches are not re-evaluated


def test_duplicate_instances_in_one_predictions_file_are_rejected():
    rows = [{"instance_id": "a", "generated_patch": "P", "original_patch": None}] * 2
    with pytest.raises(ValueError):
        build_predictions(rows, "job", "final")


def test_harness_command_keeps_images_and_names_instances():
    cmd = harness_command("py", "data.json", "test", "preds.jsonl", "run1", ["i1", "i2"], 4, 1800, "reports")
    assert cmd[:3] == ["py", "-m", "swebench.harness.run_evaluation"]
    assert cmd[cmd.index("--cache_level") + 1] == "instance" and cmd[cmd.index("--clean") + 1] == "False"
    assert cmd[cmd.index("--namespace") + 1] == "swebench" and cmd[-2:] == ["i1", "i2"]
    assert instance_log_dir(Path("/e"), "run1", "a/b", "i1") == Path("/e/logs/run_evaluation/run1/a__b/i1")
