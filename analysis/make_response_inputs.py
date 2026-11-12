#!/usr/bin/env python3
"""
Write the generated parts of the response document:

  paper/mdis/response/generated/refs.tex               section and table numbers of the manuscript
  paper/mdis/response/generated/appendix-instances.tex the sample: method, counts, identifiers
  paper/mdis/response/generated/appendix-prompts.tex   the prompts, taken from the harness code

The prompts are not copied by hand: the system and user messages are built by the runner,
and the request for a final answer and the reflection prompt are captured from the agent
loop driven by a scripted stand-in for a model. Instance-specific parts are replaced by
bracketed placeholders.

Usage (after compiling the manuscript with --keep-intermediates, so that main.aux exists):
    .venv/bin/python analysis/make_response_inputs.py
"""
import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmark.runner import ExperimentRunner, WorkItem  # noqa: E402
from app.core.base_provider import BaseAIProvider, ParsedResponse  # noqa: E402
from app.core.tool_manager import initialize_tools  # noqa: E402
from app.core.tools import tool_registry  # noqa: E402
from app.models.benchmark import ExperimentMode  # noqa: E402
from app.models.requests import ChatMessage, ToolCallRequest  # noqa: E402
from app.models.responses import TokenUsage  # noqa: E402
from app.utils.repo_context import current_repo_path  # noqa: E402

PAPER = PROJECT_ROOT / "paper" / "mdis"
OUT = PAPER / "response" / "generated"
SELECTION = PROJECT_ROOT / "data" / "instances_seed20261007.json"

REPO = "[repository, e.g. django/django]"
COMMIT = "[base commit hash]"
ISSUE = "[text of the GitHub issue]"
SUMMARY = "[summary of the exploration so far: files read, searches made with their number of matches, directories listed]"
ANSWER = "a.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"
ANSWER_PLACEHOLDER = "[the model's answer, from its first edit block to the end]"


class ScriptedProvider(BaseAIProvider):
    """Stands in for a model: returns scripted replies and records the messages it is sent."""
    retry_base_seconds = 0.0

    def __init__(self, script):
        super().__init__(api_key="none", model="scripted")
        self.script = list(script)
        self.calls = []

    @property
    def provider_name(self):
        return "scripted"

    def _validate_api_key(self):
        pass

    def _is_retryable(self, exc):
        return False

    def _build_exploration_summary(self, tool_calls):
        return SUMMARY

    async def _make_api_call(self, messages, temperature=0.7, max_tokens=1000, tools=None, tool_choice="auto", **kwargs):
        self.calls.append(list(messages))
        return self.script.pop(0)

    def _parse_response(self, raw_response):
        return raw_response


def reply(text="", calls=()):
    return ParsedResponse(
        text=text,
        tool_calls=[ToolCallRequest(id=f"c{i}", name=name, arguments=args) for i, (name, args) in enumerate(calls)],
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        stop_reason="stop", api_model_id="scripted",
    )


def analysis_messages(mode: ExperimentMode):
    work = WorkItem(
        instance_id="x__x-1", provider="openai", model="m", tool_set_name="all_tools",
        enabled_tools=["read_file", "search_code", "list_directory"], repo=REPO, base_commit=COMMIT,
        problem_statement=ISSUE, hints_text="", patch="", fail_to_pass=[], pass_to_pass=[],
        split="lite", experiment_mode=mode,
    )
    return ExperimentRunner(storage=None)._build_analysis_messages(work)


async def captured_prompts() -> dict:
    """The request for a final answer (per regime) and the reflection prompt, as the loop sends them."""
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.py").write_text("x = 1\n")
        token = current_repo_path.set(tmp)
        try:
            for mode in ("baseline", "forced", "self_reflect"):
                # First reply: neither a tool call nor an answer, so the loop asks for the final answer
                script = [reply("I will look at this."), reply(ANSWER)]
                if mode == "self_reflect":
                    script.append(reply("<<<PATCH_APPROVED>>>"))
                provider = ScriptedProvider(script)
                await provider.generate_response(
                    [ChatMessage(role="system", content="S"), ChatMessage(role="user", content="U")],
                    max_tool_iterations=10, experiment_mode=mode,
                )
                out[f"final_{mode}"] = provider.calls[1][-1].content
                if mode == "self_reflect":
                    out["reflection"] = provider.calls[2][-1].content.replace(ANSWER, ANSWER_PLACEHOLDER)
        finally:
            current_repo_path.reset(token)
    assert out["final_baseline"] == out["final_self_reflect"], "baseline and self-reflection must share the request"
    assert ANSWER_PLACEHOLDER in out["reflection"]
    return out


def verbatim(text: str) -> str:
    assert "\\end{Verbatim}" not in text
    non_ascii = sorted({c for c in text if ord(c) > 126})
    assert not non_ascii, f"non-ASCII characters in a prompt: {non_ascii}"
    return "\\begin{Verbatim}[breaklines=true,breakanywhere=false,fontsize=\\footnotesize,frame=single,framesep=4pt]\n" \
           + text.rstrip("\n") + "\n\\end{Verbatim}\n"


def tex_escape(text: str) -> str:
    text = text.replace("\\", "\0")
    for a, b in (("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("{", r"\{"), ("}", r"\}"),
                 ("$", r"\$"), ("^", r"\^{}"), ("~", r"\~{}")):
        text = text.replace(a, b)
    return text.replace("\0", r"\textbackslash{}")


def write_prompts() -> None:
    initialize_tools()
    baseline = analysis_messages(ExperimentMode.BASELINE)
    forced = analysis_messages(ExperimentMode.FORCED)
    reflect = analysis_messages(ExperimentMode.SELF_REFLECT)
    assert [m.content for m in baseline] == [m.content for m in reflect]
    assert baseline[1].content == forced[1].content
    captured = asyncio.run(captured_prompts())

    parts = [
        "The texts below are produced from the harness code by a script; only the parts in square "
        "brackets stand for content that differs between instances. The description of the answer "
        "format (from ``Write your patch as one or more edit blocks'' to the last rule) is the same "
        "text in every prompt in which it appears. The maximum number of tool iterations was 10.\n",
        "\\subsection*{B.1 System prompt: baseline and self-reflection regimes (identical)}\n",
        verbatim(baseline[0].content),
        "\\subsection*{B.2 System prompt: forced regime}\n",
        verbatim(forced[0].content),
        "\\subsection*{B.3 First user message (all regimes)}\n",
        verbatim(baseline[1].content),
        "\\subsection*{B.4 Request for the final answer: baseline and self-reflection regimes}\n"
        "Sent as a user message, with tool use disabled, when the model stops calling tools without "
        "giving an answer or when the tenth iteration ends. The request opens with a list, built by the "
        "harness from the run's tool calls, of the files read, the searches made and the directories listed; "
        "the conversation itself is left as it is.\n",
        verbatim(captured["final_baseline"]),
        "\\subsection*{B.5 Request for the final answer: forced regime}\n",
        verbatim(captured["final_forced"]),
        "\\subsection*{B.6 Reflection prompt (self-reflection regime only)}\n"
        "Sent as a user message after the answer, in the same conversation, with tool use disabled. "
        "It is sent only when the answer contains an edit and no abstention marker; the model is shown "
        "the edit as it wrote it.\n",
        verbatim(captured["reflection"]),
        "\\subsection*{B.7 Tool declarations}\n"
        "The three tools are declared to each model through the function-calling interface of its API "
        "with the following names, descriptions and parameters.\n",
    ]
    rows = []
    for definition in sorted(tool_registry.get_tool_definitions(), key=lambda d: d.name):
        spec = definition.model_dump()
        schema = spec.get("parameters") or spec.get("parameters_schema") or {}
        required = set(schema.get("required", []))
        rows.append(f"\\item \\code{{{tex_escape(spec['name'])}}}: {tex_escape(spec['description'])}\n\\begin{{itemize}}[leftmargin=*,itemsep=0pt,topsep=1pt]")
        for name, prop in schema.get("properties", {}).items():
            kind = prop.get("type", "")
            note = "required" if name in required else "optional"
            if "default" in prop:
                note += f", default {prop['default']}"
            rows.append(f"  \\item \\code{{{tex_escape(name)}}} ({tex_escape(str(kind))}, {tex_escape(note)}): {tex_escape(prop.get('description', ''))}")
        rows.append("\\end{itemize}")
    parts.append("\\begin{itemize}[leftmargin=*,itemsep=3pt]\n" + "\n".join(rows) + "\n\\end{itemize}\n")
    (OUT / "appendix-prompts.tex").write_text("% Written by analysis/make_response_inputs.py. Do not edit by hand.\n" + "\n".join(parts))


def write_instances() -> None:
    selection = json.loads(SELECTION.read_text())
    ids = selection["instance_ids"]
    population, allocation = selection["population"], selection["allocation"]
    assert len(ids) == selection["n"] == sum(allocation.values())
    order = sorted(allocation, key=lambda repo: (-population[repo], repo))

    lines = [
        "% Written by analysis/make_response_inputs.py. Do not edit by hand.",
        f"\\textbf{{Population.}} The {sum(population.values())} instances of the test split of SWE-bench Lite "
        f"as published on the Hugging Face Hub (dataset revision \\code{{{selection['dataset_revision'][:12]}}}).",
        "",
        f"\\textbf{{Method.}} Stratified by repository. Each repository receives a number of instances proportional "
        f"to its share of the split (rounded down, at least one); the remaining places go to the repositories with "
        f"the largest remainders (ties go to the larger repository, then to the first by name). Within a repository, "
        f"the sorted identifiers are sampled without replacement by "
        f"\\code{{random.Random(\"{selection['seed']}:<repository>\").sample(identifiers, k)}} "
        f"(Python {selection['python_version']}). The seed ({selection['seed']}) was fixed before any run. "
        f"The SHA-256 of the sorted identifiers, joined by newlines, begins \\code{{{selection['instance_ids_sha256'][:16]}}}.",
        "",
        "\\textbf{Development instances.} Eight further instances were drawn with the same seed from the instances "
        "\\emph{not} selected (two each from Django and SymPy, one each from matplotlib, scikit-learn, pytest and "
        "Sphinx). The harness was developed and debugged on these before the runs; they are listed last and are "
        "not part of any reported result. Three rules for reading answers were added later, during the first "
        "repetition on the evaluated instances (see the response to R2.1.2).",
        "",
        "\\begin{center}\\small",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Repository & Lite test split & Sample \\\\",
        "\\midrule",
    ]
    for repo in order:
        lines.append(f"\\code{{{tex_escape(repo)}}} & {population[repo]} & {allocation[repo]} \\\\")
    lines += ["\\midrule", f"Total & {sum(population.values())} & {sum(allocation.values())} \\\\", "\\bottomrule",
              "\\end{tabular}", "\\end{center}", "", f"\\textbf{{The {len(ids)} evaluated instances.}}", ""]

    by_repo = {}
    for instance_id in ids:
        by_repo.setdefault(instance_id.rsplit("-", 1)[0], []).append(instance_id)
    key_of = {repo.replace("/", "__"): repo for repo in allocation}
    assert set(by_repo) == set(key_of), (set(by_repo) ^ set(key_of))
    for repo in order:
        key = repo.replace("/", "__")
        numbers = sorted(by_repo[key], key=lambda i: int(i.rsplit("-", 1)[1]))
        assert len(numbers) == allocation[repo]
        listed = ", ".join(i.rsplit("-", 1)[1] for i in numbers)
        lines.append(f"\\code{{{tex_escape(key)}-}}\\,: {listed}.\\par")
    lines += ["", "\\begin{flushleft}", "\\textbf{The eight development instances.} "
              + ", ".join(f"\\code{{{tex_escape(i)}}}" for i in selection["smoke_instance_ids"]) + ".",
              "\\end{flushleft}"]
    (OUT / "appendix-instances.tex").write_text("\n".join(lines) + "\n")


def write_refs() -> None:
    """Numbers of the manuscript's labels, so the response cannot drift from the manuscript."""
    aux = PAPER / "main.aux"
    lines = ["% Written by analysis/make_response_inputs.py from main.aux. Do not edit by hand."]
    labels = {}
    if aux.exists():
        for match in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}", aux.read_text()):
            labels[match.group(1)] = match.group(2)
    for label, number in sorted(labels.items()):
        if re.fullmatch(r"[A-Za-z0-9:_-]+", label) and number:
            lines.append(f"\\expandafter\\def\\csname mref@{label}\\endcsname{{{number}}}")
    (OUT / "refs.tex").write_text("\n".join(lines) + "\n")
    print(f"refs: {len(labels)} labels from main.aux" if labels else "refs: main.aux not found, no labels written")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_refs()
    write_instances()
    write_prompts()
    print(f"written: {', '.join(sorted(p.name for p in OUT.iterdir()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
