"""LLM-as-a-judge for one focused failure mode: does a RagEngine answer state
anything not supported by its own cited excerpts?

Mirrors the Week 4 Session 1 pattern -- a judge scoped to exactly one failure mode,
rather than a general "is this answer good" grader. A judge with one job is easier to
trust and easier to check for human/judge alignment; it's also the same idea as
RAGAS's Faithfulness metric in the Week 4 Session 2 tutorial, hand-rolled here to
reuse planpeek's own Databricks integration instead of adding ragas as a dependency
(and ragas 0.4.3 needs its own compatibility shim to even import cleanly against a
current langchain-community -- see that tutorial's ragas_compat.py).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from planpeek.rag import RagError, build_databricks_chat_model, resolve_databricks_settings

SYSTEM_PROMPT = """You are an impartial evaluator checking a research assistant's answer \
for unsupported claims.

You are given the assistant's answer and the filing excerpts it was allowed to use. Your \
only job: does the answer state any specific fact, number, or claim that is NOT present in \
or reasonably inferable from the excerpts?

Mark UNSUPPORTED only when the answer asserts something the excerpts do not contain. Do not \
mark UNSUPPORTED for reasonable summarization, rounding, or phrasing that stays faithful to \
what the excerpts say.

Output exactly this JSON, no markdown formatting, no commentary outside the JSON object:
{"label": "SUPPORTED or UNSUPPORTED", "reasoning": "one sentence; cite the specific \
unsupported claim if UNSUPPORTED"}"""


@dataclass(frozen=True, slots=True)
class FaithfulnessVerdict:
    label: str  # "SUPPORTED" | "UNSUPPORTED"
    reasoning: str


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def _parse_verdict(text: str) -> FaithfulnessVerdict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RagError("The faithfulness judge did not return valid JSON.") from exc
    label = str(payload.get("label", "")).strip().upper()
    if label not in {"SUPPORTED", "UNSUPPORTED"}:
        raise RagError(f"The faithfulness judge returned an unrecognized label: {label!r}")
    return FaithfulnessVerdict(label=label, reasoning=str(payload.get("reasoning", "")))


def judge_faithfulness(
    answer_text: str,
    excerpts: list[str],
    *,
    llm: Any | None = None,
) -> FaithfulnessVerdict:
    """Judge whether `answer_text` states anything not supported by `excerpts`.

    Requires the same Databricks environment variables as planpeek.rag when `llm` is
    not provided (see README). Pass `llm` directly (e.g. in tests) to bypass that.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    model = llm
    if model is None:
        settings = resolve_databricks_settings(os.getenv)
        model = build_databricks_chat_model(settings, max_tokens=300)

    context = "\n\n---\n\n".join(excerpts) if excerpts else "(no excerpts were retrieved)"
    user_prompt = f"Answer:\n{answer_text}\n\nExcerpts:\n{context}"
    try:
        response = model.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
        )
    except Exception as exc:
        raise RagError(f"The faithfulness judge request failed: {exc}") from exc

    return _parse_verdict(_message_text(response))
