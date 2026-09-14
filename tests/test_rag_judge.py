from __future__ import annotations

import json

import pytest

from planpeek.evals.rag_judge import judge_faithfulness
from planpeek.rag import RagError


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return _FakeResponse(self.content)


def test_judge_faithfulness_parses_supported_verdict() -> None:
    llm = _FakeLLM(json.dumps({"label": "SUPPORTED", "reasoning": "Matches the excerpt."}))
    verdict = judge_faithfulness("Total assets were $5.25M.", ["Total assets: $5.25M"], llm=llm)
    assert verdict.label == "SUPPORTED"
    assert verdict.reasoning == "Matches the excerpt."


def test_judge_faithfulness_parses_unsupported_verdict() -> None:
    payload = {"label": "UNSUPPORTED", "reasoning": "Invented a headcount figure."}
    llm = _FakeLLM(json.dumps(payload))
    verdict = judge_faithfulness("There are 500 employees.", ["Total assets: $5.25M"], llm=llm)
    assert verdict.label == "UNSUPPORTED"


def test_judge_faithfulness_strips_markdown_fence() -> None:
    fenced = "```json\n" + json.dumps({"label": "SUPPORTED", "reasoning": "ok"}) + "\n```"
    llm = _FakeLLM(fenced)
    verdict = judge_faithfulness("x", ["y"], llm=llm)
    assert verdict.label == "SUPPORTED"


def test_judge_faithfulness_sends_excerpts_and_answer_in_prompt() -> None:
    llm = _FakeLLM(json.dumps({"label": "SUPPORTED", "reasoning": "ok"}))
    judge_faithfulness("my answer text", ["excerpt one", "excerpt two"], llm=llm)
    user_message = llm.messages[1].content
    assert "my answer text" in user_message
    assert "excerpt one" in user_message
    assert "excerpt two" in user_message


def test_judge_faithfulness_handles_no_excerpts() -> None:
    llm = _FakeLLM(json.dumps({"label": "UNSUPPORTED", "reasoning": "nothing retrieved"}))
    verdict = judge_faithfulness("some answer", [], llm=llm)
    assert verdict.label == "UNSUPPORTED"


def test_judge_faithfulness_raises_on_invalid_json() -> None:
    llm = _FakeLLM("not json at all")
    with pytest.raises(RagError, match="did not return valid JSON"):
        judge_faithfulness("x", ["y"], llm=llm)


def test_judge_faithfulness_raises_on_unrecognized_label() -> None:
    llm = _FakeLLM(json.dumps({"label": "MAYBE", "reasoning": "unsure"}))
    with pytest.raises(RagError, match="unrecognized label"):
        judge_faithfulness("x", ["y"], llm=llm)


def test_judge_faithfulness_wraps_model_invocation_errors() -> None:
    class _RaisingLLM:
        def invoke(self, messages):
            raise RuntimeError("connection reset")

    with pytest.raises(RagError, match="judge request failed"):
        judge_faithfulness("x", ["y"], llm=_RaisingLLM())
