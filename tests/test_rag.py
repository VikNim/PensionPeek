from __future__ import annotations

from types import SimpleNamespace

import pytest

from planpeek.models import TextChunk
from planpeek.rag import (
    DATABRICKS_QUERY_INSTRUCTION,
    RagEngine,
    RagError,
    _collection_name,
    _DatabricksEmbeddings,
    _DatabricksOAuthToken,
    normalize_databricks_base_url,
    resolve_databricks_settings,
)


def test_resolve_databricks_settings_falls_back_to_documented_defaults() -> None:
    settings = resolve_databricks_settings(lambda key: "")
    assert settings["base_url"] == "https://dbc-7b106152-caf3.cloud.databricks.com/serving-endpoints"
    assert settings["profile"] == "dbc-7b106152-caf3"
    assert settings["model_name"] == "databricks-claude-haiku-4-5"
    assert settings["embedding_model"] == "databricks-qwen3-embedding-0-6b"
    assert settings["api_key"] == ""


def test_resolve_databricks_settings_prefers_provided_values() -> None:
    values = {
        "DATABRICKS_FM_BASE_URL": "https://example.cloud.databricks.com/serving-endpoints",
        "LLM_MODEL": "custom-llm",
        "EMBEDDING_MODEL": "custom-embed",
        "DATABRICKS_PROFILE": "custom-profile",
        "DATABRICKS_FM_TOKEN": "secret-token",
    }
    settings = resolve_databricks_settings(lambda key: values.get(key, ""))
    assert settings["base_url"] == "https://example.cloud.databricks.com/serving-endpoints"
    assert settings["model_name"] == "custom-llm"
    assert settings["embedding_model"] == "custom-embed"
    assert settings["profile"] == "custom-profile"
    assert settings["api_key"] == "secret-token"


def test_resolve_databricks_settings_falls_back_to_config_profile() -> None:
    values = {"DATABRICKS_CONFIG_PROFILE": "legacy-profile"}
    settings = resolve_databricks_settings(lambda key: values.get(key, ""))
    assert settings["profile"] == "legacy-profile"


class FakeEmbeddingsResource:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        count = len(inputs) if isinstance(inputs, list) else 1
        data = [
            SimpleNamespace(index=index, embedding=[float(index), 1.0])
            for index in reversed(range(count))
        ]
        return SimpleNamespace(data=data)


def test_normalizes_databricks_serving_base() -> None:
    assert normalize_databricks_base_url("https://dbc-example.cloud.databricks.com") == (
        "https://dbc-example.cloud.databricks.com/serving-endpoints"
    )
    assert normalize_databricks_base_url("dbc-example.cloud.databricks.com") == (
        "https://dbc-example.cloud.databricks.com/serving-endpoints"
    )
    assert normalize_databricks_base_url(
        "https://dbc-example.cloud.databricks.com/ai-gateway/mlflow/v1/"
    ).endswith("/ai-gateway/mlflow/v1")


@pytest.mark.parametrize(
    "url",
    [
        "http://dbc-example.cloud.databricks.com/serving-endpoints",
        "https://example.com/serving-endpoints",
        "https://dbc-example.cloud.databricks.com/untrusted/path",
    ],
)
def test_rejects_untrusted_databricks_base(url: str) -> None:
    with pytest.raises(RagError):
        normalize_databricks_base_url(url)


def test_databricks_embeddings_batch_documents_and_instruct_query() -> None:
    resource = FakeEmbeddingsResource()
    client = SimpleNamespace(embeddings=resource)
    embeddings = _DatabricksEmbeddings(
        api_key="unused-in-test",
        base_url="https://dbc-example.cloud.databricks.com/serving-endpoints",
        model_name="databricks-qwen3-embedding-0-6b",
        batch_size=2,
        client=client,
    )

    documents = embeddings.embed_documents(["one", "two", "three"])
    query = embeddings.embed_query("total plan assets")

    assert documents == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert query == [0.0, 1.0]
    assert "extra_body" not in resource.calls[0]
    assert resource.calls[-1]["extra_body"] == {"instruction": DATABRICKS_QUERY_INSTRUCTION}


def test_databricks_oauth_profile_returns_bearer_token() -> None:
    config = SimpleNamespace(
        host="https://dbc-example.cloud.databricks.com",
        authenticate=lambda: {"Authorization": "Bearer oauth-access-token"},
    )
    credential = _DatabricksOAuthToken(
        base_url="https://dbc-example.cloud.databricks.com/serving-endpoints",
        profile="development",
        workspace_client=SimpleNamespace(config=config),
    )

    assert credential() == "oauth-access-token"


def test_databricks_oauth_profile_must_match_serving_host() -> None:
    workspace_client = SimpleNamespace(
        config=SimpleNamespace(host="https://different.cloud.databricks.com")
    )

    with pytest.raises(RagError, match="model-serving URL"):
        _DatabricksOAuthToken(
            base_url="https://dbc-example.cloud.databricks.com/serving-endpoints",
            profile="wrong-workspace",
            workspace_client=workspace_client,
        )


def test_databricks_oauth_failure_is_actionable_and_sanitized() -> None:
    def fail_authentication() -> dict[str, str]:
        raise RuntimeError("internal credential details")

    config = SimpleNamespace(
        host="https://dbc-example.cloud.databricks.com",
        authenticate=fail_authentication,
    )
    credential = _DatabricksOAuthToken(
        base_url="https://dbc-example.cloud.databricks.com/serving-endpoints",
        profile="development",
        workspace_client=SimpleNamespace(config=config),
    )

    with pytest.raises(RagError, match="databricks auth login") as error:
        credential()
    assert "internal credential details" not in str(error.value)


def test_chroma_collection_names_are_unique_per_engine() -> None:
    first = _collection_name("same-filing")
    second = _collection_name("same-filing")

    assert first.startswith("pp_")
    assert first != second


def test_rebuilding_same_filing_uses_isolated_chroma_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _DatabricksEmbeddings,
        "embed_documents",
        lambda _self, texts: [[1.0, float(index)] for index, _text in enumerate(texts)],
    )
    settings = {
        "filing_key": "same-filing",
        "chunks": [
            TextChunk(
                chunk_id="p1-c1",
                page=1,
                text="Masked ABCDEFGHI -123456789012345; filing value $42,000",
            )
        ],
        "provider": "Databricks",
        "api_key": "not-sent-in-test",
        "model_name": "databricks-claude-haiku-4-5",
        "embedding_backend": "Databricks",
        "embedding_model": "databricks-qwen3-embedding-0-6b",
        "databricks_base_url": ("https://dbc-example.cloud.databricks.com/serving-endpoints"),
    }

    first = RagEngine(**settings)
    second = RagEngine(**settings)
    try:
        assert first._collection_name != second._collection_name
        assert first.collection.count() == second.collection.count() == 1
        stored_document = first.collection.get(include=["documents"])["documents"][0]
        assert "ABCDEFGHI" not in stored_document
        assert "-123456789012345" not in stored_document
        assert "$42,000" in stored_document
    finally:
        first.close()
        second.close()


def test_answer_prompt_rejects_and_output_removes_mask_artifacts() -> None:
    class FakeLlm:
        def __init__(self) -> None:
            self.messages: list[object] = []

        def invoke(self, messages: list[object]) -> SimpleNamespace:
            self.messages = messages
            return SimpleNamespace(
                content="Masked ABCDEFGHI -123456789012345. Supported value is $42,000 [p. 1]."
            )

    engine = object.__new__(RagEngine)
    engine.provider = "Databricks"
    engine.llm = FakeLlm()

    result = engine._answer(
        {
            "query": "What is the value?",
            "conversation": [],
            "retrieved": [
                {
                    "text": "Masked ABCDEFGHI; valid value $42,000",
                    "page": 1,
                    "chunk_id": "p1-c1",
                    "distance": 0.1,
                }
            ],
        }
    )

    assert "ABCDEFGHI" not in result["answer"]
    assert "-123456789012345" not in result["answer"]
    assert "$42,000" in result["answer"]
    assert "ABCDEFGHI" not in result["citations"][0]["excerpt"]
    system_prompt = str(engine.llm.messages[0].content)
    user_prompt = str(engine.llm.messages[1].content)
    assert "Never quote" in system_prompt
    assert "numeric sentinels" in system_prompt
    assert "ABCDEFGHI" not in user_prompt
