from __future__ import annotations

from types import SimpleNamespace

import pytest

from pensionpeek.rag import (
    DATABRICKS_QUERY_INSTRUCTION,
    RagError,
    _DatabricksEmbeddings,
    normalize_databricks_base_url,
)


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
