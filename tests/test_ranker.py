import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack import Document
from haystack.utils.auth import Secret
from voyageai.error import InvalidRequestError

from haystack_integrations.components.rankers.voyage import VoyageRanker


class TestVoyageTextReranker:
    @pytest.mark.unit
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        reranker = VoyageRanker(model="rerank-2")

        assert reranker._client is None
        assert reranker._async_client is None
        assert reranker.model == "rerank-2"
        assert reranker.truncate is None
        assert reranker.prefix == ""
        assert reranker.suffix == ""
        assert reranker.top_k is None
        assert reranker.meta_fields_to_embed == []
        assert reranker.meta_data_separator == "\n"

    @pytest.mark.unit
    def test_init_with_parameters(self):
        reranker = VoyageRanker(
            model="model",
            api_key=Secret.from_token("fake-api-key"),
            truncate=True,
            top_k=10,
            prefix="prefix",
            suffix="suffix",
            meta_fields_to_embed=["meta_field_1", "meta_field_2"],
            meta_data_separator=",",
        )
        assert reranker._client is None
        assert reranker._async_client is None
        assert reranker.model == "model"
        assert reranker.truncate is True
        assert reranker.top_k == 10
        assert reranker.prefix == "prefix"
        assert reranker.suffix == "suffix"
        assert reranker.meta_fields_to_embed == ["meta_field_1", "meta_field_2"]
        assert reranker.meta_data_separator == ","

    @pytest.mark.unit
    def test_init_with_explicit_timeout_and_max_retries(self):
        reranker = VoyageRanker(
            model="rerank-2",
            api_key=Secret.from_token("fake-api-key"),
            timeout=60,
            max_retries=3,
        )
        assert reranker._client is None
        assert reranker._timeout == 60
        assert reranker._max_retries == 3

    @pytest.mark.unit
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        embedder = VoyageRanker(model="rerank-2")
        # Init succeeds, but warm_up() should fail
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            embedder.warm_up()

    @pytest.mark.unit
    def test_warm_up(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageRanker(model="rerank-2")

        assert embedder._client is None
        assert embedder._async_client is None
        embedder.warm_up()
        assert embedder._client is not None
        assert embedder._async_client is not None

        # Idempotent
        embedder.warm_up()
        assert embedder._client is not None

    @pytest.mark.unit
    def test_client_property(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageRanker(model="rerank-2")

        assert embedder._client is None
        client = embedder.client
        assert client is not None
        assert embedder._client is not None

        # Second access returns the same client (short-circuit branch)
        client2 = embedder.client
        assert client2 is not None

    @pytest.mark.unit
    def test_async_client_property(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageRanker(model="rerank-2")

        assert embedder._async_client is None
        async_client = embedder.async_client
        assert async_client is not None
        assert embedder._async_client is not None

    @pytest.mark.unit
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        component = VoyageRanker(model="rerank-2")
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.voyage.ranker.VoyageRanker",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "rerank-2",
                "truncate": None,
                "top_k": None,
                "prefix": "",
                "suffix": "",
                "meta_fields_to_embed": [],
                "meta_data_separator": "\n",
            },
        }

    @pytest.mark.unit
    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.rankers.voyage.ranker.VoyageRanker",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "rerank-2",
                "truncate": None,
                "top_k": 10,
                "prefix": "",
                "suffix": "",
                "meta_fields_to_embed": None,
                "meta_data_separator": "\n",
            },
        }

        reranker = VoyageRanker.from_dict(data)
        assert reranker._client is None
        assert reranker._async_client is None
        assert reranker.top_k == 10
        assert reranker.model == "rerank-2"
        assert reranker.truncate is None
        assert reranker.prefix == ""
        assert reranker.suffix == ""
        assert reranker.meta_fields_to_embed == []
        assert reranker.meta_data_separator == "\n"

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        component = VoyageRanker(
            model="model",
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            truncate=True,
            top_k=10,
            prefix="prefix",
            suffix="suffix",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.voyage.ranker.VoyageRanker",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "model",
                "truncate": True,
                "top_k": 10,
                "prefix": "prefix",
                "suffix": "suffix",
                "meta_data_separator": "\n",
                "meta_fields_to_embed": [],
            },
        }

    @pytest.mark.unit
    def test_from_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.rankers.voyage.ranker.VoyageRanker",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "model",
                "truncate": True,
                "top_k": 10,
                "prefix": "prefix",
                "suffix": "suffix",
            },
        }

        reranker = VoyageRanker.from_dict(data)
        assert reranker._client is None
        assert reranker._async_client is None
        assert reranker.model == "model"
        assert reranker.truncate is True
        assert reranker.top_k == 10
        assert reranker.prefix == "prefix"
        assert reranker.suffix == "suffix"

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    @pytest.mark.flaky(reruns=3, reruns_delay=60)
    @pytest.mark.asyncio
    async def test_run(self):
        model = "rerank-2.5"

        documents = [
            Document(content="Paris is in France"),
            Document(content="Berlin is in Germany"),
            Document(content="Lyon is in France"),
        ]

        reranker = VoyageRanker(model=model, prefix="prefix ", suffix=" suffix")
        result = reranker.run(query="The food was delicious", documents=documents, top_k=2)

        assert len(result["documents"]) == 2
        assert all(isinstance(x, Document) for x in result["documents"])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_wrong_input_format(self):
        reranker = VoyageRanker(model="rerank-2", api_key=Secret.from_token("fake-api-key"))

        integer_input = 1
        documents = [
            Document(content="Paris is in France"),
            Document(content="Berlin is in Germany"),
            Document(content="Lyon is in France"),
        ]

        # Mock the sync client to see the error from voyageai
        mock_client = MagicMock()
        mock_client.rerank.side_effect = InvalidRequestError("not a valid string")
        reranker._client = mock_client
        reranker._async_client = MagicMock()

        with pytest.raises(InvalidRequestError, match="not a valid string"):
            reranker.run(query=integer_input, documents=documents)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_with_negative_top_k(self):
        reranker = VoyageRanker(model="rerank-2", api_key=Secret.from_token("fake-api-key"))

        documents = [
            Document(content="Paris is in France"),
            Document(content="Berlin is in Germany"),
        ]

        with pytest.raises(ValueError, match="top_k must be > 0"):
            reranker.run(query="test query", documents=documents, top_k=-1)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_with_zero_top_k(self):
        # When top_k is set in __init__ and we pass 0 to run(),
        # the logic `top_k = top_k or self.top_k` will use self.top_k (5)
        # So we need to test with a default top_k that's 0
        reranker = VoyageRanker(model="rerank-2", api_key=Secret.from_token("fake-api-key"), top_k=0)

        documents = [
            Document(content="Paris is in France"),
            Document(content="Berlin is in Germany"),
        ]

        with pytest.raises(ValueError, match="top_k must be > 0"):
            reranker.run(query="test query", documents=documents)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_async_with_mocked_client(self):
        """Test run_async with mocked async API client."""
        reranker = VoyageRanker(
            model="rerank-2.5",
            api_key=Secret.from_token("fake-api-key"),
            prefix="prefix ",
            suffix=" suffix",
        )

        documents = [
            Document(content="Paris is in France"),
            Document(content="Berlin is in Germany"),
            Document(content="Lyon is in France"),
        ]

        mock_outputs = [MagicMock(index=0, relevance_score=0.95), MagicMock(index=1, relevance_score=0.85)]

        mock_response = MagicMock()
        mock_response.results = mock_outputs

        reranker._async_client = MagicMock()
        reranker._async_client.rerank = AsyncMock(return_value=mock_response)
        reranker._client = MagicMock()

        result = await reranker.run_async(query="What is the capital of France?", documents=documents, top_k=2)

        assert len(result["documents"]) == 2
        assert all(isinstance(x, Document) for x in result["documents"])
        assert result["documents"][0].score == 0.95
        assert result["documents"][1].score == 0.85

        reranker._async_client.rerank.assert_called_once()
        call_kwargs = reranker._async_client.rerank.call_args[1]
        assert call_kwargs["query"] == "What is the capital of France?"
        assert call_kwargs["top_k"] == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_async_with_exceeding_document_count(self):
        """Test run_async truncates documents exceeding MAX_NUM_DOCS."""
        reranker = VoyageRanker(model="rerank-2", api_key=Secret.from_token("fake-api-key"))

        documents = [Document(content=f"Content {i}") for i in range(1100)]

        mock_outputs = [MagicMock(index=i, relevance_score=0.95 - (i * 0.01)) for i in range(10)]

        mock_response = MagicMock()
        mock_response.results = mock_outputs

        reranker._async_client = MagicMock()
        reranker._async_client.rerank = AsyncMock(return_value=mock_response)
        reranker._client = MagicMock()

        result = await reranker.run_async(query="test query", documents=documents, top_k=10)

        reranker._async_client.rerank.assert_called_once()
        call_kwargs = reranker._async_client.rerank.call_args[1]
        assert len(call_kwargs["documents"]) == 1000
        assert len(result["documents"]) == 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_async_with_negative_top_k(self):
        """Test run_async raises ValueError for negative top_k."""
        reranker = VoyageRanker(model="rerank-2", api_key=Secret.from_token("fake-api-key"))

        documents = [Document(content="Paris is in France")]

        with pytest.raises(ValueError, match="top_k must be > 0"):
            await reranker.run_async(query="test query", documents=documents, top_k=-1)

    @pytest.mark.unit
    def test_prepare_input_docs_with_metadata(self):
        """Test _prepare_input_docs concatenates metadata fields correctly."""
        reranker = VoyageRanker(
            model="rerank-2",
            api_key=Secret.from_token("fake-api-key"),
            meta_fields_to_embed=["title", "author"],
            meta_data_separator=" | ",
        )

        documents = [
            Document(content="Content about Paris", meta={"title": "Paris Guide", "author": "Alice"}),
            Document(content="Content about Berlin", meta={"title": "Berlin Guide", "author": "Bob"}),
        ]

        result = reranker._prepare_input_docs(documents)

        assert result == [
            "Paris Guide | Alice | Content about Paris",
            "Berlin Guide | Bob | Content about Berlin",
        ]

    @pytest.mark.unit
    def test_prepare_input_docs_with_empty_fields(self):
        """Test _prepare_input_docs handles missing metadata fields gracefully."""
        reranker = VoyageRanker(
            model="rerank-2",
            api_key=Secret.from_token("fake-api-key"),
            meta_fields_to_embed=["title"],
        )

        documents = [
            Document(content="Content only"),  # No meta at all
            Document(content="Content with meta", meta={"title": ""}),  # Empty title
        ]

        result = reranker._prepare_input_docs(documents)

        # Missing keys and empty values are skipped
        assert result[0] == "Content only"
        assert result[1] == "Content with meta"  # Empty title is falsy and skipped

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_with_exceeding_document_count(self):
        reranker = VoyageRanker(model="rerank-2", api_key=Secret.from_token("fake-api-key"))

        # Create 1100 documents to exceed MAX_NUM_DOCS (1000)
        documents = [Document(content=f"Content {i}") for i in range(1100)]

        # Mock the sync client's rerank method
        mock_outputs = [MagicMock(index=i, relevance_score=0.95 - (i * 0.01)) for i in range(10)]  # Return 10 results

        mock_response = MagicMock()
        mock_response.results = mock_outputs

        mock_client = MagicMock()
        mock_client.rerank = MagicMock(return_value=mock_response)
        reranker._client = mock_client
        reranker._async_client = MagicMock()

        result = reranker.run(query="test query", documents=documents, top_k=10)

        # Verify that rerank was called with only the first 1000 documents
        reranker._client.rerank.assert_called_once()
        call_kwargs = reranker._client.rerank.call_args[1]
        assert len(call_kwargs["documents"]) == 1000

        # Verify results are returned correctly
        assert len(result["documents"]) == 10
        assert all(isinstance(doc, Document) for doc in result["documents"])
        assert result["documents"][0].score == 0.95
        assert result["documents"][-1].score == 0.95 - (9 * 0.01)
