import os
from unittest.mock import Mock, patch

import pytest
from haystack import Document
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.voyage_embedders import VoyageDocumentEmbedder


class TestVoyageDocumentEmbedder:
    @pytest.mark.unit
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageDocumentEmbedder(model="voyage-3")

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.input_type is None
        assert embedder.model == "voyage-3"
        assert embedder.truncate is True
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.output_dimension is None
        assert embedder.output_dtype == "float"
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.metadata_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    @pytest.mark.unit
    def test_init_with_parameters(self):
        embedder = VoyageDocumentEmbedder(
            model="voyage-3-large",
            api_key=Secret.from_token("fake-api-key"),
            input_type="document",
            truncate=False,
            prefix="prefix",
            suffix="suffix",
            batch_size=4,
            output_dimension=2048,
            output_dtype="int8",
            progress_bar=False,
            metadata_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "voyage-3-large"
        assert embedder.input_type == "document"
        assert embedder.truncate is False
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.output_dimension == 2048
        assert embedder.output_dtype == "int8"
        assert embedder.batch_size == 4
        assert embedder.progress_bar is False
        assert embedder.metadata_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    @pytest.mark.unit
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            VoyageDocumentEmbedder(model="voyage-3")

    @pytest.mark.unit
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        component = VoyageDocumentEmbedder(model="voyage-3")
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_document_embedder."
            "VoyageDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-3",
                "input_type": None,
                "truncate": True,
                "prefix": "",
                "suffix": "",
                "output_dimension": None,
                "output_dtype": "float",
                "batch_size": 32,
                "progress_bar": True,
                "metadata_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

    @pytest.mark.unit
    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_document_embedder."
            "VoyageDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-3",
                "input_type": None,
                "truncate": True,
                "prefix": "",
                "suffix": "",
                "output_dimension": None,
                "output_dtype": "float",
                "batch_size": 32,
                "progress_bar": True,
                "metadata_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

        embedder = VoyageDocumentEmbedder.from_dict(data)

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "voyage-3"
        assert embedder.input_type is None
        assert embedder.truncate is True
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.output_dimension is None
        assert embedder.output_dtype == "float"
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.metadata_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        component = VoyageDocumentEmbedder(
            model="voyage-3-large",
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            truncate=False,
            input_type="document",
            prefix="prefix",
            suffix="suffix",
            output_dimension=2048,
            output_dtype="int8",
            batch_size=4,
            progress_bar=False,
            metadata_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_document_embedder."
            "VoyageDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "voyage-3-large",
                "truncate": False,
                "input_type": "document",
                "prefix": "prefix",
                "suffix": "suffix",
                "output_dimension": 2048,
                "output_dtype": "int8",
                "batch_size": 4,
                "progress_bar": False,
                "metadata_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }

    @pytest.mark.unit
    def test_from_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_document_embedder."
            "VoyageDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "voyage-3-large",
                "truncate": False,
                "input_type": "document",
                "prefix": "prefix",
                "suffix": "suffix",
                "output_dimension": 2048,
                "output_dtype": "int8",
                "batch_size": 4,
                "progress_bar": False,
                "metadata_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }

        embedder = VoyageDocumentEmbedder.from_dict(data)

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "voyage-3-large"
        assert embedder.input_type == "document"
        assert embedder.truncate is False
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.output_dimension == 2048
        assert embedder.output_dtype == "int8"
        assert embedder.batch_size == 4
        assert embedder.progress_bar is False
        assert embedder.metadata_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    @pytest.mark.unit
    def test_prepare_texts_to_embed_w_metadata(self):
        documents = [
            Document(content=f"document number {i}: content", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = VoyageDocumentEmbedder(
            model="voyage-3",
            api_key=Secret.from_token("fake-api-key"),
            metadata_fields_to_embed=["meta_field"],
            embedding_separator=" | ",
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "meta_value 0 | document number 0: content",
            "meta_value 1 | document number 1: content",
            "meta_value 2 | document number 2: content",
            "meta_value 3 | document number 3: content",
            "meta_value 4 | document number 4: content",
        ]

    @pytest.mark.unit
    def test_prepare_texts_to_embed_w_suffix(self):
        documents = [Document(content=f"document number {i}") for i in range(5)]

        embedder = VoyageDocumentEmbedder(
            model="voyage-3", api_key=Secret.from_token("fake-api-key"), prefix="my_prefix ", suffix=" my_suffix"
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "my_prefix document number 0 my_suffix",
            "my_prefix document number 1 my_suffix",
            "my_prefix document number 2 my_suffix",
            "my_prefix document number 3 my_suffix",
            "my_prefix document number 4 my_suffix",
        ]

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        embedder = VoyageDocumentEmbedder(model="voyage-3", api_key=Secret.from_token("fake-api-key"))

        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="VoyageDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="VoyageDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

    @pytest.mark.unit
    def test_run_on_empty_list(self):
        embedder = VoyageDocumentEmbedder(model="voyage-3", api_key=Secret.from_token("fake-api-key"))

        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list

    @pytest.mark.unit
    def test_run_with_mocked_api(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        embedder = VoyageDocumentEmbedder(
            model="voyage-3",
            prefix="prefix ",
            suffix=" suffix",
            metadata_fields_to_embed=["topic"],
            embedding_separator=" | ",
            api_key=Secret.from_token("fake-api-key"),
        )

        # Mock the client.embed method
        mock_response = Mock()
        mock_response.embeddings = [
            [0.1] * 1024,  # 1024 dimensions
            [0.4] * 1024,  # 1024 dimensions
        ]
        mock_response.total_tokens = 18

        with patch.object(embedder.client, "embed", return_value=mock_response):
            result = embedder.run(documents=docs)

        documents_with_embeddings = result["documents"]
        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == 2
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 1024
        assert result["meta"]["total_tokens"] == 18

    @pytest.mark.unit
    def test_run_with_mocked_api_batch_processing(self):
        docs = [Document(content=f"content {i}") for i in range(5)]

        embedder = VoyageDocumentEmbedder(
            model="voyage-3",
            api_key=Secret.from_token("fake-api-key"),
            batch_size=2,
        )

        # Mock the client.embed method to return embeddings for each batch
        def mock_embed(*args, **kwargs):
            mock_response = Mock()
            texts = kwargs.get("texts", args[0] if args else [])
            mock_response.embeddings = [[0.1] * 1024 for _ in range(len(texts))]
            mock_response.total_tokens = len(texts) * 6
            return mock_response

        with patch.object(embedder.client, "embed", side_effect=mock_embed):
            result = embedder.run(documents=docs)

        assert len(result["documents"]) == 5
        assert all(len(doc.embedding) == 1024 for doc in result["documents"])

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    @pytest.mark.flaky(reruns=3, reruns_delay=60)
    @pytest.mark.parametrize("model", ["voyage-4", "voyage-4-large", "voyage-4-lite"])
    def test_run_voyage_4(self, model):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        embedder = VoyageDocumentEmbedder(model=model, timeout=600, max_retries=1200)
        result = embedder.run(documents=docs)

        assert len(result["documents"]) == len(docs)
        for doc in result["documents"]:
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 1024
            assert all(isinstance(x, float) for x in doc.embedding)
        assert result["meta"]["total_tokens"] > 0

        # Custom dimensions
        embedder_dim = VoyageDocumentEmbedder(model=model, output_dimension=512, timeout=600, max_retries=1200)
        result_dim = embedder_dim.run(documents=[Document(content="I love cheese")])
        assert len(result_dim["documents"][0].embedding) == 512

        # Quantized output
        embedder_int8 = VoyageDocumentEmbedder(model=model, output_dtype="int8", timeout=600, max_retries=1200)
        result_int8 = embedder_int8.run(documents=[Document(content="I love cheese")])
        assert len(result_int8["documents"][0].embedding) == 1024

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    @pytest.mark.flaky(reruns=3, reruns_delay=60)
    def test_run(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        model = "voyage-3"
        embedder = VoyageDocumentEmbedder(
            model=model,
            prefix="prefix ",
            suffix=" suffix",
            metadata_fields_to_embed=["topic"],
            embedding_separator=" | ",
            timeout=120,
            max_retries=10,
        )

        result = embedder.run(documents=docs)

        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 1024
            assert all(isinstance(x, float) for x in doc.embedding)
        assert result["meta"]["total_tokens"] == 18, "Total tokens does not match"
