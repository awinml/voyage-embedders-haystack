import os

import pytest
from haystack import Document
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.voyage_embedders import VoyageContextualizedDocumentEmbedder


class TestVoyageContextualizedDocumentEmbedder:
    @pytest.mark.unit
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageContextualizedDocumentEmbedder()

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.input_type is None
        assert embedder.model == "voyage-context-3"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.output_dimension is None
        assert embedder.output_dtype == "float"
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.metadata_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.source_id_field == "source_id"
        assert embedder.chunk_fn is None

    @pytest.mark.unit
    def test_init_with_parameters(self):
        embedder = VoyageContextualizedDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="voyage-context-3",
            input_type="document",
            prefix="prefix",
            suffix="suffix",
            batch_size=4,
            output_dimension=2048,
            output_dtype="int8",
            progress_bar=False,
            metadata_fields_to_embed=["test_field"],
            embedding_separator=" | ",
            source_id_field="custom_source_field",
        )

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "voyage-context-3"
        assert embedder.input_type == "document"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.output_dimension == 2048
        assert embedder.output_dtype == "int8"
        assert embedder.batch_size == 4
        assert embedder.progress_bar is False
        assert embedder.metadata_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "
        assert embedder.source_id_field == "custom_source_field"

    @pytest.mark.unit
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            VoyageContextualizedDocumentEmbedder()

    @pytest.mark.unit
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        component = VoyageContextualizedDocumentEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.voyage_embedders."
            "voyage_contextualized_document_embedder.VoyageContextualizedDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-context-3",
                "input_type": None,
                "prefix": "",
                "suffix": "",
                "output_dimension": None,
                "output_dtype": "float",
                "batch_size": 32,
                "progress_bar": True,
                "metadata_fields_to_embed": [],
                "embedding_separator": "\n",
                "source_id_field": "source_id",
                "chunk_fn": None,
            },
        }

    @pytest.mark.unit
    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.voyage_embedders."
            "voyage_contextualized_document_embedder.VoyageContextualizedDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-context-3",
                "input_type": None,
                "prefix": "",
                "suffix": "",
                "output_dimension": None,
                "output_dtype": "float",
                "batch_size": 32,
                "progress_bar": True,
                "metadata_fields_to_embed": [],
                "embedding_separator": "\n",
                "source_id_field": "source_id",
                "chunk_fn": None,
            },
        }

        embedder = VoyageContextualizedDocumentEmbedder.from_dict(data)

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "voyage-context-3"
        assert embedder.input_type is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.output_dimension is None
        assert embedder.output_dtype == "float"
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.metadata_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.source_id_field == "source_id"
        assert embedder.chunk_fn is None

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        component = VoyageContextualizedDocumentEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="voyage-context-3",
            input_type="document",
            prefix="prefix",
            suffix="suffix",
            output_dimension=2048,
            output_dtype="int8",
            batch_size=4,
            progress_bar=False,
            metadata_fields_to_embed=["test_field"],
            embedding_separator=" | ",
            source_id_field="custom_source_field",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.voyage_embedders."
            "voyage_contextualized_document_embedder.VoyageContextualizedDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "voyage-context-3",
                "input_type": "document",
                "prefix": "prefix",
                "suffix": "suffix",
                "output_dimension": 2048,
                "output_dtype": "int8",
                "batch_size": 4,
                "progress_bar": False,
                "metadata_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
                "source_id_field": "custom_source_field",
                "chunk_fn": None,
            },
        }

    @pytest.mark.unit
    def test_from_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.voyage_embedders."
            "voyage_contextualized_document_embedder.VoyageContextualizedDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "voyage-context-3",
                "input_type": "document",
                "prefix": "prefix",
                "suffix": "suffix",
                "output_dimension": 2048,
                "output_dtype": "int8",
                "batch_size": 4,
                "progress_bar": False,
                "metadata_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
                "source_id_field": "custom_source_field",
                "chunk_fn": None,
            },
        }

        embedder = VoyageContextualizedDocumentEmbedder.from_dict(data)

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "voyage-context-3"
        assert embedder.input_type == "document"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.output_dimension == 2048
        assert embedder.output_dtype == "int8"
        assert embedder.batch_size == 4
        assert embedder.progress_bar is False
        assert embedder.metadata_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "
        assert embedder.source_id_field == "custom_source_field"

    @pytest.mark.unit
    def test_prepare_texts_to_embed_w_metadata(self):
        documents = [
            Document(content=f"document number {i}: content", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = VoyageContextualizedDocumentEmbedder(
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

        embedder = VoyageContextualizedDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"), prefix="my_prefix ", suffix=" my_suffix"
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
    def test_group_documents_by_source(self):
        documents = [
            Document(content="doc1_chunk1", meta={"source_id": "doc1"}),
            Document(content="doc1_chunk2", meta={"source_id": "doc1"}),
            Document(content="doc2_chunk1", meta={"source_id": "doc2"}),
            Document(content="doc1_chunk3", meta={"source_id": "doc1"}),
            Document(content="doc2_chunk2", meta={"source_id": "doc2"}),
        ]

        embedder = VoyageContextualizedDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))

        grouped_docs, source_order = embedder._group_documents_by_source(documents)

        assert source_order == ["doc1", "doc2"]
        assert len(grouped_docs["doc1"]) == 3
        assert len(grouped_docs["doc2"]) == 2
        assert grouped_docs["doc1"][0].content == "doc1_chunk1"
        assert grouped_docs["doc1"][1].content == "doc1_chunk2"
        assert grouped_docs["doc1"][2].content == "doc1_chunk3"
        assert grouped_docs["doc2"][0].content == "doc2_chunk1"
        assert grouped_docs["doc2"][1].content == "doc2_chunk2"

    @pytest.mark.unit
    def test_group_documents_by_source_missing_field(self):
        documents = [
            Document(content="doc1_chunk1", meta={"source_id": "doc1"}),
            Document(content="doc2_chunk1", meta={}),  # Missing source_id
        ]

        embedder = VoyageContextualizedDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))

        with pytest.raises(
            ValueError, match="Document is missing the 'source_id' metadata field"
        ):
            embedder._group_documents_by_source(documents)

    @pytest.mark.unit
    def test_group_documents_by_source_custom_field(self):
        documents = [
            Document(content="doc1_chunk1", meta={"custom_field": "doc1"}),
            Document(content="doc1_chunk2", meta={"custom_field": "doc1"}),
            Document(content="doc2_chunk1", meta={"custom_field": "doc2"}),
        ]

        embedder = VoyageContextualizedDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"), source_id_field="custom_field"
        )

        grouped_docs, source_order = embedder._group_documents_by_source(documents)

        assert source_order == ["doc1", "doc2"]
        assert len(grouped_docs["doc1"]) == 2
        assert len(grouped_docs["doc2"]) == 1

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        embedder = VoyageContextualizedDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))

        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(
            TypeError, match="VoyageContextualizedDocumentEmbedder expects a list of Documents as input"
        ):
            embedder.run(documents=string_input)

        with pytest.raises(
            TypeError, match="VoyageContextualizedDocumentEmbedder expects a list of Documents as input"
        ):
            embedder.run(documents=list_integers_input)

    @pytest.mark.unit
    def test_run_on_empty_list(self):
        embedder = VoyageContextualizedDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))

        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list
        assert result["meta"]["total_tokens"] == 0

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    def test_run(self):
        docs = [
            Document(content="Introduction to quantum computing.", meta={"source_id": "doc1", "topic": "Quantum"}),
            Document(content="Quantum bits or qubits are the basic unit.", meta={"source_id": "doc1", "topic": "Quantum"}),
            Document(content="Classical computers use binary bits.", meta={"source_id": "doc2", "topic": "Classical"}),
            Document(content="Binary systems have two states: 0 and 1.", meta={"source_id": "doc2", "topic": "Classical"}),
        ]

        model = "voyage-context-3"
        embedder = VoyageContextualizedDocumentEmbedder(
            model=model,
            prefix="prefix ",
            suffix=" suffix",
            metadata_fields_to_embed=["topic"],
            embedding_separator=" | ",
            timeout=600,
            max_retries=1200,
        )

        result = embedder.run(documents=docs)

        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 1024  # Default dimension for voyage-context-3
            assert all(isinstance(x, float) for x in doc.embedding)

        # Verify that the embeddings are different (contextualized)
        assert documents_with_embeddings[0].embedding != documents_with_embeddings[2].embedding

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    def test_run_with_single_source(self):
        docs = [
            Document(content="First chunk of content.", meta={"source_id": "single_doc"}),
            Document(content="Second chunk of content.", meta={"source_id": "single_doc"}),
            Document(content="Third chunk of content.", meta={"source_id": "single_doc"}),
        ]

        embedder = VoyageContextualizedDocumentEmbedder(model="voyage-context-3")

        result = embedder.run(documents=docs)

        documents_with_embeddings = result["documents"]

        assert len(documents_with_embeddings) == 3
        for doc in documents_with_embeddings:
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 1024
