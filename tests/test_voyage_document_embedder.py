from typing import List
from unittest.mock import patch

import numpy as np
import pytest
import voyageai
from haystack.dataclasses import Document

from voyage_embedders.voyage_document_embedder import VoyageDocumentEmbedder


def mock_voyageai_response(list_of_text: List[str], model: str = "voyage-01", **kwargs) -> List[List[float]]:  # noqa
    response = [np.random.rand(1024).tolist() for i in range(len(list_of_text))]
    return response


class TestVoyageDocumentEmbedder:
    @pytest.mark.unit
    def test_init_default(self, monkeypatch):
        voyageai.api_key = None
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageDocumentEmbedder()

        assert voyageai.api_key == "fake-api-key"

        assert embedder.model_name == "voyage-01"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 8
        assert embedder.progress_bar is True
        assert embedder.metadata_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    @pytest.mark.unit
    def test_init_with_parameters(self):
        embedder = VoyageDocumentEmbedder(
            api_key="fake-api-key",
            model_name="model",
            prefix="prefix",
            suffix="suffix",
            batch_size=4,
            progress_bar=False,
            metadata_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        assert voyageai.api_key == "fake-api-key"

        assert embedder.model_name == "model"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 4
        assert embedder.progress_bar is False
        assert embedder.metadata_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    @pytest.mark.unit
    def test_init_fail_wo_api_key(self, monkeypatch):
        voyageai.api_key = None
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="VoyageDocumentEmbedder expects an VoyageAI API key"):
            VoyageDocumentEmbedder()

    @pytest.mark.unit
    def test_to_dict(self):
        component = VoyageDocumentEmbedder(api_key="fake-api-key")
        data = component.to_dict()
        assert data == {
            "type": "voyage_embedders.voyage_document_embedder.VoyageDocumentEmbedder",
            "init_parameters": {
                "model_name": "voyage-01",
                "prefix": "",
                "suffix": "",
                "batch_size": 8,
                "progress_bar": True,
                "metadata_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = VoyageDocumentEmbedder(
            api_key="fake-api-key",
            model_name="model",
            prefix="prefix",
            suffix="suffix",
            batch_size=4,
            progress_bar=False,
            metadata_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        data = component.to_dict()
        assert data == {
            "type": "voyage_embedders.voyage_document_embedder.VoyageDocumentEmbedder",
            "init_parameters": {
                "model_name": "model",
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 4,
                "progress_bar": False,
                "metadata_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }

    @pytest.mark.unit
    def test_prepare_texts_to_embed_w_metadata(self):
        documents = [
            Document(content=f"document number {i}: content", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = VoyageDocumentEmbedder(
            api_key="fake-api-key", metadata_fields_to_embed=["meta_field"], embedding_separator=" | "
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        # note that newline is replaced by space
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

        embedder = VoyageDocumentEmbedder(api_key="fake-api-key", prefix="my_prefix ", suffix=" my_suffix")

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "my_prefix document number 0 my_suffix",
            "my_prefix document number 1 my_suffix",
            "my_prefix document number 2 my_suffix",
            "my_prefix document number 3 my_suffix",
            "my_prefix document number 4 my_suffix",
        ]

    @pytest.mark.unit
    def test_embed_batch(self):
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        with patch("voyage_embedders.voyage_document_embedder.get_embeddings") as voyageai_embedding_patch:
            voyageai_embedding_patch.side_effect = mock_voyageai_response
            embedder = VoyageDocumentEmbedder(api_key="fake-api-key", model_name="model")

            embeddings = embedder._embed_batch(texts_to_embed=texts, batch_size=2)

            assert voyageai_embedding_patch.call_count == 3

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 1024
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.unit
    def test_run(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        model = "voyage-01-lite"
        with patch("voyage_embedders.voyage_document_embedder.get_embeddings") as voyageai_embedding_patch:
            voyageai_embedding_patch.side_effect = mock_voyageai_response
            embedder = VoyageDocumentEmbedder(
                api_key="fake-api-key",
                model_name=model,
                prefix="prefix ",
                suffix=" suffix",
                metadata_fields_to_embed=["topic"],
                embedding_separator=" | ",
            )

            result = embedder.run(documents=docs)

            voyageai_embedding_patch.assert_called_once_with(
                model=model,
                list_of_text=[
                    "prefix Cuisine | I love cheese suffix",
                    "prefix ML | A transformer is a deep learning architecture suffix",
                ],
                batch_size=8,
                input_type="document",
            )
        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 1024
            assert all(isinstance(x, float) for x in doc.embedding)

    @pytest.mark.unit
    def test_run_custom_batch_size(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        model = "voyage-01-lite"
        with patch("voyage_embedders.voyage_document_embedder.get_embeddings") as voyageai_embedding_patch:
            voyageai_embedding_patch.side_effect = mock_voyageai_response
            embedder = VoyageDocumentEmbedder(
                api_key="fake-api-key",
                model_name=model,
                prefix="prefix ",
                suffix=" suffix",
                metadata_fields_to_embed=["topic"],
                embedding_separator=" | ",
                batch_size=1,
            )

            result = embedder.run(documents=docs)

            assert voyageai_embedding_patch.call_count == 2

        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 1024
            assert all(isinstance(x, float) for x in doc.embedding)

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        embedder = VoyageDocumentEmbedder(api_key="fake-api-key")

        # wrong formats
        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="VoyageDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="VoyageDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

    @pytest.mark.unit
    def test_run_on_empty_list(self):
        embedder = VoyageDocumentEmbedder(api_key="fake-api-key")

        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list
