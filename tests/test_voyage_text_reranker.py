import os

import pytest
from haystack import Document
from haystack.utils.auth import Secret
from voyageai.error import InvalidRequestError

from haystack_integrations.components.rerankers.voyage_rerankers import VoyageRanker


class TestVoyageTextReranker:
    @pytest.mark.unit
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        reranker = VoyageRanker()

        assert reranker.client.api_key == "fake-api-key"
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
            api_key=Secret.from_token("fake-api-key"),
            model="model",
            truncate=True,
            top_k=10,
            prefix="prefix",
            suffix="suffix",
            meta_fields_to_embed=["meta_field_1", "meta_field_2"],
            meta_data_separator=",",
        )
        assert reranker.client.api_key == "fake-api-key"
        assert reranker.model == "model"
        assert reranker.truncate is True
        assert reranker.top_k == 10
        assert reranker.prefix == "prefix"
        assert reranker.suffix == "suffix"
        assert reranker.meta_fields_to_embed == ["meta_field_1", "meta_field_2"]
        assert reranker.meta_data_separator == ","

    @pytest.mark.unit
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            VoyageRanker()

    @pytest.mark.unit
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        component = VoyageRanker()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rerankers.voyage_rerankers.voyage_text_reranker."
            "VoyageRanker",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "rerank-2",
                "truncate": None,
                "top_k": None,
                "prefix": "",
                "suffix": "",
                "meta_fields_to_embed": [],
                "meta_data_separator": "\n"
            },
        }

    @pytest.mark.unit
    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.rerankers.voyage_rerankers.voyage_text_reranker."
            "VoyageRanker",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "rerank-2",
                "truncate": None,
                "top_k": 10,
                "prefix": "",
                "suffix": "",
                "meta_fields_to_embed": None,
                "meta_data_separator": "\n"
            },
        }

        reranker = VoyageRanker.from_dict(data)
        assert reranker.client.api_key == "fake-api-key"
        assert reranker.top_k == 10
        assert reranker.model == "rerank-2"
        assert reranker.truncate is None
        assert reranker.prefix == ""
        assert reranker.suffix == ""
        assert reranker.meta_fields_to_embed == []
        assert  reranker.meta_data_separator == '\n'

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        component = VoyageRanker(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="model",
            truncate=True,
            top_k=10,
            prefix="prefix",
            suffix="suffix",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rerankers.voyage_rerankers.voyage_text_reranker."
            "VoyageRanker",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "model",
                "truncate": True,
                "top_k": 10,
                "prefix": "prefix",
                "suffix": "suffix",
                'meta_data_separator': '\n',
                'meta_fields_to_embed': [],

            },
        }

    @pytest.mark.unit
    def test_from_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.rerankers.voyage_rerankers.voyage_text_reranker."
            "VoyageRanker",
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
        assert reranker.client.api_key == "fake-api-key"
        assert reranker.model == "model"
        assert reranker.truncate is True
        assert reranker.top_k == 10
        assert reranker.prefix == "prefix"
        assert reranker.suffix == "suffix"

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    def test_run(self):
        model = "rerank-2"

        documents = [
            Document(id="abcd", content="Paris is in France"),
            Document(id="efgh", content="Berlin is in Germany"),
            Document(id="ijkl", content="Lyon is in France"),
        ]

        reranker = VoyageRanker(model=model, prefix="prefix ", suffix=" suffix")
        result = reranker.run(query="The food was delicious", documents=documents, top_k=2)

        assert len(result["documents"]) == 2
        assert all(isinstance(x, Document) for x in result["documents"])

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        reranker = VoyageRanker(api_key=Secret.from_token("fake-api-key"))

        integer_input = 1
        documents = [
            Document(id="abcd", content="Paris is in France"),
            Document(id="efgh", content="Berlin is in Germany"),
            Document(id="ijkl", content="Lyon is in France"),
        ]

        with pytest.raises(InvalidRequestError, match=f"not a valid string"):
            reranker.run(query=integer_input, documents=documents)
