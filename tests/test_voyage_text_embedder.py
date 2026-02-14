import os
from unittest.mock import Mock, patch

import pytest
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.voyage_embedders import VoyageTextEmbedder


class TestVoyageTextEmbedder:
    @pytest.mark.unit
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageTextEmbedder(model="voyage-3")

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.input_type is None
        assert embedder.model == "voyage-3"
        assert embedder.truncate is True
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.output_dimension is None
        assert embedder.output_dtype == "float"

    @pytest.mark.unit
    def test_init_with_parameters(self):
        embedder = VoyageTextEmbedder(
            model="voyage-3-large",
            api_key=Secret.from_token("fake-api-key"),
            input_type="document",
            truncate=False,
            prefix="prefix",
            suffix="suffix",
            output_dimension=2048,
            output_dtype="int8",
        )
        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "voyage-3-large"
        assert embedder.truncate is False
        assert embedder.input_type == "document"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.output_dimension == 2048
        assert embedder.output_dtype == "int8"

    @pytest.mark.unit
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            VoyageTextEmbedder(model="voyage-3")

    @pytest.mark.unit
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        component = VoyageTextEmbedder(model="voyage-3")
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_text_embedder."
            "VoyageTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-3",
                "truncate": True,
                "input_type": None,
                "prefix": "",
                "suffix": "",
                "output_dimension": None,
                "output_dtype": "float",
            },
        }

    @pytest.mark.unit
    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_text_embedder."
            "VoyageTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-3",
                "truncate": True,
                "input_type": None,
                "prefix": "",
                "suffix": "",
                "output_dimension": None,
                "output_dtype": "float",
            },
        }

        embedder = VoyageTextEmbedder.from_dict(data)
        assert embedder.client.api_key == "fake-api-key"
        assert embedder.input_type is None
        assert embedder.model == "voyage-3"
        assert embedder.truncate is True
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.output_dimension is None
        assert embedder.output_dtype == "float"

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        component = VoyageTextEmbedder(
            model="voyage-3-large",
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            truncate=False,
            input_type="document",
            prefix="prefix",
            suffix="suffix",
            output_dimension=2048,
            output_dtype="int8",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_text_embedder."
            "VoyageTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "voyage-3-large",
                "truncate": False,
                "input_type": "document",
                "prefix": "prefix",
                "suffix": "suffix",
                "output_dimension": 2048,
                "output_dtype": "int8",
            },
        }

    @pytest.mark.unit
    def test_from_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_text_embedder."
            "VoyageTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "voyage-3-large",
                "truncate": False,
                "input_type": "document",
                "prefix": "prefix",
                "suffix": "suffix",
                "output_dimension": 2048,
                "output_dtype": "int8",
            },
        }

        embedder = VoyageTextEmbedder.from_dict(data)
        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "voyage-3-large"
        assert embedder.truncate is False
        assert embedder.input_type == "document"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.output_dimension == 2048
        assert embedder.output_dtype == "int8"

    @pytest.mark.unit
    def test_run_with_mocked_api(self):
        embedder = VoyageTextEmbedder(
            model="voyage-3",
            prefix="prefix ",
            suffix=" suffix",
            api_key=Secret.from_token("fake-api-key"),
        )

        # Mock the client.embed method
        mock_response = Mock()
        mock_response.embeddings = [[0.1] * 1024]  # 1024 dimensions
        mock_response.total_tokens = 6

        with patch.object(embedder.client, "embed", return_value=mock_response):
            result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 1024
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"]["total_tokens"] == 6

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    @pytest.mark.flaky(reruns=3, reruns_delay=60)
    def test_run(self):
        model = "voyage-3-large"

        embedder = VoyageTextEmbedder(
            model=model,
            prefix="prefix ",
            suffix=" suffix",
            timeout=120,
            max_retries=10,
        )
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 1024
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"]["total_tokens"] == 6, "Total tokens does not match"

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    @pytest.mark.parametrize("model", ["voyage-4", "voyage-4-large", "voyage-4-lite"])
    def test_run_voyage_4(self, model):
        embedder = VoyageTextEmbedder(model=model, timeout=600, max_retries=1200)
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 1024
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"]["total_tokens"] > 0

        # Custom dimensions
        embedder_dim = VoyageTextEmbedder(model=model, output_dimension=512, timeout=600, max_retries=1200)
        result_dim = embedder_dim.run(text="The food was delicious")
        assert len(result_dim["embedding"]) == 512

        # Quantized output
        embedder_int8 = VoyageTextEmbedder(model=model, output_dtype="int8", timeout=600, max_retries=1200)
        result_int8 = embedder_int8.run(text="The food was delicious")
        assert len(result_int8["embedding"]) == 1024

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        embedder = VoyageTextEmbedder(model="voyage-3", api_key=Secret.from_token("fake-api-key"))

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="VoyageTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)
