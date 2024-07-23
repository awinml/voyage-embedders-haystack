import os

import pytest
from haystack.utils.auth import Secret
from haystack_integrations.components.embedders.voyage_embedders import VoyageTextEmbedder


class TestVoyageTextEmbedder:
    @pytest.mark.unit
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageTextEmbedder()

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.input_type == "query"
        assert embedder.model == "voyage-2"
        assert embedder.truncate is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    @pytest.mark.unit
    def test_init_with_parameters(self):
        embedder = VoyageTextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="model",
            input_type="document",
            truncate=True,
            prefix="prefix",
            suffix="suffix",
        )
        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "model"
        assert embedder.truncate is True
        assert embedder.input_type == "document"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"

    @pytest.mark.unit
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            VoyageTextEmbedder()

    @pytest.mark.unit
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        component = VoyageTextEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_text_embedder."
            "VoyageTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-2",
                "truncate": None,
                "input_type": "query",
                "prefix": "",
                "suffix": "",
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
                "model": "voyage-2",
                "truncate": None,
                "input_type": "query",
                "prefix": "",
                "suffix": "",
            },
        }

        embedder = VoyageTextEmbedder.from_dict(data)
        assert embedder.client.api_key == "fake-api-key"
        assert embedder.input_type == "query"
        assert embedder.model == "voyage-2"
        assert embedder.truncate is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        component = VoyageTextEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="model",
            truncate=True,
            input_type="document",
            prefix="prefix",
            suffix="suffix",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_text_embedder."
            "VoyageTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "model",
                "truncate": True,
                "input_type": "document",
                "prefix": "prefix",
                "suffix": "suffix",
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
                "model": "model",
                "truncate": True,
                "input_type": "document",
                "prefix": "prefix",
                "suffix": "suffix",
            },
        }

        embedder = VoyageTextEmbedder.from_dict(data)
        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "model"
        assert embedder.truncate is True
        assert embedder.input_type == "document"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    def test_run(self):
        model = "voyage-large-2-instruct"

        embedder = VoyageTextEmbedder(model=model, prefix="prefix ", suffix=" suffix")
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 1024
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"]["total_tokens"] == 8, "Total tokens does not match"

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        embedder = VoyageTextEmbedder(api_key=Secret.from_token("fake-api-key"))

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="VoyageTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)
