import os

import pytest

from voyage_embedders.voyage_text_embedder import VoyageTextEmbedder


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
            api_key="fake-api-key",
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
        with pytest.raises(ValueError, match="VoyageTextEmbedder expects an VoyageAI API key"):
            VoyageTextEmbedder()

    @pytest.mark.unit
    def test_to_dict(self):
        component = VoyageTextEmbedder(api_key="fake-api-key")
        data = component.to_dict()
        assert data == {
            "type": "voyage_embedders.voyage_text_embedder.VoyageTextEmbedder",
            "init_parameters": {
                "model": "voyage-2",
                "truncate": None,
                "input_type": "query",
                "prefix": "",
                "suffix": "",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = VoyageTextEmbedder(
            api_key="fake-api-key",
            model="model",
            truncate=True,
            input_type="document",
            prefix="prefix",
            suffix="suffix",
        )
        data = component.to_dict()
        assert data == {
            "type": "voyage_embedders.voyage_text_embedder.VoyageTextEmbedder",
            "init_parameters": {
                "model": "model",
                "truncate": True,
                "input_type": "document",
                "prefix": "prefix",
                "suffix": "suffix",
            },
        }

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    def test_run(self):
        model = "voyage-lite-02-instruct"

        embedder = VoyageTextEmbedder(model=model, prefix="prefix ", suffix=" suffix")
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 1024
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"]["total_tokens"] == 8, "Total tokens does not match"

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        embedder = VoyageTextEmbedder(api_key="fake-api-key")

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="VoyageTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)
