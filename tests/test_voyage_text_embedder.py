from typing import List
from unittest.mock import patch

import numpy as np
import pytest
import voyageai

from voyage_embedders.voyage_text_embedder import VoyageTextEmbedder


def mock_voyageai_response(text: str, model: str = "voyage-01", **kwargs) -> List[float]:  # noqa
    response = np.random.rand(1024).tolist()
    return response


class TestVoyageTextEmbedder:
    @pytest.mark.unit
    def test_init_default(self, monkeypatch):
        voyageai.api_key = None
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageTextEmbedder()

        assert voyageai.api_key == "fake-api-key"
        assert embedder.input_type == "query"
        assert embedder.model_name == "voyage-01"
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    @pytest.mark.unit
    def test_init_with_parameters(self):
        embedder = VoyageTextEmbedder(
            api_key="fake-api-key",
            model_name="model",
            input_type="document",
            prefix="prefix",
            suffix="suffix",
        )
        assert voyageai.api_key == "fake-api-key"
        assert embedder.model_name == "model"
        assert embedder.input_type == "document"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"

    @pytest.mark.unit
    def test_init_fail_wo_api_key(self, monkeypatch):
        voyageai.api_key = None
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
                "model_name": "voyage-01",
                "input_type": "query",
                "prefix": "",
                "suffix": "",
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        component = VoyageTextEmbedder(
            api_key="fake-api-key",
            model_name="model",
            input_type="document",
            prefix="prefix",
            suffix="suffix",
        )
        data = component.to_dict()
        assert data == {
            "type": "voyage_embedders.voyage_text_embedder.VoyageTextEmbedder",
            "init_parameters": {
                "model_name": "model",
                "input_type": "document",
                "prefix": "prefix",
                "suffix": "suffix",
            },
        }

    @pytest.mark.unit
    def test_run(self):
        model = "voyage-01-lite"

        with patch("voyage_embedders.voyage_text_embedder.get_embedding") as voyageai_embedding_patch:
            voyageai_embedding_patch.side_effect = mock_voyageai_response

            embedder = VoyageTextEmbedder(api_key="fake-api-key", model_name=model, prefix="prefix ", suffix=" suffix")
            result = embedder.run(text="The food was delicious")

            voyageai_embedding_patch.assert_called_once_with(
                model=model, text="prefix The food was delicious suffix", input_type="query"
            )

        assert len(result["embedding"]) == 1024
        assert all(isinstance(x, float) for x in result["embedding"])

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        embedder = VoyageTextEmbedder(api_key="fake-api-key")

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="VoyageTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)
