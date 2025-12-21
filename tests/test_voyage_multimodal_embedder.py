import os

import pytest
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.voyage_embedders import VoyageMultimodalEmbedder


class TestVoyageMultimodalEmbedder:
    @pytest.mark.unit
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageMultimodalEmbedder()

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "voyage-multimodal-3.5"
        assert embedder.input_type is None
        assert embedder.truncate is True
        assert embedder.output_dimension is None
        assert embedder.output_dtype is None
        assert embedder.batch_size == 8
        assert embedder.progress_bar is True

    @pytest.mark.unit
    def test_init_with_parameters(self):
        embedder = VoyageMultimodalEmbedder(
            model="voyage-multimodal-3.5",
            api_key=Secret.from_token("fake-api-key"),
            input_type="document",
            truncate=False,
            output_dimension=2048,
            output_dtype="float",
            batch_size=4,
            progress_bar=False,
        )

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "voyage-multimodal-3.5"
        assert embedder.input_type == "document"
        assert embedder.truncate is False
        assert embedder.output_dimension == 2048
        assert embedder.output_dtype == "float"
        assert embedder.batch_size == 4
        assert embedder.progress_bar is False

    @pytest.mark.unit
    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            VoyageMultimodalEmbedder()

    @pytest.mark.unit
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        component = VoyageMultimodalEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_multimodal_embedder."
            "VoyageMultimodalEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-multimodal-3.5",
                "input_type": None,
                "truncate": True,
                "output_dimension": None,
                "output_dtype": None,
                "batch_size": 8,
                "progress_bar": True,
            },
        }

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        component = VoyageMultimodalEmbedder(
            model="voyage-multimodal-3.5",
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            input_type="query",
            truncate=False,
            output_dimension=512,
            output_dtype="float",
            batch_size=4,
            progress_bar=False,
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_multimodal_embedder."
            "VoyageMultimodalEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "voyage-multimodal-3.5",
                "input_type": "query",
                "truncate": False,
                "output_dimension": 512,
                "output_dtype": "float",
                "batch_size": 4,
                "progress_bar": False,
            },
        }

    @pytest.mark.unit
    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_multimodal_embedder."
            "VoyageMultimodalEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["VOYAGE_API_KEY"], "strict": True, "type": "env_var"},
                "model": "voyage-multimodal-3.5",
                "input_type": None,
                "truncate": True,
                "output_dimension": None,
                "output_dtype": None,
                "batch_size": 8,
                "progress_bar": True,
            },
        }

        embedder = VoyageMultimodalEmbedder.from_dict(data)

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "voyage-multimodal-3.5"
        assert embedder.input_type is None
        assert embedder.truncate is True
        assert embedder.output_dimension is None
        assert embedder.output_dtype is None
        assert embedder.batch_size == 8
        assert embedder.progress_bar is True

    @pytest.mark.unit
    def test_from_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.embedders.voyage_embedders.voyage_multimodal_embedder."
            "VoyageMultimodalEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "voyage-multimodal-3.5",
                "input_type": "document",
                "truncate": False,
                "output_dimension": 2048,
                "output_dtype": "float",
                "batch_size": 4,
                "progress_bar": False,
            },
        }

        embedder = VoyageMultimodalEmbedder.from_dict(data)

        assert embedder.client.api_key == "fake-api-key"
        assert embedder.model == "voyage-multimodal-3.5"
        assert embedder.input_type == "document"
        assert embedder.truncate is False
        assert embedder.output_dimension == 2048
        assert embedder.output_dtype == "float"
        assert embedder.batch_size == 4
        assert embedder.progress_bar is False

    @pytest.mark.unit
    def test_run_wrong_input_format(self):
        embedder = VoyageMultimodalEmbedder(api_key=Secret.from_token("fake-api-key"))

        # Not a list
        with pytest.raises(TypeError, match="VoyageMultimodalEmbedder expects a list of inputs"):
            embedder.run(inputs="text")

        # List but items are not lists
        with pytest.raises(TypeError, match="Each input must be a list of content items"):
            embedder.run(inputs=["text1", "text2"])

    @pytest.mark.unit
    def test_run_on_empty_list(self):
        embedder = VoyageMultimodalEmbedder(api_key=Secret.from_token("fake-api-key"))

        result = embedder.run(inputs=[])

        assert result["embeddings"] == []
        assert result["meta"]["text_tokens"] == 0
        assert result["meta"]["image_pixels"] == 0
        assert result["meta"]["video_pixels"] == 0
        assert result["meta"]["total_tokens"] == 0

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    def test_run_text_only(self):
        """Test embedding text-only inputs."""
        embedder = VoyageMultimodalEmbedder(
            model="voyage-multimodal-3.5",
            timeout=120,
            max_retries=3,
        )

        result = embedder.run(
            inputs=[
                ["What is machine learning?"],
                ["How does natural language processing work?"],
            ]
        )

        assert len(result["embeddings"]) == 2
        assert len(result["embeddings"][0]) == 1024  # Default dimension
        assert all(isinstance(x, float) for x in result["embeddings"][0])
        assert result["meta"]["text_tokens"] > 0
        assert result["meta"]["total_tokens"] > 0

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    def test_run_with_output_dimension(self):
        """Test embedding with custom output dimension."""
        embedder = VoyageMultimodalEmbedder(
            model="voyage-multimodal-3.5",
            output_dimension=512,
            timeout=120,
            max_retries=3,
        )

        result = embedder.run(inputs=[["Test embedding with custom dimension"]])

        assert len(result["embeddings"]) == 1
        assert len(result["embeddings"][0]) == 512
        assert all(isinstance(x, float) for x in result["embeddings"][0])

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    def test_run_with_input_type(self):
        """Test embedding with input_type specified."""
        embedder = VoyageMultimodalEmbedder(
            model="voyage-multimodal-3.5",
            input_type="query",
            timeout=120,
            max_retries=3,
        )

        result = embedder.run(inputs=[["Search query for documents"]])

        assert len(result["embeddings"]) == 1
        assert len(result["embeddings"][0]) == 1024
        assert all(isinstance(x, float) for x in result["embeddings"][0])
