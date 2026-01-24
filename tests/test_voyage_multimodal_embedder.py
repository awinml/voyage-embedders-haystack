import io
import os
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import ByteStream
from haystack.utils.auth import Secret
from PIL import Image
from voyageai.video_utils import Video

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

    @pytest.mark.unit
    def test_convert_content_item_string(self):
        """Test that string content passes through unchanged."""
        embedder = VoyageMultimodalEmbedder(api_key=Secret.from_token("fake-api-key"))

        result = embedder._convert_content_item("test text")

        assert result == "test text"

    @pytest.mark.unit
    def test_convert_content_item_bytestream(self):
        """Test that ByteStream is converted to PIL Image."""
        embedder = VoyageMultimodalEmbedder(api_key=Secret.from_token("fake-api-key"))

        # Create a small test image as ByteStream
        img = Image.new("RGB", (10, 10), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        bytestream = ByteStream(data=buffer.getvalue())

        result = embedder._convert_content_item(bytestream)

        assert isinstance(result, Image.Image)
        assert result.size == (10, 10)

    @pytest.mark.unit
    def test_convert_content_item_pil_image(self):
        """Test that PIL Image passes through unchanged."""
        embedder = VoyageMultimodalEmbedder(api_key=Secret.from_token("fake-api-key"))

        img = Image.new("RGB", (10, 10), color="blue")

        result = embedder._convert_content_item(img)

        assert result is img
        assert isinstance(result, Image.Image)

    @pytest.mark.unit
    def test_convert_content_item_unsupported_type(self):
        """Test that unsupported types raise TypeError."""
        embedder = VoyageMultimodalEmbedder(api_key=Secret.from_token("fake-api-key"))

        with pytest.raises(TypeError, match="Unsupported content type"):
            embedder._convert_content_item(12345)

        with pytest.raises(TypeError, match="Unsupported content type"):
            embedder._convert_content_item({"key": "value"})

    @pytest.mark.unit
    def test_prepare_inputs(self):
        """Test input preparation with mixed content types."""
        embedder = VoyageMultimodalEmbedder(api_key=Secret.from_token("fake-api-key"))

        # Create a test image as ByteStream
        img = Image.new("RGB", (10, 10), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        bytestream = ByteStream(data=buffer.getvalue())

        inputs = [
            ["text only"],
            ["text with image", bytestream],
        ]

        result = embedder._prepare_inputs(inputs)

        assert len(result) == 2
        assert result[0] == ["text only"]
        assert result[1][0] == "text with image"
        assert isinstance(result[1][1], Image.Image)

    @pytest.mark.unit
    def test_run_with_mocked_client(self, monkeypatch):
        """Test run method with mocked API client."""
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")

        embedder = VoyageMultimodalEmbedder()

        # Mock the client's multimodal_embed method
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_response.text_tokens = 10
        mock_response.image_pixels = 0
        mock_response.video_pixels = 0
        mock_response.total_tokens = 10

        embedder.client.multimodal_embed = MagicMock(return_value=mock_response)

        result = embedder.run(inputs=[["Hello world"], ["Test input"]])

        assert len(result["embeddings"]) == 2
        assert result["embeddings"][0] == [0.1, 0.2, 0.3]
        assert result["embeddings"][1] == [0.4, 0.5, 0.6]
        assert result["meta"]["text_tokens"] == 10
        assert result["meta"]["total_tokens"] == 10

        # Verify API was called with correct parameters
        embedder.client.multimodal_embed.assert_called_once()
        call_kwargs = embedder.client.multimodal_embed.call_args[1]
        assert call_kwargs["model"] == "voyage-multimodal-3.5"
        assert call_kwargs["truncation"] is True

    @pytest.mark.unit
    def test_run_with_all_parameters_mocked(self, monkeypatch):
        """Test run method with all optional parameters."""
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")

        embedder = VoyageMultimodalEmbedder(
            input_type="query",
            output_dimension=512,
            output_dtype="int8",
        )

        # Mock the client
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2]]
        mock_response.text_tokens = 5
        mock_response.image_pixels = 0
        mock_response.video_pixels = 0
        mock_response.total_tokens = 5

        embedder.client.multimodal_embed = MagicMock(return_value=mock_response)

        embedder.run(inputs=[["Query text"]])

        # Verify API was called with all parameters
        call_kwargs = embedder.client.multimodal_embed.call_args[1]
        assert call_kwargs["input_type"] == "query"
        assert call_kwargs["output_dimension"] == 512
        assert call_kwargs["output_dtype"] == "int8"

    @pytest.mark.unit
    def test_run_with_batching(self, monkeypatch):
        """Test that large inputs are processed in batches."""
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")

        embedder = VoyageMultimodalEmbedder(batch_size=2, progress_bar=False)

        # Create mock responses for each batch
        mock_response1 = MagicMock()
        mock_response1.embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_response1.text_tokens = 5
        mock_response1.image_pixels = 0
        mock_response1.video_pixels = 0
        mock_response1.total_tokens = 5

        mock_response2 = MagicMock()
        mock_response2.embeddings = [[0.5, 0.6]]
        mock_response2.text_tokens = 3
        mock_response2.image_pixels = 0
        mock_response2.video_pixels = 0
        mock_response2.total_tokens = 3

        embedder.client.multimodal_embed = MagicMock(side_effect=[mock_response1, mock_response2])

        # 3 inputs with batch_size=2 should result in 2 API calls
        result = embedder.run(inputs=[["text1"], ["text2"], ["text3"]])

        assert len(result["embeddings"]) == 3
        assert embedder.client.multimodal_embed.call_count == 2
        assert result["meta"]["text_tokens"] == 8
        assert result["meta"]["total_tokens"] == 8

    @pytest.mark.unit
    def test_run_with_image_content(self, monkeypatch):
        """Test run method with image content."""
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")

        embedder = VoyageMultimodalEmbedder(progress_bar=False)

        # Create a test image
        img = Image.new("RGB", (10, 10), color="red")

        # Mock the client
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_response.text_tokens = 2
        mock_response.image_pixels = 100
        mock_response.video_pixels = 0
        mock_response.total_tokens = 12

        embedder.client.multimodal_embed = MagicMock(return_value=mock_response)

        result = embedder.run(inputs=[["Describe this image:", img]])

        assert len(result["embeddings"]) == 1
        assert result["meta"]["image_pixels"] == 100
        assert result["meta"]["total_tokens"] == 12

        # Verify the image was passed correctly
        call_kwargs = embedder.client.multimodal_embed.call_args[1]
        inputs = call_kwargs["inputs"]
        assert len(inputs) == 1
        assert inputs[0][0] == "Describe this image:"
        assert isinstance(inputs[0][1], Image.Image)

    @pytest.mark.unit
    def test_init_with_env_vars(self, monkeypatch):
        """Test initialization with environment variables for timeout and max_retries."""
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        monkeypatch.setenv("VOYAGE_TIMEOUT", "60")
        monkeypatch.setenv("VOYAGE_MAX_RETRIES", "10")

        with patch(
            "haystack_integrations.components.embedders.voyage_embedders.voyage_multimodal_embedder.Client"
        ) as mock_client:
            VoyageMultimodalEmbedder()

            mock_client.assert_called_once_with(
                api_key="fake-api-key",
                max_retries=10,
                timeout=60,
            )

    @pytest.mark.unit
    def test_convert_content_item_video(self):
        """Test that Video objects pass through unchanged."""
        embedder = VoyageMultimodalEmbedder(api_key=Secret.from_token("fake-api-key"))

        # Create a mock Video object
        mock_video = MagicMock(spec=Video)

        result = embedder._convert_content_item(mock_video)

        assert result is mock_video

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
