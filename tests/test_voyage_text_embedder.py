import os
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.voyage_embedders import VoyageTextEmbedder


class TestVoyageTextEmbedder:
    @pytest.mark.unit
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageTextEmbedder(model="voyage-3")

        # Client is not created at init - must be lazy
        assert embedder._client is None
        assert embedder._async_client is None
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
        assert embedder._client is None
        assert embedder._async_client is None
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
        embedder = VoyageTextEmbedder(model="voyage-3")
        # Init succeeds, but warm_up() should fail
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            embedder.warm_up()

    @pytest.mark.unit
    def test_warm_up(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageTextEmbedder(model="voyage-3")

        assert embedder._client is None
        embedder.warm_up()
        assert embedder._client is not None
        assert embedder._async_client is not None

        # Idempotent
        embedder.warm_up()
        assert embedder._client is not None

    @pytest.mark.unit
    def test_client_property(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageTextEmbedder(model="voyage-3")

        assert embedder._client is None
        client = embedder.client
        assert client is not None
        assert embedder._client is not None

    @pytest.mark.unit
    def test_async_client_property(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "fake-api-key")
        embedder = VoyageTextEmbedder(model="voyage-3")

        assert embedder._async_client is None
        async_client = embedder.async_client
        assert async_client is not None
        assert embedder._async_client is not None

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
        assert embedder._client is None
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
        assert embedder._client is None
        assert embedder.model == "voyage-3-large"
        assert embedder.truncate is False
        assert embedder.input_type == "document"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.output_dimension == 2048
        assert embedder.output_dtype == "int8"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_with_mocked_api(self):
        embedder = VoyageTextEmbedder(
            model="voyage-3",
            prefix="prefix ",
            suffix=" suffix",
            api_key=Secret.from_token("fake-api-key"),
        )

        mock_response = Mock()
        mock_response.embeddings = [[0.1] * 1024]  # 1024 dimensions
        mock_response.total_tokens = 6
        embedder._async_client = MagicMock()
        embedder._client = MagicMock()
        embedder._client.embed = MagicMock(return_value=mock_response)

        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 1024
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"]["total_tokens"] == 6

    @pytest.mark.skipif(os.environ.get("VOYAGE_API_KEY", "") == "", reason="VOYAGE_API_KEY is not set")
    @pytest.mark.integration
    @pytest.mark.flaky(reruns=3, reruns_delay=60)
    @pytest.mark.asyncio
    async def test_run(self):
        embedder = VoyageTextEmbedder(
            model="voyage-4",
            prefix="prefix ",
            suffix=" suffix",
            timeout=120,
            max_retries=10,
        )
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 1024
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"]["total_tokens"] > 0

        # Custom output dimension
        embedder_dim = VoyageTextEmbedder(model="voyage-4", output_dimension=512, timeout=120, max_retries=10)
        result_dim = embedder_dim.run(text="test")
        assert len(result_dim["embedding"]) == 512

        # Quantized output
        embedder_int8 = VoyageTextEmbedder(model="voyage-4", output_dtype="int8", timeout=120, max_retries=10)
        result_int8 = embedder_int8.run(text="test")
        assert len(result_int8["embedding"]) == 1024

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_async_with_mocked_api(self):
        """Test run_async method with mocked async API client."""
        embedder = VoyageTextEmbedder(
            model="voyage-3",
            prefix="prefix ",
            suffix=" suffix",
            api_key=Secret.from_token("fake-api-key"),
        )

        mock_response = Mock()
        mock_response.embeddings = [[0.1] * 1024]
        mock_response.total_tokens = 6
        embedder._async_client = MagicMock()
        embedder._async_client.embed = AsyncMock(return_value=mock_response)
        embedder._client = MagicMock()

        result = await embedder.run_async(text="The food was delicious")

        assert len(result["embedding"]) == 1024
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"]["total_tokens"] == 6

        # Verify the async client was called
        embedder._async_client.embed.assert_called_once()
        call_kwargs = embedder._async_client.embed.call_args[1]
        assert call_kwargs["texts"] == ["prefix The food was delicious suffix"]
        assert call_kwargs["model"] == "voyage-3"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_async_wrong_input_format(self):
        """Test run_async raises TypeError for non-string input."""
        embedder = VoyageTextEmbedder(model="voyage-3", api_key=Secret.from_token("fake-api-key"))

        with pytest.raises(TypeError, match="VoyageTextEmbedder expects a string as an input"):
            await embedder.run_async(text=[1, 2, 3])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_wrong_input_format(self):
        embedder = VoyageTextEmbedder(model="voyage-3", api_key=Secret.from_token("fake-api-key"))

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="VoyageTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)
