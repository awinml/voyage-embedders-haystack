import io
import os
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace
from tqdm import tqdm
from voyageai import Client

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from voyageai.video_utils import Video

    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    Video = None  # type: ignore[misc, assignment]


logger = logging.getLogger(__name__)


# Type alias for multimodal content items
# Each input can contain text strings, PIL Images, Videos, or ByteStreams
MultimodalContent = Union[str, "Image.Image", "Video", ByteStream]


@component
class VoyageMultimodalEmbedder:
    """
    A component for computing embeddings using VoyageAI's multimodal embedding models.

    The multimodal embedder can process inputs containing text, images, and videos,
    creating embeddings in a shared vector space that enables cross-modal similarity search.

    Each input is a list of content items (text strings, PIL Images, Videos, or ByteStreams),
    and the component returns one embedding per input.

    Usage example:
    ```python
    from haystack.dataclasses import ByteStream
    from haystack_integrations.components.embedders.voyage_embedders import VoyageMultimodalEmbedder

    # Text-only embedding
    embedder = VoyageMultimodalEmbedder(model="voyage-multimodal-3.5")
    result = embedder.run(inputs=[["What is in this image?"]])
    print(result["embeddings"][0][:5])  # First 5 dimensions

    # Mixed text and image embedding
    image_bytes = ByteStream.from_file_path("image.jpg")
    result = embedder.run(inputs=[["Describe this image:", image_bytes]])
    print(result["embeddings"][0][:5])

    # Video embedding (requires voyageai >= 0.3.6)
    from voyageai.video_utils import Video
    video = Video.from_path("video.mp4", model="voyage-multimodal-3.5")
    result = embedder.run(inputs=[["Describe this video:", video]])
    print(result["embeddings"][0][:5])
    ```
    """

    def __init__(
        self,
        model: str = "voyage-multimodal-3.5",
        api_key: Secret = Secret.from_env_var("VOYAGE_API_KEY"),
        input_type: str | None = None,
        truncate: bool = True,
        output_dimension: int | None = None,
        output_dtype: str | None = None,
        batch_size: int = 8,
        progress_bar: bool = True,
        timeout: int | None = None,
        max_retries: int | None = None,
    ):
        """
        Create a VoyageMultimodalEmbedder component.

        :param model:
            The name of the VoyageAI multimodal model to use. Defaults to "voyage-multimodal-3.5".
            For more details, see [Voyage Multimodal Embeddings](https://docs.voyageai.com/docs/multimodal-embeddings).
        :param api_key:
            The VoyageAI API key. It can be explicitly provided or automatically read from the
            environment variable VOYAGE_API_KEY (recommended).
        :param input_type:
            Type of the input. For retrieval/search purposes, it is recommended to specify
            whether your inputs are intended as queries or documents:
            - None: No special prompt is added (default)
            - "query": Optimizes embeddings for query-side retrieval
            - "document": Optimizes embeddings for document-side retrieval
        :param truncate:
            Whether to truncate inputs to fit within the model's context length. Defaults to True.
            If False, an error will be raised if any input exceeds the context length.
        :param output_dimension:
            The dimension of the output embedding. Defaults to None (1024 for voyage-multimodal-3.5).
            voyage-multimodal-3.5 supports: 256, 512, 1024 (default), and 2048.
        :param output_dtype:
            The data type for the embeddings. Defaults to None (float).
            Options: "float", "int8", "uint8", "binary", "ubinary".
        :param batch_size:
            Number of inputs to embed at once. Defaults to 8.
            Note: Multimodal inputs can be large, so smaller batch sizes may be needed.
        :param progress_bar:
            Whether to show a progress bar during embedding. Defaults to True.
        :param timeout:
            Timeout for VoyageAI Client calls in seconds. If not set, uses the VOYAGE_TIMEOUT
            environment variable or defaults to 30.
        :param max_retries:
            Maximum retries for failed API calls. If not set, uses the VOYAGE_MAX_RETRIES
            environment variable or defaults to 5.
        """
        if not PIL_AVAILABLE:
            msg = "The 'pillow' package is required for multimodal embeddings. Install it with: pip install pillow"
            raise ImportError(msg)

        self.api_key = api_key
        self.model = model
        self.input_type = input_type
        self.truncate = truncate
        self.output_dimension = output_dimension
        self.output_dtype = output_dtype
        self.batch_size = batch_size
        self.progress_bar = progress_bar

        if timeout is None:
            timeout = int(os.environ.get("VOYAGE_TIMEOUT", "30"))
        if max_retries is None:
            max_retries = int(os.environ.get("VOYAGE_MAX_RETRIES", "5"))

        self._timeout = timeout
        self._max_retries = max_retries
        self.client = Client(api_key=api_key.resolve_value(), max_retries=max_retries, timeout=timeout)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model,
            input_type=self.input_type,
            truncate=self.truncate,
            output_dimension=self.output_dimension,
            output_dtype=self.output_dtype,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoyageMultimodalEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _convert_content_item(self, item: MultimodalContent) -> Union[str, "Image.Image", "Video"]:
        """
        Convert a content item to a format accepted by the VoyageAI SDK.

        :param item:
            A content item: string, PIL Image, Video, or ByteStream.
        :returns:
            A string, PIL Image, or Video suitable for the VoyageAI API.
        """
        if isinstance(item, str):
            return item
        elif isinstance(item, ByteStream):
            # Convert ByteStream to PIL Image
            return Image.open(io.BytesIO(item.data))
        elif PIL_AVAILABLE and isinstance(item, Image.Image):
            return item
        elif VIDEO_AVAILABLE and Video is not None and isinstance(item, Video):
            # Pass Video objects through directly
            return item
        else:
            msg = f"Unsupported content type: {type(item)}. Expected str, PIL.Image.Image, Video, or ByteStream."
            raise TypeError(msg)

    def _prepare_inputs(self, inputs: list[list[MultimodalContent]]) -> list[list[Union[str, "Image.Image", "Video"]]]:
        """
        Prepare inputs for the VoyageAI multimodal API.

        :param inputs:
            List of inputs, where each input is a list of content items.
        :returns:
            List of prepared inputs suitable for the VoyageAI API.
        """
        prepared = []
        for input_items in inputs:
            prepared_items = [self._convert_content_item(item) for item in input_items]
            prepared.append(prepared_items)
        return prepared

    def _embed_batch(
        self, inputs: list[list[Union[str, "Image.Image", "Video"]]], batch_size: int
    ) -> tuple[list[list[float]], dict[str, Any]]:
        """
        Embed inputs in batches.

        :param inputs:
            List of prepared inputs.
        :param batch_size:
            Number of inputs per batch.
        :returns:
            Tuple of (embeddings, metadata).
        """
        all_embeddings: list[list[float]] = []
        meta: dict[str, Any] = {
            "text_tokens": 0,
            "image_pixels": 0,
            "video_pixels": 0,
            "total_tokens": 0,
        }

        for i in tqdm(
            range(0, len(inputs), batch_size),
            disable=not self.progress_bar,
            desc="Calculating multimodal embeddings",
        ):
            batch = inputs[i : i + batch_size]

            # Build API call parameters
            api_params: dict[str, Any] = {
                "inputs": batch,
                "model": self.model,
                "truncation": self.truncate,
            }

            if self.input_type is not None:
                api_params["input_type"] = self.input_type
            if self.output_dimension is not None:
                api_params["output_dimension"] = self.output_dimension
            if self.output_dtype is not None:
                api_params["output_dtype"] = self.output_dtype

            response = self.client.multimodal_embed(**api_params)

            all_embeddings.extend(response.embeddings)
            meta["text_tokens"] += response.text_tokens
            meta["image_pixels"] += response.image_pixels
            meta["video_pixels"] += response.video_pixels
            meta["total_tokens"] += response.total_tokens

        return all_embeddings, meta

    @component.output_types(embeddings=list[list[float]], meta=dict[str, Any])
    def run(
        self,
        inputs: list[list[MultimodalContent]],
    ):
        """
        Embed multimodal inputs.

        Each input is a list of content items (text strings, PIL Images, Videos, or ByteStreams).
        The component returns one embedding per input.

        :param inputs:
            List of inputs to embed. Each input is a list of content items that can include:
            - Text strings
            - PIL Image objects
            - Video objects (from voyageai.video_utils.Video)
            - ByteStream objects (will be converted to PIL Images)

        :returns:
            A dictionary with the following keys:
            - `embeddings`: List of embeddings, one per input
            - `meta`: Metadata including token/pixel usage:
              - `text_tokens`: Number of text tokens processed
              - `image_pixels`: Number of image pixels processed
              - `video_pixels`: Number of video pixels processed
              - `total_tokens`: Total tokens (text + image + video equivalent)
        """
        if not isinstance(inputs, list):
            msg = "VoyageMultimodalEmbedder expects a list of inputs."
            raise TypeError(msg)

        if not inputs:
            return {
                "embeddings": [],
                "meta": {
                    "text_tokens": 0,
                    "image_pixels": 0,
                    "video_pixels": 0,
                    "total_tokens": 0,
                },
            }

        # Validate that each input is a list
        for idx, inp in enumerate(inputs):
            if not isinstance(inp, list):
                msg = f"Each input must be a list of content items. Input at index {idx} is {type(inp).__name__}."
                raise TypeError(msg)

        # Prepare inputs for the API
        prepared_inputs = self._prepare_inputs(inputs)

        # Embed in batches
        embeddings, meta = self._embed_batch(prepared_inputs, self.batch_size)

        return {"embeddings": embeddings, "meta": meta}
