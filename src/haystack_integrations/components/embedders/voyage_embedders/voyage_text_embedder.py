import os
from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from voyageai import Client


@component
class VoyageTextEmbedder:
    """
    A component for embedding strings using Voyage models.

    Usage example:
    ```python
    from haystack_integrations.components.embedders.voyage_embedders import VoyageTextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = VoyageTextEmbedder(model="voyage-3")

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    ```
    """

    def __init__(
        self,
        model: str,
        api_key: Secret = Secret.from_env_var("VOYAGE_API_KEY"),
        input_type: str | None = None,
        truncate: bool = True,
        prefix: str = "",
        suffix: str = "",
        output_dimension: int | None = None,
        output_dtype: str = "float",
        timeout: int | None = None,
        max_retries: int | None = None,
    ):
        """
        Create an VoyageTextEmbedder component.

        :param model:
            The name of the Voyage model to use.
            For more details on the available models,
            see [Voyage Embeddings documentation](https://docs.voyageai.com/embeddings/).
        :param api_key:
            The VoyageAI API key. It can be explicitly provided or automatically read from the environment variable
            VOYAGE_API_KEY (recommended).
        :param input_type:
            Type of the input text. This is used to prepend different prompts to the text. For retrieval/search
            purposes, where a "query" is used to search for relevant information among a collection of data, referred
            to as "documents", it is recommended to specify whether your inputs (texts) are intended as queries or
            documents by setting `input_type` to `"query"` or `"document"` , respectively.
            - Defaults to `None`. This means the embedding model directly converts the inputs (texts) into numerical
                vectors. No prompt is added.
            - Can be set to `"query"`. This will prepend the text with, "Represent the query for retrieving
                supporting documents: ".
            - Can be set to `"document"`. For document, the prompt is "Represent the document for retrieval: ".
        :param truncate:
            Whether to truncate the input texts to fit within the context length. Defaults to `True`.
            - If `True`, over-length input texts will be truncated to fit within the context length, before vectorized
                by the embedding model.
            - If `False`, an error will be raised if any given text exceeds the context length.
        :param output_dimension:
            The dimension of the output embedding. Defaults to `None`.
            - Most models only support a single default dimension, used when `output_dimension` is set to `None` (see
            [model embedding dimensions](https://docs.voyageai.com/docs/embeddings#model-choices) for more details).
            - `voyage-3-large`, `voyage-code-3`, `voyage-4`, `voyage-4-large`, and `voyage-4-lite` support the
            following `output_dimension` values: 2048, 1024 (default), 512, and 256.
        :param output_dtype: The data type for the embeddings to be returned. Defaults to `"float"`.
            Options: "float", "int8", "uint8", "binary", "ubinary". "float" is supported for all models.
            "int8", "uint8", "binary", and "ubinary" are supported by voyage-3-large, voyage-code-3, voyage-4,
            voyage-4-large, and voyage-4-lite.
            Please see the [FAQ](https://docs.voyageai.com/docs/faq#what-is-quantization-and-output-data-types) for
            more details about output data types.
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param timeout:
            Timeout for VoyageAI Client calls, if not set it is inferred from the `VOYAGE_TIMEOUT` environment variable
            or set to 30.
        :param max_retries:
            Maximum retries to establish contact with VoyageAI if it returns an internal error, if not set it is
            inferred from the `VOYAGE_MAX_RETRIES` environment variable or set to 5.
        """
        self.api_key = api_key
        self.model = model
        self.input_type = input_type
        self.truncate = truncate
        self.prefix = prefix
        self.suffix = suffix
        self.output_dimension = output_dimension
        self.output_dtype = output_dtype

        if timeout is None:
            timeout = int(os.environ.get("VOYAGE_TIMEOUT", "30"))
        if max_retries is None:
            max_retries = int(os.environ.get("VOYAGE_MAX_RETRIES", "5"))

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
            prefix=self.prefix,
            suffix=self.suffix,
            output_dimension=self.output_dimension,
            output_dtype=self.output_dtype,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoyageTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    def run(self, text: str) -> dict[str, Any]:
        """
        Embed a single string.

        :param text:
            Text to embed.

        :returns:
            A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
            - `meta`: Information about the usage of the model.
        """
        if not isinstance(text, str):
            msg = (
                "VoyageTextEmbedder expects a string as an input. "
                "In case you want to embed a list of Documents, please use the VoyageDocumentEmbedder."
            )
            raise TypeError(msg)

        text_to_embed = self.prefix + text + self.suffix

        response = self.client.embed(
            texts=[text_to_embed],
            model=self.model,
            input_type=self.input_type,
            truncation=self.truncate,
            output_dtype=self.output_dtype,
            output_dimension=self.output_dimension,
        )
        embedding = response.embeddings[0]
        meta = {"total_tokens": response.total_tokens}

        # Note: output_dtype can produce list[int] for quantized types (int8, uint8, binary, ubinary),
        # but we declare the output type as list[float] for Haystack pipeline compatibility.
        # The component respects the output_dtype parameter for API optimization, but the type contract
        # is always list[float] to ensure compatibility with downstream components like Retriever.
        return {"embedding": embedding, "meta": meta}
