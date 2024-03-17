from typing import Any, Dict, List, Optional

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

    text_embedder = VoyageTextEmbedder()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("VOYAGE_API_KEY"),
        model: str = "voyage-2",
        input_type: str = "query",
        truncate: Optional[bool] = None,
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Create an VoyageTextEmbedder component.

        :param api_key:
            The VoyageAI API key. It can be explicitly provided or automatically read from the environment variable
            VOYAGE_API_KEY (recommended).
        :param model:
        The name of the Voyage model to use. Defaults to "voyage-2".
        For more details on the available models,
        see [Voyage Embeddings documentation](https://docs.voyageai.com/embeddings/).
        :param input_type:
            Type of the input text. This is used to prepend different prompts to the text.
            - Defaults to `"query"`. This will prepend the text with, "Represent the query for retrieving
              supporting documents: ".
            - Can be set to `"document"`. For document, the prompt is "Represent the document for retrieval: ".
            - Can be set to `None` for no prompt.
        for the document prompt.
        :param truncate:
            Whether to truncate the input texts to fit within the context length.
            - If `True`, over-length input texts will be truncated to fit within the context length, before vectorized
              by the embedding model.
            - If False, an error will be raised if any given text exceeds the context length.
            - Defaults to `None`, which will truncate the input text before sending it to the embedding model if it
              slightly exceeds the context window length. If it significantly exceeds the context window length, an
              error will be raised.
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        """
        self.api_key = api_key
        self.model = model
        self.input_type = input_type
        self.truncate = truncate
        self.prefix = prefix
        self.suffix = suffix

        self.client = Client(api_key=api_key.resolve_value())

    def to_dict(self) -> Dict[str, Any]:
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
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoyageTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str):
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
            texts=[text_to_embed], model=self.model, input_type=self.input_type, truncation=self.truncate
        )
        embedding = response.embeddings[0]
        meta = {"total_tokens": response.total_tokens}

        return {"embedding": embedding, "meta": meta}
