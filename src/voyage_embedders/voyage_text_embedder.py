import os
from typing import Any, Dict, List, Optional

import voyageai
from haystack.preview import component, default_to_dict
from voyageai import get_embedding


@component
class VoyageTextEmbedder:
    """
    A component for embedding strings using Voyage models.

    Usage example:
    ```python
    from haystack.preview.components.embedders import VoyageTextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = VoyageTextEmbedder()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "voyage-01",
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Create an VoyageTextEmbedder component.

        :param api_key: The VoyageAI API key. It can be explicitly provided or automatically read from the
            environment variable VOYAGE_API_KEY (recommended).
        :param model_name: The name of the Voyage model to use. Defaults to "voyage-01".
        For more details on the available models,
            see [Voyage Embeddings documentation](https://docs.voyageai.com/embeddings/).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        """
        # if the user does not provide the API key, check if it is set in the module client
        api_key = api_key or voyageai.api_key
        if api_key is None:
            try:
                api_key = os.environ["VOYAGE_API_KEY"]
            except KeyError as e:
                msg = "VoyageTextEmbedder expects an VoyageAI API key. Set the VOYAGE_API_KEY environment variable (recommended) or pass it explicitly."  # noqa
                raise ValueError(msg) from e

        voyageai.api_key = api_key

        self.model_name = model_name
        self.prefix = prefix
        self.suffix = suffix

    def to_dict(self) -> Dict[str, Any]:
        """
        This method overrides the default serializer in order to avoid leaking the `api_key` value passed
        to the constructor.
        """

        return default_to_dict(self, model_name=self.model_name, prefix=self.prefix, suffix=self.suffix)

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """Embed a string."""
        if not isinstance(text, str):
            msg = "VoyageTextEmbedder expects a string as an input.In case you want to embed a list of Documents, please use the VoyageDocumentEmbedder."  # noqa
            raise TypeError(msg)

        text_to_embed = self.prefix + text + self.suffix

        embedding = get_embedding(text=text_to_embed, model=self.model_name)

        return {"embedding": embedding}
