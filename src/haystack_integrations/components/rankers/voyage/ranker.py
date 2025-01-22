import os
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict, Document, logging
from haystack.utils import Secret, deserialize_secrets_inplace
from voyageai import Client

logger = logging.getLogger(__name__)

MAX_NUM_DOCS = 1000


@component
class VoyageRanker:
    """
    A component for reranking using Voyage models.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.rerankers.voyage_rerankers.voyage_text_reranker import VoyageRanker

    ranker = VoyageRanker(model="rerank-2", top_k=2)

    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "What is the capital of germany?"
    output = ranker.run(query=query, documents=docs)
    docs = output["documents"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("VOYAGE_API_KEY"),
        model: str = "rerank-2",
        truncate: Optional[bool] = None,
        top_k: Optional[int] = None,
        prefix: str = "",
        suffix: str = "",
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        meta_fields_to_embed: Optional[List[str]] = None,
        meta_data_separator: str = "\n",
    ):
        """
        Create an VoyageRanker component.

        :param api_key:
            The VoyageAI API key. It can be explicitly provided or automatically read from the environment variable
            VOYAGE_API_KEY (recommended).
        :param model:
        The name of the Voyage model to use. Defaults to "voyage-2".
        For more details on the available models,
        see [Voyage Rerankers documentation](https://docs.voyageai.com/docs/reranker).
        :param truncate:
            Whether to truncate the input texts to fit within the context length.
            - If `True`, over-length input texts will be truncated to fit within the context length, before vectorized
              by the reranker model.
            - If False, an error will be raised if any given text exceeds the context length.
            - Defaults to `None`, which will truncate the input text before sending it to the reranker model if it
              slightly exceeds the context window length. If it significantly exceeds the context window length, an
              error will be raised.
        :param top_k:
            The number of most relevant documents to return.
            If not specified, the reranking results of all documents will be returned.
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
        self.top_k = top_k
        self.truncate = truncate
        self.prefix = prefix
        self.suffix = suffix
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.meta_data_separator = meta_data_separator

        if timeout is None:
            timeout = int(os.environ.get("VOYAGE_TIMEOUT", 30))
        if max_retries is None:
            max_retries = int(os.environ.get("VOYAGE_MAX_RETRIES", 5))

        self.client = Client(api_key=api_key.resolve_value(), max_retries=max_retries, timeout=timeout)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model,
            top_k=self.top_k,
            truncate=self.truncate,
            prefix=self.prefix,
            suffix=self.suffix,
            api_key=self.api_key.to_dict(),
            meta_fields_to_embed=self.meta_fields_to_embed,
            meta_data_separator=self.meta_data_separator,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoyageRanker":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_input_docs(self, documents: List[Document]) -> List[str]:
        """
        Prepare the input by concatenating the document text with the metadata fields specified.
        :param documents: The list of Document objects.

        :return: A list of strings to be given as input to Voyage AI model.
        """
        concatenated_input_list = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta.get(key)
            ]
            concatenated_input = self.meta_data_separator.join([*meta_values_to_embed, doc.content or ""])
            concatenated_input_list.append(concatenated_input)

        return concatenated_input_list

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Use the Voyage AI Reranker to re-rank the list of documents based on the query.

        :param query:
            Query string.
        :param documents:
            List of Documents.
        :param top_k:
            The maximum number of Documents you want the Ranker to return.
        :returns:
            A dictionary with the following keys:
            - `documents`: List of Documents most similar to the given query in descending order of similarity.

        :raises ValueError: If `top_k` is not > 0.
        """
        top_k = top_k or self.top_k
        if top_k is not None and top_k <= 0:
            msg = f"top_k must be > 0, but got {top_k}"
            raise ValueError(msg)

        input_docs = self._prepare_input_docs(documents)
        if len(input_docs) > MAX_NUM_DOCS:
            logger.warning(
                f"The Voyage AI reranking endpoint only supports {MAX_NUM_DOCS} documents.\
                The number of documents has been truncated to {MAX_NUM_DOCS} \
                from {len(input_docs)}."
            )
            input_docs = input_docs[:MAX_NUM_DOCS]

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=input_docs,
            top_k=top_k,
        )
        indices = [output.index for output in response.results]
        scores = [output.relevance_score for output in response.results]
        sorted_docs = []
        for idx, score in zip(indices, scores):
            doc = documents[idx]
            doc.score = score
            sorted_docs.append(documents[idx])
        return {"documents": sorted_docs}