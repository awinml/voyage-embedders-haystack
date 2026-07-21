from dataclasses import replace as dataclass_replace
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components._voyage_client_mixin import VoyageClientMixin

logger = logging.getLogger(__name__)

MAX_NUM_DOCS = 1000


@component
class VoyageRanker(VoyageClientMixin):
    """
    A component for reranking using Voyage models.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.rankers.voyage.ranker import VoyageRanker

    ranker = VoyageRanker(model="rerank-2.5", top_k=2)

    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "What is the capital of germany?"
    output = ranker.run(query=query, documents=docs)
    docs = output["documents"]
    ```
    """

    def __init__(
        self,
        model: str,
        api_key: Secret = Secret.from_env_var("VOYAGE_API_KEY"),
        truncate: bool | None = None,
        top_k: int | None = None,
        prefix: str = "",
        suffix: str = "",
        timeout: int | None = None,
        max_retries: int | None = None,
        meta_fields_to_embed: list[str] | None = None,
        meta_data_separator: str = "\n",
    ):
        """
        Create an VoyageRanker component.

        :param model:
            The name of the Voyage model to use.
            For more details on the available models,
            see [Voyage Rerankers documentation](https://docs.voyageai.com/docs/reranker).
        :param api_key:
            The VoyageAI API key. It can be explicitly provided or automatically read from the environment variable
            VOYAGE_API_KEY (recommended).
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
        self.model = model
        self.top_k = top_k
        self.truncate = truncate
        self.prefix = prefix
        self.suffix = suffix
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.meta_data_separator = meta_data_separator

        self._init_client_lifecycle(api_key, timeout, max_retries)

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "VoyageRanker":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_input_docs(self, documents: list[Document]) -> list[str]:
        """
        Prepare the input by concatenating the document text with the metadata fields specified.
        :param documents:
            The list of Document objects.

        :return:
            A list of strings to be given as input to Voyage AI model.
        """
        concatenated_input_list = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta.get(key)
            ]
            concatenated_input = self.meta_data_separator.join([*meta_values_to_embed, doc.content or ""])
            concatenated_input_list.append(concatenated_input)

        return concatenated_input_list

    def _validate_top_k(self, top_k: int | None) -> int | None:
        """Resolve top_k from the argument or instance default, validating it's positive."""
        resolved = top_k if top_k is not None else self.top_k
        if resolved is not None and resolved <= 0:
            msg = f"top_k must be > 0, but got {resolved}"
            raise ValueError(msg)
        return resolved

    def _truncate_documents(self, input_docs: list[str]) -> list[str]:
        """Truncate documents to MAX_NUM_DOCS if needed."""
        if len(input_docs) > MAX_NUM_DOCS:
            logger.warning(
                f"The Voyage AI reranking endpoint only supports {MAX_NUM_DOCS} documents."
                f" The number of documents has been truncated to {MAX_NUM_DOCS}"
                f" from {len(input_docs)}."
            )
            input_docs = input_docs[:MAX_NUM_DOCS]
        return input_docs

    def _build_rerank_response(
        self,
        documents: list[Document],
        response_results: list[Any],
    ) -> list[Document]:
        """Map rerank API response back to Document objects with scores."""
        sorted_docs = []
        for output in response_results:
            sorted_docs.append(dataclass_replace(documents[output.index], score=output.relevance_score))
        return sorted_docs

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document], top_k: int | None = None) -> dict[str, list[Document]]:
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
        top_k = self._validate_top_k(top_k)
        input_docs = self._prepare_input_docs(documents)
        input_docs = self._truncate_documents(input_docs)

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=input_docs,
            top_k=top_k,
        )
        sorted_docs = self._build_rerank_response(documents, response.results)
        return {"documents": sorted_docs}

    @component.output_types(documents=list[Document])
    async def run_async(
        self, query: str, documents: list[Document], top_k: int | None = None
    ) -> dict[str, list[Document]]:
        """
        Use the Voyage AI Reranker to re-rank the list of documents based on the query (async).

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
        top_k = self._validate_top_k(top_k)
        input_docs = self._prepare_input_docs(documents)
        input_docs = self._truncate_documents(input_docs)

        response = await self.async_client.rerank(
            model=self.model,
            query=query,
            documents=input_docs,
            top_k=top_k,
        )
        sorted_docs = self._build_rerank_response(documents, response.results)
        return {"documents": sorted_docs}
