import os
from typing import Any, Dict, List, Optional, Tuple

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from tqdm import tqdm
from voyageai import Client


@component
class VoyageDocumentEmbedder:
    """
    A component for computing Document embeddings using Voyage Embedding models.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.voyage_embedders import VoyageDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = VoyageDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("VOYAGE_API_KEY"),
        model: str = "voyage-3",
        input_type: Optional[str] = None,
        truncate: bool = True,
        prefix: str = "",
        suffix: str = "",
        output_dimension: Optional[int] = None,
        output_dtype: str = "float",
        batch_size: int = 32,
        metadata_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        progress_bar: bool = True,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Create a VoyageDocumentEmbedder component.

        :param api_key:
            The VoyageAI API key. It can be explicitly provided or automatically read from the environment variable
            VOYAGE_API_KEY (recommended).
        :param model:
            The name of the model to use. Defaults to "voyage-3".
            For more details on the available models,
            see [Voyage Embeddings documentation](https://docs.voyageai.com/embeddings/).
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
            - `voyage-3-large` and `voyage-code-3` support the following `output_dimension` values: 2048,
            1024 (default), 512, and 256.
        :param output_dtype: The data type for the embeddings to be returned. Defaults to `"float"`.
            Options: "float", "int8", "uint8", "binary", "ubinary". "float" is supported for all models.
            "int8", "uint8", "binary", and "ubinary" are supported by voyage-3-large and voyage-code-3.
            Please see the [FAQ](https://docs.voyageai.com/docs/faq#what-is-quantization-and-output-data-types) for
            more details about output data types.
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param batch_size:
            Number of Documents to encode at once.
        :param metadata_fields_to_embed:
            List of meta fields that should be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        :param progress_bar:
            Whether to show a progress bar or not. Can be helpful to disable in production deployments to keep the logs
            clean.
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
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.metadata_fields_to_embed = metadata_fields_to_embed or []
        self.embedding_separator = embedding_separator

        if timeout is None:
            timeout = int(os.environ.get("VOYAGE_TIMEOUT", "30"))
        if max_retries is None:
            max_retries = int(os.environ.get("VOYAGE_MAX_RETRIES", "5"))

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
            input_type=self.input_type,
            truncate=self.truncate,
            prefix=self.prefix,
            suffix=self.suffix,
            output_dimension=self.output_dimension,
            output_dtype=self.output_dtype,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            metadata_fields_to_embed=self.metadata_fields_to_embed,
            embedding_separator=self.embedding_separator,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoyageDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key])
                for key in self.metadata_fields_to_embed
                if key in doc.meta and doc.meta[key] is not None
            ]

            text_to_embed = (
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix
            )

            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def _embed_batch(self, texts_to_embed: List[str], batch_size: int) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """

        all_embeddings = []
        meta: Dict[str, Any] = {}
        meta["total_tokens"] = 0
        for i in tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]
            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type=self.input_type,
                truncation=self.truncate,
                output_dtype=self.output_dtype,
                output_dimension=self.output_dimension,
            )
            all_embeddings.extend(response.embeddings)
            meta["total_tokens"] += response.total_tokens

        return all_embeddings, meta

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings
            - `meta`: Information about the usage of the model.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "VoyageDocumentEmbedder expects a list of Documents as input."
                " In case you want to embed a string, please use the VoyageTextEmbedder."
            )
            raise TypeError(msg)

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings, meta = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": meta}
