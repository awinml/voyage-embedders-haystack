import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from tqdm import tqdm
from voyageai import Client


@component
class VoyageContextualizedDocumentEmbedder:
    """
    A component for computing contextualized Document embeddings using Voyage's contextualized embedding models.

    Unlike standard embeddings, contextualized embeddings encode document chunks "in context" with other chunks
    from the same parent document, preserving semantic relationships and reducing context loss.

    Documents should have a metadata field (default: 'source_id') to indicate which chunks belong to the same
    parent document. Documents with the same source_id will be embedded together as a group.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.voyage_embedders import VoyageContextualizedDocumentEmbedder

    # Documents with same source_id will be embedded together
    docs = [
        Document(content="Introduction to quantum computing.", meta={"source_id": "doc1"}),
        Document(content="Quantum bits or qubits are the basic unit.", meta={"source_id": "doc1"}),
        Document(content="Classical computers use binary bits.", meta={"source_id": "doc2"}),
    ]

    embedder = VoyageContextualizedDocumentEmbedder(model="voyage-context-3")
    result = embedder.run(docs)
    print(result['documents'][0].embedding)
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("VOYAGE_API_KEY"),
        model: str = "voyage-context-3",
        input_type: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
        output_dimension: Optional[int] = None,
        output_dtype: str = "float",
        batch_size: int = 32,
        metadata_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        source_id_field: str = "source_id",
        chunk_fn: Optional[Callable] = None,
        progress_bar: bool = True,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Create a VoyageContextualizedDocumentEmbedder component.

        :param api_key:
            The VoyageAI API key. It can be explicitly provided or automatically read from the environment variable
            VOYAGE_API_KEY (recommended).
        :param model:
            The name of the model to use. Defaults to "voyage-context-3".
            For more details, see [Voyage Contextualized Embeddings](https://docs.voyageai.com/docs/contextualized-chunk-embeddings).
        :param input_type:
            Type of the input text. Can be "query", "document", or None.
            - None: No special prompt is added
            - "query": Prepends "Represent the query for retrieving supporting documents: "
            - "document": Prepends "Represent the document for retrieval: "
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param output_dimension:
            The dimension of the output embedding. Defaults to None (1024 for voyage-context-3).
            voyage-context-3 supports: 2048, 1024 (default), 512, and 256.
        :param output_dtype:
            The data type for the embeddings. Defaults to "float".
            Options: "float", "int8", "uint8", "binary", "ubinary".
        :param batch_size:
            Number of document groups to process at once. Each group contains multiple chunks from
            the same source document.
        :param metadata_fields_to_embed:
            List of meta fields that should be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        :param source_id_field:
            The metadata field name used to group documents. Documents with the same value in this field
            will be embedded together as contextualized chunks. Defaults to "source_id".
        :param chunk_fn:
            Optional custom chunking function to apply. If provided, it will be passed to the API.
        :param progress_bar:
            Whether to show a progress bar. Can be helpful to disable in production deployments.
        :param timeout:
            Timeout for VoyageAI Client calls, if not set it is inferred from the VOYAGE_TIMEOUT
            environment variable or set to 30.
        :param max_retries:
            Maximum retries to establish contact with VoyageAI if it returns an internal error.
            If not set, inferred from VOYAGE_MAX_RETRIES environment variable or set to 5.
        """
        self.api_key = api_key
        self.model = model
        self.input_type = input_type
        self.prefix = prefix
        self.suffix = suffix
        self.output_dimension = output_dimension
        self.output_dtype = output_dtype
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.metadata_fields_to_embed = metadata_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.source_id_field = source_id_field
        self.chunk_fn = chunk_fn

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
            prefix=self.prefix,
            suffix=self.suffix,
            output_dimension=self.output_dimension,
            output_dtype=self.output_dtype,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            metadata_fields_to_embed=self.metadata_fields_to_embed,
            embedding_separator=self.embedding_separator,
            source_id_field=self.source_id_field,
            chunk_fn=self.chunk_fn,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoyageContextualizedDocumentEmbedder":
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

    def _group_documents_by_source(self, documents: List[Document]) -> Tuple[Dict[str, List[Document]], List[str]]:
        """
        Group documents by their source_id field.

        :returns:
            A tuple of (grouped_documents, source_order) where:
            - grouped_documents: Dictionary mapping source_id to list of documents
            - source_order: List of source_ids in order they were first encountered
        """
        grouped_docs: Dict[str, List[Document]] = defaultdict(list)
        source_order: List[str] = []

        for doc in documents:
            source_id = doc.meta.get(self.source_id_field)
            if source_id is None:
                msg = (
                    f"Document is missing the '{self.source_id_field}' metadata field. "
                    f"All documents must have this field to group contextualized chunks. "
                    f"You can change the field name with the 'source_id_field' parameter."
                )
                raise ValueError(msg)

            source_id_str = str(source_id)
            if source_id_str not in grouped_docs:
                source_order.append(source_id_str)
            grouped_docs[source_id_str].append(doc)

        return dict(grouped_docs), source_order

    def _embed_batch(self, grouped_texts: List[List[str]], batch_size: int) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Embed groups of texts using contextualized embeddings.

        :param grouped_texts:
            List of text groups, where each group is a list of related text chunks.
        :param batch_size:
            Number of groups to process at once.
        :returns:
            Tuple of (all_embeddings, metadata) where all_embeddings is a flat list of embeddings
            corresponding to the flattened input texts.
        """
        all_embeddings = []
        meta: Dict[str, Any] = {}
        meta["total_tokens"] = 0

        for i in tqdm(
            range(0, len(grouped_texts), batch_size),
            disable=not self.progress_bar,
            desc="Calculating contextualized embeddings",
        ):
            batch = grouped_texts[i : i + batch_size]

            # Prepare API call parameters
            api_params = {
                "inputs": batch,
                "model": self.model,
            }

            if self.input_type is not None:
                api_params["input_type"] = self.input_type
            if self.output_dtype is not None:
                api_params["output_dtype"] = self.output_dtype
            if self.output_dimension is not None:
                api_params["output_dimension"] = self.output_dimension
            if self.chunk_fn is not None:
                api_params["chunk_fn"] = self.chunk_fn

            response = self.client.contextualized_embed(**api_params)

            # Flatten embeddings from all groups in this batch
            for result in response.results:
                all_embeddings.extend(result.embeddings)

            meta["total_tokens"] += response.total_tokens

        return all_embeddings, meta

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents using contextualized embeddings.

        Documents are grouped by their source_id metadata field, and each group is embedded
        together to preserve context between related chunks.

        :param documents:
            Documents to embed. Each must have a metadata field (default: 'source_id') indicating
            which chunks belong to the same parent document.
        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings
            - `meta`: Information about the usage of the model
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "VoyageContextualizedDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the VoyageTextEmbedder."
            )
            raise TypeError(msg)

        if not documents:
            return {"documents": [], "meta": {"total_tokens": 0}}

        # Group documents by source_id
        grouped_docs, source_order = self._group_documents_by_source(documents)

        # Prepare texts for each group
        grouped_texts = []
        doc_mapping = []  # Maps flattened position to original document

        for source_id in source_order:
            docs = grouped_docs[source_id]
            texts = self._prepare_texts_to_embed(docs)
            grouped_texts.append(texts)
            doc_mapping.extend(docs)

        # Get embeddings
        embeddings, meta = self._embed_batch(grouped_texts, batch_size=self.batch_size)

        # Assign embeddings back to documents
        for doc, emb in zip(doc_mapping, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": meta}
