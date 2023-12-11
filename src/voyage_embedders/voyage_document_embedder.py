import os
from typing import Any, Dict, List, Optional

import voyageai
from haystack.dataclasses import Document
from haystack.core.component import component
from haystack.core.serialization import default_to_dict
from tqdm import tqdm
from voyageai import get_embeddings

MAX_BATCH_SIZE = 8


@component
class VoyageDocumentEmbedder:
    """
    A component for computing Document embeddings using Voyage Embedding models.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack.preview import Document
    from haystack.preview.components.embedders import VoyageDocumentEmbedder

    doc = Document(text="I love pizza!")

    document_embedder = VoyageDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "voyage-01",
        input_type: str = "document",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 8,
        metadata_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        progress_bar: bool = True,  # noqa
    ):
        """
        Create a VoyageDocumentEmbedder component.
        :param api_key: The VoyageAI API key. It can be explicitly provided or automatically read from the
                        environment variable VOYAGE_API_KEY (recommended).
        :param model_name: The name of the model to use. Defaults to "voyage-01".
        For more details on the available models,
            see [Voyage Embeddings documentation](https://docs.voyageai.com/embeddings/).
        :param input_type: Type of the input text. Defaults to `"document"`. This will set the prepend the text with,
        "Represent the document for retrieval: ". Can be set to `None` for no prompt or `"query"` for the query prompt.
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param batch_size: Number of Documents to encode at once.
        :param metadata_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        :param progress_bar: Whether to show a progress bar or not. Can be helpful to disable in production deployments
                             to keep the logs clean.
        """
        # if the user does not provide the API key, check if it is set in the module client
        api_key = api_key or voyageai.api_key
        if api_key is None:
            try:
                api_key = os.environ["VOYAGE_API_KEY"]
            except KeyError as e:
                msg = "VoyageDocumentEmbedder expects an VoyageAI API key. Set the VOYAGE_API_KEY environment variable (recommended) or pass it explicitly."  # noqa
                raise ValueError(msg) from e

        voyageai.api_key = api_key

        self.model_name = model_name
        self.input_type = input_type
        self.prefix = prefix
        self.suffix = suffix

        if batch_size <= MAX_BATCH_SIZE:
            self.batch_size = batch_size
        else:
            err_msg = f"""VoyageDocumentEmbedder has a maximum batch size of {MAX_BATCH_SIZE}. Set the Set the batch_size to {MAX_BATCH_SIZE} or less."""  # noqa
            raise ValueError(err_msg)

        self.progress_bar = progress_bar
        self.metadata_fields_to_embed = metadata_fields_to_embed or []
        self.embedding_separator = embedding_separator

    def to_dict(self) -> Dict[str, Any]:
        """
        This method overrides the default serializer in order to avoid leaking the `api_key` value passed
        to the constructor.
        """
        return default_to_dict(
            self,
            model_name=self.model_name,
            input_type=self.input_type,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            metadata_fields_to_embed=self.metadata_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

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

    def _embed_batch(self, texts_to_embed: List[str], batch_size: int) -> List[List[float]]:
        """
        Embed a list of texts in batches.
        """

        all_embeddings = []
        for i in tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]
            embeddings = get_embeddings(
                list_of_text=batch, batch_size=batch_size, model=self.model_name, input_type=self.input_type
            )
            all_embeddings.extend(embeddings)

        return all_embeddings

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.
        The embedding of each Document is stored in the `embedding` field of the Document.

        :param documents: A list of Documents to embed.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            msg = "VoyageDocumentEmbedder expects a list of Documents as input.In case you want to embed a string, please use the VoyageTextEmbedder."  # noqa
            raise TypeError(msg)

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents}
