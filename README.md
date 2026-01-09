<h1 align="center"> <a href="https://github.com/awinml/voyage-embedders-haystack"> Voyage Embedders and Rankers - Haystack </a> </h1>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/voyage-embedders-haystack)](https://pypi.org/project/voyage-embedders-haystack/)
[![Downloads](https://img.shields.io/pypi/dm/voyage-embedders-haystack?color=blue&logo=pypi&logoColor=gold)](https://pypi.org/project/voyage-embedders-haystack/)
[![License](https://img.shields.io/github/license/awinml/voyage-embedders-haystack?color=green)](LICENSE)
[![Tests](https://github.com/awinml/voyage-embedders-haystack/workflows/Test/badge.svg)](https://github.com/awinml/voyage-embedders-haystack/actions)
[![Coverage](https://coveralls.io/repos/github/awinml/voyage-embedders-haystack/badge.svg?branch=main)](https://coveralls.io/github/awinml/voyage-embedders-haystack?branch=main)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/awinml/voyage-embedders-haystack/main.svg)](https://results.pre-commit.ci/latest/github/awinml/voyage-embedders-haystack/main)
[![Types](https://img.shields.io/badge/types-ty-blue.svg)](https://docs.astral.sh/ty/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

Custom components for [Haystack](https://github.com/deepset-ai/haystack) for creating embeddings and reranking documents using the [Voyage Models](https://voyageai.com/).

Voyage’s embedding models are state-of-the-art in retrieval accuracy. These models outperform top performing embedding models like `intfloat/e5-mistral-7b-instruct` and `OpenAI/text-embedding-3-large` on the [MTEB Benchmark](https://github.com/embeddings-benchmark/mteb).

#### What's New

- **[v1.8.0 - 07/11/25]:**

  - The new `VoyageContextualizedDocumentEmbedder` component supports Voyage's contextualized chunk embeddings.
  - Contextualized embeddings encode document chunks "in context" with other chunks from the same document, preserving semantic relationships and reducing context loss for improved retrieval accuracy.

- **[v1.5.0 - 22/01/25]:**

  - The new `VoyageRanker` component can be used to rerank documents using the `Voyage Reranker` models.
  - Matryoshka Embeddings and Quantized Embeddings can now be created using the `output_dimension` and `output_dtype` parameters.

- **[v1.4.0 - 24/07/24]:**

  - The maximum timeout and number of retries made by the Client can now be set for the embedders using the `timeout` and `max_retries` parameters.

- **[v1.3.0 - 18/03/24]:**

  - **Breaking Change:** The import path for the embedders has been changed to `haystack_integrations.components.embedders.voyage_embedders`.
    Please replace all instances of `from voyage_embedders.voyage_document_embedder import VoyageDocumentEmbedder` and `from voyage_embedders.voyage_text_embedder import VoyageTextEmbedder` with  
    `from haystack_integrations.components.embedders.voyage_embedders import VoyageDocumentEmbedder, VoyageTextEmbedder`.
  - The embedders now use the Haystack `Secret` API for authentication. For more information please see the [Secret Management Documentation](https://docs.haystack.deepset.ai/docs/secret-management).

- **[v1.2.0 - 02/02/24]:**

  - **Breaking Change:** `VoyageDocumentEmbedder` and `VoyageTextEmbedder` now accept the `model` parameter instead of `model_name`.
  - The embedders have been use the new `voyageai.Client.embed()` method instead of the deprecated `get_embedding` and `get_embeddings` methods of the global namespace.
  - Support for the new `truncate` parameter has been added.
  - The embedders now return the total number of tokens used as part of the `"total_tokens"` in the metadata.

- **[v1.1.0 - 13/12/23]:** Added support for `input_type` parameter in `VoyageTextEmbedder` and `VoyageDocument Embedder`.

- **[v1.0.0 - 21/11/23]:** Added `VoyageTextEmbedder` and `VoyageDocument Embedder` to embed strings and documents.

## Requirements

- Python 3.10 or higher
- [Voyage AI API Key](https://voyageai.com/)

## Installation

```bash
pip install voyage-embedders-haystack
```

## Usage

You can use Voyage Embedding models with multiple components:

- **[VoyageTextEmbedder](https://github.com/awinml/voyage-embedders-haystack/blob/main/src/voyage_embedders/voyage_text_embedder.py)**: For generating embeddings for queries.
- **[VoyageDocumentEmbedder](https://github.com/awinml/voyage-embedders-haystack/blob/main/src/voyage_embedders/voyage_document_embedder.py)**: For creating semantic embeddings for documents in your indexing pipeline.
- **[VoyageContextualizedDocumentEmbedder](https://github.com/awinml/voyage-embedders-haystack/blob/main/src/haystack_integrations/components/embedders/voyage_embedders/voyage_contextualized_document_embedder.py)**: For creating contextualized embeddings where document chunks are embedded together to preserve context and improve retrieval accuracy.

The Voyage Reranker models can be used with the [VoyageRanker](https://github.com/awinml/voyage-embedders-haystack/blob/main/src/haystack_integrations/components/rankers/voyage/ranker.py) component.

### Contextualized Chunk Embeddings

The `VoyageContextualizedDocumentEmbedder` uses Voyage's contextualized embedding models to encode document chunks "in context" with other chunks from the same document. This approach preserves semantic relationships between chunks and reduces context loss, leading to improved retrieval accuracy.

**Key features:**
- Documents are grouped by a metadata field (default: `source_id`)
- Chunks from the same source document are embedded together
- Maintains semantic connections between related chunks
- Recommended model: `voyage-context-3`

For detailed usage examples, see the [contextualized embedder example](https://github.com/awinml/voyage-embedders-haystack/blob/main/examples/contextualized_embedder_example.py).

Once you've selected the suitable component for your specific use case, initialize the component with the model name and VoyageAI API key. You can also
set the environment variable `VOYAGE_API_KEY` instead of passing the API key as an argument.
To get an API key, please see the [Voyage AI website.](https://www.voyageai.com/)

Information about the supported models, can be found on the [Voyage AI Documentation.](https://docs.voyageai.com/)

## Example

You can find all the examples in the [`examples`](https://github.com/awinml/voyage-embedders-haystack/tree/main/examples) folder.

Below is the example Semantic Search pipeline that uses the [Simple Wikipedia](https://huggingface.co/datasets/pszemraj/simple_wikipedia) Dataset from HuggingFace.

Load the dataset:

```python
# Install HuggingFace Datasets using "pip install datasets"
from datasets import load_dataset
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Import Voyage Embedders
from haystack_integrations.components.embedders.voyage_embedders import VoyageDocumentEmbedder, VoyageTextEmbedder

# Load first 100 rows of the Simple Wikipedia Dataset from HuggingFace
dataset = load_dataset("pszemraj/simple_wikipedia", split="validation[:100]")

docs = [
    Document(
        content=doc["text"],
        meta={
            "title": doc["title"],
            "url": doc["url"],
        },
    )
    for doc in dataset
]
```

Index the documents to the `InMemoryDocumentStore` using the `VoyageDocumentEmbedder` and `DocumentWriter`:

```python
doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
retriever = InMemoryEmbeddingRetriever(document_store=doc_store)
doc_writer = DocumentWriter(document_store=doc_store)

doc_embedder = VoyageDocumentEmbedder(
    model="voyage-2",
    input_type="document",
)
text_embedder = VoyageTextEmbedder(model="voyage-2", input_type="query")

# Indexing Pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=doc_embedder, name="DocEmbedder")
indexing_pipeline.add_component(instance=doc_writer, name="DocWriter")
indexing_pipeline.connect("DocEmbedder", "DocWriter")

indexing_pipeline.run({"DocEmbedder": {"documents": docs}})

print(f"Number of documents in Document Store: {len(doc_store.filter_documents())}")
print(f"First Document: {doc_store.filter_documents()[0]}")
print(f"Embedding of first Document: {doc_store.filter_documents()[0].embedding}")
```

Query the Semantic Search Pipeline using the `InMemoryEmbeddingRetriever` and `VoyageTextEmbedder`:

```python
text_embedder = VoyageTextEmbedder(model="voyage-2", input_type="query")

# Query Pipeline
query_pipeline = Pipeline()
query_pipeline.add_component(instance=text_embedder, name="TextEmbedder")
query_pipeline.add_component(instance=retriever, name="Retriever")
query_pipeline.connect("TextEmbedder.embedding", "Retriever.query_embedding")

# Search
results = query_pipeline.run({"TextEmbedder": {"text": "Which year did the Joker movie release?"}})

# Print text from top result
top_result = results["Retriever"]["documents"][0].content
print("The top search result is:")
print(top_result)
```

## Contributing

We welcome contributions from the community! Please take a look at our [contributing guide](CONTRIBUTING.md) for more details on how to get started.

Pull requests are welcome. For major changes, please open an issue first to discuss the proposed changes.

## License

`voyage-embedders-haystack` is distributed under the terms of the [Apache-2.0 license](https://github.com/awinml/voyage-embedders-haystack/blob/main/LICENSE).

Maintained by [Ashwin Mathur](https://github.com/awinml).
