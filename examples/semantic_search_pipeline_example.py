"""Example: Semantic search pipeline with Voyage embedders.

This example demonstrates a complete semantic search pipeline using Haystack,
Voyage Embedders, and the Simple Wikipedia dataset.

This example requires a Voyage AI API key. Set it via the VOYAGE_API_KEY
environment variable or in a .env file at the project root.
"""

import asyncio

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from datasets import load_dataset
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Import Voyage Embedders
from haystack_integrations.components.embedders.voyage_embedders import VoyageDocumentEmbedder, VoyageTextEmbedder


async def main():
    # Load first 10 rows of the Simple Wikipedia Dataset from HuggingFace
    dataset = load_dataset("pszemraj/simple_wikipedia", split="validation[:10]")

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

    doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    retriever = InMemoryEmbeddingRetriever(document_store=doc_store)
    doc_writer = DocumentWriter(document_store=doc_store)

    doc_embedder = VoyageDocumentEmbedder(
        model="voyage-4",
        input_type="document",
        timeout=600,
        max_retries=1200,
    )

    # Indexing Pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(instance=doc_embedder, name="DocEmbedder")
    indexing_pipeline.add_component(instance=doc_writer, name="DocWriter")
    indexing_pipeline.connect("DocEmbedder", "DocWriter")

    await indexing_pipeline.run_async({"DocEmbedder": {"documents": docs}})

    print(f"Number of documents in Document Store: {len(doc_store.filter_documents())}")
    print(f"First Document: {doc_store.filter_documents()[0]}")
    print(f"Embedding of first Document: {doc_store.filter_documents()[0].embedding}")

    text_embedder = VoyageTextEmbedder(
        model="voyage-4",
        input_type="query",
        timeout=600,
        max_retries=1200,
    )

    # Query Pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_component(instance=text_embedder, name="TextEmbedder")
    query_pipeline.add_component(instance=retriever, name="Retriever")
    query_pipeline.connect("TextEmbedder.embedding", "Retriever.query_embedding")

    # Search
    results = await query_pipeline.run_async({"TextEmbedder": {"text": "Which year did the Joker movie release?"}})

    # Print text from top result
    top_result = results["Retriever"]["documents"][0].content
    print("The top search result is:")
    print(top_result)


if __name__ == "__main__":
    asyncio.run(main())
