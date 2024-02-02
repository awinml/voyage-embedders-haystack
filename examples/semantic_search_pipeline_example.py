# Install HuggingFace Datasets using "pip install datasets"
from datasets import load_dataset
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Import Voyage Embedders
from voyage_embedders.voyage_document_embedder import VoyageDocumentEmbedder
from voyage_embedders.voyage_text_embedder import VoyageTextEmbedder

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

doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
doc_embedder = VoyageDocumentEmbedder(
    model="voyage-2",
    input_type="document",
)
text_embedder = VoyageTextEmbedder(model="voyage-2", input_type="query")

# Indexing Pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=doc_embedder, name="DocEmbedder")
indexing_pipeline.add_component(instance=DocumentWriter(document_store=doc_store), name="DocWriter")
indexing_pipeline.connect(connect_from="DocEmbedder", connect_to="DocWriter")

indexing_pipeline.run({"DocEmbedder": {"documents": docs}})

print(f"Number of documents in Document Store: {len(doc_store.filter_documents())}")
print(f"First Document: {doc_store.filter_documents()[0]}")
print(f"Embedding of first Document: {doc_store.filter_documents()[0].embedding}")


# Query Pipeline
query_pipeline = Pipeline()
query_pipeline.add_component("TextEmbedder", text_embedder)
query_pipeline.add_component("Retriever", InMemoryEmbeddingRetriever(document_store=doc_store))
query_pipeline.connect("TextEmbedder.embedding", "Retriever.query_embedding")


# Search
results = query_pipeline.run({"TextEmbedder": {"text": "Which year did the Joker movie release?"}})

# Print text from top result
top_result = results["Retriever"]["documents"][0].content
print("The top search result is:")
print(top_result)
