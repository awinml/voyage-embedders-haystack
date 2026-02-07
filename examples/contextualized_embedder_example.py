"""
Example usage of VoyageContextualizedDocumentEmbedder.

This example demonstrates how to use contextualized chunk embeddings to improve
retrieval quality by preserving context between related document chunks.

Contextualized embeddings encode chunks "in context" with other chunks from the
same document, reducing context loss that occurs when chunks are embedded independently.
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from haystack import Document

from haystack_integrations.components.embedders.voyage_embedders import (
    VoyageContextualizedDocumentEmbedder,
)

# Set API key (alternatively, set VOYAGE_API_KEY environment variable or load from .env file)
# os.environ["VOYAGE_API_KEY"] = "your-api-key"


def basic_example():
    """Basic example showing how to use contextualized embeddings."""
    print("=== Basic Contextualized Embeddings Example ===\n")

    # Create documents with source_id metadata to group related chunks
    # Chunks with the same source_id will be embedded together
    docs = [
        # Document 1: Three chunks about Leafy Inc.
        Document(
            content="Leafy Inc. is a sustainable agriculture company founded in 2020.",
            meta={"source_id": "leafy_inc", "chunk_index": 0},
        ),
        Document(
            content="The company's revenue increased by 15% in Q2 2024.",
            meta={"source_id": "leafy_inc", "chunk_index": 1},
        ),
        Document(
            content="Leafy Inc. specializes in organic farming solutions.",
            meta={"source_id": "leafy_inc", "chunk_index": 2},
        ),
        # Document 2: Two chunks about TechCorp
        Document(
            content="TechCorp is a leading software development company.",
            meta={"source_id": "techcorp", "chunk_index": 0},
        ),
        Document(
            content="TechCorp launched three new products in Q2 2024.", meta={"source_id": "techcorp", "chunk_index": 1}
        ),
    ]

    # Initialize the contextualized embedder
    embedder = VoyageContextualizedDocumentEmbedder(
        model="voyage-context-3",
        input_type="document",  # Specify that these are documents, not queries
    )

    # Embed the documents
    result = embedder.run(documents=docs)

    print(f"Embedded {len(result['documents'])} documents")
    print(f"Total tokens used: {result['meta']['total_tokens']}")
    print(f"Embedding dimension: {len(result['documents'][0].embedding)}\n")

    # The embeddings now preserve context between chunks from the same document
    # For example, when searching "What was the revenue growth for Leafy Inc. in Q2 2024?",
    # the second chunk will rank higher because it maintains its connection to "Leafy Inc."
    # through contextualized embedding


def advanced_example_with_metadata():
    """Advanced example showing metadata embedding and custom parameters."""
    print("=== Advanced Contextualized Embeddings Example ===\n")

    # Create documents with additional metadata fields
    docs = [
        Document(
            content="Introduction to neural networks.",
            meta={"source_id": "ml_guide", "category": "Machine Learning", "chapter": 1},
        ),
        Document(
            content="Neural networks consist of layers of interconnected nodes.",
            meta={"source_id": "ml_guide", "category": "Machine Learning", "chapter": 1},
        ),
        Document(
            content="Deep learning uses multiple layers to learn hierarchical representations.",
            meta={"source_id": "ml_guide", "category": "Machine Learning", "chapter": 2},
        ),
        Document(
            content="Introduction to Python programming.",
            meta={"source_id": "python_guide", "category": "Programming", "chapter": 1},
        ),
        Document(
            content="Python is a high-level, interpreted programming language.",
            meta={"source_id": "python_guide", "category": "Programming", "chapter": 1},
        ),
    ]

    # Initialize with custom parameters
    embedder = VoyageContextualizedDocumentEmbedder(
        model="voyage-context-3",
        input_type="document",
        output_dimension=512,  # Use smaller embedding dimension
        metadata_fields_to_embed=["category"],  # Embed category with the text
        embedding_separator=" | ",
        prefix="Document: ",  # Add prefix to each text
        progress_bar=True,
        source_id_field="source_id",  # Field used to group chunks (default)
    )

    result = embedder.run(documents=docs)

    print(f"Embedded {len(result['documents'])} documents")
    print(f"Total tokens used: {result['meta']['total_tokens']}")
    print(f"Embedding dimension: {len(result['documents'][0].embedding)}")
    print(f"Number of source documents: {len({doc.meta['source_id'] for doc in docs})}\n")


def custom_source_field_example():
    """Example showing how to use a custom field for grouping chunks."""
    print("=== Custom Source Field Example ===\n")

    # In this example, we use 'parent_doc' instead of 'source_id'
    docs = [
        Document(content="First chunk of document A.", meta={"parent_doc": "doc_a", "position": 1}),
        Document(content="Second chunk of document A.", meta={"parent_doc": "doc_a", "position": 2}),
        Document(content="First chunk of document B.", meta={"parent_doc": "doc_b", "position": 1}),
    ]

    # Specify the custom field name with source_id_field parameter
    embedder = VoyageContextualizedDocumentEmbedder(
        model="voyage-context-3",
        source_id_field="parent_doc",  # Use 'parent_doc' instead of default 'source_id'
    )

    result = embedder.run(documents=docs)

    print(f"Embedded {len(result['documents'])} documents using 'parent_doc' field")
    print(f"Total tokens used: {result['meta']['total_tokens']}\n")


def comparison_with_standard_embeddings():
    """
    Example showing the difference between standard and contextualized embeddings.

    Note: This is conceptual - you would need to actually run queries to see the difference.
    """
    print("=== Comparison: Standard vs Contextualized ===\n")

    # Sample document chunks that benefit from contextualization
    docs = [
        Document(content="Apple Inc. released their Q1 earnings report.", meta={"source_id": "apple_news"}),
        Document(content="Revenue increased by 12% year over year.", meta={"source_id": "apple_news"}),
        Document(content="The company announced a new product line.", meta={"source_id": "apple_news"}),
    ]

    print("With STANDARD embeddings:")
    print('  - Query: "What was Apple\'s revenue growth?"')
    print("  - The second chunk might rank lower because it lacks explicit mention of 'Apple'")
    print("  - Context is lost when chunks are embedded independently\n")

    # Use contextualized embeddings
    embedder = VoyageContextualizedDocumentEmbedder(model="voyage-context-3", input_type="document")

    result = embedder.run(documents=docs)

    print("With CONTEXTUALIZED embeddings:")
    print('  - Query: "What was Apple\'s revenue growth?"')
    print("  - The second chunk ranks higher because it maintains connection to 'Apple Inc.'")
    print("  - Context is preserved through the contextualized embedding process")
    print(f"\nSuccessfully embedded {len(result['documents'])} documents with context preservation\n")


if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("VOYAGE_API_KEY"):
        print("Warning: VOYAGE_API_KEY environment variable is not set.")
        print("Set it with: export VOYAGE_API_KEY='your-api-key'\n")

    # Run all examples
    try:
        basic_example()
        advanced_example_with_metadata()
        custom_source_field_example()
        comparison_with_standard_embeddings()

        print("=== All examples completed successfully! ===")
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set the VOYAGE_API_KEY environment variable.")
