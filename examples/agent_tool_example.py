"""Example: Use a Voyage search pipeline as an Agent tool.

This wraps a retrieve-and-rerank pipeline (VoyageTextEmbedder ->
InMemoryEmbeddingRetriever -> VoyageRanker) as a `ComponentTool` and hands it to
a Haystack `Agent`. The Agent decides when to search the indexed corpus and
grounds its answer in the retrieved passages.

Requires:
- A Voyage AI API key, via the VOYAGE_API_KEY environment variable or a .env
  file at the project root.
- An LLM for the Agent. By default this uses OpenAI (`gpt-4o-mini`), so set
  OPENAI_API_KEY. To use any OpenAI-compatible endpoint (a local proxy, Ollama,
  vLLM, ...) instead, set OPENAI_API_BASE_URL and OPENAI_MODEL (OPENAI_API_KEY
  is sent as the bearer token).
"""

import asyncio
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from haystack import Pipeline, SuperComponent
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import ComponentTool

from haystack_integrations.components.embedders.voyage_embedders import VoyageDocumentEmbedder, VoyageTextEmbedder
from haystack_integrations.components.rankers.voyage.ranker import VoyageRanker

DOCS = [
    Document(content="The Joker movie directed by Todd Phillips was released in the year 2019."),
    Document(content="Bananas are a good source of potassium and dietary fiber."),
    Document(content="The Eiffel Tower is a wrought-iron lattice tower located in Paris, France."),
    Document(content="Water boils at 100 degrees Celsius at sea-level atmospheric pressure."),
]


def format_documents(documents):
    """Format the retrieved documents into the text the Agent's LLM will read."""
    return "\n\n".join(f"[{i + 1}] {doc.content}" for i, doc in enumerate(documents))


async def main():
    # Index the corpus with the Voyage document embedder.
    doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    doc_embedder = VoyageDocumentEmbedder(model="voyage-4", input_type="document")
    embedded = (await doc_embedder.run_async(documents=DOCS))["documents"]
    doc_store.write_documents(embedded)

    # Query pipeline: embed -> retrieve -> rerank.
    rerank_pipeline = Pipeline()
    rerank_pipeline.add_component(
        instance=VoyageTextEmbedder(model="voyage-4", input_type="query"), name="TextEmbedder"
    )
    rerank_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=doc_store, top_k=5), name="Retriever"
    )
    rerank_pipeline.add_component(instance=VoyageRanker(model="rerank-2.5", top_k=3), name="Ranker")
    rerank_pipeline.connect("TextEmbedder.embedding", "Retriever.query_embedding")
    rerank_pipeline.connect("Retriever.documents", "Ranker.documents")

    # Expose the pipeline as a single component: `input_mapping` sends the query
    # to both the embedder and the ranker; `output_mapping` surfaces the docs.
    search_component = SuperComponent(
        pipeline=rerank_pipeline,
        input_mapping={"query": ["TextEmbedder.text", "Ranker.query"]},
        output_mapping={"Ranker.documents": "documents"},
    )

    # ComponentTool auto-generates the tool's JSON schema from the component inputs.
    search_tool = ComponentTool(
        component=search_component,
        name="voyage_search",
        description="Search the indexed corpus for passages relevant to a query.",
        outputs_to_string={"source": "documents", "handler": format_documents},
    )

    agent = Agent(
        chat_generator=OpenAIChatGenerator(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_base_url=os.getenv("OPENAI_API_BASE_URL"),  # None -> OpenAI
        ),
        tools=[search_tool],
        system_prompt=(
            "You are a helpful assistant. Use the voyage_search tool to find relevant passages, "
            "then answer the user's question based only on those passages. "
            "If the search results don't contain the answer, say you couldn't find it."
        ),
    )

    result = await agent.run_async(messages=[ChatMessage.from_user("Which year did the Joker movie release?")])
    print(result["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())
