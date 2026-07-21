"""Example: Document reranking with VoyageRanker.

This example requires a Voyage AI API key. Set it via the VOYAGE_API_KEY
environment variable or in a .env file at the project root.
"""

import asyncio

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from haystack import Document

from haystack_integrations.components.rankers.voyage.ranker import VoyageRanker


async def main():
    ranker = VoyageRanker(model="rerank-2.5", top_k=2)

    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "What is the capital of germany?"
    output = await ranker.run(query=query, documents=docs)
    docs = output["documents"]

    for doc in docs:
        print(f"{doc.content} - {doc.score}")


if __name__ == "__main__":
    asyncio.run(main())
