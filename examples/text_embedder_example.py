"""Example: Text embedding with VoyageTextEmbedder.

This example requires a Voyage AI API key. Set it via the VOYAGE_API_KEY
environment variable or in a .env file at the project root.
"""

import asyncio

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from haystack_integrations.components.embedders.voyage_embedders import VoyageTextEmbedder

# Example text from the Amazon Reviews Polarity Dataset (https://huggingface.co/datasets/amazon_polarity)
text = (
    "It clearly says on line this will work on a Mac OS system. The disk comes and it does not, only Windows."
    " Do Not order this if you have a Mac!!"
)
instruction = "Represent the Amazon comment for classifying the sentence as positive or negative"


async def main():
    text_embedder = VoyageTextEmbedder(
        model="voyage-4",
        input_type="query",
        timeout=600,
        max_retries=1200,
    )

    result = await text_embedder.run_async(text=text)
    print(f"Embedding: {result['embedding']}")
    print(f"Embedding Dimension: {len(result['embedding'])}")


if __name__ == "__main__":
    asyncio.run(main())
