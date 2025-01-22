from haystack_integrations.components.embedders.voyage_embedders import VoyageTextEmbedder

# Example text from the Amazon Reviews Polarity Dataset (https://huggingface.co/datasets/amazon_polarity)
text = (
    "It clearly says on line this will work on a Mac OS system. The disk comes and it does not, only Windows."
    " Do Not order this if you have a Mac!!"
)
instruction = "Represent the Amazon comment for classifying the sentence as positive or negative"

text_embedder = VoyageTextEmbedder(
    model="voyage-3",
    input_type="query",
    timeout=600,
    max_retries=1200,
)

result = text_embedder.run(text)
print(f"Embedding: {result['embedding']}")
print(f"Embedding Dimension: {len(result['embedding'])}")
