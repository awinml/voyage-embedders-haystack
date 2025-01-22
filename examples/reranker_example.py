from haystack import Document

from haystack_integrations.components.rankers.voyage.voyage_text_reranker import VoyageRanker

ranker = VoyageRanker(model="rerank-2", top_k=2)

docs = [Document(content="Paris"), Document(content="Berlin")]
query = "What is the capital of germany?"
output = ranker.run(query=query, documents=docs)
docs = output["documents"]

for doc in docs:
    print(f"{doc.content} - {doc.score}")
