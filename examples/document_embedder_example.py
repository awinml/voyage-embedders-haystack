from haystack.dataclasses import Document
from haystack_integrations.components.embedders.voyage_embedders import VoyageDocumentEmbedder

# Text taken from PubMed QA Dataset (https://huggingface.co/datasets/pubmed_qa)
document_list = [
    Document(
        content="Oxidative stress generated within inflammatory joints can produce autoimmune phenomena and joint "
        "destruction. Radical species with oxidative activity, including reactive nitrogen species, "
        "represent mediators of inflammation and cartilage damage.",
        meta={
            "pubid": "25,445,628",
            "long_answer": "yes",
        },
    ),
    Document(
        content="Plasma levels of pancreatic polypeptide (PP) rise upon food intake. Although other pancreatic islet"
        "hormones, such as insulin and glucagon, have been extensively investigated, PP secretion and actions are still"
        " poorly understood.",
        meta={
            "pubid": "25,445,712",
            "long_answer": "yes",
        },
    ),
    Document(
        content="Disturbed sleep is associated with mood disorders. Both depression and insomnia may increase the risk "
        "of disability retirement. The longitudinal links among insomnia, depression and work incapacity are poorly "
        "known.",
        meta={
            "pubid": "25,451,441",
            "long_answer": "yes",
        },
    ),
]

doc_embedder = VoyageDocumentEmbedder(
    model="voyage-2",
)

result = doc_embedder.run(document_list)

print(f"Document Text: {result['documents'][0].content}")
print(f"Document Embedding: {result['documents'][0].embedding}")
print(f"Embedding Dimension: {len(result['documents'][0].embedding)}")
