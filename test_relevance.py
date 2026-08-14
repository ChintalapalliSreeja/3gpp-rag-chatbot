from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Load embedding model
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load FAISS
print("Loading FAISS database...")

vector_store = FAISS.load_local(
    "data/3gpp/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


questions = [
    "What is the role of the AMF?",
    "What is the role of the SMF?",
    "What is PDU Session Establishment?",
    "What is the capital of India?",
    "What is the salary of an AI Engineer?"
]


for question in questions:

    print("\n" + "=" * 80)
    print("QUESTION:", question)

    results = vector_store.similarity_search_with_score(
        question,
        k=3
    )

    for i, (document, score) in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("Distance:", round(score, 4))
        print("Page:", document.metadata.get("page"))
        print("Text:", document.page_content[:300])