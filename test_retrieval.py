from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# 1. Load the embedding model
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Load the existing FAISS database
print("Loading FAISS database...")

vector_store = FAISS.load_local(
    "data/3gpp/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# 3. Ask a question
question = "What is the role of the AMF?"

print("\nQuestion:")
print(question)


# 4. Retrieve the most relevant chunks
results = vector_store.similarity_search_with_score(
    question,
    k=5
)


# 5. Display results
print("\nRetrieved documents:\n")

for i, (document, score) in enumerate(results, start=1):

    print("=" * 70)

    print(f"Result {i}")
    print(f"Similarity score: {score}")

    print(f"Source: {document.metadata.get('source')}")
    print(f"Page: {document.metadata.get('page')}")

    print("\nText:")
    print(document.page_content[:1500])

    print()