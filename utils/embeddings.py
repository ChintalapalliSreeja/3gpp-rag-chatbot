from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Load the embedding model used for semantic search.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings