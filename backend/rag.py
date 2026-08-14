import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


# ============================================================
# ENVIRONMENT
# ============================================================

# Project root:
# 3gpp-rag-chatbot/
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from project root
load_dotenv(BASE_DIR / ".env")


# ============================================================
# CONFIGURATION
# ============================================================

FAISS_PATH = BASE_DIR / "data" / "3gpp" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "gpt-4o-mini"

TEMPERATURE = 0

# Number of documents retrieved from FAISS
TOP_K = 8

# FAISS similarity_search_with_score returns distance.
# Lower distance = more similar.
RELEVANCE_THRESHOLD = 1.10


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a 3GPP telecommunications technical documentation assistant.

Your job is to answer questions using ONLY the provided 3GPP
documentation context.

Rules:

1. Use only the supplied context.
2. Do not use outside knowledge.
3. If the context contains enough information to answer the
   question, answer clearly and directly.
4. Do not say that information is unavailable when the context
   actually contains the answer.
5. If the context genuinely does not contain enough information,
   say:

   I couldn't find sufficient information in the provided
   3GPP documentation to answer this question.

6. Do not invent specifications, interfaces, procedures,
   functions, or values.
7. Keep technical terminology such as AMF, SMF, UPF, NRF,
   N2, N3, PDU Session, etc. unchanged.
8. When possible, explain the answer in 2-5 concise paragraphs
   or bullet points.
"""


# ============================================================
# 3GPP RAG CLASS
# ============================================================

class ThreeGPPRAG:

    def __init__(self):

        print("=" * 70)
        print("3GPP RAG INITIALIZATION")
        print("=" * 70)

        # ----------------------------------------------------
        # Check OpenAI API key
        # ----------------------------------------------------

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing.\n"
                "Please add it to the .env file in the project root."
            )

        print("\nLoading embedding model...")

        # ----------------------------------------------------
        # Embedding model
        # ----------------------------------------------------

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        print("Embedding model loaded.")

        # ----------------------------------------------------
        # Load FAISS
        # ----------------------------------------------------

        print("\nLoading FAISS database...")

        if not FAISS_PATH.exists():

            raise FileNotFoundError(
                f"\nFAISS database not found at:\n{FAISS_PATH}\n\n"
                "Run this command first:\n"
                "python utils/vector_store.py"
            )

        self.vector_store = FAISS.load_local(
            str(FAISS_PATH),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        print("FAISS database loaded successfully.")

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        print("\nLoading LLM...")

        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            api_key=api_key
        )

        print("LLM loaded successfully.")

        print("\n" + "=" * 70)
        print("3GPP RAG READY")
        print("=" * 70)


    # ========================================================
    # PAGE NUMBER
    # ========================================================

    def get_page_number(self, metadata):

        page = metadata.get("page")

        if page is None:
            return "N/A"

        try:
            # PyPDFLoader normally stores pages as zero-based.
            # Convert to human-readable page number.
            return int(page) + 1

        except (ValueError, TypeError):
            return page


    # ========================================================
    # DOCUMENT NAME
    # ========================================================

    def get_document_name(self, metadata):

        source = metadata.get("source")

        if source:
            return Path(str(source)).name

        return metadata.get(
            "document",
            "Unknown document"
        )


    # ========================================================
    # RETRIEVE DOCUMENTS
    # ========================================================

    def retrieve(self, question):

        results = self.vector_store.similarity_search_with_score(
            question,
            k=TOP_K
        )

        return results


    # ========================================================
    # CHECK RELEVANCE
    # ========================================================

    def get_relevant_results(self, results):

        relevant_results = []

        for doc, distance in results:

            try:
                distance_value = float(distance)
            except (ValueError, TypeError):
                continue

            if distance_value <= RELEVANCE_THRESHOLD:

                relevant_results.append(
                    (doc, distance_value)
                )

        return relevant_results


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    def build_context(self, relevant_results):

        context_parts = []

        for index, (doc, distance) in enumerate(
            relevant_results,
            start=1
        ):

            metadata = doc.metadata

            document_name = self.get_document_name(
                metadata
            )

            page_number = self.get_page_number(
                metadata
            )

            content = doc.page_content.strip()

            context_parts.append(
                f"""
---------------- CONTEXT {index} ----------------

Document: {document_name}

Page: {page_number}

Similarity Distance: {distance:.4f}

Content:
{content}

---------------------------------------------------
"""
            )

        return "\n".join(context_parts)


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    def generate_answer(
        self,
        question,
        context
    ):

        prompt = f"""
{SYSTEM_PROMPT}

USER QUESTION:
{question}

3GPP DOCUMENTATION CONTEXT:
{context}

Now answer the user's question using ONLY the context above.

Important:
- If the context contains the answer, provide the answer.
- Do not refuse simply because the question is technical.
- Do not add unrelated information.
- Do not use outside knowledge.
"""

        response = self.llm.invoke(prompt)

        answer = response.content

        if isinstance(answer, list):

            answer = " ".join(
                str(item)
                for item in answer
            )

        return str(answer).strip()


    # ========================================================
    # ASK
    # ========================================================

    def ask(self, question):

        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

        if not question:

            return {
                "answer": "Please enter a question.",
                "sources": []
            }

        question = question.strip()

        if not question:

            return {
                "answer": "Please enter a question.",
                "sources": []
            }


        # ----------------------------------------------------
        # Print question
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("QUESTION:")
        print(question)
        print("=" * 70)


        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        results = self.retrieve(question)


        # ----------------------------------------------------
        # Display retrieved results
        # ----------------------------------------------------

        print("\nRETRIEVED RESULTS:")

        for index, (doc, distance) in enumerate(
            results,
            start=1
        ):

            metadata = doc.metadata

            page_number = self.get_page_number(
                metadata
            )

            print(
                f"{index}. "
                f"Distance={float(distance):.4f}, "
                f"Page={page_number}"
            )


        # ----------------------------------------------------
        # Filter relevant documents
        # ----------------------------------------------------

        relevant_results = self.get_relevant_results(
            results
        )


        # ----------------------------------------------------
        # No relevant context
        # ----------------------------------------------------

        if not relevant_results:

            print(
                "\nNo sufficiently relevant "
                "3GPP context found."
            )

            answer = (
                "I couldn't find sufficient information "
                "in the provided 3GPP documentation "
                "to answer this question."
            )

            print("\nANSWER:")
            print(answer)

            print("\nSOURCES:")
            print("None")

            return {
                "answer": answer,
                "sources": []
            }


        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context = self.build_context(
            relevant_results
        )


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        print("\nGenerating answer...")

        answer = self.generate_answer(
            question,
            context
        )


        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        sources = []

        for doc, distance in relevant_results:

            metadata = doc.metadata

            document_name = self.get_document_name(
                metadata
            )

            page_number = self.get_page_number(
                metadata
            )

            sources.append(
                {
                    "document": document_name,
                    "page": page_number,
                    "distance": round(
                        float(distance),
                        4
                    )
                }
            )


        # ----------------------------------------------------
        # Final output
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("FINAL RESPONSE")
        print("=" * 70)

        print(answer)

        print("\nSOURCES:")
        print("=" * 70)

        for source in sources:

            print(
                f"Document: {source['document']} | "
                f"Page: {source['page']} | "
                f"Distance: {source['distance']}"
            )

        print("=" * 70)


        # ----------------------------------------------------
        # Return API response
        # ----------------------------------------------------

        return {
            "answer": answer,
            "sources": sources
        }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    rag = ThreeGPPRAG()

    test_question = (
        "What is the role of the AMF?"
    )

    result = rag.ask(
        test_question
    )

    print("\n")
    print("=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")

    for source in result["sources"]:

        print(
            f"- {source['document']} "
            f"| Page {source['page']} "
            f"| Distance {source['distance']}"
        )