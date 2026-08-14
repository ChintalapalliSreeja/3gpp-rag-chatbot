# 3GPP Telecom RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about 3GPP telecommunications standards using official 3GPP specification documents as the primary knowledge source.

The system is designed with a strong focus on reducing hallucinations by grounding generated answers in retrieved 3GPP documentation and refusing to answer when sufficient supporting context is not available.

---

## Project Overview

The 3GPP Telecom RAG Assistant allows users to ask questions related to 5G telecommunications standards.

The knowledge base currently contains:

- 3GPP TS 23.501
- 3GPP TS 23.502
- 3GPP TS 23.503

The system retrieves relevant sections from these documents and uses them as context for generating answers.

If the retrieved information is not sufficiently relevant to the question, the system does not generate an unsupported answer. Instead, it returns a safe response indicating that sufficient information was not found in the provided 3GPP documentation.

---

## Key Features

- 3GPP standards-based question answering
- Retrieval-Augmented Generation (RAG)
- FAISS vector database
- Hugging Face sentence embeddings
- OpenAI GPT-4o-mini
- Semantic similarity-based retrieval
- Relevance threshold filtering
- Source document and page references
- Out-of-domain question detection
- Hallucination reduction
- FastAPI backend
- Streamlit frontend
- Evaluation framework with answerable and unanswerable questions

---

## Architecture

```text
                    ┌──────────────────┐
                    │      User        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Streamlit UI     │
                    │   Frontend       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ FastAPI Backend  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ User Question    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Query Embedding  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ FAISS Retriever  │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌────────────────────────┐
                  │ Relevance Filtering   │
                  └───────────┬────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
            Relevant                  Not Relevant
                 │                         │
                 ▼                         ▼
        ┌─────────────────┐       ┌─────────────────┐
        │ Retrieved 3GPP  │       │ Safe Refusal    │
        │ Context         │       │ Response        │
        └────────┬────────┘       └─────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ GPT-4o-mini LLM │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────────────┐
        │ Grounded Answer +       │
        │ Document/Page Sources   │
        └─────────────────────────┘