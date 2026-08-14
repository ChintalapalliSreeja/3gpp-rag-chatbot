from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag import ThreeGPPRAG


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="3GPP Telecom RAG Assistant",
    description="RAG chatbot grounded in 3GPP specifications",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load RAG system
# --------------------------------------------------

print("Initializing 3GPP RAG system...")

rag = ThreeGPPRAG()

print("3GPP RAG system ready.")


# --------------------------------------------------
# Request model
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "3GPP Telecom RAG Assistant is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Chat endpoint
# --------------------------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    result = rag.ask(question)

    return result