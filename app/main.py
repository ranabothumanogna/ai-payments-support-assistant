"""
main.py — the FastAPI server. This is the "Backend & APIs" part of the JD.

ENDPOINTS:
  GET  /                -> serves the simple chat frontend
  POST /chat             -> the main RAG + agent endpoint
  GET  /health            -> simple health check (good practice for deployment)

HOW A REQUEST FLOWS THROUGH THE SYSTEM:
  user question
      -> tools.py:   check if it mentions a transaction ID -> call mock tool
      -> rag.py:     embed question -> search FAISS -> get relevant FAQ chunks
      -> llm.py:     combine question + FAQ chunks + tool result -> ask LLM
      -> response returned as JSON
"""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app import rag, tools, llm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "payment_faqs.txt"

app = FastAPI(title="AI Payments Support Assistant")

# Serve the static frontend (index.html, css, js)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.on_event("startup")
def startup_event():
    """Build the FAISS index once when the server starts, not on every request."""
    rag.init_store(str(DATA_PATH))


@app.get("/")
def serve_frontend():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    used_context: str
    tool_called: bool
    tool_result: dict | None = None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.question

    # 1. Agent step: does this question need a live tool call?
    tool_result = None
    tool_called = False
    txn_id = tools.extract_transaction_id(question)
    if txn_id:
        tool_result = tools.check_transaction_status(txn_id)
        tool_called = True

    # 2. RAG step: retrieve relevant knowledge chunks
    context = rag.retrieve_context(question, top_k=3)

    # 3. Generation step: ask the LLM, grounded in context (+ tool result if any)
    answer = llm.generate_answer(question, context, tool_result)

    return ChatResponse(
        answer=answer,
        used_context=context,
        tool_called=tool_called,
        tool_result=tool_result,
    )
