"""
llm.py — talks to the LLM.

WHAT THIS FILE DOES:
Takes the user's question + retrieved context (from rag.py) + optional tool
result (from tools.py), builds a single prompt, and asks the LLM to answer
using ONLY that information. This "grounding" is the whole point of RAG —
it stops the model from making things up.

This version uses Groq — a free, fast LLM API (no credit card required).
Works both locally and when deployed to the cloud (unlike Ollama, which
needs to run on the same machine). Get a free key at console.groq.com.
"""

import os
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"

_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Create a .env file (copy .env.example) "
        "and paste your free Groq key into it before running the server."
    )

SYSTEM_PROMPT = """You are an AI support assistant for a payments platform (like Razorpay).
Answer the user's question using ONLY the context provided below.
If the context includes a live transaction status, mention it clearly.
If you don't know the answer from the context, say so honestly — do not make things up.
Keep answers concise and helpful."""


def generate_answer(question: str, retrieved_context: str, tool_result: dict | None = None) -> str:
    tool_section = ""
    if tool_result:
        tool_section = f"\n\nLive transaction data (from internal system):\n{tool_result}"

    user_prompt = f"""Context from knowledge base:
{retrieved_context}
{tool_section}

User question: {question}

Answer:"""

    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {_api_key}"},
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


