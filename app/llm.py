"""
llm.py — talks to the LLM.

WHAT THIS FILE DOES:
Takes the user's question + retrieved context (from rag.py) + optional tool
result (from tools.py), builds a single prompt, and asks the LLM to answer
using ONLY that information. This "grounding" is the whole point of RAG —
it stops the model from making things up.

This version uses Ollama — a free tool that runs an AI model directly on
your own computer. No API key, no account, no cost. See README for setup.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2"   # must match the model you pulled with `ollama pull llama3.2`

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
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]

