# AI Payments Support Assistant

A RAG-powered support chatbot for a payments platform (inspired by Razorpay), with an
agentic tool-calling layer for live transaction lookups.

Built to demonstrate: **Python · LLMs · RAG · Embeddings · Vector DB (FAISS) · AI Agents
· FastAPI · REST APIs · Docker**

---

## How it works

```
User question
     │
     ├──▶ tools.py: does it mention a transaction ID (e.g. TXN1002)?
     │         └─▶ yes → call mock "check_transaction_status" tool
     │
     ├──▶ rag.py: embed the question → search FAISS index → get top matching
     │            FAQ chunks from data/payment_faqs.txt
     │
     └──▶ llm.py: combine question + retrieved FAQ chunks + tool result
                   → send to LLM → grounded answer
```

## Project structure

```
ai-payments-assistant/
├── app/
│   ├── main.py       # FastAPI app & /chat endpoint
│   ├── rag.py         # chunking, embeddings, FAISS vector store
│   ├── tools.py        # mock "agent" tool: transaction status lookup
│   └── llm.py           # prompt assembly + OpenAI call
├── data/
│   └── payment_faqs.txt  # knowledge base the RAG system retrieves from
├── static/
│   └── index.html          # simple chat frontend
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## Setup (local)

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
cp .env.example .env
# edit .env and paste your key

# 4. Run the server
uvicorn app.main:app --reload

# 5. Open the app
# http://localhost:8000
```

Try asking: *"Why did my payment fail?"* or *"What's the status of TXN1002?"*

### Don't have an OpenAI key / don't want to pay?
Two free options — either works, no code changes needed outside `app/llm.py`:
- **Hugging Face Inference API** (free tier): swap the `openai` client in `llm.py` for
  `huggingface_hub.InferenceClient` and use a free instruct model like
  `mistralai/Mistral-7B-Instruct-v0.2`.
- **Ollama** (fully local, free, no key at all): run `ollama pull llama3` locally, then
  point `llm.py` at `http://localhost:11434/api/chat` instead of OpenAI.

Mention in your resume/interview that you designed `llm.py` as a swappable module —
that itself is a good engineering talking point.

## Run with Docker

```bash
docker build -t payments-assistant .
docker run -p 8000:8000 --env-file .env payments-assistant
```

---

## Day-by-day build plan (do this yourself to actually learn it)

**Day 1 — Environment & data**
- Set up the folder structure, `requirements.txt`, virtual env.
- Write `data/payment_faqs.txt` — 10-15 real payment support Q&As.
- Understand: what a payment gateway does, common failure reasons (read Razorpay's
  own public docs — great prep for the interview too).

**Day 2 — Embeddings & FAISS**
- Build `rag.py`. Manually test in a Python shell:
  ```python
  from app.rag import load_and_chunk, embed_texts
  chunks = load_and_chunk("data/payment_faqs.txt")
  vectors = embed_texts(chunks)
  print(vectors.shape)   # should be (num_chunks, 384)
  ```
- Understand: what an embedding actually is (a point in high-dimensional space),
  why cosine similarity = "closeness in meaning."

**Day 3 — RAG retrieval**
- Finish `VectorStore.search()`. Test that searching "payment declined" returns the
  "Why did my payment fail?" chunk even though the wording is different — that's
  the "search by meaning, not keyword" proof point.

**Day 4 — LLM integration**
- Write `llm.py`. Test that the model refuses to answer things NOT in your FAQ file
  (this proves grounding is working, not just the model's general knowledge).

**Day 5 — Agent tool-calling**
- Write `tools.py`. This is the simplest possible "agent": detect intent → call a
  function → feed the result back into the LLM prompt.
- **Stretch goal** (do this if you want to really stand out): replace the regex-based
  `extract_transaction_id` with real LLM function-calling using OpenAI's `tools`
  parameter, so the *model* decides when to call the tool, not a regex. This is the
  actual "AI Agents / Agentic AI" pattern companies mean.

**Day 6 — API & frontend**
- Build `main.py` endpoints, test with `curl` or Postman:
  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "status of TXN1002"}'
  ```
- Wire up `static/index.html`.

**Day 7 — Deployment**
- Write the `Dockerfile`, build and run it locally.
- Deploy free on Render, Railway, or Hugging Face Spaces.
- Push to GitHub with a clean README (this one!) and a demo link/GIF.

---

## Resume bullet points you can use (only after you've actually built & understood this)

- Built a Retrieval-Augmented Generation (RAG) chatbot in Python using FastAPI,
  FAISS, and Sentence-Transformer embeddings to answer payment support queries
  grounded in a custom knowledge base.
- Implemented an agentic tool-calling layer enabling the assistant to fetch live
  transaction status via a mock backend API, combining retrieved context and
  real-time data in LLM prompts.
- Containerized the application with Docker and deployed it on [Render/HF Spaces],
  exposing REST endpoints for chat and health checks.

## Interview talking points to prepare
- Why FAISS over a plain database search? (semantic vs. keyword matching)
- What happens if the retrieved context doesn't contain the answer? (how you'd
  handle hallucination / "I don't know" responses)
- How would you scale this — chunk size trade-offs, caching, async requests?
- How is this relevant to Razorpay specifically? (payments domain, RAG for support
  automation, AI agents for internal tooling)
