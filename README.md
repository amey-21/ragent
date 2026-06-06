# Support Agent - AI Customer Support with RAG

A production-grade AI customer support agent built with **LangGraph**, **hybrid RAG retrieval**, and **FastAPI**. Answers support questions from a knowledge base, checks order status, and escalates to humans when needed.

Built as a GenAI engineering portfolio project demonstrating end-to-end RAG pipeline design, agent orchestration, and LLM evaluation.

---

## Architecture

```
User Query
    │
    ▼
FastAPI (async + SSE streaming)
    │
    ▼
LangGraph ReAct Agent  ←──────────────────────┐
    │                                          │
    ├── Tool: search_knowledge_base            │
    │       │                                  │
    │       ▼                                  │
    │   Retrieval Pipeline                     │
    │   Dense (ChromaDB) + Sparse (BM25)       │
    │   → RRF Fusion → CrossEncoder Rerank     │
    │   → Top 5 chunks → LLM answer            │
    │                                          │
    ├── Tool: check_order_status               │
    │       └── Mock order DB lookup           │
    │                                          │
    └── Tool: escalate_to_human ───────────────┘
            └── Logs to escalations.json
    │
    ▼
Two-tier Memory
    ├── Short-term: last 10 messages (in-session)
    └── Long-term: conversation summaries in ChromaDB
    │
    ▼
Streamed response (SSE)
```

---

## Features

- **Hybrid retrieval** - Dense (ChromaDB cosine) + Sparse (BM25) merged with Reciprocal Rank Fusion
- **CrossEncoder reranking** - `ms-marco-MiniLM-L-6-v2` reranks top-20 candidates to top-5
- **LangGraph ReAct agent** - stateful graph with conditional tool routing and loop detection
- **Two-tier memory** - short-term sliding window + long-term ChromaDB-backed summaries
- **FastAPI + SSE streaming** - async server with token-by-token response streaming
- **LLM-as-judge evaluation** - custom faithfulness, answer quality, and context recall scoring

---

## Evaluation Results

Evaluated on a 20-question test set using a custom LLM-as-judge pipeline (direct Groq calls, no Ragas dependency):

| Metric | Score | Verdict |
|---|---|---|
| Faithfulness | **0.875** | ✓ Good |
| Answer Quality | **0.925** | ✓ Good |
| Context Recall | **0.700** | ~ Acceptable |

The evaluation pipeline itself is in `eval/evaluate.py` each metric is scored by an LLM judge with per-question reasoning logged to `eval/results.json`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Agent orchestration | LangGraph |
| Vector database | ChromaDB (local persistent) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Sparse retrieval | `rank-bm25` (BM25Okapi) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| API server | FastAPI + Uvicorn |
| Data | HuggingFace `bitext/Bitext-customer-support-llm-chatbot-training-dataset` |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourusername/support-agent.git
cd support-agent
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get a free key at console.groq.com
```

### 3. Download and index the knowledge base

```bash
# Download 27-intent customer support dataset from HuggingFace
python src/ingestion/download_data.py

# Chunk, embed, and store in ChromaDB
python src/ingestion/ingest.py
```

Expected output: `✓ 810+ chunks stored in ChromaDB`

### 4. Run the API server

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive API docs.

---

## API Usage

### Chat (sync)

```bash
curl -X POST http://localhost:8000/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I get a refund?", "session_id": "user123"}'
```

```json
{
  "response": "According to our refund policy, you can request a refund by...",
  "session_id": "user123",
  "tool_calls": ["search_knowledge_base"],
  "turn": 1
}
```

### Chat (streaming SSE)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the status of order ORD002?", "session_id": "user123"}'
```

Tokens stream word-by-word via SSE. Final event includes metadata:

```
data: {"type": "token", "content": "Your "}
data: {"type": "token", "content": "order "}
...
data: {"type": "done", "session_id": "user123", "tool_calls": ["check_order_status"], "turn": 2}
```

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "active_sessions": 2,
  "total_turns": 14,
  "uptime_seconds": 312.4,
  "escalations_logged": 3
}
```

---

## Project Structure

```
support-agent/
│
├── src/
│   ├── ingestion/
│   │   ├── download_data.py   # Downloads HuggingFace dataset
│   │   ├── chunker.py         # 3 chunking strategies (fixed/recursive/QA-aware)
│   │   ├── embedder.py        # Embeds chunks into ChromaDB
│   │   └── ingest.py          # Master pipeline: chunk → embed → store
│   │
│   ├── retrieval/
│   │   ├── dense.py           # ChromaDB cosine similarity search
│   │   ├── sparse.py          # BM25 keyword search
│   │   ├── hybrid.py          # RRF fusion of dense + sparse
│   │   └── reranker.py        # CrossEncoder reranking
│   │
│   ├── agent/
│   │   ├── tools.py           # 3 tools: RAG search, order status, escalate
│   │   ├── prompts.py         # System prompt with guardrails
│   │   ├── memory.py          # Short-term + long-term memory
│   │   └── graph.py           # LangGraph ReAct agent + SupportSession
│   │
│   └── api/
│       └── main.py            # FastAPI server with SSE streaming
│
├── eval/
│   ├── dataset.json           # 20-question evaluation set
│   └── evaluate.py            # LLM-as-judge evaluation pipeline
│
├── data/                      # Runtime data (not committed)
│   ├── raw/                   # Downloaded KB text files
│   └── chroma_db/             # Vector index
│
├── .env.example               # Environment variable template
├── requirements.txt
└── README.md
```

---

## Chunking Strategy

Three strategies were evaluated:

| Strategy | Chunks | Avg Length | Quality |
|---|---|---|---|
| Fixed-size (300 chars) | 81 | 299 chars | Poor cuts mid-sentence |
| Recursive (paragraph-aware) | 61 | ~350 chars | Better respects boundaries |
| **QA-aware** (one Q&A = one chunk) | **30** | **675 chars** | **Best semantically complete** |

QA-aware was selected because each chunk is a self-contained question-answer pair, giving the retriever a precise, complete unit of information per lookup.

---

## Key Design Decisions

**Why hybrid search?**
Dense retrieval misses exact keyword matches. BM25 misses semantic similarity. Hybrid with RRF captures both signals a chunk that ranks well in both lists gets boosted.

**Why RRF over score averaging?**
Dense scores (cosine similarity 0-1) and BM25 scores (term frequency 0-15) are on different scales. RRF uses rank position instead of raw scores, making fusion scale-invariant.

**Why LangGraph over plain LangChain?**
LangChain chains are linear. The agent needs conditional routing sometimes search KB, sometimes check order, sometimes escalate. LangGraph's state machine handles this cleanly with explicit edges and loop detection.

**Why LLM-as-judge over Ragas?**
Ragas had severe version compatibility issues with the LangChain/Groq stack. Custom LLM-as-judge gives the same metrics with full control over prompts, output format, and retry logic and produces per-question reasoning that's more useful for debugging.

---

## Running Evaluation

```bash
python eval/evaluate.py
```

Outputs per-question scores with reasoning:

```
[1/20] How do I request a refund?
  Faithful   : 1.0  - Every claim is directly supported by retrieved context
  Quality    : 1.0  - Answer directly addresses the question
  Recall     : 0.5  = Context contains partial information needed

AGGREGATE SCORES
  faithfulness      0.8750   ✓ Good    (scored 20/20)
  answer_quality    0.9250   ✓ Good    (scored 20/20)
  context_recall    0.7000   ~ Acceptable  (scored 20/20)
```

---

## What I'd Add Next

- **Semantic caching** - cache responses for queries with cosine similarity > 0.95 (estimated 30-40% cost reduction)
- **LangSmith tracing** - full trace logging for every agent run in production
- **Fine-tuning** - QLoRA fine-tune a small model (Qwen2.5-1.5B) on the support corpus via Unsloth
- **Pinecone/Qdrant** - swap ChromaDB for a managed vector store for horizontal scale
- **A/B prompt testing** - run two system prompt variants in parallel and compare eval scores

---

## License

MIT
