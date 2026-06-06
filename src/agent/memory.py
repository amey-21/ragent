# src/agent/memory.py

"""
MEMORY ARCHITECTURE
--------------------
Two-tier memory system:

TIER 1 — Short-term (in-memory, per session)
  The last 10 messages are kept in a list and passed with every request.
  This handles: follow-up questions, references to earlier context.
  Cost: a few hundred tokens per request.

TIER 2 — Long-term (ChromaDB, persistent across sessions)
  After every 6 user messages, we summarize the conversation with an LLM
  and store the summary as a vector in ChromaDB under a separate collection.
  At session start, we retrieve the top-2 relevant memories and inject
  them into the system prompt.
  Cost: one extra LLM call per 6 turns + one vector search per session.

INTERVIEW ANSWER:
"I implemented a two-tier memory system. Short-term kept the last 10
messages in the session state for follow-up handling. Long-term used
ChromaDB to store conversation summaries — after every 6 turns, the
LLM summarized what happened, and that summary was retrieved by
semantic similarity at the start of future sessions. This let the
bot remember that a user had a dispute with order ORD002 even a week later."
"""

import os
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import chromadb
from sentence_transformers import SentenceTransformer
from src.ingestion.embedder import EMBEDDING_MODEL, CHROMA_DIR

# Separate ChromaDB collection just for memory
MEMORY_COLLECTION = "agent_memory"
MEMORY_SUMMARY_EVERY_N = 6      # summarize every 6 user turns
SHORT_TERM_WINDOW = 10          # keep last 10 messages in context

# Module-level caches
_memory_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _memory_model
    if _memory_model is None:
        _memory_model = SentenceTransformer(EMBEDDING_MODEL)
    return _memory_model


def _get_memory_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=MEMORY_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


# ── Short-term memory ──────────────────────────────────────────────────────

class ShortTermMemory:
    """
    Keeps a sliding window of the last N messages.
    Stored in-memory (lost on restart — that's fine for short-term).
    """

    def __init__(self, window_size: int = SHORT_TERM_WINDOW):
        self.window_size = window_size
        self.messages: list[dict] = []         # {"role": ..., "content": ...}
        self.turn_count: int = 0

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if role == "user":
            self.turn_count += 1
        # Keep only the last N messages
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    def get_history(self) -> list[dict]:
        return self.messages.copy()

    def should_summarize(self) -> bool:
        """True every MEMORY_SUMMARY_EVERY_N user turns."""
        return self.turn_count > 0 and self.turn_count % MEMORY_SUMMARY_EVERY_N == 0


# ── Long-term memory ───────────────────────────────────────────────────────

def summarize_conversation(messages: list[dict], session_id: str) -> str:
    """
    Use LLM to summarize a conversation and store in ChromaDB.

    The summary is written to be useful for future retrieval:
    it captures key facts (order IDs, issues, resolutions) rather
    than being a generic summary.
    """
    if not messages:
        return ""

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=300)

    # Format conversation for summarization
    convo_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

    summary_prompt = f"""Summarize this customer support conversation in 2-3 sentences.
Focus on: what the customer needed, what actions were taken, and the outcome.
Include any specific details like order IDs, refund amounts, or ticket numbers.

Conversation:
{convo_text}

Summary:"""

    response = llm.invoke([HumanMessage(content=summary_prompt)])
    summary = response.content.strip()

    # Store in ChromaDB
    _store_memory(summary, session_id, messages)

    return summary


def _store_memory(summary: str, session_id: str, messages: list[dict]):
    """Embed the summary and store in the memory collection."""
    collection = _get_memory_collection()
    model = _get_model()

    embedding = model.encode(summary, convert_to_numpy=True).tolist()
    memory_id = f"memory_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    collection.upsert(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[summary],
        metadatas=[{
            "session_id":  session_id,
            "timestamp":   datetime.now().isoformat(),
            "turn_count":  len([m for m in messages if m["role"] == "user"])
        }]
    )
    print(f"  ✓ Memory stored: {memory_id}")


def retrieve_relevant_memories(query: str, top_k: int = 2) -> list[str]:
    """
    At the start of a session, retrieve past memories relevant to
    the current topic. Injected into the system prompt.
    """
    collection = _get_memory_collection()

    if collection.count() == 0:
        return []

    model = _get_model()
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    memories = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        similarity = 1 - dist
        if similarity > 0.3:        # only include relevant memories
            memories.append(doc)

    return memories