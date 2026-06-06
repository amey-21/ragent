# src/api/main.py

"""
FASTAPI SERVER
--------------
Endpoints:
  POST /chat          — main chat endpoint (streaming SSE)
  POST /chat/sync     — non-streaming version (easier to test with curl)
  GET  /health        — server health + stats
  DELETE /session     — clear a session's memory

CONCEPT: Async FastAPI
FastAPI is async by default. LLM calls are slow (1-5 seconds) so we
run them in a thread pool executor to avoid blocking the event loop.
If we blocked, one slow LLM call would freeze ALL other requests.

asyncio.to_thread() moves the blocking call to a thread while the
event loop stays free to handle other requests.
"""

import asyncio
import json
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Import our agent
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.agent.graph import SupportSession

app = FastAPI(
    title="Support Agent API",
    description="AI-powered customer support agent with RAG + memory",
    version="1.0.0"
)

# Allow all origins for local dev (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ────────────────────────────────────────────────
# session_id → SupportSession
# In production: use Redis so sessions survive restarts
SESSIONS: dict[str, SupportSession] = {}

# ── Request/Response schemas ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None   # if None, we create a new session

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[str]
    turn: int

class HealthResponse(BaseModel):
    status: str
    active_sessions: int
    total_turns: int
    uptime_seconds: float
    escalations_logged: int

# ── Startup time tracking ──────────────────────────────────────────────────
START_TIME = datetime.now()

# ── Helpers ────────────────────────────────────────────────────────────────

def get_or_create_session(session_id: str | None) -> SupportSession:
    """Get existing session or create a new one."""
    if session_id and session_id in SESSIONS:
        return SESSIONS[session_id]
    session = SupportSession(session_id=session_id)
    SESSIONS[session.session_id] = session
    return session


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """
    Non-streaming chat endpoint.
    Full response returned at once after LLM finishes.
    Use this for: testing, simple integrations, non-UI clients.
    """
    session = get_or_create_session(request.session_id)

    # Run in thread pool so we don't block the event loop
    result = await asyncio.to_thread(session.chat, request.message)

    return ChatResponse(
        response=result["response"],
        session_id=result["session_id"],
        tool_calls=result["tool_calls"],
        turn=result["turn"]
    )


@app.post("/chat")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    SSE format — each line the server sends looks like:
        data: {"type": "token", "content": "Hello"}\n\n

    The client reads these events and appends tokens to the UI.
    Final event sends tool_calls and session metadata.

    CONCEPT: Why SSE over WebSockets?
    SSE is simpler for one-directional streaming (server → client).
    WebSockets are bidirectional — overkill for chat responses.
    SSE works over plain HTTP/2, easier to deploy and debug.
    """
    session = get_or_create_session(request.session_id)

    async def event_generator():
        try:
            # Run agent in thread pool (blocking call)
            result = await asyncio.to_thread(session.chat, request.message)

            # Simulate token-by-token streaming
            # (Groq doesn't easily support streaming with tool calls,
            #  so we stream the final response word by word)
            words = result["response"].split()
            for i, word in enumerate(words):
                token = word if i == len(words) - 1 else word + " "
                event = json.dumps({"type": "token", "content": token})
                yield f"data: {event}\n\n"
                await asyncio.sleep(0.02)   # 20ms between tokens feels natural

            # Final event with metadata
            done_event = json.dumps({
                "type":       "done",
                "session_id": result["session_id"],
                "tool_calls": result["tool_calls"],
                "turn":       result["turn"]
            })
            yield f"data: {done_event}\n\n"

        except Exception as e:
            error_event = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — shows server stats. Use this in interviews to show observability."""
    uptime = (datetime.now() - START_TIME).total_seconds()
    total_turns = sum(s.memory.turn_count for s in SESSIONS.values())

    # Count escalations logged
    escalation_count = 0
    escalation_log = "data/escalations.json"
    if os.path.exists(escalation_log):
        with open(escalation_log) as f:
            try:
                escalation_count = len(json.load(f))
            except Exception:
                escalation_count = 0

    return HealthResponse(
        status="healthy",
        active_sessions=len(SESSIONS),
        total_turns=total_turns,
        uptime_seconds=round(uptime, 1),
        escalations_logged=escalation_count
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clear a session's memory. Useful for testing."""
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        return {"message": f"Session {session_id} cleared"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/")
async def root():
    return {
        "message": "Support Agent API is running",
        "docs": "/docs",
        "health": "/health"
    }


