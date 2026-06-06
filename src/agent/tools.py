# src/agent/tools.py

"""
TOOLS — what the agent can do
-------------------------------
Each tool is a Python function decorated with @tool.
The docstring is critical — the LLM reads it to decide WHEN to use the tool.
Write docstrings that describe the situation, not just the function.

We have 3 tools:
  1. search_knowledge_base  — answer questions from our RAG pipeline
  2. check_order_status     — look up a mock order by ID
  3. escalate_to_human      — log and confirm escalation to a human agent

CONCEPT: Function calling / Tool use
The LLM doesn't call Python directly. Instead:
  1. We send it a list of tool schemas (name + description + parameters as JSON Schema)
  2. It responds with a tool_call: {"name": "search_knowledge_base", "args": {"query": "..."}}
  3. We execute the Python function and send the result back
  4. The LLM reads the result and decides what to do next
"""

import json
import random
from datetime import datetime, timedelta
from langchain_core.tools import tool
from src.retrieval.reranker import rerank


# ── Tool 1: RAG Search ────────────────────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the customer support knowledge base to answer questions about:
    refunds, cancellations, orders, payments, account management,
    delivery, complaints, and general support policies.
    Use this tool whenever the customer asks a support question.
    Input should be the customer's question as a plain string.
    """
    results = rerank(query, top_k=3)

    if not results:
        return "No relevant information found in the knowledge base."

    # Format results for the LLM — include topic and content
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[Source {i} — {r['topic']}]\n{r['text']}"
        )

    return "\n\n---\n\n".join(formatted)


# ── Tool 2: Order Status ──────────────────────────────────────────────────

# Mock order database — in production this would be a real DB call
MOCK_ORDERS = {
    "ORD001": {"status": "Delivered",    "item": "Blue Sneakers",      "date": "2025-01-10", "carrier": "FedEx",  "tracking": "FX123456"},
    "ORD002": {"status": "In Transit",   "item": "Wireless Headphones","date": "2025-01-15", "carrier": "UPS",    "tracking": "UP789012"},
    "ORD003": {"status": "Processing",   "item": "Coffee Maker",       "date": "2025-01-16", "carrier": "TBD",    "tracking": "TBD"},
    "ORD004": {"status": "Cancelled",    "item": "Gaming Mouse",       "date": "2025-01-12", "carrier": "N/A",    "tracking": "N/A"},
    "ORD005": {"status": "Out for Delivery","item": "Running Shoes",   "date": "2025-01-17", "carrier": "USPS",   "tracking": "US345678"},
}


@tool
def check_order_status(order_id: str) -> str:
    """
    Check the current status of a customer's order.
    Use this tool when the customer provides an order ID and wants to know
    the status, delivery date, tracking number, or carrier information.
    Input should be the order ID (e.g. ORD001, ORD002).
    """
    order_id = order_id.strip().upper()
    order = MOCK_ORDERS.get(order_id)

    if not order:
        return (
            f"Order '{order_id}' not found. "
            f"Please verify the order ID and try again. "
            f"Valid format example: ORD001"
        )

    result = {
        "order_id":   order_id,
        "status":     order["status"],
        "item":       order["item"],
        "order_date": order["date"],
        "carrier":    order["carrier"],
        "tracking":   order["tracking"],
    }

    # Add estimated delivery for in-transit orders
    if order["status"] == "In Transit":
        eta = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        result["estimated_delivery"] = eta

    return json.dumps(result, indent=2)


# ── Tool 3: Escalate to Human ─────────────────────────────────────────────

import os

ESCALATION_LOG = "data/escalations.json"


@tool
def escalate_to_human(reason: str, customer_message: str) -> str:
    """
    Escalate the conversation to a human support agent.
    Use this tool when:
    - The customer is very frustrated or angry
    - The issue cannot be resolved with available information
    - The customer explicitly asks to speak to a human
    - The issue involves sensitive topics like fraud or legal matters
    - You have already tried to help but the customer is still unsatisfied
    Input: reason (why escalating), customer_message (their last message).
    """
    # Log the escalation
    escalation = {
        "timestamp":        datetime.now().isoformat(),
        "reason":           reason,
        "customer_message": customer_message,
        "ticket_id":        f"TKT{random.randint(10000, 99999)}",
        "status":           "pending"
    }

    # Load existing escalations
    os.makedirs("data", exist_ok=True)
    existing = []
    if os.path.exists(ESCALATION_LOG):
        with open(ESCALATION_LOG, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    existing.append(escalation)

    with open(ESCALATION_LOG, "w") as f:
        json.dump(existing, f, indent=2)

    return (
        f"Escalation logged successfully. "
        f"Ticket ID: {escalation['ticket_id']}. "
        f"A human agent will contact the customer within 2 business hours. "
        f"Reason recorded: {reason}"
    )


# Export all tools as a list for the agent
TOOLS = [search_knowledge_base, check_order_status, escalate_to_human]