# src/agent/graph.py

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import uuid

from src.agent.tools import TOOLS
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.memory import ShortTermMemory, summarize_conversation, retrieve_relevant_memories

load_dotenv()


# ── State ──────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── LLM ───────────────────────────────────────────────────────────────────

def get_llm():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1024,
    )
    return llm.bind_tools(TOOLS)


# ── Nodes ──────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    llm = get_llm()
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ── Graph ──────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


agent_graph = build_graph()


# ── Session manager — ties graph + memory together ─────────────────────────

class SupportSession:
    """
    One session = one customer conversation.
    Manages short-term memory and triggers long-term summarization.

    Usage:
        session = SupportSession()
        response = session.chat("How do I get a refund?")
        response = session.chat("What about my order ORD002?")  # has context
    """

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.memory = ShortTermMemory()
        print(f"Session started: {self.session_id}")

    def chat(self, user_message: str) -> dict:
        """Send a message and get a response, with full memory context."""

        # On first message, retrieve relevant past memories
        extra_context = ""
        if self.memory.turn_count == 0:
            memories = retrieve_relevant_memories(user_message)
            if memories:
                extra_context = (
                    "\n\nRELEVANT PAST INTERACTIONS:\n" +
                    "\n".join(f"- {m}" for m in memories)
                )

        # Build system prompt with optional memory context
        system_content = SYSTEM_PROMPT
        if extra_context:
            system_content += extra_context

        # Get conversation history from short-term memory
        history = self.memory.get_history()

        # Build LangGraph message list
        messages = [SystemMessage(content=system_content)]
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_message))

        # Run the graph
        result = agent_graph.invoke({"messages": messages})

        # Extract response and tool calls
        final_response = ""
        tool_calls_used = []

        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_used.extend([tc["name"] for tc in msg.tool_calls])
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content

        # Update short-term memory
        self.memory.add("user", user_message)
        self.memory.add("assistant", final_response)

        # Trigger long-term summarization if needed
        if self.memory.should_summarize():
            print(f"  Summarizing conversation for long-term memory...")
            summarize_conversation(self.memory.get_history(), self.session_id)

        return {
            "response":   final_response,
            "tool_calls": tool_calls_used,
            "session_id": self.session_id,
            "turn":       self.memory.turn_count,
        }