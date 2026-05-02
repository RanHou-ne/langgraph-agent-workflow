"""
memory.py

Three-tier memory system for the LangGraph agent workflow:

1. Short-term memory  – LangGraph Checkpointer (SQLite-backed) persists
   conversation state so any interrupted turn can be resumed in-place.

2. Long-term memory   – InMemoryStore (swap for a persistent backend in
   production) stores cross-session knowledge snippets and user preferences
   that are retrieved automatically when planning a new task.

3. Summary memory     – A dedicated summarisation pass compresses verbose
   history into a single high-density message, replacing the raw messages
   that have already served their purpose.  This keeps the effective context
   window wide even for very long conversations.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_SHORT_TERM_MESSAGES: int = int(os.getenv("MAX_SHORT_TERM_MESSAGES", "20"))
MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))
_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Singletons – shared across the whole process
# ---------------------------------------------------------------------------

# Short-term: in-process checkpointer (replace with SqliteSaver for
# durability across process restarts).
checkpointer = MemorySaver()

# Long-term: in-memory store for cross-session knowledge and preferences.
long_term_store = InMemoryStore()


# ---------------------------------------------------------------------------
# Long-term memory helpers
# ---------------------------------------------------------------------------


def save_to_long_term(namespace: tuple[str, ...], key: str, value: Any) -> None:
    """Persist a piece of information in the long-term store.

    Args:
        namespace: Tuple that scopes the entry, e.g. ``("user", "alice")``.
        key:       Unique identifier within the namespace.
        value:     Arbitrary JSON-serialisable payload.
    """
    long_term_store.put(namespace, key, value)


def load_from_long_term(namespace: tuple[str, ...], key: str) -> Any | None:
    """Retrieve a single entry from the long-term store.

    Returns ``None`` when the key does not exist.
    """
    item = long_term_store.get(namespace, key)
    return item.value if item else None


def search_long_term(namespace: tuple[str, ...], query: str) -> list[dict]:
    """Return all entries in *namespace* whose value matches *query*.

    The built-in :class:`InMemoryStore` does exact-string matching on
    serialised values.  Replace with a vector-store backed implementation for
    semantic search in production.

    Returns a list of ``{"key": ..., "value": ...}`` dicts.
    """
    results = long_term_store.search(namespace, query=query)
    return [{"key": item.key, "value": item.value} for item in results]


# ---------------------------------------------------------------------------
# Summarisation helper
# ---------------------------------------------------------------------------


def summarise_messages(messages: list) -> str:
    """Compress a list of messages into a single dense summary string.

    The summary is intended to replace the original messages in the state so
    that older context occupies far fewer tokens while retaining the key facts.

    Args:
        messages: A sequence of LangChain message objects.

    Returns:
        A multi-sentence summary produced by the LLM.
    """
    llm = ChatOpenAI(model=_MODEL, temperature=0)

    formatted: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
        elif isinstance(msg, SystemMessage):
            formatted.append(f"System: {msg.content}")
        else:
            formatted.append(f"[{type(msg).__name__}]: {msg.content}")

    conversation_text = "\n".join(formatted)

    prompt = (
        "Below is a conversation excerpt. "
        "Write a concise summary (≤150 words) that preserves every important "
        "decision, constraint, and intermediate result so the conversation can "
        "continue without the original messages.\n\n"
        f"{conversation_text}"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def maybe_summarise(messages: list) -> list:
    """Return a (possibly compressed) message list.

    If the number of messages exceeds ``MAX_SHORT_TERM_MESSAGES`` the oldest
    messages (everything except the most recent half) are compressed into a
    single :class:`SystemMessage` summary, reducing token consumption while
    keeping the latest exchanges intact.

    Args:
        messages: Current message list from the graph state.

    Returns:
        Potentially shortened message list with a prepended summary message.
    """
    if len(messages) <= MAX_SHORT_TERM_MESSAGES:
        return messages

    half = len(messages) // 2
    old_messages = messages[:half]
    recent_messages = messages[half:]

    summary_text = summarise_messages(old_messages)
    summary_message = SystemMessage(
        content=f"[Conversation summary – replaces earlier messages]\n{summary_text}"
    )

    return [summary_message] + recent_messages
