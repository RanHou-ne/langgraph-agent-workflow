"""
tools.py

Simple tool set available to the Executor node.

Every tool is a plain Python function decorated with ``@tool`` from
LangChain.  Add domain-specific tools here; the Executor will receive them
automatically through the graph configuration.
"""

from __future__ import annotations

import datetime
import json
import math

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result as a string.

    Supports standard arithmetic operators (+, -, *, /), exponentiation (**),
    and common ``math`` module functions (e.g. ``math.sqrt(9)``).

    Args:
        expression: A Python-compatible numeric expression, e.g. ``"2 ** 10"``
                    or ``"math.sqrt(144)"``.

    Returns:
        The result as a string, or an error message if evaluation fails.
    """
    try:
        # Restrict evaluation to a safe subset of builtins.
        result = eval(expression, {"__builtins__": {}}, {"math": math})  # noqa: S307
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Error evaluating expression: {exc}"


@tool
def get_current_datetime() -> str:
    """Return the current UTC date and time in ISO-8601 format.

    Returns:
        Current UTC datetime string, e.g. ``"2024-01-15T10:30:00"``.
    """
    return datetime.datetime.utcnow().isoformat()


@tool
def format_json(data: str) -> str:
    """Pretty-print a JSON string.

    Args:
        data: A valid JSON string.

    Returns:
        Indented JSON string, or an error message if parsing fails.
    """
    try:
        parsed = json.loads(data)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"


@tool
def word_count(text: str) -> str:
    """Count the number of words in the given text.

    Args:
        text: Any string whose words should be counted.

    Returns:
        A human-readable count, e.g. ``"42 words"``.
    """
    count = len(text.split())
    return f"{count} words"


@tool
def search_memory(query: str, user_id: str = "default") -> str:
    """Search the long-term memory store for entries related to *query*.

    This tool allows the Executor to retrieve previously saved knowledge
    snippets or user preferences during task execution.

    Args:
        query:   Free-text search query.
        user_id: Identifier for the memory namespace (default ``"default"``).

    Returns:
        JSON-formatted list of matching memory entries, or a message
        indicating that nothing was found.
    """
    from memory import search_long_term  # local import to avoid circular deps

    results = search_long_term(("users", user_id), query)
    if not results:
        return "No relevant memories found."
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def save_memory(key: str, value: str, user_id: str = "default") -> str:
    """Save a key-value pair to the long-term memory store.

    Use this tool to persist any fact, preference, or result that should be
    available in future sessions.

    Args:
        key:     Unique identifier for this memory entry.
        value:   The information to store (string).
        user_id: Identifier for the memory namespace (default ``"default"``).

    Returns:
        Confirmation message.
    """
    from memory import save_to_long_term  # local import to avoid circular deps

    save_to_long_term(("users", user_id), key, {"content": value})
    return f"Saved memory '{key}' for user '{user_id}'."


# Collected list of all tools exported to the graph.
ALL_TOOLS = [
    calculator,
    get_current_datetime,
    format_json,
    word_count,
    search_memory,
    save_memory,
]
