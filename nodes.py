"""
nodes.py

Core nodes of the LangGraph agent workflow:

* planner_node   – Decomposes the user request into an ordered list of steps.
* executor_node  – Executes the current step using the available tool set.
* evaluator_node – Judges whether the task has been completed satisfactorily.
* memory_node    – Updates short-term and long-term memory; triggers
                   summarisation when the context grows too large.
* summary_node   – Standalone summarisation node (called explicitly when
                   needed, e.g. after a long chain of exchanges).
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from memory import (
    load_from_long_term,
    maybe_summarise,
    save_to_long_term,
    search_long_term,
    summarise_messages,
)
from tools import ALL_TOOLS

load_dotenv()

_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "15"))

# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """You are a meticulous task planner.
Given the user's request and any relevant context, break the work down into a
numbered list of concrete, actionable steps. Each step should be a single,
self-contained action that can be executed independently.

Respond ONLY with a JSON object in this exact format:
{
  "steps": ["step 1 description", "step 2 description", ...]
}
"""


def planner_node(state: dict) -> dict:
    """Decompose the user task into an ordered list of steps.

    Reads the current ``messages`` and any retrieved ``context`` from long-term
    memory, then produces a ``plan`` (list of step strings) stored in the state.

    Args:
        state: Current graph state containing at minimum ``messages``.

    Returns:
        State update dict with ``plan`` and reset ``current_step_index``.
    """
    llm = ChatOpenAI(model=_MODEL, temperature=0)

    messages = state.get("messages", [])
    context = state.get("context", [])
    user_id = state.get("user_id", "default")

    # Pull relevant long-term memories to inform the plan.
    user_prefs = load_from_long_term(("users", user_id), "preferences")
    recent_knowledge = search_long_term(("users", user_id), "task")

    context_text = ""
    if user_prefs:
        context_text += f"\nUser preferences: {json.dumps(user_prefs)}"
    if recent_knowledge:
        context_text += f"\nRelevant knowledge: {json.dumps(recent_knowledge[:3])}"
    if context:
        context_text += f"\nAdditional context: {' '.join(str(c) for c in context)}"

    system_msg = SystemMessage(content=_PLANNER_SYSTEM + context_text)

    response = llm.invoke([system_msg] + messages)

    # Parse the plan from the JSON response.
    try:
        plan_data = json.loads(response.content)
        steps: list[str] = plan_data.get("steps", [])
    except (json.JSONDecodeError, AttributeError):
        # Fallback: treat the whole response as a single step.
        steps = [response.content]

    return {
        "plan": steps,
        "current_step_index": 0,
        "messages": messages + [AIMessage(content=f"Plan created: {steps}")],
    }


# ---------------------------------------------------------------------------
# Executor node
# ---------------------------------------------------------------------------

_EXECUTOR_SYSTEM = """You are a precise task executor.
Your job is to carry out ONE specific step using the tools available to you.
Be thorough but concise. Report exactly what you did and what you found.
"""


def executor_node(state: dict) -> dict:
    """Execute the current step in the plan using the ReAct agent pattern.

    Creates a short-lived ReAct agent bound to :data:`~tools.ALL_TOOLS` and
    runs it against the current step description.

    Args:
        state: Current graph state.  Must contain ``plan``,
               ``current_step_index``, and ``messages``.

    Returns:
        State update with updated ``messages``, incremented
        ``current_step_index``, and appended ``execution_results``.
    """
    messages = state.get("messages", [])
    plan = state.get("plan", [])
    step_index = state.get("current_step_index", 0)
    execution_results = state.get("execution_results", [])

    if step_index >= len(plan):
        return {"messages": messages, "current_step_index": step_index}

    current_step = plan[step_index]

    # Build a minimal ReAct agent for this step.
    llm = ChatOpenAI(model=_MODEL, temperature=0)
    agent = create_react_agent(llm, tools=ALL_TOOLS)

    step_input = {
        "messages": [
            SystemMessage(content=_EXECUTOR_SYSTEM),
            HumanMessage(content=f"Execute this step: {current_step}"),
        ]
    }

    result = agent.invoke(step_input)
    result_messages = result.get("messages", [])
    last_ai = next(
        (m for m in reversed(result_messages) if isinstance(m, AIMessage)), None
    )
    step_result = last_ai.content if last_ai else "Step completed."

    execution_results = execution_results + [
        {"step": step_index, "description": current_step, "result": step_result}
    ]

    updated_messages = messages + [
        AIMessage(content=f"Step {step_index + 1}: {current_step}\nResult: {step_result}")
    ]

    return {
        "messages": updated_messages,
        "current_step_index": step_index + 1,
        "execution_results": execution_results,
    }


# ---------------------------------------------------------------------------
# Evaluator node
# ---------------------------------------------------------------------------

_EVALUATOR_SYSTEM = """You are a strict task evaluator.
Given the original request, the plan, and the execution results so far, decide
whether the overall task has been completed satisfactorily.

Respond ONLY with a JSON object:
{
  "completed": true | false,
  "reason": "brief explanation",
  "next_action": "finish" | "continue" | "replan"
}
"""


def evaluator_node(state: dict) -> dict:
    """Assess whether the task is complete and decide the next action.

    Sets ``task_completed`` (bool) and ``next_action`` (one of
    ``"finish"``, ``"continue"``, ``"replan"``) in the returned state.

    Args:
        state: Current graph state.

    Returns:
        State update with evaluation outcome fields.
    """
    llm = ChatOpenAI(model=_MODEL, temperature=0)

    messages = state.get("messages", [])
    plan = state.get("plan", [])
    execution_results = state.get("execution_results", [])
    iteration_count = state.get("iteration_count", 0)

    # Hard stop to prevent infinite loops.
    if iteration_count >= MAX_ITERATIONS:
        return {
            "task_completed": True,
            "next_action": "finish",
            "evaluation_reason": "Maximum iteration limit reached.",
        }

    eval_prompt = (
        f"Original plan:\n{json.dumps(plan, indent=2)}\n\n"
        f"Execution results so far:\n{json.dumps(execution_results, indent=2)}\n\n"
        "Evaluate whether the task is fully complete."
    )

    system_msg = SystemMessage(content=_EVALUATOR_SYSTEM)
    eval_msg = HumanMessage(content=eval_prompt)

    response = llm.invoke([system_msg] + messages[-6:] + [eval_msg])

    try:
        eval_data = json.loads(response.content)
        completed: bool = eval_data.get("completed", False)
        reason: str = eval_data.get("reason", "")
        next_action: str = eval_data.get("next_action", "continue")
    except (json.JSONDecodeError, AttributeError):
        completed = False
        reason = "Could not parse evaluation response."
        next_action = "continue"

    return {
        "task_completed": completed,
        "next_action": next_action,
        "evaluation_reason": reason,
        "iteration_count": iteration_count + 1,
    }


# ---------------------------------------------------------------------------
# Memory node
# ---------------------------------------------------------------------------


def memory_node(state: dict) -> dict:
    """Update memory after each evaluation cycle.

    * Saves the latest execution results to long-term memory.
    * Applies summarisation to the message list when it grows too long.

    Args:
        state: Current graph state.

    Returns:
        State update with (potentially compressed) ``messages`` and updated
        ``memory_summary``.
    """
    messages = state.get("messages", [])
    execution_results = state.get("execution_results", [])
    user_id = state.get("user_id", "default")
    task_completed = state.get("task_completed", False)

    # Persist the latest results to long-term memory.
    if execution_results:
        save_to_long_term(
            ("users", user_id),
            f"task_results_{len(execution_results)}",
            {"results": execution_results, "completed": task_completed},
        )

    # Compress message history when it grows large.
    compressed = maybe_summarise(messages)

    memory_summary = state.get("memory_summary", "")
    if len(compressed) < len(messages):
        # A summary was generated; extract its text for the state.
        for msg in compressed:
            if isinstance(msg, SystemMessage) and "[Conversation summary" in msg.content:
                memory_summary = msg.content
                break

    return {"messages": compressed, "memory_summary": memory_summary}


# ---------------------------------------------------------------------------
# Summary node
# ---------------------------------------------------------------------------


def summary_node(state: dict) -> dict:
    """Force a full summarisation of the current message history.

    This node can be invoked explicitly (e.g. after a very long chain) to
    replace the entire history with a single dense summary message.

    Args:
        state: Current graph state.

    Returns:
        State update with a compressed single-message ``messages`` list.
    """
    messages = state.get("messages", [])
    if len(messages) < 2:
        return {}

    summary_text = summarise_messages(messages)
    summary_message = SystemMessage(
        content=f"[Full conversation summary]\n{summary_text}"
    )

    return {
        "messages": [summary_message],
        "memory_summary": summary_text,
    }
