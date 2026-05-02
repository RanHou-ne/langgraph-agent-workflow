"""
graph.py

LangGraph workflow definition for the multi-agent task execution system.

Flow
----
User Input
    │
    ▼
planner ──► executor ──► evaluator ──► memory
    ▲                        │
    │    (replan)             │ (finish / continue)
    └────────────────────────┘
                             │ finish
                             ▼
                            END

Routing logic
─────────────
After the evaluator:
  * "finish"   → memory → END
  * "continue" → memory → executor  (next step in existing plan)
  * "replan"   → memory → planner   (rebuild the plan from scratch)

When all steps in the plan have been executed the graph routes to the
evaluator regardless, letting it decide whether to replan or finish.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from memory import checkpointer
from nodes import (
    evaluator_node,
    executor_node,
    memory_node,
    planner_node,
    summary_node,
)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Typed state shared across all nodes."""

    # Conversation history (LangChain message objects).
    messages: list

    # User identifier used to namespace long-term memory.
    user_id: str

    # Ordered list of step descriptions produced by the planner.
    plan: list[str]

    # Index into ``plan`` pointing at the next step to execute.
    current_step_index: int

    # Accumulated results from the executor, one entry per executed step.
    execution_results: list[dict]

    # Whether the evaluator has declared the task complete.
    task_completed: bool

    # Routing signal from the evaluator: "finish" | "continue" | "replan".
    next_action: str

    # Human-readable explanation from the last evaluator pass.
    evaluation_reason: str

    # Number of evaluator cycles completed (guard against infinite loops).
    iteration_count: int

    # Latest compressed memory summary text.
    memory_summary: str

    # Extra context snippets fetched from long-term memory for the planner.
    context: list[str]


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def route_after_evaluator(state: AgentState) -> str:
    """Decide the next node after evaluation.

    Returns:
        One of ``"memory_finish"``, ``"memory_continue"``, or
        ``"memory_replan"``.  These map to the same *memory* node but the
        downstream conditional edge uses the stored ``next_action`` value.
    """
    next_action = state.get("next_action", "continue")

    # If all plan steps are done but evaluator says "continue", treat as finish.
    plan = state.get("plan", [])
    step_index = state.get("current_step_index", 0)
    if step_index >= len(plan) and next_action == "continue":
        next_action = "finish"

    if next_action == "finish":
        return "memory_finish"
    if next_action == "replan":
        return "memory_replan"
    return "memory_continue"


def route_after_memory(state: AgentState) -> str:
    """Decide the next node after memory update.

    Returns:
        ``"planner"``, ``"executor"``, or ``END``.
    """
    next_action = state.get("next_action", "continue")
    task_completed = state.get("task_completed", False)

    if task_completed or next_action == "finish":
        return END
    if next_action == "replan":
        return "planner"
    return "executor"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """Construct and compile the agent workflow graph.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for invocation.
    """
    workflow = StateGraph(AgentState)

    # Register nodes.
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("summary", summary_node)

    # Entry point.
    workflow.set_entry_point("planner")

    # Fixed edges.
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "evaluator")

    # Conditional routing after the evaluator.
    workflow.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {
            "memory_finish": "memory",
            "memory_continue": "memory",
            "memory_replan": "memory",
        },
    )

    # Conditional routing after memory update.
    workflow.add_conditional_edges(
        "memory",
        route_after_memory,
        {
            "planner": "planner",
            "executor": "executor",
            END: END,
        },
    )

    # The summary node is a dead-end utility node that feeds back to memory.
    workflow.add_edge("summary", "memory")

    # Compile with the checkpointer to enable state persistence and resumption.
    return workflow.compile(checkpointer=checkpointer)


# Module-level graph instance for easy import.
graph = build_graph()
