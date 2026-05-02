"""
main.py

Entry point for the LangGraph agent workflow.

Usage
-----
Interactive REPL (single-session multi-turn conversation)::

    python main.py

Single-shot (non-interactive) invocation::

    python main.py --task "Calculate the square root of 144 and tell me today's date"

Resume a previous session::

    python main.py --thread-id my-thread-123

Options
-------
--task        Task string to execute (skip the interactive prompt).
--thread-id   Thread ID to resume or start (default: "default").
--user-id     User identifier for long-term memory (default: "default").
"""

from __future__ import annotations

import argparse
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()


def run_task(
    task: str,
    thread_id: str = "default",
    user_id: str = "default",
) -> str:
    """Execute a single task and return the final assistant message.

    Args:
        task:      Natural-language task description.
        thread_id: LangGraph thread ID used for checkpointing.
        user_id:   User identifier for long-term memory namespacing.

    Returns:
        The last AI response text from the completed workflow run.
    """
    from graph import graph  # defer import so env vars are loaded first

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "user_id": user_id,
        "plan": [],
        "current_step_index": 0,
        "execution_results": [],
        "task_completed": False,
        "next_action": "continue",
        "evaluation_reason": "",
        "iteration_count": 0,
        "memory_summary": "",
        "context": [],
    }

    final_state = graph.invoke(initial_state, config=config)

    messages = final_state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            return msg.content

    return "Task completed."


def interactive_loop(thread_id: str, user_id: str) -> None:
    """Run a multi-turn interactive conversation loop.

    Type ``quit`` or ``exit`` (or send EOF / Ctrl-C) to end the session.

    Args:
        thread_id: Thread ID that persists state across turns.
        user_id:   User identifier for long-term memory.
    """
    from graph import graph  # defer import so env vars are loaded first

    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n🤖  LangGraph Agent Workflow")
    print(f"    Thread ID : {thread_id}")
    print(f"    User ID   : {user_id}")
    print("    Type 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        print("Agent: (thinking...)\n")

        try:
            # Fetch the existing checkpoint for this thread (if any) to build
            # an incremental state update rather than starting from scratch.
            current = graph.get_state(config)
            if current and current.values:
                existing_messages = current.values.get("messages", [])
                state_input = {
                    **current.values,
                    "messages": existing_messages + [HumanMessage(content=user_input)],
                    "plan": [],
                    "current_step_index": 0,
                    "execution_results": [],
                    "task_completed": False,
                    "next_action": "continue",
                    "evaluation_reason": "",
                    "iteration_count": 0,
                }
            else:
                state_input = {
                    "messages": [HumanMessage(content=user_input)],
                    "user_id": user_id,
                    "plan": [],
                    "current_step_index": 0,
                    "execution_results": [],
                    "task_completed": False,
                    "next_action": "continue",
                    "evaluation_reason": "",
                    "iteration_count": 0,
                    "memory_summary": "",
                    "context": [],
                }

            final_state = graph.invoke(state_input, config=config)

            messages = final_state.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    print(f"Agent: {msg.content}\n")
                    break

        except Exception as exc:  # noqa: BLE001
            print(f"Error during execution: {exc}\n")


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate execution mode."""
    parser = argparse.ArgumentParser(
        description="LangGraph multi-agent workflow runner"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Task to execute (non-interactive mode)",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="",
        help="Thread ID for checkpointing (default: random UUID)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="default",
        help="User ID for long-term memory namespacing",
    )

    args = parser.parse_args()

    thread_id = args.thread_id or str(uuid.uuid4())
    user_id = args.user_id

    if args.task:
        result = run_task(args.task, thread_id=thread_id, user_id=user_id)
        print(result)
    else:
        interactive_loop(thread_id=thread_id, user_id=user_id)


if __name__ == "__main__":
    main()
