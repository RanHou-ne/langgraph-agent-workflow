# langgraph-agent-workflow

A multi-agent task execution system built on [LangGraph](https://github.com/langchain-ai/langgraph), designed for multi-turn conversations and long-horizon tasks.

Three core nodes—**Planner**, **Executor**, and **Memory**—work together in a feedback loop:

```
User Input → Task Decomposition → Agent Execution → Result Evaluation → Memory Update
                ↑                                           │
                └───────────── replan if needed ────────────┘
```

---

## Repository Structure

```
langgraph-agent-workflow/
├── README.md          ← this file
├── requirements.txt   ← Python dependencies
├── .env.example       ← environment variable template
├── main.py            ← CLI entry point (interactive REPL + single-shot mode)
├── graph.py           ← LangGraph workflow definition & state schema
├── nodes.py           ← Planner, Executor, Evaluator, Memory, Summary nodes
├── memory.py          ← Three-tier memory system
└── tools.py           ← Tools available to the Executor
```

---

## Quick Start

### 1 — Clone & install

```bash
git clone https://github.com/RanHou-ne/langgraph-agent-workflow.git
cd langgraph-agent-workflow
pip install -r requirements.txt
```

### 2 — Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (and optionally other values)
```

### 3 — Run

**Interactive multi-turn REPL:**

```bash
python main.py
```

**Single-shot task:**

```bash
python main.py --task "Calculate the square root of 144 and tell me today's date"
```

**Resume a previous session:**

```bash
python main.py --thread-id my-thread-123
```

---

## Node Design

### Planner

Reads the user request plus any relevant long-term memories, then produces an ordered JSON list of concrete, actionable steps.  The plan is stored in the shared state and drives the Executor.

### Executor

Picks up the next step from the plan and runs it using a short-lived [ReAct](https://arxiv.org/abs/2210.03629) agent bound to the tools in `tools.py`.  Appends the result to `execution_results` in the state.

### Evaluator

After each execution step, judges whether the overall task is complete.  Returns one of three routing signals:

| Signal     | Meaning                                    |
|------------|--------------------------------------------|
| `finish`   | Task is done — route to memory then END    |
| `continue` | More plan steps remain — keep executing    |
| `replan`   | Results are unsatisfactory — rebuild plan  |

### Memory

Updates long-term memory with the latest results and applies summarisation when the message list grows beyond `MAX_SHORT_TERM_MESSAGES`.

### Summary (utility node)

Can be invoked explicitly to compress the *entire* history into a single dense summary message, freeing up context window space for long-running tasks.

---

## Three-Tier Memory System

| Tier       | Implementation                  | Purpose                                                          |
|------------|---------------------------------|------------------------------------------------------------------|
| Short-term | `MemorySaver` (Checkpointer)    | Persists full conversation state; enables in-place resumption    |
| Long-term  | `InMemoryStore`                 | Cross-session knowledge snippets & user preferences              |
| Summary    | LLM-powered compression         | Replaces old messages with a dense summary to extend the window  |

### Short-term: Checkpointer

Every state transition is checkpointed via LangGraph's `MemorySaver`.  Swap it for `SqliteSaver` or `PostgresSaver` for durability across process restarts:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
```

### Long-term: Store

Key-value store scoped by `(namespace, key)`.  Swap `InMemoryStore` for a vector store or Redis backend in production:

```python
from memory import save_to_long_term, load_from_long_term, search_long_term

save_to_long_term(("users", "alice"), "preferences", {"language": "zh"})
result = load_from_long_term(("users", "alice"), "preferences")
```

### Summary: Compression

When `len(messages) > MAX_SHORT_TERM_MESSAGES` the oldest half of the history is compressed into a single `SystemMessage` summary, keeping the most recent exchanges intact.  The compression is handled by the LLM and controlled via `memory.py::maybe_summarise()`.

---

## Configuration

All settings are read from environment variables (`.env` file):

| Variable                 | Default        | Description                                          |
|--------------------------|----------------|------------------------------------------------------|
| `OPENAI_API_KEY`         | —              | **Required.** OpenAI API key                         |
| `OPENAI_MODEL`           | `gpt-4o-mini`  | Chat model to use                                    |
| `MAX_SHORT_TERM_MESSAGES`| `20`           | Message count before triggering summarisation        |
| `MAX_CONTEXT_TOKENS`     | `8000`         | Token budget hint (informational)                    |
| `CHECKPOINT_DB_PATH`     | `./checkpoints.db` | Path for a SQLite checkpointer (if used)         |
| `MAX_ITERATIONS`         | `15`           | Hard cap on evaluator cycles to prevent infinite loops |

---

## Available Tools

| Tool                  | Description                                              |
|-----------------------|----------------------------------------------------------|
| `calculator`          | Evaluate arithmetic / math expressions                   |
| `get_current_datetime`| Return the current UTC date-time                         |
| `format_json`         | Pretty-print a JSON string                               |
| `word_count`          | Count words in a string                                  |
| `search_memory`       | Search long-term memory store                            |
| `save_memory`         | Persist a key-value entry to long-term memory            |

Add custom tools in `tools.py` and include them in `ALL_TOOLS`; they will be picked up automatically by the Executor.

---

## Extending the Workflow

* **Add a new node:** implement a function `my_node(state: dict) -> dict` in `nodes.py`, register it in `graph.py` with `workflow.add_node("my_node", my_node)`, and wire it with edges or conditional edges.
* **Swap the LLM:** change `OPENAI_MODEL` in `.env` or replace `ChatOpenAI` with any LangChain-compatible chat model.
* **Persistent checkpointing:** replace `MemorySaver` in `memory.py` with `SqliteSaver` or `PostgresSaver`.
* **Semantic long-term memory:** replace `InMemoryStore` with a vector-store backed implementation for similarity search.
