"""
LangGraph 工作流构建模块。
构建 "Planner → Executor → Evaluator → Memory Updater" 的多节点协同工作流。
"""
from langgraph.graph import StateGraph, START, END
from nodes import AgentState, planner, executor, evaluator, update_memory
import logging

logger = logging.getLogger(__name__)


def build_graph(checkpointer=None, store=None):
    """
    构建并编译 Agent 工作流图。
    
    工作流结构：
        START → Planner → Executor → Evaluator
                                        ├── (还有步骤) → Executor
                                        └── (全部完成) → UpdateMemory → END
    
    Args:
        checkpointer: 短期记忆组件（MemorySaver），用于对话状态持久化
        store: 长期记忆组件（InMemoryStore），用于跨会话事实存储
    
    Returns:
        编译后的 LangGraph 可执行图
    """
    logger.info("🔧 构建 Agent 工作流图...")
    
    # 创建状态图
    workflow = StateGraph(AgentState)
    
    # 注册节点
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("evaluator", evaluator)
    workflow.add_node("update_memory", update_memory)
    
    # 定义边：START → Planner
    workflow.add_edge(START, "planner")
    
    # 定义边：Planner → Executor
    workflow.add_edge("planner", "executor")
    
    # 定义边：Executor → Evaluator
    workflow.add_edge("executor", "evaluator")
    
    # 条件边：Evaluator 根据执行状态决定下一步
    def should_continue(state: AgentState) -> str:
        """
        判断是否还有未执行的步骤：
        - 如果 cur_step < len(plan)，还有步骤需要执行 → 回到 executor
        - 否则所有步骤完成 → 进入 update_memory
        """
        if state["cur_step"] < len(state["plan"]):
            logger.info(f"  ↩️ 还有 {len(state['plan']) - state['cur_step']} 步待执行，返回 Executor")
            return "executor"
        else:
            logger.info("  ✅ 所有步骤完成，进入记忆更新")
            return "update_memory"
    
    workflow.add_conditional_edges(
        "evaluator",
        should_continue,
        {
            "executor": "executor",
            "update_memory": "update_memory",
        }
    )
    
    # 定义边：UpdateMemory → END
    workflow.add_edge("update_memory", END)
    
    # 编译图，绑定 checkpointer 和 store
    graph = workflow.compile(
        checkpointer=checkpointer,
        store=store,
    )
    
    logger.info("🔧 工作流图构建完成")
    return graph