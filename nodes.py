"""
多节点协同模块：Planner + Executor + Evaluator + Memory Updater
实现 "用户输入 → 任务拆解 → Agent执行 → 结果评估 → 记忆更新" 闭环。
"""
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import TypedDict
import logging

from llm_provider import get_llm

from memory import (
    generate_summary,
    extract_fact,
    put_long_term_memory,
    get_long_term_memories,
    get_context_from_memory,
    trim_messages,
)
from tools import ALL_TOOLS, TOOL_MAP, execute_tool

logger = logging.getLogger(__name__)


# ─────────────────────────── 状态定义 ───────────────────────────

class AgentState(MessagesState):
    """Agent 工作流状态，继承 MessagesState 并增加任务专用字段。"""
    summary: str          # 对话摘要（摘要记忆）
    task: str             # 用户原始任务
    plan: list[str]       # 拆解后的执行计划
    cur_step: int         # 当前执行步骤索引
    final_answer: str     # 最终回答
    step_results: list[str]  # 每步执行结果
    iteration: int        # 当前迭代轮次


# ─────────────────────────── 工具调用辅助 ───────────────────────────

def _parse_tool_call(step_text: str) -> tuple[str, dict]:
    """
    从步骤描述中解析应该调用的工具和参数。
    使用 LLM 判断应该调用哪个工具。
    """
    model = get_llm(temperature=0)
    
    tool_descriptions = "\n".join([
        f"- {t.name}: {t.description}" for t in ALL_TOOLS
    ])
    
    prompt = f"""根据以下步骤描述，判断应该调用哪个工具以及参数。

可用工具：
{tool_descriptions}

步骤描述：{step_text}

请严格返回 JSON 格式：{{"tool_name": "工具名", "args": {{"参数名": "参数值"}}}}
如果不需要调用任何工具，返回：{{"tool_name": null, "args": {{}}}}
只返回 JSON，不要其他内容。"""

    try:
        response = model.invoke(prompt)
        import json
        # 提取 JSON 部分
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        tool_name = result.get("tool_name")
        args = result.get("args", {})
        return tool_name, args
    except Exception as e:
        logger.warning(f"解析工具调用失败: {e}")
        return None, {}


# ─────────────────────────── 节点实现 ───────────────────────────

def planner(state: AgentState, config) -> dict:
    """
    规划器节点：拆解用户任务为可执行的步骤计划。
    结合长期记忆和对话摘要生成个性化计划。
    """
    logger.info("📋 [Planner] 开始任务规划...")
    
    store = config.get("configurable", {}).get("store")
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    
    # 获取上下文信息
    context = get_context_from_memory(
        store, user_id,
        state.get("summary", ""),
        state["task"]
    )
    
    system_prompt = f"""你是一个智能任务规划器。你的职责是将用户的任务拆解为清晰、可执行的步骤。

{context}

规划规则：
1. 将任务拆解为 2~5 个具体步骤
2. 每个步骤应该是独立可执行的
3. 步骤之间要有逻辑顺序
4. 考虑用户的偏好和历史记忆
5. 每行一个步骤，不要编号（系统会自动编号）
6. 步骤描述要具体，便于执行器理解应该调用什么工具

示例输出：
查询北京今天的天气
搜索北京评分最高的中餐厅
推荐最近上映的热门电影"""

    model = get_llm(temperature=0)
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"请为以下任务制定执行计划：\n{state['task']}")
    ])
    
    # 解析计划
    plan = [
        step.strip().lstrip("0123456789.、）)")
        for step in response.content.strip().split("\n")
        if step.strip() and len(step.strip()) > 2
    ]
    
    logger.info(f"📋 [Planner] 生成 {len(plan)} 步计划:")
    for i, step in enumerate(plan, 1):
        logger.info(f"  {i}. {step}")
    
    return {
        "plan": plan,
        "cur_step": 0,
        "step_results": [],
        "iteration": state.get("iteration", 0) + 1,
    }


def executor(state: AgentState, config) -> dict:
    """
    执行器节点：执行当前步骤，调用合适的工具获取结果。
    """
    idx = state["cur_step"]
    step = state["plan"][idx]
    
    logger.info(f"⚡ [Executor] 执行步骤 {idx + 1}/{len(state['plan'])}: {step}")
    
    # 解析并调用工具
    tool_name, args = _parse_tool_call(step)
    
    if tool_name and tool_name in TOOL_MAP:
        result = execute_tool(tool_name, **args)
        logger.info(f"🔧 [Executor] 调用工具 {tool_name}，参数: {args}")
    else:
        # 如果没有匹配的工具，使用 LLM 直接回答
        model = get_llm(temperature=0)
        response = model.invoke([
            SystemMessage(content="你是一个智能助手。请根据以下步骤描述，提供简洁有用的回答。"),
            HumanMessage(content=f"步骤：{step}")
        ])
        result = response.content.strip()
        logger.info(f"💬 [Executor] 使用 LLM 直接回答")
    
    # 构建执行消息
    msg = AIMessage(content=f"📌 步骤 {idx + 1}: {step}\n✅ 执行结果: {result}")
    
    # 记录步骤结果
    step_results = state.get("step_results", [])
    step_results.append(f"步骤{idx + 1}({step}): {result}")
    
    logger.info(f"⚡ [Executor] 步骤 {idx + 1} 执行完成")
    
    return {
        "messages": [msg],
        "cur_step": idx + 1,
        "step_results": step_results,
    }


def evaluator(state: AgentState, config) -> dict:
    """
    评估器节点：评估执行结果，决定是否继续执行或生成最终回答。
    """
    logger.info(f"🔍 [Evaluator] 评估执行状态: {state['cur_step']}/{len(state['plan'])} 步完成")
    
    # 如果还有未执行的步骤，继续
    if state["cur_step"] < len(state["plan"]):
        logger.info("🔍 [Evaluator] 还有未完成的步骤，继续执行...")
        return {}
    
    # 所有步骤完成，生成最终回答
    logger.info("🔍 [Evaluator] 所有步骤完成，生成最终回答...")
    
    model = get_llm(temperature=0)
    
    # 汇总所有步骤结果
    results_summary = "\n".join(state.get("step_results", []))
    
    final_prompt = f"""你是一个智能助手。请根据以下任务执行结果，给用户一个完整、自然、有条理的中文回答。

用户任务：{state['task']}

执行结果汇总：
{results_summary}

要求：
1. 回答要完整、自然、有条理
2. 将各步骤的结果整合为连贯的回答
3. 如果有推荐或建议，要给出理由
4. 语气友好、专业
5. 使用适当的 emoji 增强可读性"""

    response = model.invoke([HumanMessage(content=final_prompt)])
    final_answer = response.content.strip()
    
    logger.info(f"✅ [Evaluator] 最终回答生成完成，长度: {len(final_answer)} 字符")
    
    return {"final_answer": final_answer}


def update_memory(state: AgentState, config) -> dict:
    """
    记忆更新节点：更新摘要记忆和长期记忆，裁剪消息列表。
    实现三级记忆的闭环管理。
    """
    logger.info("🧠 [Memory] 开始更新记忆...")
    
    msgs = state["messages"]
    previous_summary = state.get("summary", "")
    store = config.get("configurable", {}).get("store")
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    
    # 1. 更新摘要记忆
    new_summary = generate_summary(msgs, previous_summary)
    logger.info(f"📝 [Memory] 摘要更新完成")
    
    # 2. 提取并存储长期记忆
    fact = extract_fact(msgs)
    if store and fact:
        put_long_term_memory(store, user_id, fact)
        logger.info(f"🧠 [Memory] 长期记忆已更新: {fact[:50]}...")
    
    # 3. 消息裁剪：保留最近消息 + 摘要
    trimmed = trim_messages(msgs, new_summary, keep_recent=4)
    
    logger.info("🧠 [Memory] 记忆更新完成")
    
    return {
        "summary": new_summary,
        "messages": trimmed,
    }