"""
三级记忆管理模块：
  1. 短期记忆（Checkpointer）- 对话级别的状态持久化，由 LangGraph 内置支持
  2. 长期记忆（Store）- 跨会话的事实/偏好存储
  3. 摘要记忆（Summary）- 对话摘要压缩，减少上下文窗口占用
"""
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.store.base import BaseStore
import uuid
import logging

from llm_provider import get_llm

logger = logging.getLogger(__name__)

# ─────────────────────────── Prompt 模板 ───────────────────────────

SUMMARY_PROMPT = """你是一个对话摘要助手。请根据下面的对话历史和已有摘要，生成一段新的简洁摘要。

要求：
1. 涵盖关键事实、决策和用户偏好
2. 保留重要的上下文信息
3. 摘要应控制在 200 字以内
4. 只返回摘要文本，不要额外解释

已有摘要：{previous_summary}

对话历史：
{history}

新摘要："""

EXTRACT_FACT_PROMPT = """你是一个信息提取专家。从以下对话中提取可以存入长期记忆的关键信息。

提取规则：
1. 优先提取用户偏好（如口味、兴趣、习惯）
2. 其次提取重要事实（如地点、时间、决策）
3. 每次只提取一个最重要的事实
4. 用简洁的一句话表述
5. 如果没有值得记忆的信息，返回"无"

对话：
{text}

提取的事实："""

CONTEXT_RECALL_PROMPT = """根据以下长期记忆和对话摘要，为当前任务提供相关的上下文信息。

长期记忆：
{memories}

对话摘要：{summary}

当前任务：{task}

请用简洁的几句话总结与当前任务最相关的上下文信息："""


# ─────────────────────────── 摘要记忆 ───────────────────────────

def generate_summary(messages, previous_summary: str = "") -> str:
    """
    生成对话摘要，用于压缩上下文窗口。
    
    Args:
        messages: 对话消息列表
        previous_summary: 已有的摘要文本
    
    Returns:
        新的摘要文本
    """
    try:
        model = get_llm(temperature=0)
        history = "\n".join([
            f"{msg.type}: {msg.content[:500]}"  # 限制单条消息长度
            for msg in messages
            if hasattr(msg, 'content') and msg.content
        ])
        
        if not history.strip():
            return previous_summary or "暂无对话记录。"
        
        prompt = SUMMARY_PROMPT.format(
            previous_summary=previous_summary or "无",
            history=history
        )
        response = model.invoke(prompt)
        new_summary = response.content.strip()
        
        logger.info(f"生成摘要成功，长度: {len(new_summary)} 字符")
        return new_summary
        
    except Exception as e:
        logger.error(f"生成摘要失败: {e}")
        return previous_summary or "摘要生成失败。"


# ─────────────────────────── 长期记忆 ───────────────────────────

def extract_fact(messages) -> str | None:
    """
    从对话中提取可存储的长期记忆事实。
    
    Args:
        messages: 对话消息列表
    
    Returns:
        提取的事实字符串，如果没有值得记忆的内容则返回 None
    """
    try:
        model = get_llm(temperature=0)
        # 只看最近 8 条消息，避免上下文过长
        recent_msgs = messages[-8:] if len(messages) > 8 else messages
        text = "\n".join([
            f"{msg.type}: {msg.content[:300]}"
            for msg in recent_msgs
            if hasattr(msg, 'content') and msg.content
        ])
        
        if not text.strip():
            return None
        
        prompt = EXTRACT_FACT_PROMPT.format(text=text)
        fact = model.invoke(prompt).content.strip()
        
        # 过滤无效结果
        if not fact or fact in ("无", "无。", "没有", "暂无"):
            return None
        
        logger.info(f"提取事实: {fact[:50]}...")
        return fact
        
    except Exception as e:
        logger.error(f"提取事实失败: {e}")
        return None


def put_long_term_memory(store: BaseStore, user_id: str, fact: str, category: str = "general"):
    """
    将事实存入长期记忆存储。
    
    Args:
        store: LangGraph BaseStore 实例
        user_id: 用户标识
        fact: 要存储的事实
        category: 记忆分类（general, preference, schedule 等）
    """
    try:
        key = str(uuid.uuid4())
        store.put(
            ("memories", user_id),
            key=key,
            value={
                "fact": fact,
                "category": category,
                "access_count": 0,
            }
        )
        logger.info(f"长期记忆已存储: [{category}] {fact[:50]}...")
    except Exception as e:
        logger.error(f"存储长期记忆失败: {e}")


def get_long_term_memories(store: BaseStore, user_id: str, limit: int = 10) -> list[str]:
    """
    获取用户的长期记忆列表。
    
    Args:
        store: LangGraph BaseStore 实例
        user_id: 用户标识
        limit: 返回的最大记忆条数
    
    Returns:
        记忆事实字符串列表
    """
    try:
        items = store.search(("memories", user_id))
        facts = [item.value["fact"] for item in items if "fact" in item.value]
        # 返回最近的记忆（限制数量）
        return facts[-limit:] if len(facts) > limit else facts
    except Exception as e:
        logger.error(f"获取长期记忆失败: {e}")
        return []


def get_context_from_memory(
    store: BaseStore | None,
    user_id: str,
    summary: str,
    task: str
) -> str:
    """
    综合长期记忆和摘要，为当前任务生成上下文信息。
    
    Args:
        store: LangGraph BaseStore 实例
        user_id: 用户标识
        summary: 当前对话摘要
        task: 当前任务描述
    
    Returns:
        格式化的上下文文本
    """
    memories = get_long_term_memories(store, user_id) if store else []
    memory_text = "\n".join([f"  • {m}" for m in memories]) if memories else "  暂无长期记忆"
    
    context = f"""📋 对话摘要：{summary or '暂无'}

🧠 长期记忆：
{memory_text}

📌 当前任务：{task}"""
    
    return context


# ─────────────────────────── 消息裁剪 ───────────────────────────

def trim_messages(messages: list, summary: str, keep_recent: int = 4) -> list:
    """
    裁剪消息列表，用摘要替代旧消息以节省上下文窗口。
    
    Args:
        messages: 原始消息列表
        summary: 当前摘要
        keep_recent: 保留最近的消息数量
    
    Returns:
        裁剪后的消息列表（包含摘要消息 + 最近消息）
    """
    if not messages:
        return [SystemMessage(content=f"对话摘要：{summary}")] if summary else []
    
    # 保留最近的消息
    recent = messages[-keep_recent:] if len(messages) > keep_recent else messages
    
    # 添加摘要作为系统消息
    trimmed = []
    if summary:
        trimmed.append(SystemMessage(content=f"📋 对话摘要：{summary}"))
    trimmed.extend(recent)
    
    logger.info(f"消息裁剪: {len(messages)} -> {len(trimmed)} 条")
    return trimmed