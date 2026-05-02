"""
多 Agent 协作模块。
实现专业化 Agent 角色，支持多 Agent 间协作完成复杂任务。

支持的 Agent 角色：
  - Researcher: 信息搜索和数据收集
  - Coder: 代码生成和计算任务
  - Writer: 内容撰写和总结

协作模式：
  - 串行协作：Agent 按顺序执行，前一个的输出作为后一个的输入
  - 并行协作：多个 Agent 同时执行，结果汇总
  - 主从协作：主 Agent 分配任务，从 Agent 执行并汇报
"""
import json
import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from llm_provider import get_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


# ─────────────────────────── Agent 角色定义 ───────────────────────────

class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    CODER = "coder"
    WRITER = "writer"
    COORDINATOR = "coordinator"


AGENT_PROMPTS = {
    AgentRole.RESEARCHER: """你是一个专业的信息研究员。你的职责是：
1. 使用搜索工具查找最新、最准确的信息
2. 整理和验证搜索结果的可靠性
3. 提供结构化的信息摘要
4. 标注信息来源

请用中文回答，保持客观、准确。""",

    AgentRole.CODER: """你是一个专业的程序员和技术专家。你的职责是：
1. 编写清晰、高效的代码
2. 解决技术问题和算法挑战
3. 进行数学计算和数据分析
4. 提供技术方案和最佳实践

请用中文回答，代码部分使用标准格式。""",

    AgentRole.WRITER: """你是一个专业的内容创作者。你的职责是：
1. 撰写清晰、有条理的文档和报告
2. 将复杂信息转化为易懂的内容
3. 总结和归纳各方面的信息
4. 确保内容的连贯性和可读性

请用中文回答，保持专业且友好的语气。""",

    AgentRole.COORDINATOR: """你是一个任务协调者。你的职责是：
1. 分析复杂任务，确定需要哪些专业 Agent
2. 将任务拆解并分配给合适的 Agent
3. 整合各 Agent 的输出，形成完整回答
4. 确保最终回答的质量和完整性

请用中文回答。""",
}


# ─────────────────────────── Agent 实例 ───────────────────────────

@dataclass
class AgentResult:
    """Agent 执行结果。"""
    role: str
    content: str
    success: bool = True
    error: str = ""


class SpecializedAgent:
    """专业化 Agent 基类。"""

    def __init__(self, role: AgentRole, tools: list = None):
        self.role = role
        self.tools = tools or []
        self.system_prompt = AGENT_PROMPTS.get(role, "")
        self.llm = get_llm(temperature=0)

    def run(self, task: str, context: str = "") -> AgentResult:
        """
        执行任务。

        Args:
            task: 任务描述
            context: 额外上下文信息

        Returns:
            AgentResult 对象
        """
        try:
            messages = [SystemMessage(content=self.system_prompt)]

            if context:
                messages.append(HumanMessage(content=f"背景信息：\n{context}"))

            messages.append(HumanMessage(content=task))

            response = self.llm.invoke(messages)
            return AgentResult(
                role=self.role.value,
                content=response.content.strip(),
                success=True,
            )
        except Exception as e:
            logger.error(f"Agent {self.role.value} 执行失败: {e}")
            return AgentResult(
                role=self.role.value,
                content="",
                success=False,
                error=str(e),
            )


class ResearcherAgent(SpecializedAgent):
    """研究员 Agent - 专注于信息搜索和数据收集。"""

    def __init__(self, tools: list = None):
        super().__init__(AgentRole.RESEARCHER, tools)

    def search_and_summarize(self, query: str) -> AgentResult:
        """搜索并总结信息。"""
        task = f"请搜索并总结以下主题的相关信息：\n{query}\n\n要求：\n1. 提供 3-5 个关键点\n2. 标注信息来源\n3. 评估信息的时效性"
        return self.run(task)


class CoderAgent(SpecializedAgent):
    """程序员 Agent - 专注于代码生成和计算。"""

    def __init__(self, tools: list = None):
        super().__init__(AgentRole.CODER, tools)

    def solve_problem(self, problem: str) -> AgentResult:
        """解决技术问题。"""
        task = f"请解决以下技术问题：\n{problem}\n\n要求：\n1. 提供清晰的解决方案\n2. 如需代码，请使用标准格式\n3. 解释关键步骤"
        return self.run(task)


class WriterAgent(SpecializedAgent):
    """作家 Agent - 专注于内容创作和总结。"""

    def __init__(self, tools: list = None):
        super().__init__(AgentRole.WRITER, tools)

    def write_report(self, topic: str, sources: list[str]) -> AgentResult:
        """撰写报告。"""
        sources_text = "\n".join([f"- {s}" for s in sources])
        task = f"请根据以下信息撰写一份报告：\n\n主题：{topic}\n\n参考资料：\n{sources_text}\n\n要求：\n1. 结构清晰，有引言、正文、结论\n2. 语言专业但易懂\n3. 适当使用 emoji 增强可读性"
        return self.run(task)

    def summarize(self, content: str) -> AgentResult:
        """总结内容。"""
        task = f"请总结以下内容的要点：\n\n{content}\n\n要求：\n1. 提取 3-5 个核心要点\n2. 使用简洁的语言\n3. 保持原意"
        return self.run(task)


# ─────────────────────────── 多 Agent 协作器 ───────────────────────────

class MultiAgentCoordinator:
    """
    多 Agent 协调器。
    根据任务类型自动选择合适的 Agent 组合和协作模式。
    """

    def __init__(self, tools: list = None):
        self.tools = tools or []
        self.researcher = ResearcherAgent(tools)
        self.coder = CoderAgent(tools)
        self.writer = WriterAgent(tools)
        self.coordinator_llm = get_llm(temperature=0)

    def classify_task(self, task: str) -> dict:
        """
        分类任务，确定需要哪些 Agent。

        Returns:
            包含 agent_roles 和 collaboration_mode 的字典
        """
        prompt = f"""分析以下任务，确定需要哪些专业 Agent 协作完成。

可用 Agent：
- researcher: 信息搜索、数据收集、事实验证
- coder: 代码编写、数学计算、技术问题
- writer: 内容撰写、报告生成、信息总结

任务：{task}

请返回 JSON 格式：
{{"agent_roles": ["role1", "role2"], "collaboration_mode": "serial|parallel", "reasoning": "选择理由"}}

只返回 JSON，不要其他内容。"""

        try:
            response = self.coordinator_llm.invoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content.strip())
            return result
        except Exception as e:
            logger.warning(f"任务分类失败，使用默认配置: {e}")
            # 默认使用研究员 + 作家串行协作
            return {
                "agent_roles": ["researcher", "writer"],
                "collaboration_mode": "serial",
                "reasoning": "默认配置：先搜索信息，再总结输出",
            }

    def run_collaborative(self, task: str, context: str = "") -> str:
        """
        执行多 Agent 协作任务。

        Args:
            task: 任务描述
            context: 额外上下文

        Returns:
            最终回答文本
        """
        logger.info(f"🤝 [MultiAgent] 开始多 Agent 协作: {task[:50]}...")

        # 1. 分类任务
        classification = self.classify_task(task)
        agent_roles = classification.get("agent_roles", ["researcher", "writer"])
        mode = classification.get("collaboration_mode", "serial")
        reasoning = classification.get("reasoning", "")

        logger.info(f"🤝 [MultiAgent] 任务分类: {agent_roles}, 模式: {mode}")
        logger.info(f"🤝 [MultiAgent] 选择理由: {reasoning}")

        # 2. 获取 Agent 实例
        agents = self._get_agents_by_roles(agent_roles)

        # 3. 根据模式执行协作
        if mode == "parallel":
            results = self._run_parallel(agents, task, context)
        else:
            results = self._run_serial(agents, task, context)

        # 4. 整合结果
        final_answer = self._integrate_results(task, results)

        logger.info(f"🤝 [MultiAgent] 协作完成，结果长度: {len(final_answer)} 字符")
        return final_answer

    def _get_agents_by_roles(self, roles: list[str]) -> list[SpecializedAgent]:
        """根据角色列表获取 Agent 实例。"""
        agent_map = {
            "researcher": self.researcher,
            "coder": self.coder,
            "writer": self.writer,
        }
        return [agent_map[r] for r in roles if r in agent_map]

    def _run_serial(
        self,
        agents: list[SpecializedAgent],
        task: str,
        context: str,
    ) -> list[AgentResult]:
        """串行执行：前一个 Agent 的输出作为后一个的上下文。"""
        results = []
        current_context = context

        for agent in agents:
            logger.info(f"🤝 [MultiAgent] 串行执行: {agent.role.value}")
            result = agent.run(task, current_context)
            results.append(result)

            if result.success:
                # 将当前结果添加到上下文
                current_context = f"{current_context}\n\n{agent.role.value} 的分析结果：\n{result.content}"
            else:
                logger.warning(f"🤝 [MultiAgent] {agent.role.value} 执行失败: {result.error}")

        return results

    def _run_parallel(
        self,
        agents: list[SpecializedAgent],
        task: str,
        context: str,
    ) -> list[AgentResult]:
        """并行执行：所有 Agent 同时执行（简化版，实际可使用线程池）。"""
        results = []

        for agent in agents:
            logger.info(f"🤝 [MultiAgent] 并行执行: {agent.role.value}")
            result = agent.run(task, context)
            results.append(result)

        return results

    def _integrate_results(self, task: str, results: list[AgentResult]) -> str:
        """整合多个 Agent 的结果，生成最终回答。"""
        # 收集成功的结果
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return "抱歉，所有 Agent 执行失败，请稍后重试。"

        # 构建整合提示
        results_text = "\n\n".join([
            f"【{r.role} 的分析】\n{r.content}"
            for r in successful_results
        ])

        prompt = f"""请整合以下多个专业 Agent 的分析结果，为用户提供一个完整、连贯的回答。

用户任务：{task}

各 Agent 的分析结果：
{results_text}

要求：
1. 整合各方面的信息，形成完整回答
2. 保持信息的准确性和完整性
3. 使用清晰的结构组织内容
4. 语气友好、专业
5. 使用适当的 emoji 增强可读性"""

        try:
            response = self.coordinator_llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"结果整合失败: {e}")
            # 降级：直接拼接结果
            return f"以下是多 Agent 协作的结果：\n\n{results_text}"


# ─────────────────────────── 便捷函数 ───────────────────────────

_coordinator: Optional[MultiAgentCoordinator] = None


def get_coordinator(tools: list = None) -> MultiAgentCoordinator:
    """获取多 Agent 协调器单例。"""
    global _coordinator
    if _coordinator is None:
        _coordinator = MultiAgentCoordinator(tools)
    return _coordinator


def run_multi_agent_task(task: str, context: str = "", tools: list = None) -> str:
    """
    执行多 Agent 协作任务的便捷函数。

    Args:
        task: 任务描述
        context: 额外上下文
        tools: 可用工具列表

    Returns:
        最终回答文本
    """
    coordinator = get_coordinator(tools)
    return coordinator.run_collaborative(task, context)