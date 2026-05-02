"""
LangGraph Agent 工作流 - Streamlit 前端 UI。
提供交互式聊天界面、任务历史、指标监控等功能。

启动方式：
    streamlit run app.py
"""
import os
import uuid
import time
import logging
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────── 页面配置 ───────────────────────────

st.set_page_config(
    page_title="LangGraph Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .agent-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1976d2;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── 初始化 Session State ───────────────────────────

def init_session_state():
    """初始化 Streamlit session state。"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "user_id" not in st.session_state:
        st.session_state.user_id = "streamlit_user"
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "task_history" not in st.session_state:
        st.session_state.task_history = []
    if "current_plan" not in st.session_state:
        st.session_state.current_plan = []
    if "current_results" not in st.session_state:
        st.session_state.current_results = []
    if "multi_agent_mode" not in st.session_state:
        st.session_state.multi_agent_mode = False


def get_agent():
    """获取或创建 Agent 实例（懒加载）。"""
    if st.session_state.agent is None:
        try:
            from main import LangGraphAgent
            st.session_state.agent = LangGraphAgent(user_id=st.session_state.user_id)
            logger.info("Agent 实例已创建")
        except Exception as e:
            st.error(f"Agent 初始化失败: {e}")
            return None
    return st.session_state.agent


# ─────────────────────────── 侧边栏 ───────────────────────────

def render_sidebar():
    """渲染侧边栏。"""
    with st.sidebar:
        st.title("🤖 LangGraph Agent")
        st.markdown("---")

        # 用户设置
        st.subheader("👤 用户设置")
        new_user_id = st.text_input("用户 ID", value=st.session_state.user_id)
        if new_user_id != st.session_state.user_id:
            st.session_state.user_id = new_user_id
            st.session_state.agent = None  # 重新创建 Agent

        # 会话管理
        st.subheader("💬 会话管理")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 新会话", use_container_width=True):
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.session_state.current_plan = []
                st.session_state.current_results = []
                st.rerun()
        with col2:
            if st.button("🗑️ 清空聊天", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.markdown(f"**会话 ID:** `{st.session_state.thread_id[:8]}...`")

        # 模式切换
        st.markdown("---")
        st.subheader("⚙️ 运行模式")
        st.session_state.multi_agent_mode = st.checkbox(
            "启用多 Agent 协作",
            value=st.session_state.multi_agent_mode,
            help="启用后，复杂任务将由多个专业 Agent 协作完成"
        )

        # LLM 配置
        st.markdown("---")
        st.subheader("🧠 LLM 配置")
        provider = os.getenv("LLM_PROVIDER", "openai")
        model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        st.info(f"**提供商:** {provider}\n\n**模型:** {model}")

        # API 状态
        st.markdown("---")
        st.subheader("📡 API 状态")
        try:
            from tools import get_api_status
            api_status = get_api_status()
            search_status = api_status["search"]
            weather_status = api_status["weather"]

            search_icon = "✅" if search_status["tavily_configured"] or search_status["serpapi_configured"] else "⚠️"
            weather_icon = "✅" if weather_status["openweathermap_configured"] else "⚠️"

            st.markdown(f"{search_icon} 搜索 API: `{search_status['provider']}`")
            st.markdown(f"{weather_icon} 天气 API: `{weather_status['provider']}`")
        except Exception:
            st.warning("无法获取 API 状态")

        # 记忆查看
        st.markdown("---")
        st.subheader("🧠 长期记忆")
        if st.button("查看记忆", use_container_width=True):
            agent = get_agent()
            if agent:
                memories = agent.get_memories()
                if memories:
                    for i, m in enumerate(memories, 1):
                        st.markdown(f"{i}. {m}")
                else:
                    st.info("暂无长期记忆")

        # 指标概览
        st.markdown("---")
        st.subheader("📊 运行指标")
        try:
            from monitoring import get_collector
            collector = get_collector()
            stats = collector.get_realtime_stats()
            agg = stats["aggregate"]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("总任务", agg["total_tasks"])
                st.metric("LLM 调用", agg["total_llm_calls"])
            with col2:
                st.metric("成功率", agg["success_rate"])
                st.metric("工具调用", agg["total_tool_calls"])
        except Exception:
            st.info("指标暂不可用")

        # 版本信息
        st.markdown("---")
        st.caption("v2.0.0 | LangGraph Agent Workflow")


# ─────────────────────────── 主聊天界面 ───────────────────────────

def render_chat():
    """渲染主聊天界面。"""
    st.title("💬 对话")
    st.markdown("输入任务描述，Agent 将自动规划和执行。")

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 用户输入
    if prompt := st.chat_input("输入你的任务..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 执行任务
        with st.chat_message("assistant"):
            with st.spinner("Agent 正在思考和执行..."):
                response = execute_task(prompt)
            st.markdown(response)

        # 添加 Agent 回复
        st.session_state.messages.append({"role": "assistant", "content": response})


def execute_task(task: str) -> str:
    """执行任务并返回结果。"""
    start_time = time.time()

    try:
        if st.session_state.multi_agent_mode:
            # 多 Agent 模式
            from multi_agent import run_multi_agent_task
            response = run_multi_agent_task(task)
        else:
            # 单 Agent 模式
            agent = get_agent()
            if not agent:
                return "❌ Agent 初始化失败，请检查配置。"

            # 记录任务开始
            task_id = str(uuid.uuid4())
            try:
                from monitoring import get_collector
                collector = get_collector()
                collector.start_task(task_id, st.session_state.user_id)
            except Exception:
                pass

            # 执行任务
            result = agent.run_task(task)
            response = result.get("final_answer", "抱歉，未能生成回答。")

            # 更新会话状态
            st.session_state.current_plan = result.get("plan", [])
            st.session_state.current_results = result.get("step_results", [])

            # 记录任务完成
            try:
                collector.complete_task(
                    task_id,
                    plan_steps=len(result.get("plan", [])),
                    completed_steps=len(result.get("step_results", [])),
                )
            except Exception:
                pass

            # 添加到历史
            st.session_state.task_history.append({
                "task": task,
                "answer": response[:200],
                "timestamp": datetime.now().isoformat(),
                "duration": round(time.time() - start_time, 2),
            })

        return response

    except Exception as e:
        logger.error(f"任务执行失败: {e}")
        return f"❌ 任务执行失败: {str(e)}"


# ─────────────────────────── 执行详情面板 ───────────────────────────

def render_execution_details():
    """渲染执行详情面板。"""
    if not st.session_state.current_plan and not st.session_state.current_results:
        return

    st.markdown("---")
    st.subheader("📋 执行详情")

    # 执行计划
    if st.session_state.current_plan:
        with st.expander("📝 执行计划", expanded=False):
            for i, step in enumerate(st.session_state.current_plan, 1):
                st.markdown(f"**{i}.** {step}")

    # 步骤结果
    if st.session_state.current_results:
        with st.expander("📊 步骤结果", expanded=False):
            for result in st.session_state.current_results:
                st.markdown(f"- {result}")


# ─────────────────────────── 指标仪表盘 ───────────────────────────

def render_metrics_dashboard():
    """渲染指标仪表盘页面。"""
    st.subheader("📊 运行指标仪表盘")

    try:
        from monitoring import get_collector
        collector = get_collector()
        stats = collector.get_realtime_stats()
        agg = stats["aggregate"]

        # 概览卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 总任务数", agg["total_tasks"])
        with col2:
            st.metric("✅ 完成数", agg["completed_tasks"])
        with col3:
            st.metric("📈 成功率", agg["success_rate"])
        with col4:
            st.metric("⏱️ 平均耗时", f"{agg['avg_duration_seconds']}s")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🤖 LLM 调用", agg["total_llm_calls"])
        with col2:
            st.metric("🔧 工具调用", agg["total_tool_calls"])
        with col3:
            st.metric("❌ 工具错误", agg["total_tool_errors"])
        with col4:
            st.metric("📊 LLM/任务", agg["avg_llm_calls_per_task"])

        # 工具使用统计
        if agg.get("tool_usage"):
            st.markdown("---")
            st.subheader("🔧 工具使用统计")
            tool_data = agg["tool_usage"]
            st.bar_chart(tool_data)

        # 最近任务
        st.markdown("---")
        st.subheader("📋 最近任务")
        recent = collector.get_recent_tasks(10)
        if recent:
            for task in recent:
                status_icon = "✅" if task["status"] == "completed" else "❌"
                with st.expander(f"{status_icon} {task['task_id'][:8]}... ({task['status']})"):
                    st.json(task)
        else:
            st.info("暂无任务记录")

    except Exception as e:
        st.error(f"无法加载指标: {e}")


# ─────────────────────────── 任务历史页面 ───────────────────────────

def render_task_history():
    """渲染任务历史页面。"""
    st.subheader("📜 任务历史")

    # 从 session state 获取历史
    if st.session_state.task_history:
        for i, task in enumerate(reversed(st.session_state.task_history), 1):
            with st.expander(f"任务 {i}: {task['task'][:50]}..."):
                st.markdown(f"**任务:** {task['task']}")
                st.markdown(f"**回答:** {task['answer']}")
                st.markdown(f"**时间:** {task['timestamp']}")
                st.markdown(f"**耗时:** {task['duration']}s")
    else:
        st.info("暂无任务历史")

    # 从数据库获取历史（如果可用）
    st.markdown("---")
    st.subheader("💾 数据库历史")
    try:
        from database import TaskHistoryManager
        history_mgr = TaskHistoryManager()
        tasks = history_mgr.get_user_tasks(st.session_state.user_id, limit=20)

        if tasks:
            for task in tasks:
                status_icon = "✅" if task["status"] == "completed" else "❌" if task["status"] == "failed" else "⏳"
                with st.expander(f"{status_icon} {task['task'][:50]}..."):
                    st.json(task)
        else:
            st.info("数据库中暂无任务记录")
    except Exception as e:
        st.warning(f"无法访问数据库: {e}")


# ─────────────────────────── 系统信息页面 ───────────────────────────

def render_system_info():
    """渲染系统信息页面。"""
    st.subheader("ℹ️ 系统信息")

    # LLM 配置
    st.markdown("### 🧠 LLM 配置")
    try:
        from llm_provider import get_provider_info
        info = get_provider_info()
        st.json(info)
    except Exception as e:
        st.error(f"无法获取 LLM 信息: {e}")

    # API 状态
    st.markdown("### 📡 API 状态")
    try:
        from tools import get_api_status
        st.json(get_api_status())
    except Exception as e:
        st.error(f"无法获取 API 状态: {e}")

    # 环境变量
    st.markdown("### 🔧 环境变量")
    env_vars = {
        "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "openai"),
        "LLM_MODEL": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        "SEARCH_API": os.getenv("SEARCH_API", "mock"),
        "DATABASE_URL": os.getenv("DATABASE_URL", "sqlite:///data/agent.db"),
        "OPENAI_API_KEY": "✅ 已配置" if os.getenv("OPENAI_API_KEY") else "❌ 未配置",
        "TAVILY_API_KEY": "✅ 已配置" if os.getenv("TAVILY_API_KEY") else "❌ 未配置",
        "OPENWEATHERMAP_API_KEY": "✅ 已配置" if os.getenv("OPENWEATHERMAP_API_KEY") else "❌ 未配置",
    }
    st.json(env_vars)

    # 项目结构
    st.markdown("### 📁 项目结构")
    st.code("""
langgraph-agent-workflow/
├── main.py          # 主入口
├── graph.py         # 工作流构建
├── nodes.py         # 节点实现
├── memory.py        # 记忆管理
├── tools.py         # 工具集
├── llm_provider.py  # LLM 抽象层
├── database.py      # 数据库持久化
├── monitoring.py    # 监控指标
├── multi_agent.py   # 多 Agent 协作
├── app.py           # Streamlit UI
└── requirements.txt # 依赖
    """)


# ─────────────────────────── 主应用 ───────────────────────────

def main():
    """主应用入口。"""
    init_session_state()
    render_sidebar()

    # 主内容区域使用标签页
    tab1, tab2, tab3, tab4 = st.tabs(["💬 对话", "📊 指标", "📜 历史", "ℹ️ 系统"])

    with tab1:
        render_chat()
        render_execution_details()

    with tab2:
        render_metrics_dashboard()

    with tab3:
        render_task_history()

    with tab4:
        render_system_info()


if __name__ == "__main__":
    main()