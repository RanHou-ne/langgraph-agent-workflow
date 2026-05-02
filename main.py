"""
LangGraph Agent 工作流 - 主入口
支持三种运行模式：
  1. 交互式对话模式（默认）
  2. 单次任务模式（--task）
  3. Web API 服务模式（--serve）

架构：Planner + Executor + Evaluator + Memory Updater
记忆：短期（Checkpointer）+ 长期（Store）+ 摘要（Summary）
增强：LLM 多模型支持 | 真实 API | 数据库持久化 | 监控指标 | 多 Agent 协作
"""
import os
import sys
import uuid
import time
import logging
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from graph import build_graph

# ─────────────────────────── 日志配置 ───────────────────────────

def setup_logging(verbose: bool = False):
    """配置日志格式和级别。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

logger = logging.getLogger(__name__)


# ─────────────────────────── 存储后端选择 ───────────────────────────

def create_memory_components(use_database: bool = False):
    """
    创建记忆组件。
    
    Args:
        use_database: 是否使用数据库持久化
    
    Returns:
        (checkpointer, store) 元组
    """
    if use_database:
        try:
            from database import DatabaseStore, get_db_manager
            db_manager = get_db_manager()
            store = DatabaseStore(db_manager)
            # 短期记忆仍使用内存（LangGraph 的 SqliteSaver 需要额外配置）
            checkpointer = MemorySaver()
            logger.info("📦 使用数据库持久化存储")
            return checkpointer, store
        except Exception as e:
            logger.warning(f"数据库初始化失败，降级为内存存储: {e}")
    
    # 默认使用内存存储
    return MemorySaver(), InMemoryStore()


# ─────────────────────────── Agent 封装 ───────────────────────────

class LangGraphAgent:
    """
    LangGraph Agent 工作流封装。
    管理三级记忆组件和工作流图的生命周期。
    支持数据库持久化和监控指标收集。
    """
    
    def __init__(self, user_id: str = "default_user", use_database: bool = False):
        self.user_id = user_id
        self.use_database = use_database
        
        # 创建记忆组件
        self.checkpointer, self.store = create_memory_components(use_database)
        
        # 构建工作流图
        self.graph = build_graph(self.checkpointer, self.store)
        
        # 会话 ID
        self.thread_id = str(uuid.uuid4())
        
        # 任务历史管理器（数据库模式）
        self._task_history = None
        if use_database:
            try:
                from database import TaskHistoryManager
                self._task_history = TaskHistoryManager()
            except Exception:
                pass
        
        # 监控指标收集器
        self._metrics = None
        try:
            from monitoring import get_collector
            self._metrics = get_collector()
        except Exception:
            pass
        
        logger.info(f"🤖 Agent 初始化完成 (user_id={user_id}, thread_id={self.thread_id[:8]}..., db={'on' if use_database else 'off'})")
    
    def run_task(self, task: str, verbose: bool = False) -> dict:
        """
        执行单次任务。
        
        Args:
            task: 用户任务描述
            verbose: 是否显示详细执行过程
        
        Returns:
            包含执行结果的字典
        """
        task_id = str(uuid.uuid4())
        start_time = time.time()
        
        # 记录监控指标
        if self._metrics:
            self._metrics.start_task(task_id, self.user_id)
        
        # 记录数据库任务历史
        db_task_id = None
        if self._task_history:
            db_task_id = self._task_history.create_task(
                self.user_id, self.thread_id, task
            )
        
        initial_state = {
            "messages": [],
            "summary": "",
            "task": task,
            "plan": [],
            "cur_step": 0,
            "final_answer": "",
            "step_results": [],
            "iteration": 0,
        }
        
        config = {
            "configurable": {
                "thread_id": self.thread_id,
                "store": self.store,
                "user_id": self.user_id,
            }
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 开始执行任务: {task}")
        logger.info(f"{'='*60}\n")
        
        try:
            result = self.graph.invoke(
                initial_state,
                config,
                config={"recursion_limit": 50}
            )
            
            duration = time.time() - start_time
            
            # 更新监控指标
            if self._metrics:
                self._metrics.complete_task(
                    task_id,
                    plan_steps=len(result.get("plan", [])),
                    completed_steps=len(result.get("step_results", [])),
                )
            
            # 更新数据库任务历史
            if self._task_history and db_task_id:
                self._task_history.complete_task(
                    db_task_id,
                    plan=result.get("plan", []),
                    step_results=result.get("step_results", []),
                    final_answer=result.get("final_answer", ""),
                    summary=result.get("summary", ""),
                    duration_seconds=duration,
                    llm_calls=result.get("plan", []).__len__() + 2,  # 估算
                    tool_calls=len(result.get("step_results", [])),
                )
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ 任务执行完成 (耗时 {duration:.1f}s)")
            logger.info(f"{'='*60}")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ 任务执行失败: {e}")
            
            # 更新监控指标
            if self._metrics:
                self._metrics.fail_task(task_id, str(e))
            
            # 更新数据库任务历史
            if self._task_history and db_task_id:
                self._task_history.fail_task(db_task_id, str(e))
            
            return {
                "final_answer": f"抱歉，任务执行过程中出现错误：{str(e)}",
                "plan": [],
                "summary": "",
                "step_results": [],
            }
    
    def chat(self, user_input: str, thread_id: str = None) -> str:
        """
        交互式对话模式，支持多轮对话。
        
        Args:
            user_input: 用户输入
            thread_id: 可选的会话 ID（用于恢复之前的会话）
        
        Returns:
            Agent 的回复文本
        """
        if thread_id:
            self.thread_id = thread_id
        
        result = self.run_task(user_input)
        return result.get("final_answer", "抱歉，未能生成回答。")
    
    def get_memories(self) -> list[str]:
        """获取当前用户的长期记忆列表。"""
        try:
            items = self.store.search(("memories", self.user_id))
            return [item.value.get("fact", "") for item in items if item.value.get("fact")]
        except Exception:
            return []


# ─────────────────────────── 交互式模式 ───────────────────────────

def run_interactive(agent: LangGraphAgent):
    """运行交互式对话模式。"""
    print("\n" + "="*60)
    print("🤖 LangGraph Agent 工作流 - 交互式对话模式")
    print("="*60)
    print("💡 输入任务描述，Agent 将自动规划和执行")
    print("💡 输入 'memory' 查看长期记忆")
    print("💡 输入 'metrics' 查看运行指标")
    print("💡 输入 'multi <任务>' 使用多 Agent 协作模式")
    print("💡 输入 'new' 开始新的会话")
    print("💡 输入 'quit' 或 'exit' 退出")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("👤 你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit", "q"):
                print("\n👋 再见！感谢使用 LangGraph Agent。")
                break
            
            if user_input.lower() == "memory":
                memories = agent.get_memories()
                if memories:
                    print("\n🧠 长期记忆:")
                    for i, m in enumerate(memories, 1):
                        print(f"  {i}. {m}")
                else:
                    print("\n🧠 长期记忆为空。")
                print()
                continue
            
            if user_input.lower() == "metrics":
                try:
                    from monitoring import get_metrics_summary
                    print(f"\n{get_metrics_summary()}")
                except Exception:
                    print("\n📊 监控模块不可用。")
                print()
                continue
            
            if user_input.lower() == "new":
                agent.thread_id = str(uuid.uuid4())
                print("\n🔄 已开始新的会话。\n")
                continue
            
            # 多 Agent 模式
            if user_input.lower().startswith("multi "):
                task = user_input[6:].strip()
                if task:
                    try:
                        from multi_agent import run_multi_agent_task
                        print(f"\n🤝 多 Agent 协作模式执行中...")
                        response = run_multi_agent_task(task)
                        print(f"\n🤖 Agent: {response}\n")
                    except Exception as e:
                        print(f"\n❌ 多 Agent 执行失败: {e}\n")
                continue
            
            # 执行任务
            response = agent.chat(user_input)
            print(f"\n🤖 Agent: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


# ─────────────────────────── 单次任务模式 ───────────────────────────

def run_single_task(agent: LangGraphAgent, task: str):
    """执行单次任务并输出结果。"""
    result = agent.run_task(task)
    
    print("\n" + "="*60)
    print("📋 执行计划:")
    for i, step in enumerate(result.get("plan", []), 1):
        print(f"  {i}. {step}")
    
    print("\n📊 执行结果:")
    for sr in result.get("step_results", []):
        print(f"  • {sr}")
    
    print("\n💬 最终回答:")
    print(result.get("final_answer", "无"))
    
    print("\n📝 对话摘要:")
    print(result.get("summary", "无"))
    
    # 显示长期记忆
    memories = agent.get_memories()
    if memories:
        print("\n🧠 长期记忆:")
        for m in memories:
            print(f"  • {m}")
    else:
        print("\n🧠 长期记忆为空。")
    
    # 显示监控指标
    try:
        from monitoring import get_metrics_summary
        print(f"\n{get_metrics_summary()}")
    except Exception:
        pass
    
    print("="*60)


# ─────────────────────────── Web API 模式 ───────────────────────────

def run_server(agent: LangGraphAgent, host: str = "0.0.0.0", port: int = 8000):
    """启动 Web API 服务。"""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("❌ Web 服务模式需要安装 fastapi 和 uvicorn:")
        print("   pip install fastapi uvicorn")
        sys.exit(1)
    
    app = FastAPI(
        title="LangGraph Agent API",
        description="基于 LangGraph 的 AI Agent 工作流 API",
        version="2.0.0",
    )
    
    class TaskRequest(BaseModel):
        task: str
        user_id: str = "default_user"
        thread_id: str = None
    
    class TaskResponse(BaseModel):
        answer: str
        plan: list[str]
        summary: str
        step_results: list[str]
    
    class ChatRequest(BaseModel):
        message: str
        user_id: str = "default_user"
        thread_id: str = None
    
    class MultiAgentRequest(BaseModel):
        task: str
        context: str = ""
    
    @app.get("/")
    async def root():
        return {
            "name": "LangGraph Agent API",
            "version": "2.0.0",
            "status": "running",
            "endpoints": {
                "POST /task": "执行单次任务",
                "POST /chat": "交互式对话",
                "POST /multi-agent": "多 Agent 协作任务",
                "GET /memories/{user_id}": "获取用户长期记忆",
                "GET /health": "健康检查",
                "GET /metrics": "运行指标",
                "GET /tasks/{user_id}": "任务历史",
                "GET /api-status": "API 配置状态",
            }
        }
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    @app.post("/task", response_model=TaskResponse)
    async def execute_task(request: TaskRequest):
        try:
            if request.user_id != agent.user_id:
                agent.user_id = request.user_id
            if request.thread_id:
                agent.thread_id = request.thread_id
            
            result = agent.run_task(request.task)
            return TaskResponse(
                answer=result.get("final_answer", ""),
                plan=result.get("plan", []),
                summary=result.get("summary", ""),
                step_results=result.get("step_results", []),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/chat")
    async def chat(request: ChatRequest):
        try:
            if request.user_id != agent.user_id:
                agent.user_id = request.user_id
            
            response = agent.chat(request.message, request.thread_id)
            return {
                "response": response,
                "thread_id": agent.thread_id,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/multi-agent")
    async def multi_agent_task(request: MultiAgentRequest):
        try:
            from multi_agent import run_multi_agent_task
            result = run_multi_agent_task(request.task, request.context)
            return {"answer": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/memories/{user_id}")
    async def get_memories(user_id: str):
        try:
            agent.user_id = user_id
            memories = agent.get_memories()
            return {"user_id": user_id, "memories": memories}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/metrics")
    async def get_metrics():
        try:
            from monitoring import get_collector
            collector = get_collector()
            return collector.get_realtime_stats()
        except Exception as e:
            return {"error": str(e)}
    
    @app.get("/tasks/{user_id}")
    async def get_tasks(user_id: str, limit: int = 20):
        try:
            if agent._task_history:
                tasks = agent._task_history.get_user_tasks(user_id, limit=limit)
                return {"user_id": user_id, "tasks": tasks}
            return {"user_id": user_id, "tasks": [], "message": "数据库未启用"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api-status")
    async def api_status():
        try:
            from tools import get_api_status
            from llm_provider import get_provider_info
            return {
                "api": get_api_status(),
                "llm": get_provider_info(),
            }
        except Exception as e:
            return {"error": str(e)}
    
    print(f"\n🚀 LangGraph Agent API v2.0.0 启动中...")
    print(f"📡 地址: http://{host}:{port}")
    print(f"📖 文档: http://{host}:{port}/docs")
    print(f"🔗 健康检查: http://{host}:{port}/health")
    print(f"📊 运行指标: http://{host}:{port}/metrics")
    print(f"🤖 多 Agent: POST http://{host}:{port}/multi-agent\n")
    
    uvicorn.run(app, host=host, port=port)


# ─────────────────────────── 主入口 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Agent 工作流 v2.0 - 多节点协同 AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                              # 交互式对话模式
  python main.py --task "查查北京天气"         # 单次任务模式
  python main.py --serve                      # 启动 Web API 服务
  python main.py --serve --port 9000          # 指定端口启动服务
  python main.py --user alice                 # 指定用户 ID
  python main.py --verbose                    # 显示详细日志
  python main.py --database                   # 启用数据库持久化
  python main.py --multi-agent "研究AI趋势"   # 多 Agent 协作模式

环境变量:
  LLM_PROVIDER=openai|anthropic|ollama       # LLM 提供商
  LLM_MODEL=gpt-3.5-turbo                    # 模型名称
  SEARCH_API=tavily|serpapi|mock             # 搜索 API
  DATABASE_URL=sqlite:///data/agent.db       # 数据库连接
        """
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="执行单次任务（指定任务描述）"
    )
    parser.add_argument(
        "--serve", "-s",
        action="store_true",
        help="启动 Web API 服务模式"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Web 服务监听地址（默认: 0.0.0.0）"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Web 服务端口（默认: 8000）"
    )
    parser.add_argument(
        "--user", "-u",
        type=str,
        default="default_user",
        help="用户 ID（默认: default_user）"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志"
    )
    parser.add_argument(
        "--database", "-d",
        action="store_true",
        help="启用数据库持久化（默认使用内存存储）"
    )
    parser.add_argument(
        "--multi-agent", "-m",
        type=str,
        help="使用多 Agent 协作模式执行任务"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(args.verbose)
    
    # 检查 API Key
    if not os.getenv("OPENAI_API_KEY") and os.getenv("LLM_PROVIDER", "openai") == "openai":
        print("⚠️  警告: 未设置 OPENAI_API_KEY 环境变量")
        print("   请在 .env 文件中设置，或通过环境变量导出")
        print("   示例: set OPENAI_API_KEY=your_api_key_here")
        print()
    
    # 多 Agent 模式
    if args.multi_agent:
        try:
            from multi_agent import run_multi_agent_task
            print(f"\n🤝 多 Agent 协作模式")
            print(f"📋 任务: {args.multi_agent}")
            print(f"{'='*60}\n")
            
            result = run_multi_agent_task(args.multi_agent)
            
            print(f"\n💬 协作结果:")
            print(result)
            print(f"\n{'='*60}")
        except Exception as e:
            print(f"❌ 多 Agent 执行失败: {e}")
        return
    
    # 创建 Agent
    agent = LangGraphAgent(user_id=args.user, use_database=args.database)
    
    # 根据模式运行
    if args.serve:
        run_server(agent, host=args.host, port=args.port)
    elif args.task:
        run_single_task(agent, args.task)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()