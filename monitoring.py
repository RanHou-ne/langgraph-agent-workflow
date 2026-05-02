"""
任务执行监控和指标收集模块。
收集 LLM 调用、工具调用、任务执行等指标，支持实时监控和历史统计。
"""
import time
import threading
import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# ─────────────────────────── 指标数据结构 ───────────────────────────

@dataclass
class TaskMetrics:
    """单次任务的执行指标。"""
    task_id: str = ""
    user_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    plan_steps: int = 0
    completed_steps: int = 0
    status: str = "pending"  # pending | running | completed | failed
    error_message: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "duration_seconds": round(self.duration_seconds, 2),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "plan_steps": self.plan_steps,
            "completed_steps": self.completed_steps,
            "status": self.status,
            "error_message": self.error_message,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
        }


@dataclass
class AggregateMetrics:
    """聚合指标（全局统计）。"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_tool_errors: int = 0
    total_duration: float = 0.0
    tool_usage: dict = field(default_factory=lambda: defaultdict(int))
    tool_errors_by_name: dict = field(default_factory=lambda: defaultdict(int))

    def to_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": (
                f"{(self.completed_tasks / self.total_tasks * 100):.1f}%"
                if self.total_tasks > 0 else "N/A"
            ),
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_tool_errors": self.total_tool_errors,
            "avg_duration_seconds": (
                round(self.total_duration / self.completed_tasks, 2)
                if self.completed_tasks > 0 else 0
            ),
            "avg_llm_calls_per_task": (
                round(self.total_llm_calls / self.total_tasks, 1)
                if self.total_tasks > 0 else 0
            ),
            "tool_usage": dict(self.tool_usage),
            "tool_errors_by_name": dict(self.tool_errors_by_name),
        }


# ─────────────────────────── 指标收集器 ───────────────────────────

class MetricsCollector:
    """
    指标收集器（单例模式）。
    线程安全地收集和聚合任务执行指标。
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._metrics_lock = threading.Lock()
        self._aggregate = AggregateMetrics()
        self._active_tasks: dict[str, TaskMetrics] = {}
        self._completed_tasks: list[TaskMetrics] = []
        self._max_history = 1000  # 保留最近 1000 条记录

        logger.info("📊 指标收集器已初始化")

    # ─────────── 任务生命周期 ───────────

    def start_task(self, task_id: str, user_id: str = "") -> TaskMetrics:
        """记录任务开始。"""
        metrics = TaskMetrics(
            task_id=task_id,
            user_id=user_id,
            start_time=time.time(),
            status="running",
        )
        with self._metrics_lock:
            self._active_tasks[task_id] = metrics
            self._aggregate.total_tasks += 1

        logger.debug(f"📊 任务开始: {task_id[:8]}...")
        return metrics

    def complete_task(self, task_id: str, plan_steps: int = 0, completed_steps: int = 0):
        """记录任务完成。"""
        with self._metrics_lock:
            metrics = self._active_tasks.pop(task_id, None)
            if not metrics:
                return

            metrics.end_time = time.time()
            metrics.duration_seconds = metrics.end_time - metrics.start_time
            metrics.status = "completed"
            metrics.plan_steps = plan_steps
            metrics.completed_steps = completed_steps

            self._aggregate.completed_tasks += 1
            self._aggregate.total_duration += metrics.duration_seconds

            self._completed_tasks.append(metrics)
            if len(self._completed_tasks) > self._max_history:
                self._completed_tasks = self._completed_tasks[-self._max_history:]

        logger.debug(
            f"📊 任务完成: {task_id[:8]}... "
            f"(耗时 {metrics.duration_seconds:.1f}s, "
            f"LLM {metrics.llm_calls}次, 工具 {metrics.tool_calls}次)"
        )

    def fail_task(self, task_id: str, error_message: str = ""):
        """记录任务失败。"""
        with self._metrics_lock:
            metrics = self._active_tasks.pop(task_id, None)
            if not metrics:
                return

            metrics.end_time = time.time()
            metrics.duration_seconds = metrics.end_time - metrics.start_time
            metrics.status = "failed"
            metrics.error_message = error_message

            self._aggregate.failed_tasks += 1

            self._completed_tasks.append(metrics)
            if len(self._completed_tasks) > self._max_history:
                self._completed_tasks = self._completed_tasks[-self._max_history:]

        logger.debug(f"📊 任务失败: {task_id[:8]}... ({error_message[:50]})")

    # ─────────── 计数器 ───────────

    def record_llm_call(self, task_id: str = ""):
        """记录一次 LLM 调用。"""
        with self._metrics_lock:
            self._aggregate.total_llm_calls += 1
            if task_id and task_id in self._active_tasks:
                self._active_tasks[task_id].llm_calls += 1

    def record_tool_call(self, task_id: str, tool_name: str, success: bool = True):
        """记录一次工具调用。"""
        with self._metrics_lock:
            self._aggregate.total_tool_calls += 1
            self._aggregate.tool_usage[tool_name] += 1

            if task_id and task_id in self._active_tasks:
                self._active_tasks[task_id].tool_calls += 1

            if not success:
                self._aggregate.total_tool_errors += 1
                self._aggregate.tool_errors_by_name[tool_name] += 1
                if task_id and task_id in self._active_tasks:
                    self._active_tasks[task_id].tool_errors += 1

    # ─────────── 查询接口 ───────────

    def get_task_metrics(self, task_id: str) -> Optional[dict]:
        """获取指定任务的指标。"""
        with self._metrics_lock:
            # 先查活跃任务
            if task_id in self._active_tasks:
                return self._active_tasks[task_id].to_dict()
            # 再查已完成任务
            for m in reversed(self._completed_tasks):
                if m.task_id == task_id:
                    return m.to_dict()
        return None

    def get_active_tasks(self) -> list[dict]:
        """获取所有活跃任务的指标。"""
        with self._metrics_lock:
            return [m.to_dict() for m in self._active_tasks.values()]

    def get_aggregate_metrics(self) -> dict:
        """获取聚合指标。"""
        with self._metrics_lock:
            return self._aggregate.to_dict()

    def get_recent_tasks(self, limit: int = 20) -> list[dict]:
        """获取最近完成的任务指标。"""
        with self._metrics_lock:
            tasks = self._completed_tasks[-limit:]
            return [m.to_dict() for m in reversed(tasks)]

    def get_realtime_stats(self) -> dict:
        """获取实时统计信息。"""
        with self._metrics_lock:
            active_count = len(self._active_tasks)
            aggregate = self._aggregate.to_dict()

            return {
                "active_tasks": active_count,
                "aggregate": aggregate,
                "recent_tasks": [
                    m.to_dict() for m in self._completed_tasks[-5:]
                ],
                "timestamp": datetime.now().isoformat(),
            }

    def reset(self):
        """重置所有指标（用于测试）。"""
        with self._metrics_lock:
            self._aggregate = AggregateMetrics()
            self._active_tasks.clear()
            self._completed_tasks.clear()


# ─────────────────────────── 便捷函数 ───────────────────────────

def get_collector() -> MetricsCollector:
    """获取指标收集器单例。"""
    return MetricsCollector()


def get_metrics_summary() -> str:
    """获取格式化的指标摘要文本。"""
    collector = get_collector()
    stats = collector.get_realtime_stats()
    agg = stats["aggregate"]

    lines = [
        "📊 Agent 运行指标摘要",
        "=" * 40,
        f"🔄 活跃任务数: {stats['active_tasks']}",
        f"📋 总任务数: {agg['total_tasks']}",
        f"✅ 完成: {agg['completed_tasks']}",
        f"❌ 失败: {agg['failed_tasks']}",
        f"📈 成功率: {agg['success_rate']}",
        f"⏱️ 平均耗时: {agg['avg_duration_seconds']}s",
        f"🤖 LLM 调用总数: {agg['total_llm_calls']}",
        f"🔧 工具调用总数: {agg['total_tool_calls']}",
        f"📊 平均 LLM 调用/任务: {agg['avg_llm_calls_per_task']}",
    ]

    if agg.get("tool_usage"):
        lines.append("\n🔧 工具使用统计:")
        for tool_name, count in sorted(
            agg["tool_usage"].items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  • {tool_name}: {count} 次")

    return "\n".join(lines)