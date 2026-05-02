"""
数据库持久化模块。
支持 SQLite（默认）和 PostgreSQL，提供：
  - 长期记忆持久化（替代 InMemoryStore）
  - 短期记忆持久化（替代 MemorySaver）
  - 任务执行历史记录
  - 会话管理

通过 DATABASE_URL 环境变量配置数据库连接：
  - SQLite: sqlite:///data/agent.db
  - PostgreSQL: postgresql://user:pass@localhost:5432/agent_db
"""
import os
import uuid
import json
import logging
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ─────────────────────────── 数据库配置 ───────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/agent.db")

# ─────────────────────────── SQLAlchemy 模型 ───────────────────────────

from sqlalchemy import (
    create_engine, Column, String, Text, Integer, Float,
    DateTime, JSON, Boolean, Index, func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()


class MemoryRecord(Base):
    """长期记忆记录表。"""
    __tablename__ = "memories"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(128), nullable=False, index=True)
    namespace = Column(String(256), nullable=False, default="memories")
    fact = Column(Text, nullable=False)
    category = Column(String(64), nullable=False, default="general")
    access_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_memories_user_ns", "user_id", "namespace"),
    )


class TaskHistory(Base):
    """任务执行历史记录表。"""
    __tablename__ = "task_history"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(128), nullable=False, index=True)
    thread_id = Column(String(36), nullable=False, index=True)
    task = Column(Text, nullable=False)
    plan = Column(JSON, default=list)
    step_results = Column(JSON, default=list)
    final_answer = Column(Text, default="")
    summary = Column(Text, default="")
    status = Column(String(32), default="pending")  # pending | running | completed | failed
    error_message = Column(Text, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    llm_calls = Column(Integer, default=0)
    tool_calls = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_task_user_created", "user_id", "created_at"),
    )


class SessionRecord(Base):
    """会话记录表。"""
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(128), nullable=False, index=True)
    thread_id = Column(String(36), nullable=False, unique=True, index=True)
    title = Column(String(256), default="")
    message_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active_at = Column(DateTime, default=datetime.utcnow)


class CheckpointRecord(Base):
    """短期记忆检查点表（用于替代 MemorySaver）。"""
    __tablename__ = "checkpoints"

    thread_id = Column(String(36), primary_key=True)
    checkpoint_ns = Column(String(128), primary_key=True, default="")
    checkpoint_id = Column(String(36), primary_key=True)
    parent_checkpoint_id = Column(String(36), nullable=True)
    type = Column(String(64), nullable=True)
    checkpoint = Column(JSON, nullable=False)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_checkpoint_thread", "thread_id", "checkpoint_ns"),
    )


# ─────────────────────────── 数据库管理器 ───────────────────────────

class DatabaseManager:
    """
    数据库管理器。
    负责引擎创建、会话管理和表初始化。
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or DATABASE_URL
        self._engine = None
        self._session_factory = None

    @property
    def engine(self):
        if self._engine is None:
            connect_args = {}
            # SQLite 需要特殊配置
            if self.database_url.startswith("sqlite"):
                connect_args["check_same_thread"] = False
                # 确保数据目录存在
                db_path = self.database_url.replace("sqlite:///", "")
                if db_path and "/" in db_path:
                    os.makedirs(os.path.dirname(db_path), exist_ok=True)

            self._engine = create_engine(
                self.database_url,
                connect_args=connect_args,
                echo=False,
                pool_pre_ping=True,
            )
            logger.info(f"📦 数据库引擎已创建: {self.database_url.split('://')[0]}")
        return self._engine

    @property
    def session_factory(self):
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
            )
        return self._session_factory

    def init_db(self):
        """创建所有表（如果不存在）。"""
        Base.metadata.create_all(self.engine)
        logger.info("📦 数据库表初始化完成")

    @contextmanager
    def get_session(self) -> Session:
        """获取数据库会话的上下文管理器。"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self):
        """关闭数据库引擎。"""
        if self._engine:
            self._engine.dispose()
            logger.info("📦 数据库连接已关闭")


# ─────────────────────────── 全局实例 ───────────────────────────

_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """获取全局数据库管理器实例（懒加载）。"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.init_db()
    return _db_manager


def reset_db_manager():
    """重置全局数据库管理器（用于测试）。"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
    _db_manager = None


# ─────────────────────────── 长期记忆 Store ───────────────────────────

class DatabaseStore:
    """
    基于数据库的长期记忆存储。
    实现与 LangGraph InMemoryStore 兼容的接口。
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_db_manager()

    def put(self, namespace: tuple, key: str, value: dict):
        """存储一条记忆。"""
        user_id = namespace[1] if len(namespace) > 1 else "default_user"
        ns = namespace[0] if len(namespace) > 0 else "memories"

        with self.db.get_session() as session:
            record = MemoryRecord(
                id=key,
                user_id=user_id,
                namespace=ns,
                fact=value.get("fact", ""),
                category=value.get("category", "general"),
                access_count=value.get("access_count", 0),
            )
            session.merge(record)  # upsert

        logger.debug(f"数据库存储记忆: [{user_id}] {value.get('fact', '')[:50]}...")

    def search(self, namespace: tuple, limit: int = 50) -> list:
        """搜索记忆。返回兼容 InMemoryStore 的对象列表。"""
        user_id = namespace[1] if len(namespace) > 1 else "default_user"
        ns = namespace[0] if len(namespace) > 0 else "memories"

        with self.db.get_session() as session:
            records = (
                session.query(MemoryRecord)
                .filter(
                    MemoryRecord.user_id == user_id,
                    MemoryRecord.namespace == ns,
                )
                .order_by(MemoryRecord.created_at.desc())
                .limit(limit)
                .all()
            )

            # 返回兼容格式
            items = []
            for r in records:
                item = _MemoryItem(
                    key=r.id,
                    value={
                        "fact": r.fact,
                        "category": r.category,
                        "access_count": r.access_count,
                    },
                )
                items.append(item)

            return items

    def get(self, namespace: tuple, key: str) -> Optional[dict]:
        """获取单条记忆。"""
        with self.db.get_session() as session:
            record = session.get(MemoryRecord, key)
            if record:
                return {
                    "fact": record.fact,
                    "category": record.category,
                    "access_count": record.access_count,
                }
            return None

    def delete(self, namespace: tuple, key: str):
        """删除单条记忆。"""
        with self.db.get_session() as session:
            record = session.get(MemoryRecord, key)
            if record:
                session.delete(record)


class _MemoryItem:
    """模拟 LangGraph StoreItem 的兼容对象。"""

    def __init__(self, key: str, value: dict):
        self.key = key
        self.value = value


# ─────────────────────────── 任务历史管理 ───────────────────────────

class TaskHistoryManager:
    """任务执行历史管理器。"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_db_manager()

    def create_task(
        self,
        user_id: str,
        thread_id: str,
        task: str,
    ) -> str:
        """创建新的任务记录，返回任务 ID。"""
        task_id = str(uuid.uuid4())
        with self.db.get_session() as session:
            record = TaskHistory(
                id=task_id,
                user_id=user_id,
                thread_id=thread_id,
                task=task,
                status="running",
            )
            session.add(record)
        return task_id

    def complete_task(
        self,
        task_id: str,
        plan: list,
        step_results: list,
        final_answer: str,
        summary: str,
        duration_seconds: float = 0,
        llm_calls: int = 0,
        tool_calls: int = 0,
    ):
        """标记任务完成。"""
        with self.db.get_session() as session:
            record = session.get(TaskHistory, task_id)
            if record:
                record.plan = plan
                record.step_results = step_results
                record.final_answer = final_answer
                record.summary = summary
                record.status = "completed"
                record.duration_seconds = duration_seconds
                record.llm_calls = llm_calls
                record.tool_calls = tool_calls
                record.completed_at = datetime.utcnow()

    def fail_task(self, task_id: str, error_message: str):
        """标记任务失败。"""
        with self.db.get_session() as session:
            record = session.get(TaskHistory, task_id)
            if record:
                record.status = "failed"
                record.error_message = error_message
                record.completed_at = datetime.utcnow()

    def get_user_tasks(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """获取用户的任务历史列表。"""
        with self.db.get_session() as session:
            records = (
                session.query(TaskHistory)
                .filter(TaskHistory.user_id == user_id)
                .order_by(TaskHistory.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "task": r.task,
                    "status": r.status,
                    "duration_seconds": r.duration_seconds,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "final_answer": r.final_answer[:200] if r.final_answer else "",
                }
                for r in records
            ]

    def get_task_detail(self, task_id: str) -> Optional[dict]:
        """获取任务详情。"""
        with self.db.get_session() as session:
            record = session.get(TaskHistory, task_id)
            if not record:
                return None
            return {
                "id": record.id,
                "user_id": record.user_id,
                "thread_id": record.thread_id,
                "task": record.task,
                "plan": record.plan,
                "step_results": record.step_results,
                "final_answer": record.final_answer,
                "summary": record.summary,
                "status": record.status,
                "error_message": record.error_message,
                "duration_seconds": record.duration_seconds,
                "llm_calls": record.llm_calls,
                "tool_calls": record.tool_calls,
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            }

    def get_stats(self, user_id: Optional[str] = None) -> dict:
        """获取任务统计信息。"""
        with self.db.get_session() as session:
            query = session.query(TaskHistory)
            if user_id:
                query = query.filter(TaskHistory.user_id == user_id)

            total = query.count()
            completed = query.filter(TaskHistory.status == "completed").count()
            failed = query.filter(TaskHistory.status == "failed").count()

            avg_duration = (
                query.filter(
                    TaskHistory.status == "completed",
                    TaskHistory.duration_seconds.isnot(None),
                )
                .with_entities(func.avg(TaskHistory.duration_seconds))
                .scalar()
            )

            avg_llm_calls = (
                query.filter(TaskHistory.status == "completed")
                .with_entities(func.avg(TaskHistory.llm_calls))
                .scalar()
            )

            return {
                "total_tasks": total,
                "completed": completed,
                "failed": failed,
                "success_rate": f"{(completed / total * 100):.1f}%" if total > 0 else "N/A",
                "avg_duration_seconds": round(avg_duration, 2) if avg_duration else 0,
                "avg_llm_calls": round(avg_llm_calls, 1) if avg_llm_calls else 0,
            }