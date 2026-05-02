"""
LLM 模型提供商抽象层。
支持多种 LLM 后端：OpenAI、Anthropic Claude、本地 Ollama。
通过环境变量统一配置，一处修改全局生效。
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0,
    **kwargs,
):
    """
    LLM 工厂函数：根据提供商返回对应的 LLM 实例。

    优先级：参数 > 环境变量 > 默认值

    Args:
        provider: 提供商名称 ("openai" | "anthropic" | "ollama")
        model: 模型名称（各提供商的默认值不同）
        temperature: 温度参数
        **kwargs: 传递给底层 LLM 的额外参数

    Returns:
        LangChain 兼容的 LLM 实例

    Raises:
        ValueError: 不支持的提供商
        ImportError: 缺少必要的依赖包
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai").lower().strip()

    if provider == "openai":
        return _get_openai_llm(model, temperature, **kwargs)
    elif provider == "anthropic":
        return _get_anthropic_llm(model, temperature, **kwargs)
    elif provider == "ollama":
        return _get_ollama_llm(model, temperature, **kwargs)
    else:
        raise ValueError(
            f"不支持的 LLM 提供商: '{provider}'。"
            f"支持的提供商: openai, anthropic, ollama"
        )


def _get_openai_llm(model: Optional[str], temperature: float, **kwargs):
    """获取 OpenAI LLM 实例。"""
    from langchain_openai import ChatOpenAI

    model = model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    base_url = os.getenv("OPENAI_BASE_URL")

    params = {
        "model": model,
        "temperature": temperature,
        **kwargs,
    }
    if base_url:
        params["base_url"] = base_url

    logger.info(f"🤖 使用 OpenAI 模型: {model}")
    return ChatOpenAI(**params)


def _get_anthropic_llm(model: Optional[str], temperature: float, **kwargs):
    """获取 Anthropic Claude LLM 实例。"""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "使用 Anthropic Claude 需要安装 langchain-anthropic：\n"
            "  pip install langchain-anthropic"
        )

    model = model or os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
    api_key = os.getenv("ANTHROPIC_API_KEY")

    params = {
        "model": model,
        "temperature": temperature,
        **kwargs,
    }
    if api_key:
        params["api_key"] = api_key

    logger.info(f"🤖 使用 Anthropic Claude 模型: {model}")
    return ChatAnthropic(**params)


def _get_ollama_llm(model: Optional[str], temperature: float, **kwargs):
    """获取 Ollama 本地模型 LLM 实例。"""
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        raise ImportError(
            "使用 Ollama 本地模型需要安装 langchain-community：\n"
            "  pip install langchain-community"
        )

    model = model or os.getenv("LLM_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    params = {
        "model": model,
        "temperature": temperature,
        "base_url": base_url,
        **kwargs,
    }

    logger.info(f"🤖 使用 Ollama 本地模型: {model} ({base_url})")
    return ChatOllama(**params)


def get_provider_info() -> dict:
    """
    获取当前 LLM 配置信息（用于诊断和展示）。

    Returns:
        包含 provider、model、base_url 等信息的字典
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()

    info = {
        "provider": provider,
        "model": os.getenv("LLM_MODEL", _get_default_model(provider)),
        "base_url": None,
    }

    if provider == "openai":
        info["base_url"] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    elif provider == "ollama":
        info["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return info


def _get_default_model(provider: str) -> str:
    """获取各提供商的默认模型名称。"""
    defaults = {
        "openai": "gpt-3.5-turbo",
        "anthropic": "claude-3-5-sonnet-20241022",
        "ollama": "llama3",
    }
    return defaults.get(provider, "gpt-3.5-turbo")