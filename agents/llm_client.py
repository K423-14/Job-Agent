"""共享 LangChain ChatOpenAI 客户端工厂"""
import json
from pathlib import Path

import httpx
from langchain_openai import ChatOpenAI

_LLM_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "llm_config.json"


def get_llm(**overrides) -> ChatOpenAI:
    """读取 llm_config.json 创建 ChatOpenAI 实例，支持参数覆盖。"""
    with open(_LLM_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    # httpx 细粒度超时：connect=10s，首个 token 等待(read)=60s
    _http_client = httpx.Client(
        timeout=httpx.Timeout(
            connect=10.0,
            read=overrides.pop("read_timeout", 60.0),
            write=30.0,
            pool=5.0,
        )
    )
    return ChatOpenAI(
        model=overrides.get("model", cfg["model"]),
        api_key=cfg["apiKey"],
        base_url=cfg["apiBase"],
        max_tokens=overrides.get("max_tokens", cfg["maxTokens"]),
        temperature=overrides.get("temperature", cfg["temperature"]),
        streaming=overrides.get("streaming", False),
        http_client=_http_client,
    )
