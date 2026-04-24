"""共享 LangChain ChatOpenAI 客户端工厂"""
import json
from pathlib import Path

import httpx
from langchain_openai import ChatOpenAI

_LLM_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "llm_config.json"


def get_llm(**overrides) -> ChatOpenAI:
    """读取 llm_config.json 创建 ChatOpenAI 实例，支持参数覆盖。

    额外参数：
      json_mode=True  → 启用 GLM JSON 输出模式（response_format: json_object）
      read_timeout    → httpx read 超时秒数（默认 60）
    """
    with open(_LLM_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    json_mode = overrides.pop("json_mode", False)
    _http_client = httpx.Client(
        timeout=httpx.Timeout(
            connect=10.0,
            read=overrides.pop("read_timeout", 60.0),
            write=30.0,
            pool=5.0,
        )
    )
    model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    return ChatOpenAI(
        model=overrides.get("model", cfg["model"]),
        api_key=cfg["apiKey"],
        base_url=cfg["apiBase"],
        max_tokens=overrides.get("max_tokens", cfg["maxTokens"]),
        temperature=overrides.get("temperature", cfg["temperature"]),
        streaming=overrides.get("streaming", False),
        http_client=_http_client,
        model_kwargs=model_kwargs,
    )
