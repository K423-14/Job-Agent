"""
Query 扩展：Multi-Query + HyDE

Multi-Query：同一意图换 N 种表达，从不同角度召回，解决"表述不准"问题。
HyDE：生成假设岗位描述文本，用 doc-doc 相似度代替 query-doc，解决"词汇鸿沟"问题。
"""
import json
from pathlib import Path
from langchain_openai import ChatOpenAI

_LLM_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "llm_config.json"

_MULTI_QUERY_PROMPT = """将以下招聘查询改写为 {n} 个语义相近但措辞不同的版本，帮助从不同角度召回相关岗位。
返回严格 JSON：{{"queries": ["改写1", "改写2", ...]}}，不含原始查询。

原始查询：{question}"""

_HYDE_PROMPT = """根据以下求职意图，生成一段100字以内的岗位JD描述，包含该岗位的典型职责和技术要求关键词。
只输出岗位描述文本，不要其他内容。

求职意图：{question}"""


def _llm(temperature: float, json_mode: bool = False) -> ChatOpenAI:
    with open(_LLM_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    return ChatOpenAI(
        model=cfg.get("lightModel", cfg["model"]),  # 改写/HyDE 用轻量模型
        api_key=cfg["apiKey"],
        base_url=cfg["apiBase"],
        temperature=temperature,
        model_kwargs=model_kwargs,
    )


def multi_query(question: str, n: int = 3) -> list[str]:
    """让 LLM 生成 n 个改写 query，temperature 稍高保证多样性。"""
    try:
        resp = _llm(temperature=0.5, json_mode=True).invoke(
            _MULTI_QUERY_PROMPT.format(n=n, question=question)
        )
        queries = json.loads(resp.content).get("queries", [])
        return [q for q in queries if isinstance(q, str) and q.strip()][:n]
    except Exception:
        return []


def hyde(question: str) -> str:
    """HyDE：生成假设文档，用于替代原始 query 做 vector 检索。"""
    try:
        resp = _llm(temperature=0.1).invoke(
            _HYDE_PROMPT.format(question=question)
        )
        return resp.content.strip()
    except Exception:
        return ""
