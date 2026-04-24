"""
Cross-encoder Reranker（精排）。
使用 BAAI/bge-reranker-base，支持中英文，CPU 可运行。

首次调用会从 HuggingFace 下载模型（~560MB）。
国内访问慢时，可在终端 export HF_ENDPOINT=https://hf-mirror.com 再启动。

全局单例，进程内只加载一次。
"""
from __future__ import annotations
from langchain_core.documents import Document

_MODEL_ID = "BAAI/bge-reranker-base"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        import os
        # 禁止运行时联网检查，完全走本地缓存，避免在代理/防火墙环境下挂起
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        print(f"[Reranker] 加载模型 {_MODEL_ID}（离线模式，从本地缓存读取）...", flush=True)
        _model = CrossEncoder(_MODEL_ID, local_files_only=True)
        print("[Reranker] 模型加载完成", flush=True)
    return _model


def rerank(
    query: str,
    results: list[tuple[Document, float]],
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """
    Cross-encoder 精排：用 (query, passage) 拼接过模型，输出相关性分数。
    比 bi-encoder 慢但精度高，适合在粗召回（~20条）后使用。
    """
    if not results:
        return results

    model = _get_model()
    pairs = [(query, doc.page_content) for doc, _ in results]
    scores = model.predict(pairs)  # numpy array

    ranked = sorted(
        zip(results, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    return [(doc, float(score)) for (doc, _), score in ranked[:top_k]]
