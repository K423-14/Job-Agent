"""
Hybrid Retrieval：BM25 + Vector 双路召回，RRF 融合排序。
BM25Store 和 ChromaDB 连接均为进程级单例，首次调用时初始化，后续复用。
"""
from langchain_core.documents import Document

from .bm25_store import BM25Store
from .vector_store import load_vector_store, load_all_docs

RRF_K = 60

_bm25: BM25Store | None = None
_vstore = None


def _get_stores():
    """懒加载并缓存 BM25Store 和 ChromaDB 连接，整个进程只建一次。"""
    global _bm25, _vstore
    if _bm25 is None:
        print("[Hybrid] 初始化 BM25 索引和 vector store...", flush=True)
        docs = load_all_docs()
        _bm25 = BM25Store(docs)
        _vstore = load_vector_store()
        print(f"[Hybrid] 初始化完成，共 {len(docs)} 条文档", flush=True)
    return _bm25, _vstore


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def hybrid_search(
    query: str,
    top_k: int = 5,
    recall_k: int = 20,
    where_document: dict | None = None,
    extra_vector_queries: list[str] | None = None,
) -> list[tuple[Document, float]]:
    """
    多路召回 + RRF 融合：
      - 主 query：BM25(recall_k) + Vector(recall_k)
      - extra_vector_queries（multi-query 改写 + HyDE）：每路 Vector(recall_k//2)
    同一文档在越多路中排名靠前，RRF 分越高。
    """
    bm25, store = _get_stores()

    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    def _add(results: list[tuple[Document, float]]) -> None:
        for rank, (doc, _) in enumerate(results, start=1):
            fp = doc.metadata.get("fingerprint", doc.page_content[:40])
            scores[fp] = scores.get(fp, 0.0) + _rrf_score(rank)
            doc_map[fp] = doc

    # ── 主 query：BM25 + Vector ──
    _add(bm25.search(query, top_k=recall_k))

    vec_kwargs: dict = {"query": query, "k": recall_k}
    if where_document:
        vec_kwargs["where_document"] = where_document
    _add(store.similarity_search_with_score(**vec_kwargs))

    # ── 扩展 query：各自独立 Vector 召回（复用主路的 where_document 过滤）──
    extra_k = max(recall_k // 2, 5)
    for eq in (extra_vector_queries or []):
        if eq:
            kw = {"query": eq, "k": extra_k}
            if where_document:
                kw["where_document"] = where_document
            _add(store.similarity_search_with_score(**kw))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_map[fp], score) for fp, score in ranked]
