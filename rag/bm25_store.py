"""
BM25 关键词检索。每次从 ChromaDB 加载全量 doc 后在内存中建索引。
岗位数据量小（几千条内），重建耗时 < 1s，无需持久化。
"""
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


def _tokenize(text: str) -> list[str]:
    return list(jieba.cut(text))


class BM25Store:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        corpus = [_tokenize(d.page_content) for d in docs]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Document, float]]:
        """返回 (doc, bm25_score) 列表，按分数降序。"""
        scores = self.bm25.get_scores(_tokenize(query))
        # argsort 升序，取末尾 top_k 再翻转
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.docs[i], float(scores[i])) for i in indices]
