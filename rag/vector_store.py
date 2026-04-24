"""
vector_store.py
职责：把 Document 列表向量化后存入 ChromaDB，并提供检索接口。
"""
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import json

# 向量数据库持久化路径（存到项目目录下的 chroma_db 文件夹）
CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")

# 加载LLM配置（复用现有的配置文件，Embedding也走同一个API）
_LLM_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "llm_config.json"

def _get_embedding_model() -> OpenAIEmbeddings:
    """
    创建 Embedding 模型实例。
    
    面试要点：
    - Embedding模型和Chat模型是不同的模型
    - Chat模型负责"理解+生成"，Embedding模型只负责"文本→向量"
    - 这里用OpenAI兼容接口，实际可以换成任何Embedding模型
    """
    with open(_LLM_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    
    return OpenAIEmbeddings(
        model=cfg.get("embeddingModel", "text-embedding-ada-002"),
        api_key=cfg["apiKey"],
        base_url=cfg["apiBase"],
    )

def build_vector_store(docs: list) -> Chroma:
    """
    把Document列表向量化后存入ChromaDB。
    
    这一步做了两件事：
    1. 调用Embedding模型，把每个doc的page_content变成向量
    2. 把向量+原文+metadata一起存到ChromaDB（持久化到磁盘）
    
    面试要点：
    - 这是"离线索引"阶段，只需要跑一次（或者数据更新时重跑）
    - 类比搜索引擎：这一步相当于"建索引"
    - OpenAI API 限制：一次最多64条，需要分批处理
    """
    embedding = _get_embedding_model()
    
    # OpenAI embeddings API 一次最多处理 64 条
    BATCH_SIZE = 64
    vector_store = None
    
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        print(f"处理第 {batch_num} 批（{len(batch)} 条）...")

        # 用fingerprint作为文档ID，ChromaDB会按ID去重（相同ID不会重复插入）
        batch_ids = [doc.metadata.get("fingerprint", str(i + idx))
                     for idx, doc in enumerate(batch)]
        
        if vector_store is None:
            # 第一批：创建新的vector store
            vector_store = Chroma.from_documents(
                documents=batch, 
                embedding=embedding, 
                persist_directory=CHROMA_DIR, 
                collection_name="jobs", 
                ids=batch_ids
            )
        else:
            # 后续批次：添加到已有的vector store
            vector_store.add_documents(documents=batch)

    print(f"已索引 {len(docs)} 条岗位到 ChromaDB，路径: {CHROMA_DIR}")
    return vector_store


def load_vector_store() -> Chroma:
    """
    加载已有的向量数据库（不重新Embedding，直接读磁盘）。
    
    面试要点：
    - build 只需跑一次，后续查询只需 load
    - ChromaDB的persist_directory就是持久化目录，重启不丢数据
    """
    embedding = _get_embedding_model()

    return Chroma(
        persist_directory=CHROMA_DIR, 
        embedding_function=embedding, 
        collection_name="jobs"
    )


def load_all_docs() -> list:
    """从 ChromaDB 拉取全量文档，供 BM25 建内存索引用。"""
    store = load_vector_store()
    result = store.get(include=["documents", "metadatas"])
    docs = []
    from langchain_core.documents import Document
    for content, meta in zip(result["documents"], result["metadatas"]):
        docs.append(Document(page_content=content, metadata=meta or {}))
    return docs


def search(query: str, top_k: int = 5) -> list:
    """
    语义检索：根据用户问题，返回最相似的top_k条岗位。
    
    面试要点：
    - 底层是"余弦相似度"（cosine similarity）计算
    - query也会先经过Embedding变成向量，然后和库里所有向量比较
    - top_k就是返回最相似的k条结果
    - similarity_search_with_score 返回 (Document, score) 对
      分数越小越相似（ChromaDB用的是L2距离）
    """
    store = load_vector_store()
    results = store.similarity_search_with_score(query=query, k = top_k)
    return results


def filtered_search(query: str, top_k: int = 5, where_document: dict = None) -> list:
    """
    带过滤条件的语义检索。
    
    参数：
    - where_document: 文档内容过滤，如 {"$contains": "Java"}
                      多条件用{"$and": [{"$contains": "北京"}, {"$contains": "Java"}]}
    
    面试要点：
    - 这是"混合检索"的简易版：结构化过滤 + 语义召回
    - ChromaDB的where语法支持 $eq, $ne, $gt, $lt, $in, $contains 等
    - 先过滤缩小范围，再做向量相似度计算，效率更高
    """
    store = load_vector_store()

    kwargs = {"query": query, "k": top_k}
    if where_document:
        kwargs["where_document"] = where_document

    results = store.similarity_search_with_score(**kwargs)
    return results    