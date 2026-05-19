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

def upsert_jobs(docs: list, missing_threshold: int = 2) -> dict:
    """
    增量更新向量数据库。

    策略：
    1. 以 fingerprint（岗位ID的MD5）作为 ChromaDB 文档主键
    2. 对比 content_hash：内容变化则删旧插新（重新Embedding）
    3. 本次未出现的岗位累加 missing_count，连续 missing_threshold 次删除

    返回统计：{"added": int, "updated": int, "deleted": int, "skipped": int}
    """
    BATCH_SIZE = 64
    embedding = _get_embedding_model()
    store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding,
        collection_name="jobs"
    )

    # 1. 拉取库内全量 metadata（不需要向量，省开销）
    existing_raw = store.get(include=["metadatas"])
    existing = {
        chroma_id: (meta or {})
        for chroma_id, meta in zip(
            existing_raw.get("ids", []),
            existing_raw.get("metadatas") or []
        )
    }

    # 2. 遍历本次爬取，分类
    current_fps: set[str] = set()
    to_add_docs: list = []
    to_add_ids: list[str] = []
    to_delete_fps: list[str] = []  # 内容有变，需先删后加

    for doc in docs:
        fp = doc.metadata.get("fingerprint", "")
        if not fp:
            continue
        current_fps.add(fp)
        new_hash = doc.metadata.get("content_hash", "")

        if fp not in existing:
            doc.metadata["missing_count"] = 0
            to_add_docs.append(doc)
            to_add_ids.append(fp)
        elif existing[fp].get("content_hash", "") != new_hash:
            # 内容变更：删旧重建
            to_delete_fps.append(fp)
            doc.metadata["missing_count"] = 0
            to_add_docs.append(doc)
            to_add_ids.append(fp)
        else:
            # 内容未变；若之前被标记缺失则重置计数
            if int(existing[fp].get("missing_count", 0)) > 0:
                updated_meta = dict(existing[fp])
                updated_meta["missing_count"] = 0
                # 仅更新 metadata，不重新 Embedding
                store._collection.update(ids=[fp], metadatas=[updated_meta])

    # 3. 删除内容已变的旧文档
    if to_delete_fps:
        store.delete(ids=to_delete_fps)

    # 4. 批量插入新/更新的文档
    for i in range(0, len(to_add_docs), BATCH_SIZE):
        batch_docs = to_add_docs[i:i + BATCH_SIZE]
        batch_ids = to_add_ids[i:i + BATCH_SIZE]
        print(f"  写入第 {i // BATCH_SIZE + 1} 批（{len(batch_docs)} 条）...")
        store.add_documents(documents=batch_docs, ids=batch_ids)

    added = sum(1 for fp in to_add_ids if fp not in existing)
    updated = len(to_delete_fps)
    skipped = len(docs) - len(to_add_docs)

    # 5. 处理本次未出现的岗位（疑似下架）
    absent_fps = set(existing.keys()) - current_fps
    deleted = 0
    for fp in absent_fps:
        meta = existing[fp]
        new_missing = int(meta.get("missing_count", 0)) + 1
        if new_missing >= missing_threshold:
            store.delete(ids=[fp])
            deleted += 1
            print(f"  删除下架岗位（连续 {new_missing} 次未见）: {meta.get('title', fp)}")
        else:
            updated_meta = dict(meta)
            updated_meta["missing_count"] = new_missing
            store._collection.update(ids=[fp], metadatas=[updated_meta])

    stats = {"added": added, "updated": updated, "deleted": deleted, "skipped": skipped}
    print(f"向量库更新完成: 新增 {added}，更新 {updated}，删除(下架) {deleted}，跳过 {skipped}")
    return stats


def build_vector_store(docs: list) -> dict:
    """向后兼容入口，内部调用 upsert_jobs。"""
    return upsert_jobs(docs)


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