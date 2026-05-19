"""
job_loader.py
职责：读取 jobs.json，把每条岗位转成 LangChain Document。
"""
import hashlib
import json
from pathlib import Path
from langchain_core.documents import Document

def load_jobs(json_path: str | None = None) -> list[Document]:
    """
    读取岗位JSON文件，每条岗位转成一个Document。
    
    为什么一条岗位 = 一个Document（即一个chunk）？
    因为每条岗位信息量适中（几百字），天然就是一个完整的语义单元。
    不需要再切分（chunking），也不需要合并。
    这是最简单也最适合结构化数据的分块策略。
    """

    if json_path is None:
        json_path = str(Path(__file__).parent.parent / "jobs.json")

    with open(json_path, encoding="utf-8") as f:
        raw_jobs = json.load(f)

    docs = []
    for job in raw_jobs:
        # --- 拼成自然语言文本，供 Embedding 模型理解语义 ---
        text = (
            f"岗位名称：{job.get('title', '未知')}\n"
            f"工作地点：{job.get('location', '未知')}\n"
            f"岗位类型：{job.get('type', '未知')}\n"
            f"发布时间：{job.get('publish_time', '未知')}\n"
            f"岗位职责：{job.get('description', '无')}\n"
            f"岗位要求：{job.get('requirements', '无')}"
        )

        # --- metadata：不参与Embedding，但可以用来过滤 ---
        # fingerprint 仅基于岗位ID，改名/换地不会产生重复主键
        fingerprint = (
            job.get("fingerprint")
            or hashlib.md5(str(job.get("id", "")).encode()).hexdigest()
        )
        # content_hash 用于检测内容变更；优先取 scraper 已算好的值
        content_hash = job.get("content_hash") or hashlib.md5(text.encode()).hexdigest()

        metadata = {
            "id": job.get("id"),
            "title": job.get("title", ""),
            "location": job.get("location", ""),
            "type": job.get("type", ""),
            "publish_time": job.get("publish_time", ""),
            "source": "kuaishou",
            "fingerprint": fingerprint,
            "content_hash": content_hash,
            "missing_count": 0,
        }

        docs.append(Document(page_content=text, metadata=metadata))

    return docs