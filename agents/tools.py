"""
tools.py
把系统能力封装成标准工具函数，供Agent通过Tool Calling调用。

面试要点：
- 每个工具有名字、描述、参数Schema — LLM通过描述来理解工具能干什么
- 工具函数本身是普通Python函数，加了@tool装饰器
- LLM不直接执行工具，它只输出"我要调哪个工具+什么参数"，由框架代理执行
- 这就是Agent的核心：LLM做决策，工具做执行
"""
import json
import sys
from pathlib import Path
from langchain_core.tools import tool

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

@tool
def search_jobs(query: str, top_k: int = 5) -> str:
    """
    语义检索岗位信息。根据用户描述，从向量数据库中返回最相关的岗位。
    当用户想查找、搜索、了解岗位信息时使用此工具。
    
    参数：
    - query: 用户的搜索描述，如"北京的Java后端实习"
    - top_k: 返回结果数量，默认5
    """
    from rag.vector_store import search    

    results = search(query=query, top_k=top_k)
    if not results:
        return "未找到相关岗位。"
    
    output = []
    for i, (doc, score) in enumerate(results, 1):
        output.append(
            f"{i}. {doc.metadata.get('title', '未知')}"
            f"| 地点：{doc.metadata.get('location', '未知')}"
            f"| 相关度：{score:.2f}"
        )
    return "\n".join(output)


@tool
def search_jobs_with_filter(query: str, location: str = None, 
                            keyword: str = None, top_k: int = 5) -> str:
    """
    带过滤条件的岗位检索。支持按城市、关键词过滤后再做语义检索。
    当用户明确指定了城市或技术关键词时使用此工具。
    
    参数：
    - query: 语义检索的查询文本
    - location: 城市过滤，如"北京"、"深圳"
    - keyword: 关键词过滤，如"Java"、"算法"
    - top_k: 返回数量
    """
    from rag.vector_store import filtered_search
    
    conditions = []
    if location:
        conditions.append({"$contains": location})
    if keyword:
        conditions.append({"$contains": keyword})
    
    where_document = None
    if len(conditions) == 1:
        where_document = conditions[0]
    elif len(conditions) > 1:
        where_document = {"$and": conditions}
    
    results = filtered_search(query=query, top_k=top_k, where_document=where_document)
    
    if not results:
        return f"未找到符合条件的岗位（城市={location}, 关键词={keyword}）。"
    
    output = []
    for i, (doc, score) in enumerate(results, 1):
        output.append(
            f"{i}. {doc.metadata.get('title', '未知')} "
            f"| 地点: {doc.metadata.get('location', '未知')} "
            f"| 相关度: {score:.2f}"
        )
    return "\n".join(output)


@tool
def index_jobs(json_path: str = None) -> str:
    """
    将岗位数据向量化并存入向量数据库。
    当有新爬取的数据需要入库，或需要重建索引时使用此工具。
    
    参数：
    - json_path: 岗位JSON文件路径，默认为项目根目录的jobs.json
    """
    from rag.job_loader import load_jobs
    from rag.vector_store import build_vector_store
    
    docs = load_jobs(json_path)
    build_vector_store(docs)
    return f"索引构建完成，共 {len(docs)} 条岗位已入库。"


@tool
def crawl_website(homepage: str) -> str:
    """
    一键爬取新的招聘网站。自动执行：发现API → 生成配置 → 批量爬取 → 数据入库。
    当用户想要添加新的数据源、爬取新网站时使用此工具。
    
    参数：
    - homepage: 招聘网站首页URL，如"https://campus.kuaishou.cn"
    """
    from agents.pipeline import run_pipeline
    
    result = run_pipeline(homepage)
    if result["success"]:
        return f"爬取成功！共获取 {result['total_jobs']} 条岗位并已入库。"
    else:
        failed_step = [k for k, v in result["steps"].items() 
                       if v.get("status") == "failed"]
        return f"爬取失败，失败步骤: {failed_step}"


# 工具注册表：统一管理所有工具

ALL_TOOLS = [search_jobs, search_jobs_with_filter, index_jobs, crawl_website]

def get_tool_names() -> list[str]:
    """返回所有工具名称"""
    return [t.name for t in ALL_TOOLS]


