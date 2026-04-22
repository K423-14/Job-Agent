"""
mcp_server.py
MCP Server：把 JobMind 的能力通过 MCP 协议暴露给任何AI工具。

运行方式：
  Streamable HTTP模式（远程）：python mcp_server.py  →  端点: http://localhost:8000/mcp
  Stdio模式（本地）：由MCP Client（如Cursor）自动启动

面试要点：
- MCP是标准化的AI能力暴露协议，类比Nacos的服务注册与发现
- 底层通信协议：JSON-RPC 2.0
- 传输层两种模式：
    Stdio（本地）：进程 stdin/stdout 通信，零网络开销，被 Cursor/Claude Desktop 作为子进程启动
    Streamable HTTP（远程）：单端点 /mcp，同时支持普通HTTP响应和流式SSE推送
- 早期的 HTTP+SSE 双端点方案（/sse + /messages）已在2025年3月规范更新中被标记为deprecated
- 和直接暴露HTTP API的区别：MCP Client可以自动发现工具，不需要硬编码调用逻辑
"""
import json
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP


# 确保项目根目录在Python路径中
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))


# 创建MCP Server实例
mcp = FastMCP(
    name="JobMind", 
    instructions="智能招聘信息助手，支持岗位语义检索、条件过滤、数据爬取和向量化入库。",
)

@mcp.tool()
def search_jobs(query: str, top_k: int = 5) -> str:
    """
    语义检索岗位信息。根据描述从向量数据库中返回最相关的岗位。
    
    Args:
        query: 搜索描述，如"北京的Java后端实习"
        top_k: 返回结果数量，默认5
    """
    from rag.vector_store import search

    results = search(query=query, top_k=top_k)
    if not results:
        return "未找到相关岗位。"
    
    output = []
    for i, (doc, score) in enumerate(results, 1):
        output.append(
            f"{i}. {doc.metadata.get('title', '未知')} "
            f"| 地点: {doc.metadata.get('location', '未知')} "
            f"| 相关度: {score:.2f}"
        )
    return "\n".join(output)    
        

@mcp.tool()
def search_jobs_with_filter(query: str, location: str = None,
                            keyword: str = None, top_k: int = 5) -> str:
    """
    带过滤条件的岗位检索。支持按城市、关键词过滤后再做语义检索。
    
    Args:
        query: 语义检索的查询文本
        location: 城市过滤，如"北京"、"深圳"
        keyword: 关键词过滤，如"Java"、"算法"
        top_k: 返回数量
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


@mcp.tool()
def index_jobs(json_path: str = "") -> str:
    """
    将岗位数据向量化并存入向量数据库。数据更新后需重新调用。
    
    Args:
        json_path: 岗位JSON文件路径，为空则使用默认的jobs.json
    """
    from rag.job_loader import load_jobs
    from rag.vector_store import build_vector_store

    path = json_path if json_path else None
    docs = load_jobs(path)
    build_vector_store(docs)
    return f"索引构建完成，共 {len(docs)} 条岗位已入库。"


# 启动入口

if __name__ == "__main__":
    print("启动 JobMind MCP Server (Streamable HTTP模式)，端口 8000...")
    print("MCP端点: http://localhost:8000/mcp")
    mcp.run(transport="streamable-http")