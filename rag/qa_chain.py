"""
qa_chain.py
职责：RAG问答链 — Query理解 + 检索相关岗位 + 构造Prompt + 调用LLM生成回答。

优化点（面试必讲）：
1. Query Understanding：先用LLM解析用户意图，提取城市、关键词等结构化条件
2. 混合检索：metadata过滤（精确） + 语义召回（模糊），两者结合
3. 手动编排而非RetrievalQA，对每一步有更细粒度的控制
"""
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .vector_store import search, filtered_search


_LLM_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "llm_config.json"

# Prompt 1: Query语义理解

QUERY_UNDERSTANDING_PROMPT = """你是一个查询解析器。用户会用自然语言描述想找什么岗位，你需要结构化提取查询元数据。

请从用户问题中提取以下字段（严格Json格式返回）：
- location: 工作城市（如"北京"、"深圳"，没提到则为null）
- keywords: 岗位关键词列表（如["Java", "多模态"]），没提到则为空列表
- raw_query: 精炼后的查询语句（去掉无关语句，保留核心意图）

只返回JSON，不要其他内容。实例：
用户："我想找一个北京的Java后端开发岗位"
返回：{{"location": "北京", "keywords": ["Java", "后端"], "raw_query": "Java后端开发"}}

用户："有无算法相关的实习，我最近想找一个"
返回：{{"location": null, "keywords": ["算法"], "raw_query": "算法实习"}}

"""

# Prompt 2：回答生成

# 系统提示词
SYSTEM_PROMPT = """你是一个招聘信息助手，基于检索到的岗位数据回答用户问题

规则：
1.只根据下面提供的【检索结果】回答，不要编造不存在的岗位信息
2.如果检索中没有相关信息，明确回答“当前数据库不存在相关岗位”
3.回答简介有条理，列出岗位包含：岗位名称、工作地点、核心要求
4.用户问题不是关于岗位查询的，礼貌的引导回岗位话题

"""

# User Prompt

USER_PROMPT_TEMPLATE = """【检索结果】（共{result_count}条相关岗位）：
{context}

【用户问题】：{question}

请基于以上检索结果回答用户问题

"""


def _get_chat_llm() -> ChatOpenAI:
    """创建Chat LLM实例（用于生成回答，和Embedding模型是不同的模型）"""
    with open(_LLM_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    return ChatOpenAI(
        model=cfg["model"], 
        api_key=cfg["apiKey"], 
        base_url=cfg["apiBase"], 
        max_completion_tokens=cfg.get("maxTokens", 4096), 
        temperature=cfg.get("temperature", 0.1)
    )


def _parse_query(question: str) -> dict:
    """
    Query Understanding：用LLM解析用户意图。
    
    面试要点：
    - 这一步把自然语言转成结构化查询条件
    - 属于Agent的"思考"环节——LLM负责理解，工具负责执行
    - 如果解析失败，fallback到原始问题直接检索（容错设计）
    """
    llm = _get_chat_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_UNDERSTANDING_PROMPT), 
        ("human", "{question}"),
    ])

    messages = prompt.format_messages(question=question)
    response = llm.invoke(messages)

    try:
        # 提取JSON
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        return {
            "location": parsed.get("location"), 
            "keywords": parsed.get("keywords", []), 
            "raw_query": parsed.get("raw_query", question)
        }
    except (json.JSONDecodeError, IndexError):
        # 解析失败
        return {
            "location": None, 
            "keywords": [], 
            "raw_query": question
        }


def _format_docs(results: list) -> str:
    """
    把检索结果格式化成文本，塞进Prompt。
    
    results 是 [(Document, score), ...] 的列表。
    score 是距离分数，越小越相关。
    """
    parts = []
    for i, (doc, score) in enumerate(results, 1):
        parts.append(
            f"--- 岗位{i}（相关分数：{score:.2f}） ---\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def ask(question: str, top_k: int = 5) -> dict:
    """
    RAG问答主函数（优化版）。四步走：
    
    1. Understand：LLM解析用户意图 → 提取城市、关键词
    2. Retrieve：用结构化条件+语义检索，拿到最相关的岗位
    3. Augment：把检索结果塞进Prompt模板
    4. Generate：调用LLM，基于检索结果生成回答
    
    面试要点：
    - 第1步是Query Understanding，属于"Agent思维"
    - 第2步是混合检索（metadata filter + semantic search）
    - 整个流程是 Understand → Retrieve → Augment → Generate
    """
    # Step 1: Understand（理解用户意图）
    parsed = _parse_query(question)
    print(f"[Query理解] location={parsed['location']}, "
          f"keywords={parsed['keywords']}, raw_query={parsed['raw_query']}")

    # Step 2: Retrieve（混合检索）
    # 统一用 where_document 做文本包含过滤
    # 原因：metadata的where不支持$contains，而location字段是自由文本（"北京, 深圳"）
    # where_document 是在 page_content 里做子串匹配

    conditions = []

    if parsed["location"]:
        conditions.append({"$contains": parsed["location"]})

    # 如果解析出关键词，加metadata过滤
    if parsed["keywords"]:
        for kw in parsed["keywords"]:
            conditions.append({"$contains": kw})

    # 多个条件用$and组合
    where_document = None
    if len(conditions) == 1:
        where_document = conditions[0]
    else:
        where_document = {"$and": conditions}

    # 先尝试带过滤检索
    results = filtered_search(
        query=parsed["raw_query"], 
        top_k=top_k, 
        where_document=where_document
    )

    # 如果过滤后结果太少，fallback到纯语义检索
    if len(results) <= 0:
        print("[Fallback] 过滤结果太少，回退到纯语义检索")
        results = search(query=question, top_k=top_k)

    if not results:
        return {
            "answer": "当前数据库中没有找到相关信息。",
            "sources": [],
        }

    # Step 2: Augment（增强Prompt）
    context = _format_docs(results)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT), 
            ("human", USER_PROMPT_TEMPLATE),
        ]
    )

    messages = prompt.format_messages(
        result_count=len(results), 
        context=context, 
        question=question
    )

    # Step 3: Generate（生成回答）
    llm = _get_chat_llm()
    response = llm.invoke(messages)

    # 提取信息来源
    sources = []
    for (doc, score) in results:
        sources.append({
            "title": doc.metadata.get("title", ""),   # 没有则给空
            "location": doc.metadata.get("location", ""), 
            "score": round(score, 3),   # 四舍五入保留三位
        })

    return {
        "answer": response.content, 
        "sources": sources, 
    }
