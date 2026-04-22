"""
agent.py
ReAct Agent：根据用户输入，自主决定调用哪些工具来完成任务。

面试要点：
- 和Pipeline的区别：Pipeline是硬编码顺序，这里是LLM自主决策
- 用的是ReAct模式：Thought → Action → Observation → Thought → ...
- LangChain的create_react_agent封装了这个循环
"""
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from .tools import ALL_TOOLS

_LLM_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "llm_config.json"

# Agent 的 System Prompt
AGENT_SYSTEM_PROMPT = """你是 JobMind 智能招聘助手。你可以使用以下工具帮助用户：

1. search_jobs — 语义搜索岗位
2. search_jobs_with_filter — 带城市/关键词过滤的精确搜索
3. index_jobs — 将数据向量化入库
4. crawl_website — 一键爬取新的招聘网站

当前数据库已有数据来源：快手校园招聘（campus.kuaishou.cn）

核心决策规则：
- 用户查询的公司在数据库中 → 用 search_jobs 或 search_jobs_with_filter 检索
- 用户查询的公司不在数据库中 → 告诉用户"当前数据库没有该公司数据"，并询问是否需要爬取，同时请用户提供该公司的招聘官网URL
- 用户想爬取新网站 → 必须使用用户提供的URL，绝对不要自己编造URL。如果用户没给URL，主动询问
- 如果搜索结果为空或不相关 → 如实告诉用户没找到，不要编造岗位信息

回答要基于工具返回的真实数据，用中文回答。"""


def _get_llm() -> ChatOpenAI:
    with open(_LLM_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    return ChatOpenAI(
        model=cfg["model"],
        api_key=cfg["apiKey"],
        base_url=cfg["apiBase"],
        temperature=cfg.get("temperature", 0.1),
    )


def create_jobmind_agent():
    """
    创建一个可以自主调用工具的Agent。
    
    面试要点：
    - LangChain v1.x 用 create_agent 替代了 create_tool_calling_agent + AgentExecutor
    - create_agent 返回一个 CompiledStateGraph，内部自动完成工具调用循环
    - 调用方式：graph.invoke({"messages": [{"role": "user", "content": "..."}]})
    """
    llm = _get_llm()
    return create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )


def chat(user_input: str) -> str:
    """
    Agent 对话入口。
    
    用户说什么都行，Agent自己判断要不要调工具、调哪个。
    """
    agent = create_jobmind_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    # 最后一条消息即为 AI 回复
    return result["messages"][-1].content