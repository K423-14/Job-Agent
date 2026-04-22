"""
api.py
职责：FastAPI HTTP接口，暴露RAG问答服务。

面试要点：
- 这是Python AI服务的入口，后续Java后端通过HTTP调用这里
- 两个核心接口：
  1. POST /index  — 触发建索引（把岗位数据向量化存入ChromaDB）
  2. POST /ask    — RAG问答（接收问题，返回回答+来源）
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="JobMind RAG Service", version="0.1.0")

# 请求响应模型

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

class SourceItem(BaseModel):
    title: str
    location: str
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


# 接口：建索引

@app.post("/index")
def build_index():
    """
    读取jobs.json，向量化后存入ChromaDB。
    只需调一次，或数据更新后重新调。
    """
    from .job_loader import load_jobs
    from .vector_store import build_vector_store
    docs = load_jobs()
    build_vector_store(docs=docs)
    return {"message": f"索引构建完成，共 {len(docs)} 条岗位"}

@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """
    接收用户自然语言问题，返回基于岗位数据的回答。
    """
    from .qa_chain import ask

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    result = ask(req.question, req.top_k)
    return AskResponse(**result)

@app.get("/health")
def health():
    return {"status": "ok"}


# ─── 接口：Pipeline — 一键爬取新网站 ───
class PipelineRequest(BaseModel):
    homepage: str

@app.post("/pipeline")
def run_pipeline(req: PipelineRequest):
    """
    Multi-Agent Pipeline：输入一个网站URL，自动完成 发现API→生成配置→爬取→入库 全流程。
    """
    from agents.pipeline import run_pipeline as _run    

    result = _run(req.homepage)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result)
    return result


# ─── 接口：Agent对话 ───

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest):
    """
    Agent对话接口：用户发送自然语言，Agent自主决定调用工具并回答。
    对比 /ask 接口（固定RAG流程），/chat 接口让LLM自主决策更灵活。
    """
    from agents.agent import chat

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    reply = chat(req.message)
    return ChatResponse(reply=reply)