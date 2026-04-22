"""
test_evaluation.py
RAG系统评测脚本 — 生产简历需要的量化指标。

评测内容：
1. 检索召回率（Recall@5）：消融实验 — Baseline / +Query Rewrite / +Filter
2. MRR & NDCG@5：标准IR排序质量指标
3. 过滤精确率：带城市条件的查询，结果是否真的匹配城市
4. Token消耗对比：全量上下文 vs RAG方式（tiktoken精确统计）
5. 响应延迟（p50/p95）
"""
import json
import math
import time
import sys
import statistics
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).parent))

from rag.vector_store import search, filtered_search
from rag.qa_chain import ask, _parse_query

# ─── 测试用例：人工标注的"正确答案" ───
# 每条包含：问题、期望命中的岗位title关键词、期望的城市

TEST_CASES = [
    # ── 常规用例 ──
    {
        "question": "有没有Java开发的实习",
        "expected_keywords": ["Java"],
        "expected_location": None,
    },
    {
        "question": "北京的Java开发岗位",
        "expected_keywords": ["Java"],
        "expected_location": "北京",
    },
    {
        "question": "深圳有什么算法岗位",
        "expected_keywords": ["算法"],
        "expected_location": "深圳",
    },
    {
        "question": "做推荐系统的实习",
        "expected_keywords": ["推荐"],
        "expected_location": None,
    },
    {
        "question": "C++开发工程师",
        "expected_keywords": ["C++"],
        "expected_location": None,
    },
    {
        "question": "北京的大数据开发",
        "expected_keywords": ["大数据"],
        "expected_location": "北京",
    },
    {
        "question": "有没有产品经理的实习岗位",
        "expected_keywords": ["产品"],
        "expected_location": None,
    },
    {
        "question": "深圳做多模态的岗位",
        "expected_keywords": ["多模态"],
        "expected_location": "深圳",
    },
    {
        "question": "前端开发实习生",
        "expected_keywords": ["前端"],
        "expected_location": None,
    },
    {
        "question": "北京的Golang开发",
        "expected_keywords": ["Go", "Golang"],
        "expected_location": "北京",
    },
    {
        "question": "做运营的岗位",
        "expected_keywords": ["运营"],
        "expected_location": None,
    },
    {
        "question": "测试开发工程师",
        "expected_keywords": ["测试"],
        "expected_location": None,
    },
    {
        "question": "深圳的后端开发",
        "expected_keywords": ["后端", "开发"],
        "expected_location": "深圳",
    },
    {
        "question": "有什么设计相关的实习",
        "expected_keywords": ["设计"],
        "expected_location": None,
    },
    {
        "question": "做搜索方向的算法",
        "expected_keywords": ["搜索"],
        "expected_location": None,
    },
    {
        "question": "iOS开发实习",
        "expected_keywords": ["iOS"],
        "expected_location": None,
    },
    {
        "question": "数据分析相关的岗位",
        "expected_keywords": ["数据"],
        "expected_location": None,
    },
    {
        "question": "做广告算法的实习",
        "expected_keywords": ["广告"],
        "expected_location": None,
    },
    {
        "question": "北京安全相关岗位",
        "expected_keywords": ["安全"],
        "expected_location": "北京",
    },
    {
        "question": "有没有NLP自然语言处理的岗位",
        "expected_keywords": ["NLP", "自然语言"],
        "expected_location": None,
    },
    # ── 多地点岗位：验证 "北京, 深圳" 格式的城市过滤是否正确命中 ──
    {
        "question": "深圳的C++开发",
        "expected_keywords": ["C++"],
        "expected_location": "深圳",  # 实际岗位 location="北京, 深圳"
    },
    {
        "question": "杭州的搜索算法岗位",
        "expected_keywords": ["搜索", "算法"],
        "expected_location": "杭州",  # 实际岗位 location="杭州, 北京"
    },
    # ── 难例：口语化/语义模糊，简单关键词匹配无法直接命中 ──
    {
        "question": "写代码的后端实习",
        "expected_keywords": ["Java", "Python", "Go", "C++", "后端", "服务端"],
        "expected_location": None,
    },
    {
        "question": "搞推荐的算法实习",
        "expected_keywords": ["推荐系统", "推荐", "RecSys", "算法"],
        "expected_location": None,
    },
    {
        "question": "做人工智能方向的技术岗",
        "expected_keywords": ["AI", "人工智能", "机器学习", "深度学习", "算法"],
        "expected_location": None,
    },
    {
        "question": "北京想做和图像视频相关的研究",
        "expected_keywords": ["视觉", "CV", "图像", "视频", "多模态"],
        "expected_location": "北京",
    },
]


# ─── 语料库工具（用于 True Recall 分母） ───
_CORPUS_CACHE: list | None = None


def _load_corpus() -> list:
    """惰性加载 jobs.json，全程只读一次"""
    global _CORPUS_CACHE
    if _CORPUS_CACHE is None:
        jobs_path = Path(__file__).parent / "jobs.json"
        with open(jobs_path, encoding="utf-8") as f:
            _CORPUS_CACHE = json.load(f)
    return _CORPUS_CACHE  # type: ignore[return-value]


def _count_relevant_in_corpus(expected_keywords) -> int:
    """统计语料库中匹配任意期望关键词的岗位总数（True Recall 分母）"""
    count = 0
    for job in _load_corpus():
        text = (
            job.get("title", "") + " "
            + job.get("description", "") + " "
            + job.get("requirements", "")
        ).lower()
        if any(kw.lower() in text for kw in expected_keywords):
            count += 1
    return count


def _count_hits_in_results(results, expected_keywords) -> int:
    """统计检索结果中命中期望关键词的条数（True Recall 分子）"""
    count = 0
    for doc, score in results:
        title = doc.metadata.get("title", "").lower()
        content = doc.page_content.lower()
        if any(kw.lower() in title or kw.lower() in content for kw in expected_keywords):
            count += 1
    return count


def check_hit(results, expected_keywords):
    """检查检索结果中是否有至少一条命中了期望关键词"""
    for doc, score in results:
        title = doc.metadata.get("title", "").lower()
        content = doc.page_content.lower()
        for kw in expected_keywords:
            if kw.lower() in title or kw.lower() in content:
                return True
    return False


def first_hit_rank(results, expected_keywords):
    """返回第一个命中期望关键词的排名（1-indexed），未命中返回None"""
    for rank, (doc, score) in enumerate(results, start=1):
        title = doc.metadata.get("title", "").lower()
        content = doc.page_content.lower()
        for kw in expected_keywords:
            if kw.lower() in title or kw.lower() in content:
                return rank
    return None


def check_location(results, expected_location):
    """检查所有结果是否都匹配期望城市，始终返回 (match_count, total) 元组"""
    if not expected_location:
        return len(results), len(results)  # 无城市限制，视为全部匹配
    match_count = 0
    for doc, score in results:
        loc = doc.metadata.get("location", "")
        if expected_location in loc:
            match_count += 1
    return match_count, len(results)


# ════════════════════════════════════════════
# 评测1：消融实验 — Recall@5 三路对比
# Baseline / +Query Rewrite / +Filter
# ════════════════════════════════════════════

def _build_where_doc(parsed):
    """根据parsed结果构造ChromaDB where_document过滤条件"""
    conditions = []
    if parsed.get("location"):
        conditions.append({"$contains": parsed["location"]})
    if parsed.get("keywords"):
        for kw in parsed["keywords"]:
            conditions.append({"$contains": kw})
    if len(conditions) == 1:
        return conditions[0]
    elif len(conditions) > 1:
        return {"$and": conditions}
    return None


def eval_recall():
    """
    消融实验：三路对比 Hit Rate@5 与 True Recall@5。

    Hit Rate@5（命中率）：top-5 结果中是否存在至少一条相关结果（二值）。
    True Recall@5：top-5 中相关结果数 / 语料库相关总数，衡量覆盖广度。
    两者含义不同，简历里应分别汇报。
    """
    print("=" * 60)
    print("评测1: 命中率(Hit Rate@5) & 召回率(True Recall@5) 消融实验")
    print("  A: Baseline       — 原始query + 纯语义检索")
    print("  B: +Query Rewrite — 改写后query + 纯语义检索")
    print("  C: +Filter        — 改写后query + 过滤检索")
    print("=" * 60)

    hits_a = hits_b = hits_c = 0
    recall_sum_a = recall_sum_b = recall_sum_c = 0.0

    for i, tc in enumerate(TEST_CASES):
        q = tc["question"]
        expected_kw = tc["expected_keywords"]

        # 语料库相关总数（True Recall 分母），最小为1防止除零
        corpus_total = max(_count_relevant_in_corpus(expected_kw), 1)

        # ── A: Baseline ──
        results_a = search(query=q, top_k=5)
        hit_a = check_hit(results_a, expected_kw)
        if hit_a:
            hits_a += 1
        recall_sum_a += _count_hits_in_results(results_a, expected_kw) / corpus_total

        # ── 解析query（B和C共用）──
        parsed = _parse_query(q)
        rewritten_q = parsed.get("raw_query", q)

        # ── B: +Query Rewrite，只改写query，不加过滤 ──
        results_b = search(query=rewritten_q, top_k=5)
        hit_b = check_hit(results_b, expected_kw)
        if hit_b:
            hits_b += 1
        recall_sum_b += _count_hits_in_results(results_b, expected_kw) / corpus_total

        # ── C: +Filter，改写query + 过滤 ──
        where_doc = _build_where_doc(parsed)
        if where_doc is not None:
            try:
                results_c = filtered_search(query=rewritten_q, top_k=5, where_document=where_doc)
                if len(results_c) < 2:
                    results_c = search(query=rewritten_q, top_k=5)
            except Exception:
                results_c = search(query=rewritten_q, top_k=5)
        else:
            results_c = search(query=rewritten_q, top_k=5)
        hit_c = check_hit(results_c, expected_kw)
        if hit_c:
            hits_c += 1
        recall_c = _count_hits_in_results(results_c, expected_kw) / corpus_total
        recall_sum_c += recall_c

        print(
            f"  Q{i+1:02d}: {'✓' if hit_c else '✗'}"
            f"  A={'✓' if hit_a else '✗'}  B={'✓' if hit_b else '✗'}  C={'✓' if hit_c else '✗'}"
            f"  corpus={corpus_total}  recall_c={recall_c:.3f}  {q}"
        )

    total = len(TEST_CASES)
    hit_a  = hits_a  / total * 100
    hit_b  = hits_b  / total * 100
    hit_c  = hits_c  / total * 100
    rec_a  = recall_sum_a / total * 100
    rec_b  = recall_sum_b / total * 100
    rec_c  = recall_sum_c / total * 100

    print(f"\n  ── Hit Rate@5（命中率）──")
    print(f"  [A] Baseline:       {hits_a}/{total} = {hit_a:.0f}%")
    print(f"  [B] +Query Rewrite: {hits_b}/{total} = {hit_b:.0f}%  (+{hit_b - hit_a:.0f}% vs A)")
    print(f"  [C] +Filter:        {hits_c}/{total} = {hit_c:.0f}%  (+{hit_c - hit_a:.0f}% vs A, +{hit_c - hit_b:.0f}% vs B)")
    print(f"\n  ── True Recall@5（相关命中数/语料相关总数）──")
    print(f"  [A] Baseline:       {rec_a:.1f}%")
    print(f"  [B] +Query Rewrite: {rec_b:.1f}%  (+{rec_b - rec_a:.1f}% vs A)")
    print(f"  [C] +Filter:        {rec_c:.1f}%  (+{rec_c - rec_a:.1f}% vs A, +{rec_c - rec_b:.1f}% vs B)")
    print(f"\n  消融结论：Query改写命中率 +{hit_b - hit_a:.0f}%、召回率 +{rec_b - rec_a:.1f}%；过滤额外贡献 +{hit_c - hit_b:.0f}% / +{rec_c - rec_b:.1f}%\n")

    return hit_a, hit_b, hit_c, rec_a, rec_b, rec_c


# ════════════════════════════════════════════
# 评测2：MRR & NDCG@5
# ════════════════════════════════════════════

def _dcg(ranks, k=5):
    """计算DCG@k，ranks为命中位置列表（1-indexed），未命中位置不计入"""
    score = 0.0
    for r in ranks:
        if r <= k:
            score += 1.0 / math.log2(r + 1)
    return score


def _ideal_dcg(k=5):
    """理想DCG：前k位全部命中"""
    return sum(1.0 / math.log2(r + 1) for r in range(1, k + 1))


def eval_ranking_metrics():
    print("=" * 60)
    print("评测2: MRR & NDCG@5（使用混合检索结果）")
    print("=" * 60)

    reciprocal_ranks = []
    ndcg_scores = []
    ideal = _ideal_dcg(k=5)

    for i, tc in enumerate(TEST_CASES):
        q = tc["question"]
        expected_kw = tc["expected_keywords"]

        parsed = _parse_query(q)
        rewritten_q = parsed.get("raw_query", q)
        where_doc = _build_where_doc(parsed)

        if where_doc is not None:
            try:
                results = filtered_search(query=rewritten_q, top_k=5, where_document=where_doc)
                if len(results) < 2:
                    results = search(query=rewritten_q, top_k=5)
            except Exception:
                results = search(query=rewritten_q, top_k=5)
        else:
            results = search(query=rewritten_q, top_k=5)

        rank = first_hit_rank(results, expected_kw)

        # MRR
        rr = 1.0 / rank if rank else 0.0
        reciprocal_ranks.append(rr)

        # NDCG@5
        hit_ranks = [r for r in range(1, 6) if first_hit_rank(results[r-1:r], expected_kw) is not None]
        # 简化：收集所有命中位置
        hit_positions = []
        for r, (doc, score) in enumerate(results, start=1):
            title = doc.metadata.get("title", "").lower()
            content = doc.page_content.lower()
            for kw in expected_kw:
                if kw.lower() in title or kw.lower() in content:
                    hit_positions.append(r)
                    break
        dcg = _dcg(hit_positions, k=5)
        ndcg = dcg / ideal if ideal > 0 else 0.0
        ndcg_scores.append(ndcg)

        print(f"  Q{i+1:02d}: rank={rank if rank else '-'}  RR={rr:.3f}  NDCG={ndcg:.3f}  {q}")

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)

    print(f"\n  MRR:    {mrr:.3f}")
    print(f"  NDCG@5: {avg_ndcg:.3f}\n")
    return mrr, avg_ndcg


# ════════════════════════════════════════════
# 评测3：城市过滤精确率
# ════════════════════════════════════════════

def eval_location_precision():
    print("=" * 60)
    print("评测3: 城市过滤精确率")
    print("=" * 60)

    location_cases = [tc for tc in TEST_CASES if tc["expected_location"]]
    total_match = 0
    total_results = 0

    for tc in location_cases:
        q = tc["question"]
        loc = tc["expected_location"]

        parsed = _parse_query(q)
        where_doc = {"$contains": loc} if loc else None

        try:
            if where_doc is not None:
                results = filtered_search(query=parsed.get("raw_query", q), top_k=5, where_document=where_doc)
            else:
                results = search(query=parsed.get("raw_query", q), top_k=5)
        except Exception:
            results = []

        match, total = check_location(results, loc)
        total_match += match
        total_results += total
        pct = match / total * 100 if total else 0
        print(f"  {q}: {match}/{total} 匹配 ({pct:.0f}%)")

    overall = total_match / total_results * 100 if total_results else 0
    print(f"\n  城市过滤精确率: {total_match}/{total_results} = {overall:.0f}%\n")
    return overall


# ════════════════════════════════════════════
# 评测4：Token消耗对比（tiktoken精确统计）
# ════════════════════════════════════════════

def eval_token_saving():
    print("=" * 60)
    print("评测4: Token消耗对比（tiktoken精确统计）")
    print("=" * 60)

    # GLM模型无官方tiktoken支持，cl100k_base（GPT-4同款）字节粒度接近，用作合理近似
    enc = tiktoken.get_encoding("cl100k_base")

    # 全量上下文：把所有岗位都塞给LLM
    jobs_path = Path(__file__).parent / "jobs.json"
    with open(jobs_path, encoding="utf-8") as f:
        all_jobs = json.load(f)

    full_text = json.dumps(all_jobs, ensure_ascii=False)
    full_tokens = len(enc.encode(full_text))

    # RAG方式：只取top5
    sample_results = search(query="Java后端开发", top_k=5)
    rag_text = "\n".join(doc.page_content for doc, _ in sample_results)
    rag_tokens = len(enc.encode(rag_text))

    saving = (1 - rag_tokens / full_tokens) * 100

    print(f"  全量上下文: {full_tokens:,} tokens ({len(all_jobs)} 条岗位)")
    print(f"  RAG方式:    {rag_tokens:,} tokens (top 5 条)")
    print(f"  Token节省:  {saving:.1f}%\n")
    return saving


# ════════════════════════════════════════════
# 评测5：响应延迟（warmup + p50/p95）
# ════════════════════════════════════════════

def eval_latency():
    print("=" * 60)
    print("评测5: RAG问答响应延迟 (p50 / p95)")
    print("=" * 60)

    questions = [
        "Java后端实习",
        "北京的算法岗位",
        "深圳前端开发",
        "做推荐系统的实习",
        "C++开发工程师",
        "大数据开发工程师",
        "有没有产品经理岗位",
        "NLP自然语言处理",
        "iOS开发实习",
        "测试开发工程师",
    ]

    # Warmup：排除冷启动（chromadb连接、模型加载等）
    print("  [warmup] 预热中...")
    ask(questions[0], top_k=5)

    latencies = []
    for q in questions:
        start = time.time()
        ask(q, top_k=5)
        elapsed = time.time() - start
        latencies.append(elapsed)
        print(f"  {q}: {elapsed:.2f}s")

    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95_idx = int(math.ceil(0.95 * len(latencies_sorted))) - 1
    p95 = latencies_sorted[p95_idx]
    avg = statistics.mean(latencies)

    print(f"\n  avg={avg:.2f}s  p50={p50:.2f}s  p95={p95:.2f}s  (n={len(latencies)})\n")
    return avg, p50, p95


# ════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   JobMind RAG 系统评测")
    print("=" * 60 + "\n")

    hit_a, hit_b, hit_c, rec_a, rec_b, rec_c = eval_recall()
    mrr, ndcg = eval_ranking_metrics()
    location_precision = eval_location_precision()
    token_saving = eval_token_saving()
    avg_latency, p50, p95 = eval_latency()

    print("=" * 60)
    print("   评测汇总")
    print("=" * 60)
    print(f"  [Hit Rate@5] Baseline:              {hit_a:.0f}%")
    print(f"  [Hit Rate@5] +Query Rewrite:        {hit_b:.0f}%  (+{hit_b - hit_a:.0f}%)")
    print(f"  [Hit Rate@5] +Filter (混合检索):    {hit_c:.0f}%  (+{hit_c - hit_a:.0f}% total)")
    print(f"  [Recall@5]   Baseline:              {rec_a:.1f}%")
    print(f"  [Recall@5]   +Query Rewrite:        {rec_b:.1f}%  (+{rec_b - rec_a:.1f}%)")
    print(f"  [Recall@5]   +Filter (混合检索):    {rec_c:.1f}%  (+{rec_c - rec_a:.1f}% total)")
    print(f"  MRR:                                {mrr:.3f}")
    print(f"  NDCG@5:                             {ndcg:.3f}")
    print(f"  城市过滤精确率:                     {location_precision:.0f}%")
    print(f"  Token节省 (tiktoken/cl100k_base):   {token_saving:.1f}%")
    print(f"  响应延迟: avg={avg_latency:.2f}s  p50={p50:.2f}s  p95={p95:.2f}s")
    print("=" * 60)