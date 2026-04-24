"""
test_evaluation.py
RAG 系统评测 — 消融实验 + 排序质量 + 城市过滤 + Token 节省

优化点：
  - 预计算阶段用 ThreadPoolExecutor 并行调所有 LLM（parse/multi_query/hyde）
  - BM25 + ChromaDB 单例（hybrid_retriever 内部缓存，只建一次）
  - eval_recall 结果缓存供后续评测复用，不重复跑检索
  - 所有 print 加 flush=True，方便实时观察进度
"""
import json
import math
import time
import sys
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).parent))

from rag.vector_store import search
from rag.hybrid_retriever import hybrid_search
from rag.reranker import rerank
from rag.query_expansion import multi_query, hyde
from rag.qa_chain import _parse_query

# ─── 测试用例 ───────────────────────────────────────────────────────────────

TEST_CASES = [
    {"question": "有没有Java开发的实习",          "expected_keywords": ["Java"],                         "expected_location": None},
    {"question": "北京的Java开发岗位",             "expected_keywords": ["Java"],                         "expected_location": "北京"},
    {"question": "深圳有什么算法岗位",             "expected_keywords": ["算法"],                         "expected_location": "深圳"},
    {"question": "做推荐系统的实习",               "expected_keywords": ["推荐"],                         "expected_location": None},
    {"question": "C++开发工程师",                  "expected_keywords": ["C++"],                          "expected_location": None},
    {"question": "北京的大数据开发",               "expected_keywords": ["大数据"],                       "expected_location": "北京"},
    {"question": "有没有产品经理的实习岗位",       "expected_keywords": ["产品"],                         "expected_location": None},
    {"question": "深圳做多模态的岗位",             "expected_keywords": ["多模态"],                       "expected_location": "深圳"},
    {"question": "前端开发实习生",                 "expected_keywords": ["前端"],                         "expected_location": None},
    {"question": "北京的Golang开发",               "expected_keywords": ["Go", "Golang"],                 "expected_location": "北京"},
    {"question": "做运营的岗位",                   "expected_keywords": ["运营"],                         "expected_location": None},
    {"question": "测试开发工程师",                 "expected_keywords": ["测试"],                         "expected_location": None},
    {"question": "深圳的后端开发",                 "expected_keywords": ["后端", "开发"],                 "expected_location": "深圳"},
    {"question": "有什么设计相关的实习",           "expected_keywords": ["设计"],                         "expected_location": None},
    {"question": "做搜索方向的算法",               "expected_keywords": ["搜索"],                         "expected_location": None},
    {"question": "iOS开发实习",                    "expected_keywords": ["iOS"],                          "expected_location": None},
    {"question": "数据分析相关的岗位",             "expected_keywords": ["数据"],                         "expected_location": None},
    {"question": "做广告算法的实习",               "expected_keywords": ["广告"],                         "expected_location": None},
    {"question": "北京安全相关岗位",               "expected_keywords": ["安全"],                         "expected_location": "北京"},
    {"question": "有没有NLP自然语言处理的岗位",    "expected_keywords": ["NLP", "自然语言"],              "expected_location": None},
    {"question": "深圳的C++开发",                  "expected_keywords": ["C++"],                          "expected_location": "深圳"},
    {"question": "杭州的搜索算法岗位",             "expected_keywords": ["搜索", "算法"],                 "expected_location": "杭州"},
    {"question": "写代码的后端实习",               "expected_keywords": ["Java","Python","Go","C++","后端","服务端"], "expected_location": None},
    {"question": "搞推荐的算法实习",               "expected_keywords": ["推荐系统","推荐","RecSys","算法"], "expected_location": None},
    {"question": "做人工智能方向的技术岗",         "expected_keywords": ["AI","人工智能","机器学习","深度学习","算法"], "expected_location": None},
    {"question": "北京想做和图像视频相关的研究",   "expected_keywords": ["视觉","CV","图像","视频","多模态"], "expected_location": "北京"},
]

# ─── 语料库工具 ──────────────────────────────────────────────────────────────

_CORPUS_CACHE: list | None = None

def _load_corpus() -> list:
    global _CORPUS_CACHE
    if _CORPUS_CACHE is None:
        jobs_path = Path(__file__).parent / "jobs.json"
        with open(jobs_path, encoding="utf-8") as f:
            _CORPUS_CACHE = json.load(f)
    return _CORPUS_CACHE

def _count_relevant_in_corpus(keywords) -> int:
    count = 0
    for job in _load_corpus():
        text = (job.get("title","") + " " + job.get("description","") + " " + job.get("requirements","")).lower()
        if any(kw.lower() in text for kw in keywords):
            count += 1
    return count

def _count_hits(results, keywords) -> int:
    count = 0
    for doc, _ in results:
        text = (doc.metadata.get("title","") + " " + doc.page_content).lower()
        if any(kw.lower() in text for kw in keywords):
            count += 1
    return count

def check_hit(results, keywords) -> bool:
    return _count_hits(results, keywords) > 0

def first_hit_rank(results, keywords) -> int | None:
    for rank, (doc, _) in enumerate(results, start=1):
        text = (doc.metadata.get("title","") + " " + doc.page_content).lower()
        if any(kw.lower() in text for kw in keywords):
            return rank
    return None

def check_location(results, loc) -> tuple[int, int]:
    if not loc:
        return len(results), len(results)
    match = sum(1 for doc, _ in results if loc in doc.metadata.get("location",""))
    return match, len(results)

def _build_where_doc(parsed) -> dict | None:
    conds = []
    if parsed.get("location"):
        conds.append({"$contains": parsed["location"]})
    for kw in parsed.get("keywords", []):
        conds.append({"$contains": kw})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


# ─── 预计算：并行调所有 LLM ──────────────────────────────────────────────────

def _precompute(questions: list[str], workers: int = 6) -> dict[str, dict]:
    """
    对所有 question 并行调用 _parse_query / multi_query / hyde。
    每个 question 内部三个任务也并行，整体用一个 ThreadPoolExecutor。
    返回 {question: {parsed, mqs, hyde_doc}} 字典。
    """
    results: dict[str, dict] = {q: {} for q in questions}

    def _parse(q):    return ("parsed",    q, _parse_query(q))
    def _mq(q):       return ("mqs",       q, multi_query(q))
    def _hyde(q):     return ("hyde_doc",  q, hyde(q))

    tasks = []
    for q in questions:
        tasks += [_parse, _mq, _hyde]

    print(f"[预计算] 并行调用 LLM（{len(questions)} 个问题 × 3 任务 = {len(questions)*3} 次调用）...", flush=True)
    t0 = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, q): (fn.__name__, q) for q in questions for fn in [_parse, _mq, _hyde]}
        for fut in as_completed(futs):
            try:
                key, q, val = fut.result()
                results[q][key] = val
                completed += 1
                print(f"  [{completed}/{len(futs)}] {futs[fut][0]}({q[:18]}...) 完成", flush=True)
            except Exception as e:
                _, q = futs[fut]
                print(f"  [!] {q[:18]} 调用失败: {e}", flush=True)

    print(f"[预计算] 完成，耗时 {time.time()-t0:.1f}s\n", flush=True)
    return results


# ─── 评测1：消融实验 ─────────────────────────────────────────────────────────

# 全局缓存，供后续评测复用
_cache_A: dict[str, list] = {}
_cache_B: dict[str, list] = {}
_cache_C: dict[str, list] = {}


def eval_recall(precomp: dict) -> tuple:
    print("=" * 60, flush=True)
    print("评测1: Hit Rate@5 & True Recall@5 消融实验", flush=True)
    print("  A: Baseline      — 原始query + 纯语义 vector", flush=True)
    print("  B: +Hybrid       — BM25 + Vector + RRF", flush=True)
    print("  C: Full Pipeline — Hybrid + Multi-Query + HyDE + Rerank", flush=True)
    print("=" * 60, flush=True)

    hits_a = hits_b = hits_c = 0
    rec_sum_a = rec_sum_b = rec_sum_c = 0.0

    for i, tc in enumerate(TEST_CASES):
        q = tc["question"]
        kws = tc["expected_keywords"]
        corpus_n = max(_count_relevant_in_corpus(kws), 1)
        pc = precomp.get(q, {})

        parsed    = pc.get("parsed",   {"raw_query": q, "location": None, "keywords": []})
        mqs       = pc.get("mqs",      [])
        hyde_doc  = pc.get("hyde_doc", "")
        rewritten = parsed.get("raw_query", q)
        where_doc = _build_where_doc(parsed)

        # A: 纯 vector
        ra = search(query=q, top_k=5)
        _cache_A[q] = ra
        ha = check_hit(ra, kws)
        if ha: hits_a += 1
        rec_sum_a += _count_hits(ra, kws) / corpus_n

        # B: Hybrid（BM25+Vector+RRF，无扩展，无rerank）
        rb = hybrid_search(query=rewritten, top_k=5, recall_k=20, where_document=where_doc)
        _cache_B[q] = rb
        hb = check_hit(rb, kws)
        if hb: hits_b += 1
        rec_sum_b += _count_hits(rb, kws) / corpus_n

        # C: Full Pipeline
        extras = mqs + ([hyde_doc] if hyde_doc else [])
        candidates = hybrid_search(query=rewritten, top_k=20, recall_k=20,
                                   where_document=where_doc, extra_vector_queries=extras)
        rc = rerank(q, candidates, top_k=5)
        _cache_C[q] = rc
        hc = check_hit(rc, kws)
        if hc: hits_c += 1
        rec_c = _count_hits(rc, kws) / corpus_n
        rec_sum_c += rec_c

        print(f"  Q{i+1:02d}: A={'Y' if ha else 'N'}  B={'Y' if hb else 'N'}"
              f"  C={'Y' if hc else 'N'}  corpus={corpus_n}  {q}", flush=True)

    n = len(TEST_CASES)
    ra_, rb_, rc_ = hits_a/n*100, hits_b/n*100, hits_c/n*100
    reca, recb, recc = rec_sum_a/n*100, rec_sum_b/n*100, rec_sum_c/n*100

    print(f"\n  ── Hit Rate@5 ──", flush=True)
    print(f"  [A] Baseline:      {hits_a}/{n} = {ra_:.0f}%", flush=True)
    print(f"  [B] +Hybrid:       {hits_b}/{n} = {rb_:.0f}%  ({rb_-ra_:+.0f}% vs A)", flush=True)
    print(f"  [C] Full Pipeline: {hits_c}/{n} = {rc_:.0f}%  ({rc_-ra_:+.0f}% vs A, {rc_-rb_:+.0f}% vs B)", flush=True)
    print(f"\n  ── True Recall@5 ──", flush=True)
    print(f"  [A] Baseline:      {reca:.1f}%", flush=True)
    print(f"  [B] +Hybrid:       {recb:.1f}%  ({recb-reca:+.1f}% vs A)", flush=True)
    print(f"  [C] Full Pipeline: {recc:.1f}%  ({recc-reca:+.1f}% vs A, {recc-recb:+.1f}% vs B)\n", flush=True)

    return ra_, rb_, rc_, reca, recb, recc


# ─── 评测2：MRR & NDCG@5 ────────────────────────────────────────────────────

def _dcg(positions, k=5) -> float:
    return sum(1.0/math.log2(r+1) for r in positions if r <= k)

def _ideal_dcg(k=5) -> float:
    return sum(1.0/math.log2(r+1) for r in range(1, k+1))


def eval_ranking_metrics() -> dict:
    """对 A/B/C 三组分别算 MRR & NDCG@5，用于简历对比。"""
    print("=" * 60, flush=True)
    print("评测2: MRR & NDCG@5（三组对比）", flush=True)
    print("=" * 60, flush=True)

    ideal = _ideal_dcg(k=5)
    out: dict[str, tuple[float, float]] = {}

    for name, cache in [("Baseline", _cache_A), ("+Hybrid", _cache_B), ("Full", _cache_C)]:
        rrs, ndcgs = [], []
        for tc in TEST_CASES:
            res = cache.get(tc["question"], [])
            kws = tc["expected_keywords"]

            rank = first_hit_rank(res, kws)
            rrs.append(1.0/rank if rank else 0.0)

            hit_pos = [r for r, (doc, _) in enumerate(res, 1)
                       if any(kw.lower() in (doc.metadata.get("title","")+" "+doc.page_content).lower()
                              for kw in kws)]
            ndcgs.append(_dcg(hit_pos)/ideal if ideal else 0.0)

        mrr = sum(rrs)/len(rrs)
        ndcg = sum(ndcgs)/len(ndcgs)
        out[name] = (mrr, ndcg)
        print(f"  [{name:9s}] MRR={mrr:.3f}  NDCG@5={ndcg:.3f}", flush=True)
    print(flush=True)
    return out


# ─── 评测3：城市过滤精确率 ───────────────────────────────────────────────────

def eval_location_precision() -> dict:
    """对 A/B/C 三组分别算城市过滤精确率。"""
    print("=" * 60, flush=True)
    print("评测3: 城市过滤精确率（三组对比）", flush=True)
    print("=" * 60, flush=True)

    loc_cases = [tc for tc in TEST_CASES if tc["expected_location"]]
    out: dict[str, float] = {}

    for name, cache in [("Baseline", _cache_A), ("+Hybrid", _cache_B), ("Full", _cache_C)]:
        total_match = total_n = 0
        for tc in loc_cases:
            match, n = check_location(cache.get(tc["question"], []), tc["expected_location"])
            total_match += match
            total_n += n
        pct = total_match/total_n*100 if total_n else 0
        out[name] = pct
        print(f"  [{name:9s}] {total_match}/{total_n} = {pct:.0f}%", flush=True)
    print(flush=True)
    return out


# ─── 评测4：Token 节省 ───────────────────────────────────────────────────────

def eval_token_saving() -> float:
    print("=" * 60, flush=True)
    print("评测4: Token 消耗对比（tiktoken/cl100k_base 近似）", flush=True)
    print("=" * 60, flush=True)

    enc = tiktoken.get_encoding("cl100k_base")
    jobs_path = Path(__file__).parent / "jobs.json"
    with open(jobs_path, encoding="utf-8") as f:
        all_jobs = json.load(f)

    full_tokens = len(enc.encode(json.dumps(all_jobs, ensure_ascii=False)))

    sample = search(query="Java后端开发", top_k=5)
    rag_tokens = len(enc.encode("\n".join(doc.page_content for doc, _ in sample)))

    saving = (1 - rag_tokens/full_tokens)*100
    print(f"  全量上下文: {full_tokens:,} tokens ({len(all_jobs)} 条岗位)", flush=True)
    print(f"  RAG方式:    {rag_tokens:,} tokens (top 5 条)", flush=True)
    print(f"  Token节省:  {saving:.1f}%\n", flush=True)
    return saving


# ─── 主入口 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_total = time.time()
    print("\n" + "=" * 60, flush=True)
    print("   JobMind RAG 系统评测", flush=True)
    print("=" * 60 + "\n", flush=True)

    # ① 并行预计算所有 LLM 调用
    all_qs = [tc["question"] for tc in TEST_CASES]
    precomp = _precompute(all_qs, workers=6)

    # ② 消融实验（BM25+vector 检索 + rerank，无额外 LLM）
    rate_a, rate_b, rate_c, rec_a, rec_b, rec_c = eval_recall(precomp)

    # ③ MRR & NDCG（三组对比，用缓存）
    rank_out = eval_ranking_metrics()

    # ④ 城市过滤（三组对比，用缓存）
    loc_out = eval_location_precision()

    # ⑤ Token 节省（无 LLM）
    token_saving = eval_token_saving()

    mrr_a, ndcg_a = rank_out["Baseline"]
    mrr_b, ndcg_b = rank_out["+Hybrid"]
    mrr_c, ndcg_c = rank_out["Full"]
    loc_a, loc_b, loc_c = loc_out["Baseline"], loc_out["+Hybrid"], loc_out["Full"]

    def _delta_pct(new: float, base: float) -> str:
        return f"{(new-base)/base*100:+.0f}%" if base else "n/a"

    elapsed = time.time() - t_total
    print("=" * 60, flush=True)
    print("   评测汇总  (Baseline → +Hybrid → Full Pipeline | 相比 Baseline)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Hit Rate@5 : {rate_a:5.1f}% → {rate_b:5.1f}% → {rate_c:5.1f}%   ({_delta_pct(rate_c, rate_a)})", flush=True)
    print(f"  Recall@5   : {rec_a:5.1f}% → {rec_b:5.1f}% → {rec_c:5.1f}%   ({_delta_pct(rec_c, rec_a)})", flush=True)
    print(f"  MRR        : {mrr_a:5.3f} → {mrr_b:5.3f} → {mrr_c:5.3f}   ({_delta_pct(mrr_c, mrr_a)})", flush=True)
    print(f"  NDCG@5     : {ndcg_a:5.3f} → {ndcg_b:5.3f} → {ndcg_c:5.3f}   ({_delta_pct(ndcg_c, ndcg_a)})", flush=True)
    print(f"  城市过滤   : {loc_a:5.1f}% → {loc_b:5.1f}% → {loc_c:5.1f}%   ({_delta_pct(loc_c, loc_a)})", flush=True)
    print(f"  Token节省  : {token_saving:.1f}%", flush=True)
    print(f"  总耗时     : {elapsed:.1f}s", flush=True)
    print("=" * 60, flush=True)
