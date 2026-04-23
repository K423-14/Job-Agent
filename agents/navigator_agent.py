"""
Navigator Agent
Playwright + LangChain ReAct 循环，定位招聘网站的岗位列表 JSON API。

增强特性:
  - scroll 动作：滚动页面触发懒加载
  - 自动验证：LLM 选定候选 XHR 后，用 requests 直接调该接口核实响应是否含岗位数据
  - 验证失败则继续搜索，保证传给下一 Agent 的 XHR 是真实有效的
"""
import json
import re
import threading
import requests as _requests
from pathlib import Path
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from .llm_client import get_llm

console = Console()

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_NOISE = re.compile(
    r"(analytics|sentry|log\b|beacon|metrics|collect|track|stat\b|ping"
    r"|alertserver|yuyan|render\.alipay|/config/index\.json"
    r"|-config/[a-z_]+\.json|/h5data_|/render/|cdn-cgi)",
    re.IGNORECASE,
)
MAX_ITERATIONS = 8   # 多给 2 轮用于滚动/验证失败后重试

# ─── 岗位响应评分 ──────────────────────────────────────────────────────────────

_JOB_MARKERS = [
    "title", "positiontitle", "jobtitle", "jobname",
    "position", "岗位", "职位", "职务",
    "department", "部门",
    "location", "worklocat", "城市",
    "salary", "薪资",
    "campus", "校园", "intern", "实习",
]

def _score_job_response(data: dict) -> int:
    """对响应 JSON 评分：越高越可能是岗位列表数据（≥3 视为通过）。"""
    text = json.dumps(data, ensure_ascii=False).lower()
    score = sum(1 for m in _JOB_MARKERS if m in text)

    # 额外加分：响应体包含「对象列表」
    def _has_object_list(obj, depth: int = 0) -> bool:
        if depth > 4:
            return False
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            return True
        if isinstance(obj, dict):
            return any(_has_object_list(v, depth + 1) for v in obj.values())
        return False

    if _has_object_list(data):
        score += 2
    return score


def _load_prompt() -> ChatPromptTemplate:
    text = (_PROMPTS_DIR / "navigator.md").read_text(encoding="utf-8")
    system, human = text.split("===HUMAN===", 1)
    system = system.replace("===SYSTEM===", "").strip()
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human.strip()),
    ])


class XHRCapture:
    """单条 XHR/Fetch 记录"""

    __slots__ = ("url", "method", "status", "request_body", "response_body")

    def __init__(self, url: str, method: str, status: int,
                 request_body: str, response_body: dict):
        self.url = url
        self.method = method
        self.status = status
        self.request_body = request_body
        self.response_body = response_body

    def to_summary(self, max_body: int = 600) -> dict:
        body_str = json.dumps(self.response_body, ensure_ascii=False)
        return {
            "url": self.url,
            "method": self.method,
            "status": self.status,
            "response_preview": body_str[:max_body],
        }


class NavigatorAgent:
    """
    ReAct 循环：
      Observe  — Playwright 拦截 XHR/Fetch 响应 + 抓取页面摘要
      Think    — LangChain chain 分析，返回结构化决策
      Act      — 执行 found / goto / click / scroll / give_up
      Verify   — 对 found 候选发起真实 HTTP 请求，评分确认后才返回
    """

    def __init__(self, homepage: str):
        self.homepage = homepage
        self.xhr_log: list[XHRCapture] = []
        self._failed_indices: set[int] = set()   # 验证失败过的 xhr_index
        self._element_handles: list = []          # index → ElementHandle
        self._prompt = _load_prompt()

    # ── 响应拦截 ──────────────────────────────────────────────────────

    def _on_response(self, response) -> None:
        if "json" not in response.headers.get("content-type", ""):
            return
        if _NOISE.search(response.url):
            return
        try:
            body = response.json()
        except Exception:
            return
        if not isinstance(body, dict):
            return
        try:
            req_body = response.request.post_data or ""
        except Exception:
            req_body = ""
        self.xhr_log.append(XHRCapture(
            url=response.url,
            method=response.request.method,
            status=response.status,
            request_body=req_body[:1000],
            response_body=body,
        ))

    # ── 页面摘要 ──────────────────────────────────────────────────────

    @staticmethod
    def _page_text_snippet(page, max_len: int = 1500) -> str:
        try:
            html = page.content()
        except Exception:
            return ""
        html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", html, flags=re.DOTALL)
        html = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", html).strip()[:max_len]

    # ── 可交互元素快照 ────────────────────────────────────────────────

    _JOB_KEYWORDS = (
        "招聘", "职位", "岗位", "实习", "校园", "社会", "加入", "投递", "应聘", "求职",
        "job", "career", "position", "intern", "recruit", "apply", "campus", "talent",
        "hire", "opportunity", "more", "search", "filter", "查看更多", "筛选",
    )

    def _snapshot_elements(self, page, max_elements: int = 10) -> str:
        """提取与招聘相关的可交互元素，分配 index，存 handle 供 click 使用。"""
        self._element_handles.clear()
        rows = []
        try:
            elements = page.query_selector_all("a[href], button, [role=button], [role=tab]")
            for el in elements:
                if len(self._element_handles) >= max_elements:
                    break
                try:
                    text = (el.inner_text() or "").strip().replace("\n", " ")[:40]
                    if not text:
                        continue
                    text_lower = text.lower()
                    if not any(kw in text_lower for kw in self._JOB_KEYWORDS):
                        continue
                    tag = el.evaluate("e => e.tagName.toLowerCase()")
                    href = el.get_attribute("href") or ""
                    idx = len(self._element_handles)
                    self._element_handles.append(el)
                    hint = f"[{idx}] <{tag}> {text}"
                    if href:
                        hint += f" → {href[:60]}"
                    rows.append(hint)
                except Exception:
                    continue
        except Exception:
            pass
        return "\n".join(rows) if rows else "（无匹配元素）"

    # ── Think (流式输出) ──────────────────────────────────────────────

    _TOP_XHR = 8  # 最多送给 LLM 的 XHR 数量

    def _think(self, page) -> dict:
        # 预评分，只把最可能是岗位数据的 Top N 送给 LLM
        scored = sorted(
            ((i, x, _score_job_response(x.response_body)) for i, x in enumerate(self.xhr_log)),
            key=lambda t: t[2], reverse=True,
        )[:self._TOP_XHR]

        summaries = []
        for i, x, score in scored:
            s = x.to_summary()
            s["xhr_index"] = i          # 保留原始下标，LLM 用此索引回传
            s["pre_score"] = score
            if i in self._failed_indices:
                s["_verify_failed"] = True
            summaries.append(s)

        messages = self._prompt.format_messages(
            page_url=page.url,
            page_title=page.title(),
            page_text=self._page_text_snippet(page),
            xhr_count=len(self.xhr_log),
            xhr_list=json.dumps(summaries, ensure_ascii=False, indent=2)
                     if summaries else "（暂无）",
            elements=self._snapshot_elements(page),
        )

        llm = get_llm(streaming=True, read_timeout=45.0)
        chunks: list[str] = []
        exc_holder: list[Exception] = []

        def _stream():
            try:
                for chunk in llm.stream(messages):
                    content = chunk.content  # type: ignore[attr-defined]
                    if isinstance(content, str) and content:
                        chunks.append(content)
            except Exception as e:
                exc_holder.append(e)

        t = threading.Thread(target=_stream, daemon=True)
        try:
            with Live(console=console, refresh_per_second=8) as live:
                live.update(Spinner("dots", text=Text("Navigator LLM 思考中...", style="cyan")))
                t.start()
                elapsed = 0.0
                while t.is_alive():
                    t.join(timeout=0.3)
                    elapsed += 0.3
                    partial = "".join(chunks)
                    if partial:
                        # 显示流式输出（截取最后 120 字符避免刷屏）
                        preview = partial[-120:].replace("\n", " ")
                        live.update(Text(f"[Navigator] {preview}", style="dim cyan"))
                    if elapsed >= 50:
                        console.print("[red]Navigator LLM 超时[/red]")
                        return {"action": "give_up"}
        except KeyboardInterrupt:
            return {"action": "give_up"}

        if exc_holder:
            console.print(f"[yellow]LLM 调用异常: {exc_holder[0]}[/yellow]")
            return {"action": "give_up"}

        raw = "".join(chunks).strip()
        # 提取 JSON 块
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            console.print(f"[yellow]LLM 未返回 JSON: {raw[:200]}[/yellow]")
            return {"action": "give_up"}
        try:
            result = json.loads(m.group(0))
            return result if isinstance(result, dict) else {"action": "give_up"}
        except json.JSONDecodeError:
            console.print(f"[yellow]JSON 解析失败: {raw[:200]}[/yellow]")
            return {"action": "give_up"}

    # ── 验证候选 XHR ──────────────────────────────────────────────────

    def _verify_xhr(self, xhr: XHRCapture) -> Optional[XHRCapture]:
        """
        用 requests 直接访问该端点，检查响应是否真的含有岗位数据。
        成功返回更新了 response_body 的新 XHRCapture，失败返回 None。
        """
        console.print(f"  [dim]→ 验证接口: {xhr.method} {xhr.url}[/dim]")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, */*",
            "Content-Type": "application/json;charset=UTF-8",
        }
        try:
            if xhr.method.upper() == "POST":
                body = json.loads(xhr.request_body) if xhr.request_body else {}
                resp = _requests.post(xhr.url, json=body, headers=headers, timeout=15)
            else:
                resp = _requests.get(xhr.url, headers=headers, timeout=15)

            if resp.status_code != 200:
                console.print(f"  [yellow]接口返回 HTTP {resp.status_code}，跳过[/yellow]")
                return None

            data = resp.json()
            score = _score_job_response(data)
            console.print(f"  [dim]岗位数据评分: {score}/10[/dim]")
            if score >= 3:
                console.print(f"  [green]✓ 验证通过（评分 {score}）[/green]")
                return XHRCapture(
                    url=xhr.url,
                    method=xhr.method,
                    status=resp.status_code,
                    request_body=xhr.request_body,
                    response_body=data,
                )
            else:
                console.print(f"  [yellow]评分 {score} 不足，非岗位数据，继续搜索[/yellow]")
                return None

        except Exception as e:
            console.print(f"  [yellow]验证请求失败: {e}[/yellow]")
            return None

    # ── 主循环 ────────────────────────────────────────────────────────

    def run(self) -> Optional[XHRCapture]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            console.print("[red]请先安装: pip install playwright && playwright install chromium[/red]")
            return None

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
            ))
            page = ctx.new_page()
            page.on("response", self._on_response)

            console.print(f"[cyan]Navigator: 打开 {self.homepage}[/cyan]")
            page.goto(self.homepage, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(2000)

            for i in range(MAX_ITERATIONS):
                console.print(
                    f"[cyan]第 {i + 1}/{MAX_ITERATIONS} 轮 "
                    f"| XHR: {len(self.xhr_log)} 条（已排除 {len(self._failed_indices)} 个无效）"
                    f" | LLM 分析中...[/cyan]"
                )
                decision = self._think(page)
                action = decision.get("action", "give_up")
                console.print(f"  LLM 决策: [yellow]{decision}[/yellow]")

                if action == "found":
                    idx = decision.get("xhr_index", -1)
                    if not (0 <= idx < len(self.xhr_log)):
                        console.print("[yellow]xhr_index 越界，等待更多 XHR[/yellow]")
                        page.wait_for_timeout(2000)
                        continue

                    candidate = self.xhr_log[idx]
                    verified = self._verify_xhr(candidate)
                    if verified is not None:
                        console.print(f"[green]✓ 岗位 API 已确认: {verified.url}[/green]")
                        self._dump_xhr()
                        browser.close()
                        return verified
                    else:
                        self._failed_indices.add(idx)
                        console.print("[yellow]候选接口验证失败，继续搜索其他 XHR...[/yellow]")
                        page.wait_for_timeout(1000)

                elif action == "scroll":
                    console.print("  → 滚动页面到底部，等待懒加载...")
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(2500)
                    # 再等等新 XHR 进来
                    page.wait_for_load_state("networkidle", timeout=8000)

                elif action == "goto":
                    url = decision.get("url", "")
                    if url:
                        console.print(f"  → 跳转: {url}")
                        self.xhr_log.clear()
                        self._failed_indices.clear()
                        try:
                            page.goto(url, wait_until="networkidle", timeout=30000)
                            page.wait_for_timeout(2000)
                        except Exception as e:
                            console.print(f"  [yellow]跳转失败: {e}[/yellow]")

                elif action == "click":
                    idx = decision.get("element_index", -1)
                    handle = (self._element_handles[idx]
                              if isinstance(idx, int) and 0 <= idx < len(self._element_handles)
                              else None)
                    if handle:
                        console.print(f"  → 点击元素 [{idx}]")
                        try:
                            handle.scroll_into_view_if_needed()
                            handle.click()
                            page.wait_for_load_state("networkidle", timeout=15000)
                            page.wait_for_timeout(1500)
                        except Exception as e:
                            console.print(f"  [yellow]点击失败: {e}[/yellow]")

                elif action == "give_up":
                    console.print("[red]Navigator 放弃[/red]")
                    break

            self._dump_xhr()
            browser.close()

        console.print("[red]Navigator 达到最大迭代次数，未找到岗位 API[/red]")
        return None

    def _dump_xhr(self) -> None:
        """将捕获的 XHR 响应写到 cache/navigator_xhr_dump.json，便于人工检查。"""
        dump = [x.to_summary(max_body=2000) for x in self.xhr_log]
        cache_dir = Path(__file__).parent.parent / "cache"
        cache_dir.mkdir(exist_ok=True)
        out = cache_dir / "navigator_xhr_dump.json"
        out.write_text(
            json.dumps(dump, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"[dim]XHR 转储已写入 {out}（共 {len(dump)} 条）[/dim]")

