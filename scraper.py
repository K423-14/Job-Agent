"""
腾讯校招岗位爬虫 MVP
目标: https://join.qq.com/post.html

运行方式:
  pip install requests playwright rich
  playwright install chromium
  python scraper.py

说明:
  - 优先直接调用 API (快, 无需浏览器)
  - 若 API 有 CSRF/Sign 保护则自动切换 Playwright 拦截模式
  - 结果保存到 jobs.json 与 jobs.csv
"""

import json
import sys
import time
import csv
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# ─────────────────────────── 加载配置 ───────────────────────────

def _resolve_config_path() -> Path:
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            return Path(args[i + 1])
    return Path(__file__).parent / "site_configs" / "site_config_campus_kuaishou_cn.json"

CONFIG_PATH = _resolve_config_path()
with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = json.load(f)

API_CFG   = CONFIG["api"]["list"]
RESP_MAP  = CONFIG["response_mapping"]
CRAWL_CFG = CONFIG["crawl"]
FILTERS   = CONFIG["filters"]


# ─────────────────────────── 工具函数 ───────────────────────────

def job_fingerprint(job: dict) -> str:
    """生成岗位唯一指纹，仅基于岗位ID，确保改名/换地不产生重复记录"""
    return hashlib.md5(str(job.get('id', '')).encode()).hexdigest()


def job_content_hash(job: dict) -> str:
    """生成内容哈希，用于检测岗位信息是否发生变更"""
    stable = {k: v for k, v in job.items()
              if k not in ('fingerprint', 'content_hash', 'first_seen')}
    return hashlib.md5(
        json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def extract_field(raw: dict, path: str):
    """
    按点分路径提取字段，支持以下格式：
      普通路径:        "data.list"         → raw["data"]["list"]
      列表子字段:      "cities[].name"     → [x["name"] for x in raw["cities"]] → join 为字符串
      嵌套列表子字段:  "data.list[].name"  → 先走到 raw["data"]["list"]，再提取每个元素的 ["name"]
    """
    # 检查路径中是否含 [].subKey 后缀
    list_match = re.match(r'^(.+?)\[\]\.(.+)$', path)
    if list_match:
        list_path, sub_key = list_match.group(1), list_match.group(2)
        container = extract_field(raw, list_path)   # 递归取列表
        if isinstance(container, list):
            values = []
            for item in container:
                if isinstance(item, dict):
                    v = item.get(sub_key)
                    if v is not None:
                        values.append(str(v))
            return ", ".join(values)
        return ""

    parts = path.split(".")
    val = raw
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
            if val is None:
                return ""
        else:
            return ""
    return val if val is not None else ""


def map_job(raw_item: dict) -> dict:
    """将 API 返回的原始字段映射到统一结构"""
    fields = RESP_MAP["fields"]
    job = {}
    for key, src in fields.items():
        if not src:
            job[key] = ""
        elif src.startswith("http") or (re.search(r'\{[^}]+\}', src) and "[]" not in src):
            # URL 模板，替换占位符 {xxx} 为对应字段值
            def _replace_placeholder(m):
                field_name = m.group(1)
                return str(raw_item.get(field_name, ""))
            job[key] = re.sub(r'\{([^}]+)\}', _replace_placeholder, src)
        elif "[]." in src:
            # 列表子字段提取表达式，如 "workLocationDicts[].name"
            job[key] = extract_field(raw_item, src)
        else:
            val = raw_item.get(src, "")
            # 兜底：普通列表字段（如 ["北京", "上海"]）直接 join
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val) if val else ""
            job[key] = val

    # 若没有 publish_time，记录首次爬取时间
    if not job.get("publish_time"):
        job["first_seen"] = datetime.now().isoformat()
    else:
        job["first_seen"] = job["publish_time"]

    job["fingerprint"] = job_fingerprint(job)
    job["content_hash"] = job_content_hash(job)
    return job


def filter_job(job: dict) -> bool:
    """根据配置过滤岗位（关键词已由 API 服务端过滤，此处仅做地点等本地筛选）"""
    type_filter = FILTERS.get("type", [])
    if type_filter and job.get("type") not in type_filter:
        # 不强制过滤 type，仅打标签
        pass

    locs = FILTERS.get("location", [])
    if locs and job.get("location") not in locs:
        return False

    return True


# ─────────────────────────── 直接 API 模式 ───────────────────────────

class DirectAPIScraper:
    """直接调用 JSON API（最快）"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(API_CFG["headers"])

    def fetch_page(self, page: int) -> Optional[dict]:
        params = dict(API_CFG["params"])
        params[API_CFG["pagination"]["page_param"]] = page
        params[API_CFG["pagination"]["size_param"]] = API_CFG["pagination"]["page_size"]
        params["keyword"] = FILTERS.get("keyword", "")

        try:
            if API_CFG["method"] == "GET":
                resp = self.session.get(
                    API_CFG["url"],
                    params=params,
                    timeout=CRAWL_CFG["timeout_seconds"]
                )
            else:
                resp = self.session.post(
                    API_CFG["url"],
                    params={"timestamp": int(time.time() * 1000)},
                    json=params,
                    timeout=CRAWL_CFG["timeout_seconds"]
                )

            resp.raise_for_status()
            data = resp.json()

            # 验证响应格式
            success_val = extract_field(data, RESP_MAP["success_flag"])
            if str(success_val) != str(RESP_MAP["success_value"]):
                console.print(f"[yellow]API 返回非成功状态: {success_val}[/yellow]")
                return None

            return data

        except requests.exceptions.JSONDecodeError:
            console.print("[yellow]响应非 JSON，可能需要 Playwright 模式[/yellow]")
            return None
        except requests.exceptions.RequestException as e:
            console.print(f"[red]请求失败: {e}[/red]")
            return None

    def scrape_all(self) -> list[dict]:
        jobs = []
        page = API_CFG["pagination"]["start_page"]
        page_size = API_CFG["pagination"]["page_size"]
        total = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("正在抓取岗位列表...", total=None)

            while True:
                progress.update(task, description=f"正在抓取第 {page} 页...")
                data = self.fetch_page(page)

                if data is None:
                    console.print("[yellow]API 返回异常，停止[/yellow]")
                    break

                raw_list = extract_field(data, RESP_MAP["data_path"])
                if not isinstance(raw_list, list):
                    console.print(f"[red]data_path '{RESP_MAP['data_path']}' 提取失败[/red]")
                    console.print(f"响应 keys: {list(data.keys())}")
                    break

                if total is None:
                    total_raw = extract_field(data, RESP_MAP["total_path"])
                    try:
                        total = int(total_raw)
                        total_pages = (total + page_size - 1) // page_size
                        progress.update(task, total=total_pages)
                        console.print(f"[cyan]共 {total} 个岗位，约 {total_pages} 页[/cyan]")
                    except (ValueError, TypeError):
                        total = 0

                if not raw_list:
                    console.print("没有更多数据了")
                    break

                for item in raw_list:
                    job = map_job(item)
                    if filter_job(job):
                        jobs.append(job)

                progress.advance(task)

                # 判断是否还有下一页
                fetched = page * page_size
                if total and fetched >= total:
                    break

                page += 1
                time.sleep(CRAWL_CFG["request_delay_ms"] / 1000)

        return jobs


# ─────────────────────────── Playwright 拦截模式 ───────────────────────────

class PlaywrightAPIScraper:
    """
    通过 Playwright 启动真实浏览器，拦截页面发出的 XHR 请求，
    从而绕过 CSRF/Sign 校验，获取真实 API 响应。
    """

    def __init__(self):
        self.captured_responses = []

    def scrape_all(self) -> list[dict]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            console.print("[red]请先安装: pip install playwright && playwright install chromium[/red]")
            return []

        jobs = []
        found_api_pattern = re.compile(
            r"(position/list|post/list|getPostList|campus.*list|searchPosition)",
            re.IGNORECASE
        )

        with sync_playwright() as p:
            console.print("[cyan]启动 Chromium 浏览器...[/cyan]")
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # 拦截 API 响应
            def on_response(response):
                url = response.url
                if found_api_pattern.search(url) and "json" in response.headers.get("content-type", ""):
                    try:
                        body = response.json()
                        self.captured_responses.append({
                            "url": url,
                            "data": body
                        })
                        console.print(f"[green]✓ 捕获到 API 响应: {url}[/green]")
                    except Exception:
                        pass

            page.on("response", on_response)

            keyword = FILTERS.get("keyword", "")
            target_url = CONFIG['site']['homepage']
            if keyword:
                target_url = f"{target_url}?keyword={keyword}"
            console.print(f"[cyan]正在访问: {target_url}[/cyan]")
            page.goto(target_url, wait_until="networkidle", timeout=30000)

            # 等待页面渲染完成
            page.wait_for_timeout(3000)

            if self.captured_responses:
                console.print(f"\n[green]共捕获到 {len(self.captured_responses)} 个 API 响应[/green]")

                # 提取已发现的真实 API URL
                real_url = self.captured_responses[0]["url"]
                console.print(f"[yellow]发现真实 API: {real_url}[/yellow]")
                console.print("[yellow]请将此 URL 更新到 site_config.json 中[/yellow]")

                # 解析已捕获的数据
                for captured in self.captured_responses:
                    data = captured["data"]
                    raw_list = extract_field(data, RESP_MAP["data_path"])
                    if isinstance(raw_list, list):
                        for item in raw_list:
                            job = map_job(item)
                            if filter_job(job):
                                jobs.append(job)
            else:
                console.print("[red]未捕获到任何 API 请求，请检查网站结构[/red]")
                _LOGS_DIR = Path(__file__).parent / "logs"
                _LOGS_DIR.mkdir(exist_ok=True)
                _shot = str(_LOGS_DIR / "debug_screenshot.png")
                console.print(f"[yellow]截图已保存到 {_shot}[/yellow]")
                page.screenshot(path=_shot)

            browser.close()

        return jobs


# ─────────────────────────── 输出 ───────────────────────────

def print_jobs_table(jobs: list[dict]):
    table = Table(title=f"抓取到 {len(jobs)} 个岗位", show_lines=True)
    table.add_column("岗位名称", style="bold cyan", max_width=30)
    table.add_column("类别", max_width=15)
    table.add_column("地点", max_width=12)
    table.add_column("部门", max_width=15)
    table.add_column("类型", max_width=8)
    table.add_column("首见时间", max_width=20)

    for j in jobs:
        table.add_row(
            j.get("title", ""),
            j.get("category", ""),
            j.get("location", ""),
            j.get("department", ""),
            j.get("type", ""),
            j.get("first_seen", "")[:16],
        )
    console.print(table)


def save_json(jobs: list[dict], path: str = "jobs.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)
    console.print(f"[green]✓ 已保存 {len(jobs)} 条数据到 {path}[/green]")


def save_csv(jobs: list[dict], path: str = "jobs.csv"):
    if not jobs:
        return
    # 基础列优先，其余动态字段追加在后面
    base_keys = ["title", "category", "location", "department", "type",
                 "headcount", "first_seen", "detail_url"]
    extra_keys = [k for k in jobs[0].keys()
                  if k not in base_keys and k not in ("fingerprint", "publish_time", "update_time")]
    keys = base_keys + extra_keys
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(jobs)
    console.print(f"[green]✓ 已保存到 {path}（用 Excel 可直接打开）[/green]")


# ─────────────────────────── 主流程 ───────────────────────────

def main():
    site = CONFIG["site"]
    console.rule(f"[bold cyan]{site['company']} {site['name']} 岗位爬虫[/bold cyan]")
    console.print(f"目标网站: {site['homepage']}")
    console.print(f"搜索关键词: [yellow]{FILTERS.get('keyword', '全部')}[/yellow]")
    console.print()

    jobs = []

    # ── Step 1: 尝试直接 API 调用 ──
    console.print("[bold]Step 1 / 直接 API 模式[/bold]")
    scraper = DirectAPIScraper()
    test = scraper.fetch_page(1)

    if test is not None and extract_field(test, RESP_MAP["data_path"]):
        console.print("[green]✓ 直接 API 可用，开始批量抓取[/green]\n")
        jobs = scraper.scrape_all()
    else:
        # ── Step 2: 回退到 Playwright 拦截模式 ──
        console.print("[yellow]直接 API 不可用，切换到 Playwright 拦截模式...[/yellow]\n")
        pw_scraper = PlaywrightAPIScraper()
        jobs = pw_scraper.scrape_all()

    # ── 输出结果 ──
    console.print()
    if jobs:
        print_jobs_table(jobs)
        save_json(jobs)
        save_csv(jobs)
        console.print(f"\n[bold green]完成！共获取 {len(jobs)} 个岗位[/bold green]")
    else:
        console.print("[red]未获取到任何岗位，请参阅 TROUBLESHOOT 部分[/red]")
        _print_troubleshoot()


def _print_troubleshoot():
    console.print("\n[yellow]── 排查建议 ──[/yellow]")
    steps = [
        "1. 打开浏览器开发者工具 → Network → XHR，筛选含 'list' 或 'position' 的请求",
        "2. 将真实 API URL 更新到 site_config.json 的 api.list.url",
        "3. 将响应体的 JSON 路径更新到 response_mapping",
        "4. 如果 API 有签名参数（sign/token），需要逆向签名算法或使用 Playwright 模式",
        "5. 运行: python scraper.py --playwright  强制使用浏览器模式"
    ]
    for step in steps:
        console.print(f"  {step}")


if __name__ == "__main__":
    if "--playwright" in sys.argv:
        console.print("[yellow]强制使用 Playwright 模式[/yellow]")
        scraper = PlaywrightAPIScraper()
        jobs = scraper.scrape_all()
        if jobs:
            print_jobs_table(jobs)
            save_json(jobs)
            save_csv(jobs)
    else:
        main()
