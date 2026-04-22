"""
pipeline.py
Multi-Agent Pipeline：编排多个Agent协作完成"从零爬取一个新网站"的完整流程。

流程：
  1. Navigator Agent  → 打开网站，自动发现岗位列表API
  2. Schema Extractor → 根据API响应，自动生成site_config.json
  3. Crawler          → 基于配置，批量爬取所有岗位
  4. Indexer          → 把爬到的岗位向量化，存入ChromaDB

面试要点：
- 这是Pipeline模式的Multi-Agent，顺序编排，上一个Agent的输出是下一个的输入
- 每个Agent有独立的Prompt和专长，关注点分离
- 和微服务思想类似：一个Agent干好一件事
"""
import json
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent


def run_pipeline(homepage: str) -> dict:
    """
    Multi-Agent Pipeline 主函数。
    
    输入：一个招聘网站首页URL
    输出：完整的执行结果（各步骤状态 + 爬取到的岗位数量）
    
    面试要点：
    - 这个函数就是"编排层"（Orchestrator）
    - 它不做具体工作，只负责调度各Agent并传递数据
    - 任何一步失败都会提前终止并返回错误信息（fail-fast）
    """
    result = {
        "homepage": homepage, 
        "steps": {}, 
        "success": False, 
        "total_jobs": 0
    }

    # ════════════════════════════════════════════
    # Step 1: Navigator Agent — 发现 API 端点
    # ════════════════════════════════════════════

    console.print(Panel("Step 1/4: Navigator Agent - 寻找岗位API", style="cyan"))

    from .navigator_agent import NavigatorAgent

    navigator = NavigatorAgent(homepage)
    xhr = navigator.run()

    if xhr is None:
        console.print("[red]Navigator 未找到岗位API，Pipeline终止[/red]")
        result["steps"]["navigator"] = {"status": "failed", "reason": "未找到岗位API"}
        return result
    
    console.print(f"[green]√ 找到API：{xhr.method} {xhr.url}[/green]\n")

    # ════════════════════════════════════════════
    # Step 2: Schema Extractor — 生成爬虫配置
    # ════════════════════════════════════════════

    console.print(Panel("Step 2/4: Schema Extractor - 生成爬虫配置", style="cyan"))

    from .schema_extractor_agent import SchemaExtractorAgent

    extractor = SchemaExtractorAgent(homepage, xhr)
    config = extractor.extract()

    if config is None:
        console.print("[red]Schema Extractor 生成配置失败，Pipeline终止[/red]")
        result["steps"]["extractor"] = {"status": "failed", "reason": "配置生成失败"}
        return result

    # 保存配置文件
    from urllib.parse import urlparse
    domain = urlparse(homepage).netloc.replace(".", "_")
    config_path = ROOT_DIR / "site_configs" / f"{domain}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    result["steps"]["extractor"] = {
        "status": "success",
        "config_path": str(config_path),
    }
    console.print(f"[green]✓ 配置已保存: {config_path}[/green]\n")

    # ════════════════════════════════════════════
    # Step 3: Crawler — 批量爬取岗位
    # ════════════════════════════════════════════

    console.print(Panel("Step 3/4: Crawler — 批量爬取岗位数据", style="cyan"))

    # 动态加载scraper
    jobs = _run_crawler(config)

    if not jobs:
        console.print("[red]Crawler 未爬到任何岗位，Pipeline终止[/red]")
        result["steps"]["crawler"] = {"status": "failed", "reason": "未爬到岗位"}
        return result
    
    # 保存到json文件
    output_path = ROOT_DIR / f"jobs_{domain}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)

    result["steps"]["crawler"] = {
        "status": "success",
        "job_count": len(jobs),
        "output_path": str(output_path),
    }
    console.print(f"[green]✓ 爬取到 {len(jobs)} 条岗位，已保存到 {output_path}[/green]\n")

    # ════════════════════════════════════════════
    # Step 4: Indexer — 向量化入库
    # ════════════════════════════════════════════
    console.print(Panel("Step 4/4: Indexer — 岗位数据向量化入库", style="cyan"))

    try:
        indexed_count = _run_indexer(str(output_path))
        result["steps"]["indexer"] = {
            "status": "success",
            "indexed_count": indexed_count,
        }
        console.print(f"[green]✓ 已将 {indexed_count} 条岗位向量化入库[/green]\n")
    except Exception as e:
        console.print(f"[red]向量化入库失败: {e}[/red]")
        result["steps"]["indexer"] = {"status": "failed", "reason": str(e)}
        return result

    # ════════════════════════════════════════════
    result["success"] = True
    result["total_jobs"] = len(jobs)

    console.print(Panel(
        f"[bold green]Pipeline 完成！共爬取 {len(jobs)} 条岗位并已入库。\n"
        f"现在可以通过 /ask 接口进行问答了。[/bold green]",
        title="Pipeline Summary",
        style="green",
    ))

    return result


def _run_crawler(config: dict) -> list:
    """
    基于配置执行爬取。
    
    这里不直接 import scraper.py（因为它会在import时读取配置），
    而是把核心爬取逻辑提取出来，用我们的config驱动。
    """
    import hashlib
    import requests
    from rich.progress import Progress, SpinnerColumn, TextColumn

    api_cfg = config["api"]["list"]
    resp_map = config["response_mapping"]
    crawl_cfg = config.get("crawl", {"timeout_seconds": 15, "request_delay_ms": 500})

    session = requests.Session()
    session.headers.update(api_cfg.get("headers", {}))

    jobs = []
    page = api_cfg["pagination"]["start_page"]
    page_size = api_cfg["pagination"]["page_size"]

    # 复用scraper.py工具函数
    sys.path.insert(0, str(ROOT_DIR))
    from scraper import extract_field, map_job

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("爬取中...", total=None)

        for _ in range(50):
            progress.update(task, description=f"正在爬取第 {page} 页...")

            params = dict(api_cfg.get("params", {}))
            params[api_cfg["pagination"]["page_param"]] = page
            params[api_cfg["pagination"]["size_param"]] = page_size

            try:
                if api_cfg["method"].upper() == "GET":
                    resp = session.get(api_cfg["url"], params=params, 
                                       timeout=crawl_cfg.get("timeout_seconds", 15))
                else:
                    resp = session.post(api_cfg["url"], json=params, 
                                       timeout=crawl_cfg.get("timeout_seconds", 15))

                resp.raise_for_status()
                data = resp.json()

            except Exception as e:
                console.print(f"[yellow]第{page}页请求失败: {e}[/yellow]")
                break

            raw_list = extract_field(data, resp_map["data_path"])
            if not isinstance(raw_list, list) or not raw_list:
                break

            for item in raw_list:
                try:
                    # 由于map_job依赖全局CONFIG，这里手动做简单映射
                    job = _simple_map(item, resp_map["fields"], config)
                    jobs.append(job)
                except Exception:
                    continue
            
            # 检查是否还有下一页
            total_raw = extract_field(data, resp_map.get("total_path", ""))
            try:
                total = int(total_raw) if total_raw else 0
            except (ValueError, TypeError):
                total_raw = 0

            if total and page * page_size >= total:
                break

            page += 1
            time.sleep(crawl_cfg.get("request_delay_ms", 500) / 1000)
    
    return jobs


def _simple_map(raw_item: dict, fields: dict, config: dict) -> dict:
    """
    简单的字段映射，不依赖scraper.py的全局变量。
    """
    import hashlib
    sys.path.insert(0, str(ROOT_DIR))
    from scraper import extract_field

    job = {}
    for key, src in fields.items():
        if not src:
            job[key] = ""
        elif "[]." in src:
            job[key] = extract_field(raw_item, src)
        else:
            val = raw_item.get(src, "")
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            job[key] = val if val is not None else ""

    # 指纹去重用
    raw = f"{job.get('id', '')}{job.get('title', '')}{job.get('location', '')}"
    job["fingerprint"] = hashlib.md5(raw.encode()).hexdigest()
    return job


def _run_indexer(json_path: str) -> int:
    """
    把爬到的岗位数据向量化入库到ChromaDB。
    复用 rag 模块的现有逻辑。
    """
    sys.path.insert(0, str(ROOT_DIR))
    from rag.job_loader import load_jobs
    from rag.vector_store import build_vector_store

    docs = load_jobs(json_path)
    build_vector_store(docs)
    return len(docs)