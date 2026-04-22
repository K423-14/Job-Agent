"""
Schema Extractor Agent -- Phase 2 入口
给定任意招聘网站 URL，通过 LLM + Playwright ReAct 循环自动生成 site_config.json

运行方式:
  pip install langchain langchain-openai playwright rich
  playwright install chromium
  python schema_extractor.py <招聘网站首页URL> [输出文件路径]

示例:
  python schema_extractor.py https://talent.antgroup.com/campus-full-list
  python schema_extractor.py https://careers.bytedance.com configs/bytedance.json

Agent 模块: agents/navigator_agent.py, agents/schema_extractor_agent.py
提示词文件: prompts/navigator.md, prompts/schema_extractor.md
生成配置输出到: configs/
"""

import json
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from agents.navigator_agent import NavigatorAgent
from agents.schema_extractor_agent import SchemaExtractorAgent

console = Console()


def main(homepage: str, output_path: Optional[str] = None) -> None:
    console.rule("[bold cyan]Schema Extractor Agent -- Phase 2[/bold cyan]")
    console.print(f"目标: [yellow]{homepage}[/yellow]\n")

    # -- Step 1: Navigator Agent --
    console.print("[bold]Step 1 / Navigator Agent  --  探索页面，捕获岗位列表 API[/bold]")
    xhr = NavigatorAgent(homepage).run()

    if xhr is None:
        console.print("\n[red]未能找到岗位列表 API，建议：[/red]")
        console.print("  1. 手动用浏览器开发者工具抓包，确认 API URL")
        console.print("  2. 将 API URL 直接写入 configs/ 下对应配置文件后运行 scraper.py")
        sys.exit(1)

    console.print()

    # -- Step 2: Schema Extractor Agent --
    console.print("[bold]Step 2 / Schema Extractor Agent  --  LLM 生成 site_config[/bold]")
    agent = SchemaExtractorAgent(homepage, xhr)
    config = agent.extract()

    if config is None:
        console.print("[red]配置生成失败[/red]")
        sys.exit(1)

    config_json = json.dumps(config, ensure_ascii=False, indent=2)
    console.print()
    console.print(Panel(
        Syntax(config_json, "json", theme="monokai", word_wrap=True),
        title="[green]生成的配置[/green]",
        expand=False,
    ))

    saved_path = agent.save(config, output_path)
    console.print(f"\n[green]已保存到 {saved_path}[/green]")
    console.print(f"[dim]使用: python scraper.py --config {saved_path}[/dim]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("用法: python schema_extractor.py <URL> [输出文件路径]")
        console.print("示例: python schema_extractor.py https://talent.antgroup.com/campus-full-list")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
