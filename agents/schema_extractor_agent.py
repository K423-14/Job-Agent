"""
Schema Extractor Agent
LLM 根据捕获到的 XHR 请求/响应，生成 site_config.json。
"""
import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from .llm_client import get_llm
from .navigator_agent import XHRCapture

console = Console()

_PROMPTS_DIR      = Path(__file__).parent.parent / "prompts"
_SITE_CONFIGS_DIR = Path(__file__).parent.parent / "site_configs"
_TEMPLATE_PATH    = Path(__file__).parent.parent / "configs" / "site_config_template.json"


def _load_prompt() -> ChatPromptTemplate:
    text = (_PROMPTS_DIR / "schema_extractor.md").read_text(encoding="utf-8")
    system, human = text.split("===HUMAN===", 1)
    system = system.replace("===SYSTEM===", "").strip()
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human.strip()),
    ])


def _load_template() -> dict:
    with open(_TEMPLATE_PATH, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


class SchemaExtractorAgent:
    def __init__(self, homepage: str, xhr: XHRCapture):
        self.homepage = homepage
        self.xhr = xhr

    def extract(self) -> Optional[dict]:
        template_str = json.dumps(_load_template(), ensure_ascii=False, indent=2)
        response_str = json.dumps(self.xhr.response_body, ensure_ascii=False)[:4000]

        messages = _load_prompt().format_messages(
            homepage=self.homepage,
            domain=urlparse(self.homepage).netloc,
            api_url=self.xhr.url,
            api_method=self.xhr.method,
            request_body=self.xhr.request_body or "（无）",
            response_body=response_str,
            template_json=template_str,
        )

        llm = get_llm(streaming=True, read_timeout=60.0, temperature=0, json_mode=True)
        chunks: list[str] = []
        stream_exc: list[Exception] = []
        WALL_TIMEOUT = 90  # 整体硬超时（秒），Ctrl+C 后最多等这么久

        def _run_stream():
            try:
                for chunk in llm.stream(messages):
                    content = chunk.content  # type: ignore[attr-defined]
                    if content:
                        chunks.append(content)
            except Exception as e:
                stream_exc.append(e)

        import threading
        t = threading.Thread(target=_run_stream, daemon=True)

        try:
            with Live(console=console, refresh_per_second=4) as live:
                live.update(Spinner("dots", text=Text("Schema Extractor: 等待 LLM 首个 token...", style="cyan")))
                t.start()
                elapsed = 0
                while t.is_alive():
                    t.join(timeout=0.5)
                    elapsed += 0.5
                    char_count = sum(len(c) for c in chunks)
                    if char_count:
                        live.update(Spinner(
                            "dots",
                            text=Text(f"Schema Extractor: 生成中... {char_count} 字符  ({elapsed:.0f}s)", style="cyan"),
                        ))
                    else:
                        live.update(Spinner(
                            "dots",
                            text=Text(f"Schema Extractor: 等待首个 token... ({elapsed:.0f}s / 上限 {WALL_TIMEOUT}s)", style="cyan"),
                        ))
                    if elapsed >= WALL_TIMEOUT:
                        console.print(f"[red]超时 {WALL_TIMEOUT}s，强制中止[/red]")
                        return None
        except KeyboardInterrupt:
            console.print("\n[yellow]已手动中断[/yellow]")
            return None

        if stream_exc:
            console.print(f"[red]LLM 流式调用失败: {stream_exc[0]}[/red]")
            return None

        raw = "".join(chunks)
        if not raw:
            console.print("[red]LLM 返回空内容[/red]")
            return None

        console.print(f"[green]✓ 生成完成，共 {len(raw)} 字符[/green]")

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            console.print(f"[red]JSON 解析失败: {e}[/red]")
            console.print(raw[:600])
            return None

    def save(self, config: dict, output_path: Optional[str] = None) -> str:
        _SITE_CONFIGS_DIR.mkdir(exist_ok=True)
        if output_path is None:
            domain = urlparse(self.homepage).netloc.replace(".", "_")
            output_path = str(_SITE_CONFIGS_DIR / f"site_config_{domain}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return output_path
