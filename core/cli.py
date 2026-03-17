"""
hyw/cli.py - Rich-styled CLI for hyw (Heuristic Yield Web_search)

用法:
    python -m hyw              # 交互模式
    python -m hyw -q "问题"    # 单次问答
"""
from __future__ import annotations

import atexit
import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import textwrap
import unicodedata
from typing import Any
from urllib.parse import unquote, urlparse
from dataclasses import dataclass

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .config import (
    CONFIG_PATH,
    DEFAULT_NAME,
    ensure_config_file,
    get_model_profiles,
    load_config,
)
from .main import (
    Stats,
    build_stage_model_config,
    get_runtime_prewarm_label,
    run,
    run_stream,
    start_runtime_prewarm,
    startup_tools,
    shutdown_tools,
)

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout.processors import BeforeInput, ConditionalProcessor
    from prompt_toolkit.styles import Style as PTStyle

    _PT_AVAILABLE = True
except Exception:
    PromptSession = None  # type: ignore[assignment,misc]
    Condition = None  # type: ignore[assignment,misc]
    KeyBindings = None  # type: ignore[assignment,misc]
    Keys = None  # type: ignore[assignment,misc]
    BeforeInput = None  # type: ignore[assignment,misc]
    ConditionalProcessor = None  # type: ignore[assignment,misc]
    PTStyle = None  # type: ignore[assignment,misc]
    _PT_AVAILABLE = False

# ── Color constants ───────────────────────────────────────────
TEXT_MUTED = "rgb(194,145,92)"
TEXT_SOFT = "rgb(232,189,138)"
BORDER = "rgb(255,108,60)"
ACCENT = "rgb(255,108,60)"
INPUT_ACCENT = "#D4AF37"
INPUT_TEXT = "#E8BD8A"
INPUT_HINT = "#D4AF37"
INPUT_PLACEHOLDER = "#A08462"
PT_ACCENT = "#ff6c3c"

_QUIT = {"/exit", "/quit", "exit", "quit", "q"}
_PASTE_COMMANDS = {"/paste", "/paste-image", "/clip", "/clipboard"}
_DUMB = os.environ.get("TERM", "").lower() in {"", "dumb", "unknown"}
_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\n]+)\)")
_TASK_LIST_BLOCK_RE = re.compile(r"<task_list\b[^>]*>.*?</task_list>", flags=re.IGNORECASE | re.DOTALL)
_ARTICLE_SKELETON_BLOCK_RE = re.compile(r"<article_skeleton\b[^>]*>.*?</article_skeleton>", flags=re.IGNORECASE | re.DOTALL)
_SEARCH_REWRITE_BLOCK_RE = re.compile(r"<search_rewrite\b[^>]*>.*?</search_rewrite>", flags=re.IGNORECASE | re.DOTALL)
_KEYWORD_REWRITE_BLOCK_RE = re.compile(r"<keyword_rewrite\b[^>]*>.*?</keyword_rewrite>", flags=re.IGNORECASE | re.DOTALL)
_USER_NEED_BLOCK_RE = re.compile(r"<user_need\b[^>]*>.*?</user_need>", flags=re.IGNORECASE | re.DOTALL)
_VERIFICATION_OUTLINE_BLOCK_RE = re.compile(r"<verification_outline\b[^>]*>.*?</verification_outline>", flags=re.IGNORECASE | re.DOTALL)
_TOOL_BLOCK_RE = re.compile(r"<(?:search|wiki|sub_agent|page)\b[^>]*>.*?</(?:search|wiki|sub_agent|page)>", flags=re.IGNORECASE | re.DOTALL)
_TOOL_SELF_CLOSING_RE = re.compile(r"<(?:search|wiki|sub_agent|page)\b[^>]*/>", flags=re.IGNORECASE)
_MD_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+(?:\s*:?-{3,}:?\s*)\|?\s*$")
_PROMPT_PREFIX = "➜  "
_IMAGE_TOKEN_RE = re.compile(r"\[Image #\d+\]")
_PASTED_TEXT_TOKEN_RE = re.compile(r"\[Pasted Content \d+ chars #\d+\]")
_PASTED_TEXT_TOKEN_THRESHOLD = 280
_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
    ".heic", ".heif", ".avif", ".ico", ".icns",
}

INTRO_PROMPT_ANSI = "\x1b[38;2;255;108;60m"
INTRO_INPUT_ANSI = "\x1b[38;2;232;189;138m"
INTRO_HINT_ANSI = "\x1b[38;2;160;132;98m"

_IO_LOCK = threading.Lock()
_INPUT_RENDERED = threading.Event()
_STATUS_MAX_CELLS = 24


@dataclass
class _PromptInput:
    text: str
    image_paths: list[str]
    cleanup_paths: list[str]


@dataclass
class _PastedImage:
    token: str
    path: str
    owned: bool = False


@dataclass
class _PastedText:
    token: str
    content: str


# ── Gradient title ────────────────────────────────────────────
def _title_rgb(index: int, total: int) -> tuple[int, int, int]:
    total = max(1, int(total or 1))
    idx = max(0, min(int(index or 0), total - 1))
    start = (255, 108, 60)
    end = (232, 189, 138)
    if total == 1:
        return start
    ratio = idx / float(total - 1)
    return tuple(int(start[i] + (end[i] - start[i]) * ratio) for i in range(3))  # type: ignore[return-value]


def _gradient_title(label: str) -> Text:
    text = str(label or "").strip() or "hyw"
    title = Text()
    for i, ch in enumerate(text):
        if ch == " ":
            title.append(" ")
            continue
        r, g, b = _title_rgb(i, len(text))
        title.append(ch, style=f"bold rgb({r},{g},{b})")
    return title


def _gradient_rich_text(text: str, *, bold: bool = False) -> Text:
    value = str(text or "")
    out = Text()
    for i, ch in enumerate(value):
        if ch == " ":
            out.append(" ")
            continue
        r, g, b = _title_rgb(i, len(value))
        style = f"rgb({r},{g},{b})"
        if bold:
            style = "bold " + style
        out.append(ch, style=style)
    return out


# ── Panel helper ──────────────────────────────────────────────
def _replace_outer_border_chars(line: str, border_char: str) -> str:
    text = str(line or "")
    if not text or not border_char:
        return text
    first = text.find(border_char)
    last = text.rfind(border_char)
    if first < 0:
        return text
    chars = list(text)
    chars[first] = " "
    if last > first:
        chars[last] = " "
    return "".join(chars)


class _OpenEdgePanel(Panel):
    """Panel variant with framed title/subtitle lines and unchanged body padding."""

    def __rich_console__(self, console, options):
        from rich.cells import cell_len
        from rich.padding import Padding
        from rich.segment import Segment

        _padding = Padding.unpack(self.padding)
        renderable = Padding(self.renderable, _padding) if any(_padding) else self.renderable
        style = console.get_style(self.style)
        border_style = style + console.get_style(self.border_style)
        width = options.max_width if self.width is None else min(options.max_width, self.width)

        safe_box = console.safe_box if self.safe_box is None else self.safe_box
        panel_box = self.box.substitute(options, safe=safe_box)

        def align_text(text: Text, width: int, align: str, character: str):
            text = text.copy()
            text.truncate(width)
            excess_space = width - cell_len(text.plain)
            if text.style:
                text.stylize(console.get_style(text.style))

            if excess_space:
                if align == "left":
                    return Text.assemble(text, (character * excess_space, border_style), no_wrap=True, end="")
                if align == "center":
                    left = excess_space // 2
                    return Text.assemble(
                        (character * left, border_style),
                        text,
                        (character * (excess_space - left), border_style),
                        no_wrap=True,
                        end="",
                    )
                return Text.assemble((character * excess_space, border_style), text, no_wrap=True, end="")
            return text

        title_text = self._title
        if title_text is not None:
            title_text.stylize_before(border_style)

        child_width = (
            width - 2
            if self.expand
            else console.measure(renderable, options=options.update_width(width - 2)).maximum
        )
        child_height = self.height or options.height or None
        if child_height:
            child_height -= 2
        if title_text is not None:
            child_width = min(options.max_width - 2, max(child_width, title_text.cell_len + 2))

        width = child_width + 2
        child_options = options.update(width=child_width, height=child_height, highlight=self.highlight)
        lines = console.render_lines(renderable, child_options, style=style)

        line_start = Segment(panel_box.mid_left, border_style)
        line_end = Segment(f"{panel_box.mid_right}", border_style)
        new_line = Segment.line()
        if title_text is not None and width > 8:
            title_text = align_text(title_text, width - 8, self.title_align, panel_box.top)
            yield Segment(panel_box.top_left + (panel_box.top * 2), border_style)
            yield from console.render(title_text, child_options.update_width(width - 8))
            yield Segment((panel_box.top * 2) + panel_box.top_right, border_style)
            yield new_line
        for line in lines:
            yield line_start
            yield from line
            yield line_end
            yield new_line

        subtitle_text = self._subtitle
        if subtitle_text is not None:
            subtitle_text.stylize_before(border_style)

        if subtitle_text is None or width <= 8:
            yield Segment(
                _replace_outer_border_chars(panel_box.get_bottom([width - 2]), panel_box.bottom),
                border_style,
            )
        else:
            subtitle_text = align_text(subtitle_text, width - 8, self.subtitle_align, panel_box.bottom)
            yield Segment(panel_box.bottom_left + (panel_box.bottom * 2), border_style)
            yield from console.render(subtitle_text, child_options.update_width(width - 8))
            yield Segment((panel_box.bottom * 2) + panel_box.bottom_right, border_style)

        yield new_line


def _make_panel(
    body,
    *,
    title: str | Text | None = None,
    subtitle: str | Text | None = None,
    padding: tuple[int, int] = (0, 1),
) -> Panel:
    return _OpenEdgePanel(
        body,
        box=box.HORIZONTALS,
        border_style=BORDER,
        title=title,
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        padding=padding,
    )


# ── Markdown theme ────────────────────────────────────────────
def _build_markdown_theme() -> Theme:
    return Theme(
        {
            "markdown.text": TEXT_SOFT,
            "markdown.paragraph": TEXT_SOFT,
            "markdown.h1": f"bold {ACCENT}",
            "markdown.h2": ACCENT,
            "markdown.h3": ACCENT,
            "markdown.h4": ACCENT,
            "markdown.h5": ACCENT,
            "markdown.h6": ACCENT,
            "markdown.link": ACCENT,
            "markdown.link_url": TEXT_MUTED,
            "markdown.block_quote": TEXT_MUTED,
            "markdown.item": TEXT_SOFT,
            "markdown.item.bullet": ACCENT,
            "markdown.item.number": ACCENT,
            "markdown.table": TEXT_SOFT,
            "markdown.table.header": f"bold {ACCENT}",
            "markdown.table.cell": TEXT_SOFT,
            "markdown.table.border": TEXT_MUTED,
            "markdown.code": INPUT_HINT,
            "markdown.code_block": TEXT_SOFT,
            "markdown.hr": TEXT_MUTED,
            "markdown.em": TEXT_SOFT,
            "markdown.strong": f"bold {TEXT_SOFT}",
        }
    )


def _render_markdown(text: str):
    blocks = _split_markdown_blocks(_normalize_markdown_for_cli(text))
    renderables: list[object] = []
    for block in blocks:
        kind = block[0]
        if kind == "markdown":
            body = str(block[1] or "")
            if body.strip():
                renderables.append(
                    Markdown(
                        body,
                        code_theme="ansi_dark",
                        inline_code_theme="ansi_dark",
                    )
                )
        elif kind == "table":
            headers = block[1]
            rows = block[2]
            if renderables:
                renderables.append(Text(""))
            renderables.append(_build_rich_table(headers, rows))
            renderables.append(Text(""))

    if not renderables:
        return Text("", style=TEXT_SOFT)
    if len(renderables) == 1:
        return renderables[0]
    return Group(*renderables)


def _render_stream_preview(text: str) -> Text:
    preview = Text(style=TEXT_SOFT)
    preview.append(_normalize_markdown_for_cli(text) or "...", style=TEXT_SOFT)
    return preview


def _render_reply_body(text: str, *, preview: bool = False):
    body = _render_stream_preview(text) if preview else _render_markdown(text)
    grid = Table.grid(expand=True, padding=(0, 0))
    grid.add_column(width=2, no_wrap=True)
    grid.add_column(ratio=1, overflow="fold")
    grid.add_row(Text("•", style=f"bold {ACCENT}"), body)
    return grid


def _render_turn_transcript(
    *,
    settled: list[object] | None = None,
    reply_text: str = "",
    preview: bool = False,
    tool_renderable: object | None = None,
    note_text: str = "",
    note_style: str | None = None,
):
    blocks: list[object] = list(settled or [])
    if str(reply_text or "").strip():
        blocks.append(_render_reply_body(reply_text, preview=preview))
    if tool_renderable is not None:
        blocks.append(tool_renderable)
    if str(note_text or "").strip():
        if blocks:
            blocks.append(Text(""))
        blocks.append(Text(str(note_text), style=note_style or TEXT_MUTED))
    if not blocks:
        return Text("", style=TEXT_MUTED)
    if len(blocks) == 1:
        return blocks[0]
    return Group(*blocks)


def _parse_md_table_row(line: str) -> list[str] | None:
    stripped = str(line or "").strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [part.strip() for part in re.split(r"(?<!\\)\|", stripped)]
    if len(cells) < 2:
        return None
    return cells


def _table_rows_to_lines(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not headers or not rows:
        return []

    if len(headers) == 2:
        lines: list[str] = []
        for row in rows:
            key = str(row[0] if row else "").strip()
            value = str(row[1] if len(row) > 1 else "").strip()
            if key and value:
                lines.append(f"- {key}：{value}")
            elif key:
                lines.append(f"- {key}")
            elif value:
                lines.append(f"- {value}")
        return lines

    lines = []
    for row in rows:
        lead = str(row[0] if row else "").strip() or str(headers[0] or "").strip() or "项目"
        parts: list[str] = []
        for idx in range(1, len(headers)):
            header = str(headers[idx] or "").strip()
            value = str(row[idx] if idx < len(row) else "").strip()
            if not value:
                continue
            if header:
                parts.append(f"{header}：{value}")
            else:
                parts.append(value)
        if parts:
            lines.append(f"- {lead}：{'；'.join(parts)}")
        else:
            lines.append(f"- {lead}")
    return lines


def _build_rich_table(headers: list[str], rows: list[list[str]]) -> Table:
    table = Table(
        box=box.SIMPLE_HEAVY,
        border_style=TEXT_MUTED,
        header_style=f"bold {ACCENT}",
        style=TEXT_SOFT,
        show_edge=False,
        show_lines=False,
        pad_edge=False,
        padding=(0, 1),
        expand=True,
    )

    column_count = max(len(headers), max((len(row) for row in rows), default=0))
    for idx in range(column_count):
        header = str(headers[idx] if idx < len(headers) else "").strip()
        kwargs: dict[str, object] = {"vertical": "top", "overflow": "fold"}
        if column_count == 2 and idx == 0:
            kwargs["no_wrap"] = True
            kwargs["ratio"] = 1
            kwargs["style"] = TEXT_SOFT
        elif column_count == 2 and idx == 1:
            kwargs["ratio"] = 4
            kwargs["style"] = TEXT_SOFT
        else:
            kwargs["ratio"] = 1
            kwargs["style"] = TEXT_SOFT
        table.add_column(header or " ", **kwargs)

    for row in rows:
        cells = [str(row[idx]).strip() if idx < len(row) else "" for idx in range(column_count)]
        table.add_row(*cells)

    return table


def _split_markdown_blocks(text: str) -> list[tuple]:
    lines = str(text or "").splitlines()
    if not lines:
        return [("markdown", str(text or ""))]

    blocks: list[tuple] = []
    markdown_buffer: list[str] = []
    i = 0
    in_code_block = False

    def _flush_markdown() -> None:
        nonlocal markdown_buffer
        if markdown_buffer:
            blocks.append(("markdown", "\n".join(markdown_buffer)))
            markdown_buffer = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            markdown_buffer.append(line)
            i += 1
            continue

        if in_code_block:
            markdown_buffer.append(line)
            i += 1
            continue

        header = _parse_md_table_row(line)
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if header and _MD_TABLE_SEPARATOR_RE.match(next_line.strip()):
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines):
                row = _parse_md_table_row(lines[j])
                if not row:
                    break
                rows.append(row)
                j += 1

            if rows:
                _flush_markdown()
                blocks.append(("table", header, rows))
                i = j
                continue

        markdown_buffer.append(line)
        i += 1

    _flush_markdown()
    return blocks


def _normalize_markdown_tables(text: str) -> str:
    lines = str(text or "").splitlines()
    if not lines:
        return str(text or "")

    out: list[str] = []
    i = 0
    in_code_block = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            out.append(line)
            i += 1
            continue

        if in_code_block:
            out.append(line)
            i += 1
            continue

        header = _parse_md_table_row(line)
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if header and _MD_TABLE_SEPARATOR_RE.match(next_line.strip()):
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines):
                row = _parse_md_table_row(lines[j])
                if not row:
                    break
                rows.append(row)
                j += 1

            if rows:
                out.extend(_table_rows_to_lines(header, rows))
                i = j
                continue

        out.append(line)
        i += 1

    return "\n".join(out)


def _normalize_markdown_lists(text: str) -> str:
    lines = str(text or "").splitlines()
    if not lines:
        return str(text or "")

    out: list[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            out.append(line)
            continue

        if in_code_block:
            out.append(line)
            continue

        heading_match = re.match(r"^\s+(#{1,6}\s+.+)$", line)
        if heading_match:
            out.append(heading_match.group(1))
            continue

        quote_match = re.match(r"^\s+(>\s*.+)$", line)
        if quote_match:
            out.append(quote_match.group(1))
            continue

        bullet_match = re.match(r"^\s*[•·●▪◦▸▹►▻*-+]\s+(.+)$", line)
        if bullet_match:
            out.append(f"- {bullet_match.group(1).strip()}")
            continue

        ordered_match = re.match(r"^\s*(\d+)[.)]\s+(.+)$", line)
        if ordered_match:
            out.append(f"{ordered_match.group(1)}. {ordered_match.group(2).strip()}")
            continue

        out.append(line)

    return "\n".join(out)


def _dedent_markdown_outside_code(text: str) -> str:
    lines = str(text or "").splitlines()
    if not lines:
        return str(text or "")

    out: list[str] = []
    chunk: list[str] = []
    in_code_block = False

    def _flush_chunk() -> None:
        nonlocal chunk
        if not chunk:
            return
        out.extend(textwrap.dedent("\n".join(chunk)).splitlines())
        chunk = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            _flush_chunk()
            out.append(line)
            in_code_block = not in_code_block
            continue
        if in_code_block:
            out.append(line)
            continue
        chunk.append(line)

    _flush_chunk()
    return "\n".join(out)


def _normalize_markdown_for_cli(text: str) -> str:
    normalized = _dedent_markdown_outside_code(text)
    normalized = _normalize_markdown_tables(normalized)
    normalized = _normalize_markdown_lists(normalized)
    return normalized


# ── Suppress noisy logs ──────────────────────────────────────
def _suppress_logs() -> None:
    raw = str(os.environ.get("HYW_SHOW_TOOL_LOGS", "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass


# ── Cell width helpers (for intro status line) ───────────────
def _cell_width(ch: str) -> int:
    if not ch:
        return 0
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1


def _text_cells(text: str) -> int:
    return sum(_cell_width(ch) for ch in text)


def _tail_by_cells(text: str, max_cells: int) -> str:
    if max_cells <= 0:
        return ""
    out: list[str] = []
    used = 0
    for ch in reversed(text):
        w = _cell_width(ch)
        if used + w > max_cells:
            break
        out.append(ch)
        used += w
    return "".join(reversed(out))


def _cache_dir() -> Path:
    root = CONFIG_PATH.parent
    path = root / "cache"
    legacy_dirs = [root / "缓存", root / "clipboard"]
    for legacy in legacy_dirs:
        if legacy.exists() and not path.exists():
            try:
                legacy.rename(path)
                break
            except Exception:
                path.mkdir(parents=True, exist_ok=True)
                for item in legacy.iterdir():
                    try:
                        shutil.move(str(item), str(path / item.name))
                    except Exception:
                        pass
        else:
            path.mkdir(parents=True, exist_ok=True)
        if legacy.exists() and legacy.is_dir() and legacy != path:
            try:
                legacy.rmdir()
            except Exception:
                pass
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cleanup_cache_dir() -> None:
    cache = _cache_dir()
    for item in cache.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception:
            pass


atexit.register(_cleanup_cache_dir)


def _clipboard_dir() -> Path:
    path = _cache_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_clipboard_text() -> str:
    if sys.platform != "darwin":
        return ""
    try:
        proc = subprocess.run(["pbpaste"], capture_output=True, text=True, check=False)
    except Exception:
        return ""
    text = str(proc.stdout or "").replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s*\n\s*", " ", text).strip()


def _normalize_pasted_path(token: str) -> Path | None:
    raw = str(token or "").strip()
    if not raw:
        return None
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        raw = unquote(parsed.path or "")
    suffix = Path(raw).suffix.lower()
    looks_like_path = (
        raw.startswith(("~", ".", "/"))
        or raw.startswith("\\\\")
        or "/" in raw
        or "\\" in raw
        or suffix in _IMAGE_EXTENSIONS
    )
    if not looks_like_path:
        return None
    if len(raw) > 1024:
        return None
    path = Path(raw).expanduser()
    try:
        if not path.is_file():
            return None
    except (OSError, ValueError):
        return None
    if path.suffix.lower() not in _IMAGE_EXTENSIONS:
        return None
    return path


def _extract_image_paths_from_text(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    try:
        tokens = shlex.split(raw, posix=True)
    except Exception:
        tokens = [raw.replace("\\ ", " ")]
    if not tokens:
        return []
    paths: list[str] = []
    for token in tokens:
        path = _normalize_pasted_path(token)
        if path is None:
            return []
        paths.append(str(path))
    return paths


def _read_clipboard_image() -> str | None:
    target = _clipboard_dir() / f"paste_{int(time.time() * 1000)}.png"
    try:
        from PIL import Image, ImageGrab
    except Exception:
        return None
    try:
        grabbed = ImageGrab.grabclipboard()
    except Exception:
        return None
    if isinstance(grabbed, (list, tuple)):
        for item in grabbed:
            src = Path(str(item or "")).expanduser()
            if not src.is_file():
                continue
            try:
                with Image.open(src) as image:
                    image.save(target, format="PNG")
                return str(target)
            except Exception:
                continue
        return None
    if grabbed is None or not hasattr(grabbed, "save"):
        return None
    try:
        grabbed.save(target, format="PNG")
    except Exception:
        return None
    if target.exists():
        return str(target)
    return None


def _insert_buffer_text(buffer, text: str, *, pad: bool = False) -> None:
    value = str(text or "")
    if not value:
        return
    if pad:
        before = buffer.document.current_line_before_cursor
        after = buffer.document.current_line_after_cursor
        if before and not before[-1].isspace():
            value = " " + value
        if not value.endswith(" ") and (not after or not after[:1].isspace()):
            value = value + " "
    buffer.insert_text(value)


def _discard_pasted_image(item: _PastedImage) -> None:
    if not item.owned:
        return
    try:
        Path(item.path).unlink()
    except Exception:
        pass


def _token_number(token: str) -> int | None:
    match = re.search(r"#(\d+)\]", str(token or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _next_pasted_item_number(
    pasted_images: list[_PastedImage],
    pasted_texts: list[_PastedText],
) -> int:
    max_index = 0
    for item in pasted_images:
        index = _token_number(item.token)
        if index is None:
            continue
        max_index = max(max_index, index)
    for item in pasted_texts:
        index = _token_number(item.token)
        if index is None:
            continue
        max_index = max(max_index, index)
    return max_index + 1


def _find_pasted_token_span(text: str, cursor: int, *, backward: bool) -> tuple[int, int] | None:
    if backward:
        if cursor <= 0:
            return None
        probe = cursor - 1
        while probe >= 0 and text[probe].isspace():
            probe -= 1
        if probe < 0:
            return None
    else:
        if cursor >= len(text):
            return None
        probe = cursor
        while probe < len(text) and text[probe].isspace():
            probe += 1
        if probe >= len(text):
            return None

    for pattern in (_IMAGE_TOKEN_RE, _PASTED_TEXT_TOKEN_RE):
        for match in pattern.finditer(text):
            start, end = match.span()
            if start <= probe < end:
                return start, end
    return None


def _expanded_pasted_delete_span(text: str, start: int, end: int) -> tuple[int, int]:
    delete_start, delete_end = start, end
    has_left_space = delete_start > 0 and text[delete_start - 1].isspace()
    has_right_space = delete_end < len(text) and text[delete_end].isspace()
    if has_left_space and has_right_space:
        delete_start -= 1
    elif has_right_space:
        delete_end += 1
    elif has_left_space:
        delete_start -= 1
    return delete_start, delete_end


def _delete_pasted_token(
    buffer,
    pasted_images: list[_PastedImage],
    pasted_texts: list[_PastedText],
    *,
    backward: bool,
) -> bool:
    text = str(buffer.text or "")
    cursor = int(buffer.cursor_position or 0)
    span = _find_pasted_token_span(text, cursor, backward=backward)
    if span is None:
        return False
    token_start, token_end = span
    token_text = text[token_start:token_end]
    delete_start, delete_end = _expanded_pasted_delete_span(text, token_start, token_end)
    new_text = text[:delete_start] + text[delete_end:]
    buffer.text = new_text
    buffer.cursor_position = min(delete_start, len(new_text))

    remaining_images: list[_PastedImage] = []
    removed_image = False
    for item in pasted_images:
        if not removed_image and item.token == token_text:
            _discard_pasted_image(item)
            removed_image = True
            continue
        remaining_images.append(item)
    pasted_images[:] = remaining_images

    if not removed_image:
        remaining_texts: list[_PastedText] = []
        removed_text = False
        for item in pasted_texts:
            if not removed_text and item.token == token_text:
                removed_text = True
                continue
            remaining_texts.append(item)
        pasted_texts[:] = remaining_texts

    return True


def _should_tokenize_pasted_text(text: str) -> bool:
    raw = str(text or "")
    stripped = raw.strip()
    if not stripped:
        return False
    return len(stripped) >= _PASTED_TEXT_TOKEN_THRESHOLD


def _insert_pasted_text_token(
    buffer,
    text: str,
    pasted_texts: list[_PastedText],
    pasted_images: list[_PastedImage],
) -> None:
    content = str(text or "")
    if not content:
        return
    token = f"[Pasted Content {len(content)} chars #{_next_pasted_item_number(pasted_images, pasted_texts)}]"
    pasted_texts.append(_PastedText(token=token, content=content))
    _insert_buffer_text(buffer, token, pad=True)


def _insert_pasted_text(
    buffer,
    text: str,
    pasted_texts: list[_PastedText],
    pasted_images: list[_PastedImage],
) -> None:
    content = str(text or "")
    if not content:
        return
    if _should_tokenize_pasted_text(content):
        _insert_pasted_text_token(buffer, content, pasted_texts, pasted_images)
        return
    _insert_buffer_text(buffer, content)


def _expand_pasted_text_tokens(raw_text: str, pasted_texts: list[_PastedText]) -> str:
    text = str(raw_text or "")
    for item in pasted_texts:
        if item.token not in text:
            continue
        text = text.replace(item.token, item.content)
    return text


def _insert_image_tokens(
    buffer,
    image_paths: list[str],
    pasted_images: list[_PastedImage],
    pasted_texts: list[_PastedText],
    *,
    owned: bool = False,
) -> None:
    tokens: list[str] = []
    for image_path in image_paths:
        token = f"[Image #{_next_pasted_item_number(pasted_images, pasted_texts)}]"
        pasted_images.append(_PastedImage(token=token, path=image_path, owned=owned))
        tokens.append(token)
    _insert_buffer_text(buffer, " ".join(tokens), pad=True)


def _finalize_pasted_text(raw_text: str, pasted_texts: list[_PastedText]) -> str:
    return _expand_pasted_text_tokens(raw_text, pasted_texts)


def _normalize_prompt_input(
    raw_text: str,
    pasted_images: list[_PastedImage],
    pasted_texts: list[_PastedText],
) -> _PromptInput:
    raw = str(raw_text or "").strip()
    image_paths, cleanup_paths = _finalize_pasted_images(raw, pasted_images)
    if image_paths:
        text = _finalize_pasted_text(raw, pasted_texts)
        return _PromptInput(text=text, image_paths=image_paths, cleanup_paths=cleanup_paths)
    direct_paths = _extract_image_paths_from_text(raw)
    if direct_paths:
        tokens = [f"[Image #{idx}]" for idx in range(1, len(direct_paths) + 1)]
        return _PromptInput(text=" ".join(tokens), image_paths=direct_paths, cleanup_paths=[])
    text = _finalize_pasted_text(raw, pasted_texts)
    return _PromptInput(text=text, image_paths=[], cleanup_paths=[])


def _paste_clipboard_into_buffer(
    buffer,
    pasted_images: list[_PastedImage],
    pasted_texts: list[_PastedText],
) -> bool:
    image_path = _read_clipboard_image()
    if image_path:
        _insert_image_tokens(buffer, [image_path], pasted_images, pasted_texts, owned=True)
        return True
    clipboard_text = _read_clipboard_text()
    if clipboard_text:
        image_paths = _extract_image_paths_from_text(clipboard_text)
        if image_paths:
            _insert_image_tokens(buffer, image_paths, pasted_images, pasted_texts)
            return True
        _insert_pasted_text(buffer, clipboard_text, pasted_texts, pasted_images)
        return True
    return False


def _find_image_token_span(text: str, cursor: int, *, backward: bool) -> tuple[int, int] | None:
    return _find_pasted_token_span(text, cursor, backward=backward)


def _expanded_image_delete_span(text: str, start: int, end: int) -> tuple[int, int]:
    return _expanded_pasted_delete_span(text, start, end)


def _delete_image_token(
    buffer,
    pasted_images: list[_PastedImage],
    pasted_texts: list[_PastedText],
    *,
    backward: bool,
) -> bool:
    return _delete_pasted_token(buffer, pasted_images, pasted_texts, backward=backward)


def _finalize_pasted_images(raw_text: str, pasted_images: list[_PastedImage]) -> tuple[list[str], list[str]]:
    active: list[str] = []
    cleanup: list[str] = []
    text = str(raw_text or "")
    for item in pasted_images:
        if item.token in text:
            active.append(item.path)
            if item.owned:
                cleanup.append(item.path)
            continue
        _discard_pasted_image(item)
    return active, cleanup


def _cleanup_image_paths(paths: list[str]) -> None:
    for path_str in paths:
        try:
            Path(path_str).unlink()
        except Exception:
            pass


def _pull_clipboard_prompt_input() -> _PromptInput:
    image_path = _read_clipboard_image()
    if image_path:
        return _PromptInput(text="[Image #1]", image_paths=[image_path], cleanup_paths=[image_path])
    text = _read_clipboard_text()
    image_paths = _extract_image_paths_from_text(text)
    if image_paths:
        tokens = [f"[Image #{idx}]" for idx in range(1, len(image_paths) + 1)]
        return _PromptInput(text=" ".join(tokens), image_paths=image_paths, cleanup_paths=[])
    return _PromptInput(text=text, image_paths=[], cleanup_paths=[])


# ── Tool call formatting ─────────────────────────────────────
def _fmt_args(name: str, args: dict) -> str:
    if name in ("web_search", "web_search_wiki"):
        return str(args.get("query") or "").strip()
    if name == "page_extract":
        url = str(args.get("url") or "").strip()
        query = re.sub(r"\s+", " ", str(args.get("query") or "").strip())
        lines = str(args.get("lines") or "").strip()
        host = ""
        if url:
            parsed = urlparse(url)
            host = str(parsed.netloc or parsed.path or "").strip()
            if host.startswith("www."):
                host = host[4:]
        line_label = "all" if lines.lower() == "all" else (f"{lines}line" if lines else "")
        parts = [part for part in (host, query, line_label) if part]
        return ", ".join(parts)
    if name == "sub_agent_task":
        return _format_sub_agent_argument(args)
    return str(args)[:160]


def _sub_agent_binding_parts(args: dict) -> tuple[str, str]:
    tools = str(args.get("tools") or "").strip()
    url = str(args.get("url") or "").strip()
    host = ""
    if url:
        parsed = urlparse(url)
        host = str(parsed.netloc or parsed.path or "").strip()
        if host.startswith("www."):
            host = host[4:]
    return tools, host


def _format_sub_agent_argument(args: dict) -> str:
    tools, host = _sub_agent_binding_parts(args)
    task = re.sub(r"\s+", " ", str(args.get("task") or "").strip())
    binding = ""
    if tools and host:
        binding = f"{tools}({host})"
    elif tools:
        binding = tools
    elif host:
        binding = host
    if task:
        task = task[:120]
    if binding and task:
        return f"{binding}, {task}"
    return binding or task


def _display_tool_name(name: str, args: dict) -> str:
    return str(args.get("_display_name") or name).strip() or str(name or "").strip()


def _tool_text_line(name: str, args: dict) -> str:
    display_name = _display_tool_name(name, args)
    formatted = _fmt_args(name, args)
    level = max(0, int(args.get("_level") or 0))
    prefix = "  " * level + "> "
    line = f"{prefix}{display_name}"
    if formatted:
        line += f"({formatted})"
    return line


def _planned_tool_block(tools: list[tuple[str, dict]] | None) -> str:
    if not tools:
        return ""
    rows: list[str] = []
    for name, args in tools:
        line = _tool_text_line(name, args)
        if line:
            rows.append(line)
    return "\n".join(rows).strip()


def _tool_line(name: str, args: dict) -> Text:
    formatted = _fmt_args(name, args)
    display_name = _display_tool_name(name, args)
    count = args.get("_count")
    jina_tokens = args.get("_jina_tokens")
    ok = args.get("_ok")
    elapsed_s = args.get("_elapsed_s")
    pending = bool(args.get("_pending"))
    page_billing_mode = str(args.get("_page_billing_mode") or "").strip()
    page_cost_usd = args.get("_page_cost_usd")
    page_usage_requests = args.get("_page_usage_requests")
    page_usage_tokens = args.get("_page_usage_tokens")
    compression_cost_usd = args.get("_compression_cost_usd")
    level = max(0, int(args.get("_level") or 0))

    line = Text()
    line.append("  " + ("  " * level) + "> ", style=f"bold {ACCENT}")
    line.append_text(_gradient_title(display_name))
    if name == "sub_agent_task":
        tools, host = _sub_agent_binding_parts(args)
        task = re.sub(r"\s+", " ", str(args.get("task") or "").strip())
        if task:
            task = task[:120]
        if tools or host or task:
            line.append("(", style=TEXT_MUTED)
            if tools:
                line.append_text(_gradient_rich_text(tools))
            if host:
                line.append(f"({host})", style=TEXT_MUTED)
            if task:
                if tools or host:
                    line.append(", ", style=TEXT_MUTED)
                line.append(task, style=TEXT_MUTED)
            line.append(")", style=TEXT_MUTED)
    elif formatted:
        line.append(f"({formatted})", style=TEXT_MUTED)
    if pending:
        frame = _FRAMES[int(time.time() * 12) % len(_FRAMES)]
        line.append(f" {frame}", style=TEXT_MUTED)
    elif ok is False:
        line.append(" !", style=TEXT_MUTED)
    elif count not in (None, ""):
        line.append(" ", style=TEXT_MUTED)
        line.append(str(count), style=f"bold {ACCENT}")
        line.append(" ✓", style=TEXT_MUTED)
    elif ok is not None:
        line.append(" ✓", style=TEXT_MUTED)
    if jina_tokens not in (None, ""):
        line.append(" ", style=TEXT_MUTED)
        line.append(f"jina {jina_tokens}tok", style=TEXT_MUTED)
    if name == "sub_agent_task" and str(args.get("tools") or "").strip().lower() == "page":
        line.append(" ", style=TEXT_MUTED)
        if page_billing_mode == "free":
            line.append("page $0", style=TEXT_MUTED)
            if page_usage_tokens not in (None, ""):
                line.append(f" {page_usage_tokens}tok", style=TEXT_MUTED)
        elif page_billing_mode == "paid":
            if isinstance(page_cost_usd, (int, float)):
                line.append(f"page ${float(page_cost_usd):.6f}", style=TEXT_MUTED)
            else:
                line.append("page paid", style=TEXT_MUTED)
            if page_usage_tokens not in (None, ""):
                line.append(f" {page_usage_tokens}tok", style=TEXT_MUTED)
        elif page_billing_mode and page_billing_mode != "skipped":
            line.append(f"page {page_billing_mode}", style=TEXT_MUTED)
            if page_usage_tokens not in (None, ""):
                line.append(f" {page_usage_tokens}tok", style=TEXT_MUTED)
        if isinstance(compression_cost_usd, (int, float)):
            line.append(" ", style=TEXT_MUTED)
            line.append(f"compress ${float(compression_cost_usd):.6f}", style=TEXT_MUTED)
    if elapsed_s not in (None, ""):
        try:
            line.append(f" {float(elapsed_s):.1f}s", style=TEXT_MUTED)
        except Exception:
            pass
    if name == "page_extract" and args.get("_from_cache"):
        line.append(" cache", style=TEXT_MUTED)
    return line


# ── Clean answer ──────────────────────────────────────────────
def _strip_task_list(text: str) -> str:
    return _TASK_LIST_BLOCK_RE.sub("", str(text or ""))


def _render_article_skeleton_xml(match: re.Match[str]) -> str:
    block = str(match.group(0) or "")
    title_match = re.search(r"<title\b[^>]*>(.*?)</title>", block, flags=re.IGNORECASE | re.DOTALL)
    title = re.sub(r"\s+", " ", str(title_match.group(1) or "").strip()) if title_match else "未命名骨架"
    lines = [f"### 骨架\n**{title}**"]
    section_matches = list(re.finditer(r"<section\b([^>]*)>(.*?)</section>", block, flags=re.IGNORECASE | re.DOTALL))
    if section_matches:
        for section_match in section_matches:
            attrs = dict(
                (str(k or "").strip().lower(), str((dq if dq != "" else sq) or "").strip())
                for k, dq, sq in re.findall(r'([:\w-]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\')', str(section_match.group(1) or ""))
            )
            section_name = str(attrs.get("name") or attrs.get("title") or "").strip() or "未命名章节"
            lines.append(f"\n#### {section_name}")
            for claim_match in re.finditer(r"^\s*\[(\d+)\]\s+(.+?)\s*$", str(section_match.group(2) or ""), flags=re.MULTILINE):
                lines.append(f"- [{claim_match.group(1)}] {claim_match.group(2).strip()}")
    else:
        for claim_match in re.finditer(r"^\s*\[(\d+)\]\s+(.+?)\s*$", block, flags=re.MULTILINE):
            lines.append(f"- [{claim_match.group(1)}] {claim_match.group(2).strip()}")
    return "\n".join(lines).strip()


def _render_search_rewrite_xml(match: re.Match[str]) -> str:
    block = str(match.group(0) or "")
    terms = [
        re.sub(r"\s+", " ", str(term_match.group(1) or "").strip())
        for term_match in re.finditer(r"<term\b[^>]*>(.*?)</term>", block, flags=re.IGNORECASE | re.DOTALL)
    ]
    if not terms:
        inner = re.sub(r"</?search_rewrite\b[^>]*>", "", block, flags=re.IGNORECASE)
        for line in inner.splitlines():
            candidate = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", str(line or "").strip())
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if candidate:
                terms.append(candidate)
    if not terms:
        return ""
    return "### 搜索词重绘\n" + "\n".join(f"- {term}" for term in terms)


def _render_keyword_rewrite_xml(match: re.Match[str]) -> str:
    block = str(match.group(0) or "")
    terms = [
        re.sub(r"\s+", " ", str(term_match.group(1) or "").strip())
        for term_match in re.finditer(r"<t\b[^>]*>(.*?)</t>", block, flags=re.IGNORECASE | re.DOTALL)
    ]
    if not terms:
        return ""
    return "### Keyword Rewrite\n" + "\n".join(f"- {term}" for term in terms)


def _render_user_need_xml(match: re.Match[str]) -> str:
    block = str(match.group(0) or "")
    items = [
        re.sub(r"\s+", " ", str(item_match.group(1) or "").strip())
        for item_match in re.finditer(r"<u\b[^>]*>(.*?)</u>", block, flags=re.IGNORECASE | re.DOTALL)
    ]
    items = [item for item in items if item]
    if not items:
        return ""
    return "### User Need Reconstruction\n" + "\n".join(f"- {item}" for item in items)


def _render_verification_outline_xml(match: re.Match[str]) -> str:
    block = str(match.group(0) or "")
    lines = ["### Verification Outline"]
    for claim_match in re.finditer(r"<i\b[^>]*\bid\s*=\s*['\"]?(\d+)['\"]?[^>]*>(.*?)</i>", block, flags=re.IGNORECASE | re.DOTALL):
        claim_id = str(claim_match.group(1) or "").strip()
        claim_text = re.sub(r"\s+", " ", str(claim_match.group(2) or "").strip())
        if claim_id and claim_text:
            lines.append(f"- [{claim_id}] {claim_text}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _clean_answer(text: str) -> str:
    cleaned = re.sub(
        r"<summary>\s*(.*?)\s*</summary>",
        lambda m: "\n".join(f"> {line}" for line in [x.strip() for x in m.group(1).splitlines()] if line),
        str(text or ""),
        flags=re.I | re.S,
    )

    def _image_to_link(match: re.Match[str]) -> str:
        alt = str(match.group(1) or "").strip()
        raw_target = str(match.group(2) or "").strip()
        if not raw_target:
            return match.group(0)
        if raw_target.lower().startswith("data:image"):
            label = alt or "图片"
            return f"{label}: [inline image omitted]"
        label = alt or "图片"
        return f"{label}: [{raw_target}]({raw_target})"

    cleaned = _MD_IMAGE_RE.sub(_image_to_link, cleaned)
    cleaned = _KEYWORD_REWRITE_BLOCK_RE.sub(_render_keyword_rewrite_xml, cleaned)
    cleaned = _USER_NEED_BLOCK_RE.sub(_render_user_need_xml, cleaned)
    cleaned = _VERIFICATION_OUTLINE_BLOCK_RE.sub(_render_verification_outline_xml, cleaned)
    cleaned = _SEARCH_REWRITE_BLOCK_RE.sub(_render_search_rewrite_xml, cleaned)
    cleaned = _ARTICLE_SKELETON_BLOCK_RE.sub(_render_article_skeleton_xml, cleaned)
    cleaned = _strip_task_list(cleaned)
    cleaned = _TOOL_BLOCK_RE.sub("", cleaned)
    cleaned = _TOOL_SELF_CLOSING_RE.sub("", cleaned)
    # 清理残留的 XML 工具标签
    cleaned = re.sub(r"</?(?:search|wiki|sub_agent|page|tool_results|result|article_skeleton|search_rewrite|keyword_rewrite|user_need|verification_outline|section|title|term|t|u|i)\b[^>]*>", "", cleaned)
    return cleaned.strip()


def _gradient_name(name: str = "") -> list[tuple[str, str]]:
    """Return prompt_toolkit formatted tuples for the configured name in gradient."""
    label = name or DEFAULT_NAME
    parts: list[tuple[str, str]] = []
    for i, ch in enumerate(label):
        r, g, b = _title_rgb(i, len(label))
        parts.append((f"bold fg:#{r:02x}{g:02x}{b:02x}", ch))
    return parts


def _gradient_label(text: str) -> list[tuple[str, str]]:
    """Return prompt_toolkit formatted tuples for arbitrary text in gradient."""
    parts: list[tuple[str, str]] = []
    for i, ch in enumerate(text):
        if ch == " ":
            parts.append(("", " "))
            continue
        r, g, b = _title_rgb(i, len(text))
        parts.append((f"fg:#{r:02x}{g:02x}{b:02x}", ch))
    return parts


def _spinner_frame() -> str:
    return _FRAMES[int(time.time() * 12) % len(_FRAMES)]


def _animated_placeholder_label(text: str) -> list[tuple[str, str]]:
    label = str(text or "").strip()
    if not label:
        return []
    if label.endswith("..."):
        base = label[:-3].rstrip()
        if base:
            return [("", f"{base} "), ("class:placeholder_hint", _spinner_frame())]
        return [("class:placeholder_hint", _spinner_frame())]
    return [("", label)]


def _prompt_status(mode_state: dict | None = None) -> str:
    state = mode_state or {}
    if str(state.get("config_issue") or "").strip():
        return "!"
    ready = state.get("tools_ready")
    failed = state.get("tools_failed")
    if ready:
        return "✓"
    if failed:
        return "!"
    return "…"


def _prompt_placeholder(mode_state: dict | None = None) -> list[tuple[str, str]]:
    state = mode_state or {}
    config_issue = str(state.get("config_issue") or "").strip()
    if config_issue:
        return [("", config_issue)]
    model = str(state.get("model") or "").strip()
    runtime = state.get("_runtime_label")
    label = str(runtime() if callable(runtime) else "").strip()
    if label and label != "✓":
        return _animated_placeholder_label(label)
    if not model:
        return []
    return [("", model)]


def _prompt_parts(mode_state: dict | None = None) -> list[tuple[str, str]]:
    state = mode_state or {}
    multi = state.get("multi_turn", True)
    status_color = f"bold fg:#{255:02x}{108:02x}{60:02x}"
    parts: list[tuple[str, str]] = [("class:prompt", _PROMPT_PREFIX)]
    parts.extend(_gradient_name(str(state.get("name") or "")))
    if not multi:
        parts.append(("", " ("))
        parts.append(("class:mode", "New Session"))
        parts.append(("", ")"))
    parts.append(("", " "))
    parts.append((status_color, f"{_prompt_status(state)} "))
    return parts


def _config_issue_from(config: dict[str, Any]) -> tuple[str, str]:
    path = str(config.get("_config_path") or CONFIG_PATH.expanduser())
    exists = bool(config.get("_config_exists"))
    valid = bool(config.get("_config_valid"))
    error = str(config.get("_config_error") or "").strip()

    if valid:
        return "", ""
    if exists:
        return (f"配置损坏: {_tail_by_cells(path, 48)}", error)
    return (f"配置文件: {_tail_by_cells(path, 48)}", "")


def _provider_from_api_base(api_base: str) -> str:
    raw = str(api_base or "").strip()
    if not raw:
        return ""
    try:
        host = str(urlparse(raw).netloc or "").strip().lower()
    except Exception:
        host = ""
    if not host:
        return ""
    if "openrouter.ai" in host:
        return "openrouter"
    if "cerebras.ai" in host:
        return "cerebras"
    if "openai.com" in host:
        return "openai"
    if "anthropic.com" in host:
        return "anthropic"
    if "groq.com" in host:
        return "groq"
    if "fireworks.ai" in host:
        return "fireworks"
    if "x.ai" in host:
        return "xai"
    if "together.xyz" in host:
        return "together"
    if "aliyuncs.com" in host or "dashscope" in host:
        return "dashscope"
    if "siliconflow" in host:
        return "siliconflow"
    if "googleapis.com" in host or "generativelanguage" in host:
        return "google"
    if host.startswith(("localhost", "127.0.0.1")):
        return "local"
    if host.startswith("[::1]"):
        return "local"
    first = host.split(".", 1)[0].strip()
    return first


def _provider_from_model_id(model_id: str) -> str:
    text = str(model_id or "").strip()
    if not text or "/" not in text:
        return ""
    first = text.split("/", 1)[0].strip().lower()
    if first in {
        "openrouter",
        "cerebras",
        "openai",
        "anthropic",
        "groq",
        "fireworks",
        "xai",
        "together",
        "dashscope",
        "siliconflow",
        "google",
        "ollama",
    }:
        return first
    return ""


def _model_provider(profile: dict[str, Any]) -> str:
    provider = _provider_from_api_base(str(profile.get("api_base") or ""))
    if provider:
        return provider
    return _provider_from_model_id(str(profile.get("model") or ""))


def _compact_model_id(model_id: str, provider: str) -> str:
    text = str(model_id or "").strip()
    provider_text = str(provider or "").strip().lower()
    if not text or not provider_text:
        return text
    prefix = provider_text + "/"
    if text.lower().startswith(prefix):
        return text[len(prefix):]
    return text


def _canonical_model_label(profile: dict[str, Any]) -> str:
    provider = _model_provider(profile)
    model_id = str(profile.get("model") or "").strip()
    alias = str(profile.get("name") or "").strip()
    if model_id:
        if provider and not model_id.lower().startswith(provider.lower() + "/"):
            return f"{provider}/{model_id}"
        return model_id
    return alias


def _display_model_label(profile: dict[str, Any]) -> str:
    return _canonical_model_label(profile)


def _mode_stage_index(mode_state: dict[str, Any], key: str, default: int) -> int:
    value = mode_state.get(key)
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, int):
        return value
    return default


def _stage_model_display_label(cfg: dict[str, Any]) -> str:
    profile = {
        "model": str(cfg.get("model") or "").strip(),
        "api_base": str(cfg.get("api_base") or "").strip(),
    }
    return _display_model_label(profile)


def _stage_model_chain(
    config: dict[str, Any],
    *,
    stage1_model_index: int,
    stage2_model_index: int,
) -> tuple[str, str, str]:
    stage1_cfg = build_stage_model_config(
        config,
        "stage1",
        stage1_model_index=stage1_model_index,
        stage2_model_index=stage2_model_index,
    )
    stage2_cfg = build_stage_model_config(
        config,
        "stage2",
        stage1_model_index=stage1_model_index,
        stage2_model_index=stage2_model_index,
    )
    stage1_model_raw = str(stage1_cfg.get("model") or "").strip()
    stage2_model_raw = str(stage2_cfg.get("model") or "").strip()
    stage1_label = _stage_model_display_label(stage1_cfg)
    stage2_label = _stage_model_display_label(stage2_cfg)
    stage1_provider = _model_provider(
        {
            "model": str(stage1_cfg.get("model") or "").strip(),
            "api_base": str(stage1_cfg.get("api_base") or "").strip(),
        }
    )
    stage2_provider = _model_provider(
        {
            "model": str(stage2_cfg.get("model") or "").strip(),
            "api_base": str(stage2_cfg.get("api_base") or "").strip(),
        }
    )
    if stage1_model_raw and stage1_model_raw == stage2_model_raw:
        stage2_label = stage1_label
    elif stage1_provider and stage1_provider == stage2_provider:
        compact_stage2 = _compact_model_id(str(stage2_cfg.get("model") or "").strip(), stage2_provider)
        if compact_stage2:
            stage2_label = compact_stage2
    chain = stage1_label
    if stage2_label and stage2_label != stage1_label:
        chain = f"{stage1_label} -> {stage2_label}"
    return stage1_label, stage2_label, chain


def _apply_model_state(mode_state: dict, stage1_index: int | None = None, stage2_index: int | None = None) -> None:
    models = mode_state.get("models") or []
    raw_config = mode_state.get("config") if isinstance(mode_state.get("config"), dict) else {}
    if not isinstance(models, list) or not models:
        mode_state["stage1_model_index"] = 0
        mode_state["stage2_model_index"] = 0
        mode_state["model_count"] = 0
        mode_state["model"] = ""
        mode_state["model_id"] = ""
        mode_state["model_provider"] = ""
        mode_state["stage1_model"] = ""
        mode_state["stage2_model"] = ""
        return

    count = len(models)
    idx1 = int((0 if stage1_index is None else stage1_index) or 0) % count
    idx2_default = 1 if count > 1 else 0
    idx2 = int((idx2_default if stage2_index is None else stage2_index) or 0) % count
    current = models[idx1] if isinstance(models[idx1], dict) else {}
    provider = _model_provider(current)
    stage1_label, stage2_label, chain_label = _stage_model_chain(
        raw_config,
        stage1_model_index=idx1,
        stage2_model_index=idx2,
    )
    label = chain_label or _display_model_label(current)
    mode_state["stage1_model_index"] = idx1
    mode_state["stage2_model_index"] = idx2
    mode_state["model_count"] = len(models)
    mode_state["model"] = label
    mode_state["model_id"] = stage1_label or str(current.get("model") or "").strip()
    mode_state["model_provider"] = provider
    mode_state["stage1_model"] = stage1_label
    mode_state["stage2_model"] = stage2_label


def _cycle_model(mode_state: dict, stage: str, delta: int = 1) -> None:
    models = mode_state.get("models") or []
    if not isinstance(models, list) or len(models) <= 1:
        return
    stage_name = str(stage or "").strip().lower()
    stage1_index = _mode_stage_index(mode_state, "stage1_model_index", 0)
    stage2_index = _mode_stage_index(mode_state, "stage2_model_index", 1 if len(models) > 1 else 0)
    if stage_name == "stage2":
        stage2_index += delta
    else:
        stage1_index += delta
    _apply_model_state(mode_state, stage1_index, stage2_index)
    runtime_prewarm = mode_state.get("_runtime_prewarm")
    if callable(runtime_prewarm):
        try:
            runtime_prewarm()
        except Exception:
            pass


# ── prompt_toolkit input ──────────────────────────────────────
def _ask(session, *, mode_state: dict | None = None) -> _PromptInput:
    """Read user input. Empty-buffer arrows control model/session shortcuts."""
    pasted_images: list[_PastedImage] = []
    pasted_texts: list[_PastedText] = []

    if session is not None and _PT_AVAILABLE and PTStyle is not None and mode_state is not None:
        kb = KeyBindings()

        @kb.add("left", filter=Condition(lambda: not session.app.current_buffer.text))
        def _cycle_stage1(event):
            _cycle_model(mode_state, "stage1", 1)
            event.app.invalidate()

        @kb.add("right", filter=Condition(lambda: not session.app.current_buffer.text))
        def _cycle_stage2(event):
            _cycle_model(mode_state, "stage2", 1)
            event.app.invalidate()

        @kb.add("up", filter=Condition(lambda: not session.app.current_buffer.text))
        @kb.add("down", filter=Condition(lambda: not session.app.current_buffer.text))
        def _toggle_session(event):
            mode_state["multi_turn"] = not mode_state["multi_turn"]
            event.app.invalidate()

        @kb.add("c-v", eager=True)
        def _paste(event):
            _paste_clipboard_into_buffer(event.current_buffer, pasted_images, pasted_texts)

        @kb.add("escape", "v", eager=True)
        def _paste_alt(event):
            _paste_clipboard_into_buffer(event.current_buffer, pasted_images, pasted_texts)

        @kb.add("backspace", eager=True)
        def _delete_prev(event):
            if _delete_image_token(event.current_buffer, pasted_images, pasted_texts, backward=True):
                return
            event.current_buffer.delete_before_cursor(count=1)

        @kb.add("delete", eager=True)
        def _delete_next(event):
            if _delete_image_token(event.current_buffer, pasted_images, pasted_texts, backward=False):
                return
            event.current_buffer.delete(count=1)

        @kb.add(Keys.BracketedPaste, eager=True)
        def _paste_bracketed(event):
            pasted_text = str(event.data or "")
            if not pasted_text.strip():
                if _paste_clipboard_into_buffer(event.current_buffer, pasted_images, pasted_texts):
                    return
                return
            image_paths = _extract_image_paths_from_text(pasted_text)
            if image_paths:
                _insert_image_tokens(event.current_buffer, image_paths, pasted_images, pasted_texts, owned=False)
                return
            _insert_pasted_text(event.current_buffer, pasted_text, pasted_texts, pasted_images)

        def _get_rprompt():
            return []

        def _get_prompt():
            return _prompt_parts(mode_state)

        def _get_placeholder():
            return _prompt_placeholder(mode_state)

        input_processors = []
        if BeforeInput is not None and ConditionalProcessor is not None:
            input_processors.append(
                ConditionalProcessor(
                    BeforeInput(_get_placeholder, style="class:placeholder"),
                    filter=Condition(
                        lambda: (
                            not session.app.current_buffer.text
                            and bool(str((mode_state or {}).get("model") or "").strip())
                        )
                    ),
                )
            )

        input_style = PTStyle.from_dict(
            {
                "": INPUT_TEXT,
                "prompt": f"bold {INPUT_ACCENT}",
                "mode": "bold",
                "status": INPUT_HINT,
                "placeholder": f"fg:{INPUT_PLACEHOLDER}",
                "placeholder_hint": f"fg:{INPUT_PLACEHOLDER}",
                "bottom-toolbar": f"fg:{INPUT_PLACEHOLDER} bg:default noreverse",
                "toolbar": f"fg:{INPUT_PLACEHOLDER}",
            }
        )
        try:
            raw_text = session.prompt(
                _get_prompt,
                rprompt=_get_rprompt,
                bottom_toolbar=lambda: _prompt_bottom_toolbar(mode_state),
                multiline=False,
                style=input_style,
                input_processors=input_processors,
                default="",
                enable_open_in_editor=False,
                key_bindings=kb,
                pre_run=mode_state.get("_runtime_prewarm") if callable(mode_state.get("_runtime_prewarm")) else None,
                refresh_interval=0.15,
            )
        finally:
            pass
        return _normalize_prompt_input(str(raw_text or ""), pasted_images, pasted_texts)

    if session is not None and _PT_AVAILABLE and PTStyle is not None:
        input_style = PTStyle.from_dict(
            {
                "": INPUT_TEXT,
                "prompt": f"bold {INPUT_ACCENT}",
                "mode": "bold",
                "placeholder": f"fg:{INPUT_PLACEHOLDER}",
                "placeholder_hint": f"fg:{INPUT_PLACEHOLDER}",
                "bottom-toolbar": f"fg:{INPUT_PLACEHOLDER} bg:default noreverse",
                "toolbar": f"fg:{INPUT_PLACEHOLDER}",
            }
        )
        raw_text = session.prompt(
            _prompt_parts(mode_state),
            bottom_toolbar=lambda: _prompt_bottom_toolbar(mode_state),
            multiline=False,
            style=input_style,
            input_processors=[
                ConditionalProcessor(
                    BeforeInput(lambda: _prompt_placeholder(mode_state), style="class:placeholder"),
                    filter=Condition(
                        lambda: (
                            not session.app.current_buffer.text
                            and bool(str((mode_state or {}).get("model") or "").strip())
                        )
                    ),
                )
            ] if BeforeInput is not None and ConditionalProcessor is not None else None,
            default="",
            enable_open_in_editor=False,
            pre_run=mode_state.get("_runtime_prewarm") if callable((mode_state or {}).get("_runtime_prewarm")) else None,
            refresh_interval=0.15,
        )
        return _normalize_prompt_input(str(raw_text or ""), pasted_images, pasted_texts)
    runtime_prewarm = (mode_state or {}).get("_runtime_prewarm")
    if callable(runtime_prewarm):
        try:
            runtime_prewarm()
        except Exception:
            pass
    toolbar_text = _turn_stats_text((mode_state or {}).get("last_turn_stats"))
    if toolbar_text:
        print(f"  {toolbar_text}")
    return _normalize_prompt_input(input(_PROMPT_PREFIX).strip(), pasted_images, pasted_texts)


# ── Spinner (lightweight, for runtime status) ─────────────────
class _Spinner:
    """Simple inline spinner using \\r overwrites."""

    def __init__(self) -> None:
        self._text = ""
        self._active = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self, text: str = "") -> None:
        if not (sys.stdout.isatty() and not _DUMB):
            return
        with self._lock:
            self._text = text
            if self._active:
                return
            self._active = True

        def _loop() -> None:
            while True:
                with self._lock:
                    if not self._active:
                        break
                    t = self._text
                frame = _FRAMES[int(time.time() * 12) % len(_FRAMES)]
                with _IO_LOCK:
                    sys.stdout.write(f"\r  {INTRO_HINT_ANSI}{frame} {t}\x1b[0m\x1b[K")
                    sys.stdout.flush()
                time.sleep(0.09)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def update(self, text: str) -> None:
        with self._lock:
            self._text = text

    def stop(self) -> None:
        with self._lock:
            was_active = self._active
            self._active = False
        if self._thread:
            self._thread.join(0.3)
            self._thread = None
        if was_active:
            with _IO_LOCK:
                sys.stdout.write("\r\x1b[K")
                sys.stdout.flush()


# ── Intro status line (one-liner startup) ─────────────────────
class _IntroStatusLine:
    def __init__(self, model: str = "") -> None:
        self.model = model
        self.ready = False
        self.failed = False
        self._stop = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _frame(self) -> str:
        return _FRAMES[int(time.time() * 12) % len(_FRAMES)]

    def _status_line(self) -> str:
        prefix = f"➜  HYW({self.model})"
        if self.ready:
            return f"  {prefix} ✓"
        if self.failed:
            return f"  {prefix} ✗"
        return f"  {prefix} {self._frame()}"

    def _render_once(self) -> None:
        if not (sys.stdout.isatty() and not _DUMB):
            return
        if not _INPUT_RENDERED.is_set():
            return
        with self._lock:
            line = self._status_line()
            failed = bool(self.failed)
            ready = bool(self.ready)
        width = max(8, int(_STATUS_MAX_CELLS))
        view = _tail_by_cells(line, width)
        pad = max(0, width - _text_cells(view))
        with _IO_LOCK:
            style = INTRO_PROMPT_ANSI if failed else (INTRO_INPUT_ANSI if ready else INTRO_HINT_ANSI)
            sys.stdout.write("\r" + style + view + (" " * pad) + "\x1b[0m")
            sys.stdout.flush()

    def start(self) -> None:
        if not (sys.stdout.isatty() and not _DUMB):
            return
        with self._lock:
            self._stop = False
            if self._thread is not None and self._thread.is_alive():
                return

        def _loop() -> None:
            while True:
                with self._lock:
                    stopped = bool(self._stop)
                self._render_once()
                if stopped:
                    break
                time.sleep(0.09)

        self._thread = threading.Thread(target=_loop, name="hyw-status", daemon=True)
        self._thread.start()

    def set_ready(self) -> None:
        with self._lock:
            self.ready = True
            self.failed = False
        self._render_once()

    def set_failed(self, reason: str) -> None:
        with self._lock:
            self.ready = False
            self.failed = True
        self._render_once()

    def stop(self, *, keep_last: bool = True) -> None:
        with self._lock:
            self._stop = True
            t = self._thread
        if t is not None:
            t.join(timeout=0.25)
        if not keep_last and sys.stdout.isatty() and not _DUMB and _INPUT_RENDERED.is_set():
            width = max(8, int(_STATUS_MAX_CELLS))
            with _IO_LOCK:
                sys.stdout.write("\r" + (" " * width) + "\r")
                sys.stdout.flush()
            return
        self._render_once()


# ── Welcome (minimal setup, no panel) ────────────────────────
def _welcome_setup(console: Console) -> None:
    global _STATUS_MAX_CELLS
    if not (sys.stdout.isatty() and not _DUMB):
        return
    width = int(getattr(getattr(console, "size", None), "width", 120) or 120)
    _STATUS_MAX_CELLS = max(12, width - 4)
    _INPUT_RENDERED.set()
    _INPUT_RENDERED.set()


# ── Stats subtitle ────────────────────────────────────────────
def _stats_subtitle(stats: Stats, *, multi_turn: bool = True, elapsed: float | None = None) -> Text:
    out = Text()
    if elapsed is not None:
        if elapsed >= 60:
            out.append(f"{elapsed / 60:.1f}m ", style=f"bold {ACCENT}")
        else:
            out.append(f"{elapsed:.1f}s ", style=f"bold {ACCENT}")
        out.append(" ", style=TEXT_MUTED)
    out.append("↑ ", style=f"bold {ACCENT}")
    out.append(str(stats.prompt_tokens), style=TEXT_MUTED)
    out.append("  ", style=TEXT_MUTED)
    out.append("↓ ", style=f"bold {ACCENT}")
    out.append(str(stats.completion_tokens), style=TEXT_MUTED)
    out.append("  ", style=TEXT_MUTED)
    out.append("think ", style=f"bold {ACCENT}")
    out.append(str(stats.reasoning_tokens), style=TEXT_MUTED)
    out.append("tok", style=TEXT_MUTED)
    out.append("  ", style=TEXT_MUTED)
    out.append("cost ", style=f"bold {ACCENT}")
    cost = f"${stats.cost_usd:.6f}" if stats._has_cost else "N/A"
    out.append(cost, style=TEXT_MUTED)
    out.append("  ", style=TEXT_MUTED)
    out.append("jina ", style=f"bold {ACCENT}")
    out.append(f"{stats.jina_tokens}tok", style=TEXT_MUTED)
    out.append("  ", style=TEXT_MUTED)
    out.append("calls ", style=f"bold {ACCENT}")
    out.append(str(stats.calls), style=TEXT_MUTED)
    return out


def _thinking_status_text(meta: dict[str, Any] | None = None) -> str:
    info = dict(meta or {})
    tokens = int(info.get("reasoning_tokens") or 0)
    if tokens > 0:
        return f"Thinking ({tokens}tok)"
    return "Thinking..."


def _stats_snapshot(stats: Stats) -> dict[str, Any]:
    with stats._lock:
        return {
            "calls": int(stats.calls),
            "prompt_tokens": int(stats.prompt_tokens),
            "completion_tokens": int(stats.completion_tokens),
            "reasoning_tokens": int(stats.reasoning_tokens),
            "total_tokens": int(stats.total_tokens),
            "cost_usd": float(stats.cost_usd),
            "jina_tokens": int(stats.jina_tokens),
            "jina_calls": int(stats.jina_calls),
            "_has_cost": bool(stats._has_cost),
        }


def _stats_delta(before: dict[str, Any], after: dict[str, Any], *, elapsed: float) -> dict[str, Any]:
    prompt_tokens = max(0, int(after.get("prompt_tokens") or 0) - int(before.get("prompt_tokens") or 0))
    completion_tokens = max(0, int(after.get("completion_tokens") or 0) - int(before.get("completion_tokens") or 0))
    reasoning_tokens = max(0, int(after.get("reasoning_tokens") or 0) - int(before.get("reasoning_tokens") or 0))
    total_tokens = max(0, int(after.get("total_tokens") or 0) - int(before.get("total_tokens") or 0))
    calls = max(0, int(after.get("calls") or 0) - int(before.get("calls") or 0))
    jina_tokens = max(0, int(after.get("jina_tokens") or 0) - int(before.get("jina_tokens") or 0))
    jina_calls = max(0, int(after.get("jina_calls") or 0) - int(before.get("jina_calls") or 0))
    cost_delta = float(after.get("cost_usd") or 0.0) - float(before.get("cost_usd") or 0.0)
    has_cost = bool(after.get("_has_cost")) and calls > 0
    return {
        "elapsed": max(0.0, float(elapsed or 0.0)),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "cost_usd": max(0.0, cost_delta),
        "jina_tokens": jina_tokens,
        "jina_calls": jina_calls,
        "calls": calls,
        "_has_cost": has_cost,
    }


def _turn_stats_text(turn_stats: dict[str, Any] | None) -> str:
    data = dict(turn_stats or {})
    if not data:
        return ""
    elapsed = float(data.get("elapsed") or 0.0)
    if elapsed <= 0 and not any(int(data.get(key) or 0) for key in ("prompt_tokens", "completion_tokens", "calls", "jina_tokens", "reasoning_tokens")):
        return ""
    elapsed_text = f"{elapsed / 60:.1f}m" if elapsed >= 60 else f"{elapsed:.1f}s"
    cost_text = f"${float(data.get('cost_usd') or 0.0):.6f}" if data.get("_has_cost") else "N/A"
    return (
        f"{elapsed_text}  "
        f"↑ {int(data.get('prompt_tokens') or 0)}  "
        f"↓ {int(data.get('completion_tokens') or 0)}  "
        f"think {int(data.get('reasoning_tokens') or 0)}tok  "
        f"cost {cost_text}  "
        f"jina {int(data.get('jina_tokens') or 0)}tok  "
        f"calls {int(data.get('calls') or 0)}"
    )


def _turn_stats_toolbar_parts(turn_stats: dict[str, Any] | None) -> list[tuple[str, str]]:
    data = dict(turn_stats or {})
    if not data:
        return []
    elapsed = float(data.get("elapsed") or 0.0)
    if elapsed <= 0 and not any(int(data.get(key) or 0) for key in ("prompt_tokens", "completion_tokens", "calls", "jina_tokens", "reasoning_tokens")):
        return []

    parts: list[tuple[str, str]] = []
    if elapsed > 0:
        elapsed_text = f"{elapsed / 60:.1f}m" if elapsed >= 60 else f"{elapsed:.1f}s"
        parts.append((f"bold fg:{PT_ACCENT}", elapsed_text))
        parts.append((f"fg:{INPUT_PLACEHOLDER}", "  "))
    parts.extend(
        [
            (f"bold fg:{PT_ACCENT}", "↑ "),
            (f"fg:{INPUT_PLACEHOLDER}", str(int(data.get("prompt_tokens") or 0))),
            (f"fg:{INPUT_PLACEHOLDER}", "  "),
            (f"bold fg:{PT_ACCENT}", "↓ "),
            (f"fg:{INPUT_PLACEHOLDER}", str(int(data.get("completion_tokens") or 0))),
            (f"fg:{INPUT_PLACEHOLDER}", "  "),
            (f"bold fg:{PT_ACCENT}", "think "),
            (f"fg:{INPUT_PLACEHOLDER}", str(int(data.get("reasoning_tokens") or 0))),
            (f"fg:{INPUT_PLACEHOLDER}", "tok  "),
            (f"bold fg:{PT_ACCENT}", "cost "),
            (
                f"fg:{INPUT_PLACEHOLDER}",
                f"${float(data.get('cost_usd') or 0.0):.6f}" if data.get("_has_cost") else "N/A",
            ),
            (f"fg:{INPUT_PLACEHOLDER}", "  "),
            (f"bold fg:{PT_ACCENT}", "jina "),
            (f"fg:{INPUT_PLACEHOLDER}", str(int(data.get("jina_tokens") or 0))),
            (f"fg:{INPUT_PLACEHOLDER}", "tok  "),
            (f"bold fg:{PT_ACCENT}", "calls "),
            (f"fg:{INPUT_PLACEHOLDER}", str(int(data.get("calls") or 0))),
        ]
    )
    return parts


def _prompt_bottom_toolbar(mode_state: dict | None = None) -> list[tuple[str, str]]:
    parts = _turn_stats_toolbar_parts((mode_state or {}).get("last_turn_stats"))
    if not parts:
        return []
    plain = "".join(text for _, text in parts)
    width = shutil.get_terminal_size((120, 20)).columns
    pad = max(0, width - 2 - _text_cells(plain))
    return [("", " " * pad), *parts]


def _estimate_preview_tokens(text: str) -> int:
    body = str(text or "").strip()
    if not body:
        return 0
    return max(1, len(body) // 4)


def _render_turn_stats(delta: dict[str, Any] | None) -> Text:
    data = dict(delta or {})
    out = Text(style=TEXT_MUTED)
    elapsed = float(data.get("elapsed") or 0.0)
    if elapsed > 0:
        if elapsed >= 60:
            out.append(f"{elapsed / 60:.1f}m", style=f"bold {ACCENT}")
        else:
            out.append(f"{elapsed:.1f}s", style=f"bold {ACCENT}")
        out.append("  ", style=TEXT_MUTED)
    out.append("↑ ", style=f"bold {ACCENT}")
    out.append(str(int(data.get("prompt_tokens") or 0)), style=TEXT_MUTED)
    out.append("  ", style=TEXT_MUTED)
    out.append("↓ ", style=f"bold {ACCENT}")
    out.append(str(int(data.get("completion_tokens") or 0)), style=TEXT_MUTED)
    out.append("  ", style=TEXT_MUTED)
    out.append("think ", style=f"bold {ACCENT}")
    out.append(str(int(data.get("reasoning_tokens") or 0)), style=TEXT_MUTED)
    out.append("tok", style=TEXT_MUTED)
    out.append("  ", style=TEXT_MUTED)
    out.append("cost ", style=f"bold {ACCENT}")
    if data.get("_has_cost"):
        out.append(f"${float(data.get('cost_usd') or 0.0):.6f}", style=TEXT_MUTED)
    else:
        out.append("N/A", style=TEXT_MUTED)
    out.append("  ", style=TEXT_MUTED)
    out.append("jina ", style=f"bold {ACCENT}")
    out.append(str(int(data.get("jina_tokens") or 0)), style=TEXT_MUTED)
    out.append("tok", style=TEXT_MUTED)
    out.append("  ", style=TEXT_MUTED)
    out.append("calls ", style=f"bold {ACCENT}")
    out.append(str(int(data.get("calls") or 0)), style=TEXT_MUTED)
    return out


def _render_locked_prompt(mode_state: dict | None = None):
    state = mode_state or {}
    line = Text()
    line.append(_PROMPT_PREFIX, style=f"bold {ACCENT}")
    line.append_text(_gradient_title(str(state.get("name") or DEFAULT_NAME)))
    if not bool(state.get("multi_turn", True)):
        line.append(" (", style=TEXT_MUTED)
        line.append("New Session", style=f"bold {TEXT_MUTED}")
        line.append(")", style=TEXT_MUTED)
    line.append(" ", style=TEXT_MUTED)
    line.append("✓", style=f"bold {ACCENT}")
    model = str(state.get("model") or "").strip()
    if model:
        line.append(" ", style=TEXT_MUTED)
        line.append(model, style=TEXT_SOFT)
    return line


def _render_live_request_block(
    *,
    mode_state: dict | None,
    settled: list[object] | None = None,
    current: object | None = None,
    turn_stats: dict[str, Any] | None = None,
):
    renderables: list[object] = list(settled or [])
    if current is not None:
        renderables.append(current)
    if renderables:
        renderables.append(Text(""))
    renderables.append(_render_locked_prompt(mode_state))
    renderables.append(Align.right(_render_turn_stats(turn_stats)))
    return Group(*renderables)


# ── Slash commands ────────────────────────────────────────────
def _open_config(console: Console) -> None:
    ensure_config_file()
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    try:
        subprocess.run([editor, str(CONFIG_PATH)], check=False)
    except FileNotFoundError:
        console.print(f"  [dim]编辑器 '{editor}' 未找到, 请设置 $EDITOR[/dim]")


def _stream_enabled(config: dict[str, Any]) -> bool:
    return config.get("stream") not in (False, "false", "0", 0)


def _run_non_streaming(
    question: str,
    *,
    config: dict,
    stats: Stats,
    console: Console,
    spinner: _Spinner,
    mode_state: dict | None = None,
    images: list[str] | None = None,
    context: list[dict[str, str]] | None = None,
    multi_turn: bool = True,
) -> str:
    t_start = time.monotonic()
    turn_stats_before = _stats_snapshot(stats)
    settled: list[object] = []

    def _finish_live(
        *,
        reply_text: str = "",
        preview: bool = False,
        note_text: str = "",
        note_style: str | None = None,
    ) -> None:
        final_block = _render_turn_transcript(
            settled=settled,
            reply_text=reply_text,
            preview=preview,
            note_text=note_text,
            note_style=note_style,
        )
        try:
            live.update(final_block, refresh=True)
            live.transient = False
            live.stop()
        except Exception:
            console.print(final_block)

    live = Live(
        _render_live_request_block(
            mode_state=mode_state,
            settled=[],
            current=None,
            turn_stats=_stats_delta(turn_stats_before, turn_stats_before, elapsed=0.0),
        ),
        console=console,
        refresh_per_second=12,
        transient=True,
    )
    live.start()

    def _on_reasoning(text: str, meta: dict[str, Any]) -> None:
        del text, meta
        try:
            live.update(
                _render_live_request_block(
                    mode_state=mode_state,
                    settled=settled,
                    current=None,
                    turn_stats=_stats_delta(turn_stats_before, _stats_snapshot(stats), elapsed=time.monotonic() - t_start),
                ),
                refresh=True,
            )
        except Exception:
            pass

    def _on_tool(name: str, args: dict) -> None:
        settled.append(_tool_line(name, args))
        try:
            live.update(
                _render_live_request_block(
                    mode_state=mode_state,
                    settled=settled,
                    current=None,
                    turn_stats=_stats_delta(turn_stats_before, _stats_snapshot(stats), elapsed=time.monotonic() - t_start),
                ),
                refresh=True,
            )
        except Exception:
            pass

    try:
        answer = run(
            question,
            config=config,
            stats=stats,
            on_tool=_on_tool,
            on_reasoning=_on_reasoning,
            images=images,
            context=context,
        )
    except KeyboardInterrupt:
        _finish_live(note_text="已中断", note_style=TEXT_MUTED)
        raise
    except Exception as e:
        _finish_live(note_text=f"错误: {e}", note_style="red")
        return ""

    text = _clean_answer(answer) if answer else "未获得有效回答。"
    del multi_turn
    _finish_live(reply_text=text)
    return text


# ── Streaming answer with spinner + Live panel ───────────────
def _run_streaming(
    question: str,
    *,
    config: dict,
    stats: Stats,
    console: Console,
    spinner: _Spinner,
    mode_state: dict | None = None,
    images: list[str] | None = None,
    context: list[dict[str, str]] | None = None,
    multi_turn: bool = True,
) -> str:
    """Run one question with streaming and keep a locked next-input footer visible."""

    chunks: list[str] = []
    settled: list[object] = []
    live: Live | None = None
    last_update = [0.0]
    tool_states: list[tuple[str, dict]] = []
    tool_state_lock = threading.RLock()
    next_tool_done_index = [0]
    thinking_meta = [None]
    t_start = time.monotonic()
    turn_stats_before = _stats_snapshot(stats)

    def _current_turn_stats(preview_text: str = "") -> dict[str, Any]:
        delta = _stats_delta(turn_stats_before, _stats_snapshot(stats), elapsed=time.monotonic() - t_start)
        if preview_text and int(delta.get("completion_tokens") or 0) <= 0:
            delta["completion_tokens"] = _estimate_preview_tokens(preview_text)
        reasoning_tokens = int((thinking_meta[0] or {}).get("reasoning_tokens") or 0)
        if reasoning_tokens > int(delta.get("reasoning_tokens") or 0):
            delta["reasoning_tokens"] = reasoning_tokens
        return delta

    def _tool_status_only_renderable():
        with tool_state_lock:
            snapshot = [(name, dict(args)) for name, args in tool_states]
        if not snapshot:
            return Text("", style=TEXT_MUTED)
        return Group(*(_tool_line(name, args) for name, args in snapshot))

    def _snapshot_tool_status() -> object | None:
        tool_renderable = _tool_status_only_renderable()
        if isinstance(tool_renderable, Text) and not tool_renderable.plain.strip():
            return None
        return tool_renderable

    def _finish_live(
        *,
        reply_text: str = "",
        preview: bool = False,
        note_text: str = "",
        note_style: str | None = None,
    ) -> None:
        with tool_state_lock:
            tool_renderable = _snapshot_tool_status()
            tool_states.clear()
            next_tool_done_index[0] = 0
        final_block = _render_turn_transcript(
            settled=settled,
            reply_text=reply_text,
            preview=preview,
            tool_renderable=tool_renderable,
            note_text=note_text,
            note_style=note_style,
        )
        try:
            if live:
                live.update(final_block, refresh=True)
                live.transient = False
                live.stop()
            else:
                console.print(final_block)
        except Exception:
            console.print(final_block)

    def _current_live_renderable() -> object:
        preview_text = _clean_answer("".join(chunks))
        current: object | None = None
        tool_renderable = _snapshot_tool_status()
        if preview_text.strip():
            current = _render_reply_body(preview_text, preview=True)
        elif tool_renderable is not None:
            current = tool_renderable
        return _render_live_request_block(
            mode_state=mode_state,
            settled=settled,
            current=current,
            turn_stats=_current_turn_stats(preview_text),
        )

    def _refresh_live() -> None:
        if live is None:
            return
        try:
            live.update(_current_live_renderable(), refresh=True)
        except Exception:
            pass

    def _find_tool_index(call_id: str) -> int:
        needle = str(call_id or "").strip()
        if not needle:
            return -1
        with tool_state_lock:
            for index, (_, row_args) in enumerate(tool_states):
                if str(row_args.get("_call_id") or "").strip() == needle:
                    return index
        return -1

    def _insert_child_rows(children: list[tuple[str, dict]] | list[dict[str, Any]], parent_call_id: str) -> None:
        with tool_state_lock:
            parent_index = _find_tool_index(parent_call_id)
            if parent_index < 0:
                return
            insert_at = parent_index + 1
            while insert_at < len(tool_states):
                row_parent = str(tool_states[insert_at][1].get("_parent_call_id") or "").strip()
                if row_parent != parent_call_id:
                    break
                insert_at += 1
            for child in children:
                if isinstance(child, dict):
                    child_name = str(child.get("name") or "").strip()
                    child_args = child.get("args") if isinstance(child.get("args"), dict) else {}
                else:
                    child_name, child_args = child
                if not child_name or not isinstance(child_args, dict):
                    continue
                child_call_id = str(child_args.get("_call_id") or "").strip()
                if child_call_id and _find_tool_index(child_call_id) >= 0:
                    continue
                row_args = dict(child_args)
                row_args["_pending"] = bool(row_args.get("_pending", True))
                tool_states.insert(insert_at, (child_name, row_args))
                insert_at += 1

    def _on_status(text: str) -> None:
        del text

    def _on_chunk(delta: str) -> None:
        nonlocal live
        with tool_state_lock:
            if tool_states:
                tool_renderable = _snapshot_tool_status()
                if tool_renderable is not None:
                    settled.append(tool_renderable)
                tool_states.clear()
                next_tool_done_index[0] = 0
        chunks.append(delta)
        now = time.monotonic()
        if now - last_update[0] > 0.15:
            _refresh_live()
            last_update[0] = now

    def _on_rewind(thinking: str, tools: list[tuple[str, dict]] | None = None) -> None:
        rewind_source = str(thinking or "")
        if not rewind_source.strip():
            rewind_source = "".join(chunks)
        clean = _clean_answer(rewind_source) if rewind_source else ""
        if clean.strip():
            settled.append(_render_reply_body(clean, preview=True))
        with tool_state_lock:
            tool_states.clear()
            for name, args in tools or []:
                row_args = dict(args)
                row_args["_pending"] = True
                tool_states.append((name, row_args))
            next_tool_done_index[0] = 0
        chunks.clear()
        last_update[0] = 0.0
        _refresh_live()

    def _on_reasoning(text: str, meta: dict[str, Any]) -> None:
        del text
        thinking_meta[0] = dict(meta or {})
        _refresh_live()

    def _on_tool(name: str, args: dict) -> None:
        if name == "__tool_children__":
            children = args.get("_children") if isinstance(args, dict) else []
            parent_call_id = str((args or {}).get("_parent_call_id") or "").strip()
            if isinstance(children, list):
                _insert_child_rows(children, parent_call_id)
                _refresh_live()
            return
        call_id = str(args.get("_call_id") or "").strip()
        if call_id:
            with tool_state_lock:
                idx = _find_tool_index(call_id)
                if idx >= 0:
                    existing_name, existing_args = tool_states[idx]
                    merged_args = dict(existing_args)
                    merged_args.update(args)
                    if "_pending" not in args:
                        merged_args["_pending"] = False
                    tool_states[idx] = (name or existing_name, merged_args)
                    _refresh_live()
                    return
        with tool_state_lock:
            if next_tool_done_index[0] < len(tool_states):
                idx = next_tool_done_index[0]
                planned_name, planned_args = tool_states[idx]
                planned_args.update(args)
                planned_args["_pending"] = False
                tool_states[idx] = (planned_name, planned_args)
                next_tool_done_index[0] += 1
                _refresh_live()
                return
        settled.append(_tool_line(name, args))
        _refresh_live()

    live = Live(
        _render_live_request_block(
            mode_state=mode_state,
            settled=[],
            current=None,
            turn_stats=_current_turn_stats(),
        ),
        console=console,
        refresh_per_second=12,
        transient=True,
    )
    live.start()

    try:
        answer = run_stream(
            question,
            config=config,
            stats=stats,
            on_chunk=_on_chunk,
            on_tool=_on_tool,
            on_reasoning=_on_reasoning,
            on_status=_on_status,
            on_rewind=_on_rewind,
            images=images,
            context=context,
        )
    except KeyboardInterrupt:
        _finish_live(
            reply_text=_clean_answer("".join(chunks)),
            preview=True,
            note_text="已中断",
            note_style=TEXT_MUTED,
        )
        raise
    except Exception as e:
        _finish_live(
            reply_text=_clean_answer("".join(chunks)),
            preview=True,
            note_text=f"错误: {e}",
            note_style="red",
        )
        return ""

    text = _clean_answer(answer) if answer else "未获得有效回答。"
    del multi_turn
    _finish_live(reply_text=text)
    return text


# ── Main ──────────────────────────────────────────────────────
def main(argv: list[str] | None = None):
    _suppress_logs()
    ensure_config_file()

    parser = argparse.ArgumentParser(description="Heuristic Yield Websearch")
    parser.add_argument("-q", "--question", help="单次提问后退出")
    args = parser.parse_args(argv)

    config = load_config()
    models = get_model_profiles(config)
    headless = config.get("headless") not in (False, "false", "0", 0)
    config_issue, config_error = _config_issue_from(config)

    console = Console(theme=_build_markdown_theme())
    session_stats = Stats()
    is_tty = sys.stdin.isatty() and sys.stdout.isatty() and not _DUMB
    session = PromptSession(erase_when_done=False) if (is_tty and _PT_AVAILABLE) else None

    # ── Single-shot mode ──
    if args.question:
        try:
            startup_tools(headless=headless, config=config)
        except Exception:
            pass
        answer = run(
            args.question,
            config=config,
            stats=session_stats,
            on_tool=lambda n, a: console.print(_tool_line(n, a)),
        )
        answer = _clean_answer(answer)
        console.print(_render_reply_body(answer))
        shutdown_tools(config)
        _cleanup_cache_dir()
        return

    # ── Interactive mode ──
    mode_state = {
        "multi_turn": True,
        "config": config,
        "models": models,
        "stage1_model_index": config.get("stage1_model_index") or 0,
        "stage2_model_index": config.get("stage2_model_index") if config.get("stage2_model_index") is not None else (1 if len(models) > 1 else 0),
        "name": config.get("name") or DEFAULT_NAME,
        "config_issue": config_issue,
        "config_error": config_error,
        "tools_ready": True,
        "tools_failed": False,
        "last_turn_stats": {},
    }
    _apply_model_state(
        mode_state,
        _mode_stage_index(mode_state, "stage1_model_index", 0),
        _mode_stage_index(mode_state, "stage2_model_index", 1 if len(models) > 1 else 0),
    )

    def _session_request_config() -> dict:
        request_cfg = dict(config)
        request_cfg["stage1_model_index"] = _mode_stage_index(mode_state, "stage1_model_index", 0)
        request_cfg["stage2_model_index"] = _mode_stage_index(mode_state, "stage2_model_index", 1 if len(mode_state.get("models") or []) > 1 else 0)
        return request_cfg

    def _active_stage1_config() -> dict:
        return build_stage_model_config(
            _session_request_config(),
            "stage1",
            stage1_model_index=_mode_stage_index(mode_state, "stage1_model_index", 0),
            stage2_model_index=_mode_stage_index(mode_state, "stage2_model_index", 1 if len(mode_state.get("models") or []) > 1 else 0),
        )

    if config_issue:
        mode_state["_runtime_prewarm"] = lambda: None
        mode_state["_runtime_label"] = lambda: ""
    else:
        mode_state["_runtime_prewarm"] = lambda: start_runtime_prewarm(_active_stage1_config())
        mode_state["_runtime_label"] = lambda: get_runtime_prewarm_label(_active_stage1_config())

    spinner = _Spinner()
    last_context: list[dict[str, str]] | None = None

    if config_issue:
        console.print(f"  [{TEXT_MUTED}]{config_issue}[/{TEXT_MUTED}]")
        if config_error:
            console.print(f"  [{TEXT_MUTED}]{config_error}[/{TEXT_MUTED}]")

    try:
        while True:
            image_paths: list[str] = []
            cleanup_paths: list[str] = []
            try:
                prompt_input = _ask(session, mode_state=mode_state)
                question = prompt_input.text
                image_paths = prompt_input.image_paths
                cleanup_paths = prompt_input.cleanup_paths
            except KeyboardInterrupt:
                console.print()
                break
            except EOFError:
                console.print()
                break

            if not question and not image_paths:
                continue
            if not image_paths and question.lower() in _QUIT:
                break
            if not image_paths and question.lower() in _PASTE_COMMANDS:
                prompt_input = _pull_clipboard_prompt_input()
                question = prompt_input.text
                image_paths = prompt_input.image_paths
                cleanup_paths = prompt_input.cleanup_paths
                if not question and not image_paths:
                    console.print(f"  [{TEXT_MUTED}]剪贴板里没有可粘贴的文本或图片[/{TEXT_MUTED}]")
                    continue
            if not image_paths and question.lower() == "/stats":
                console.print(f"  [{TEXT_MUTED}]{session_stats.summary()}[/{TEXT_MUTED}]")
                continue
            if not image_paths and question.lower() == "/config":
                _open_config(console)
                config = load_config()
                mode_state["config"] = config
                mode_state["models"] = get_model_profiles(config)
                config_issue, config_error = _config_issue_from(config)
                mode_state["config_issue"] = config_issue
                mode_state["config_error"] = config_error
                _apply_model_state(
                    mode_state,
                    _mode_stage_index(mode_state, "stage1_model_index", 0),
                    _mode_stage_index(mode_state, "stage2_model_index", 1 if len(mode_state.get("models") or []) > 1 else 0),
                )
                headless = config.get("headless") not in (False, "false", "0", 0)
                if config_issue:
                    mode_state["_runtime_prewarm"] = lambda: None
                    mode_state["_runtime_label"] = lambda: ""
                    console.print(f"  [{TEXT_MUTED}]{config_issue}[/{TEXT_MUTED}]")
                    if config_error:
                        console.print(f"  [{TEXT_MUTED}]{config_error}[/{TEXT_MUTED}]")
                else:
                    mode_state["_runtime_prewarm"] = lambda: start_runtime_prewarm(_active_stage1_config())
                    mode_state["_runtime_label"] = lambda: get_runtime_prewarm_label(_active_stage1_config())
                console.print(f"  [{TEXT_MUTED}]配置已重新加载[/{TEXT_MUTED}]")
                continue
            if not image_paths and question.lower() == "/model":
                config_issue = str(mode_state.get("config_issue") or "").strip()
                config_error = str(mode_state.get("config_error") or "").strip()
                if config_issue:
                    console.print(f"  [{TEXT_MUTED}]{config_issue}[/{TEXT_MUTED}]")
                    if config_error:
                        console.print(f"  [{TEXT_MUTED}]{config_error}[/{TEXT_MUTED}]")
                    continue
                label = str(mode_state.get("model") or "").strip() or "未配置"
                console.print(f"  [{TEXT_MUTED}]{label}[/{TEXT_MUTED}]")
                continue

            multi = mode_state["multi_turn"]
            ctx = last_context if multi else None

            try:
                answer = ""
                turn_stats_before = _stats_snapshot(session_stats)
                turn_started_at = time.monotonic()
                request_config = _session_request_config()
                if _stream_enabled(request_config):
                    answer = _run_streaming(
                        question,
                        config=request_config,
                        stats=session_stats,
                        console=console,
                        spinner=spinner,
                        mode_state=mode_state,
                        images=image_paths,
                        context=ctx,
                        multi_turn=multi,
                    )
                else:
                    answer = _run_non_streaming(
                        question,
                        config=request_config,
                        stats=session_stats,
                        console=console,
                        spinner=spinner,
                        mode_state=mode_state,
                        images=image_paths,
                        context=ctx,
                        multi_turn=multi,
                    )
                mode_state["last_turn_stats"] = _stats_delta(
                    turn_stats_before,
                    _stats_snapshot(session_stats),
                    elapsed=time.monotonic() - turn_started_at,
                )
            except KeyboardInterrupt:
                mode_state["last_turn_stats"] = _stats_delta(
                    turn_stats_before,
                    _stats_snapshot(session_stats),
                    elapsed=time.monotonic() - turn_started_at,
                )
                continue
            finally:
                _cleanup_image_paths(cleanup_paths)

            # Build context for next round
            if answer:
                base_context = last_context if multi and isinstance(last_context, list) else []
                if not isinstance(base_context, list):
                    base_context = []
                last_context = [
                    *base_context,
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ][-12:]
    finally:
        shutdown_tools(config)
        _cleanup_cache_dir()


if __name__ == "__main__":
    main()
