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
import unicodedata
from urllib.parse import unquote, urlparse
from dataclasses import dataclass

from rich import box
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
    build_model_config,
    ensure_config_file,
    get_model_profiles,
    load_config,
)
from .main import (
    Stats,
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

_QUIT = {"/exit", "/quit", "exit", "quit", "q"}
_PASTE_COMMANDS = {"/paste", "/paste-image", "/clip", "/clipboard"}
_DUMB = os.environ.get("TERM", "").lower() in {"", "dumb", "unknown"}
_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\n]+)\)")
_MD_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+(?:\s*:?-{3,}:?\s*)\|?\s*$")
_PROMPT_PREFIX = "➜  "
_IMAGE_TOKEN_RE = re.compile(r"\[Image #\d+\]")
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


# ── Panel helper ──────────────────────────────────────────────
def _make_panel(
    body,
    *,
    title: str | Text,
    subtitle: str | Text | None = None,
    padding: tuple[int, int] = (0, 1),
) -> Panel:
    return Panel(
        body,
        box=box.SQUARE,
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
    blocks = _split_markdown_blocks(text)
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
    preview.append(_normalize_markdown_tables(text) or "...", style=TEXT_SOFT)
    return preview


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
    path = Path(raw).expanduser()
    if not path.is_file():
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


def _next_image_number(pasted_images: list[_PastedImage]) -> int:
    max_index = 0
    for item in pasted_images:
        match = _IMAGE_TOKEN_RE.fullmatch(item.token)
        if not match:
            continue
        try:
            index = int(item.token[len("[Image #"):-1])
        except Exception:
            continue
        max_index = max(max_index, index)
    return max_index + 1


def _find_image_token_span(text: str, cursor: int, *, backward: bool) -> tuple[int, int] | None:
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

    for match in _IMAGE_TOKEN_RE.finditer(text):
        start, end = match.span()
        if start <= probe < end:
            return start, end
    return None


def _expanded_image_delete_span(text: str, start: int, end: int) -> tuple[int, int]:
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


def _delete_image_token(buffer, pasted_images: list[_PastedImage], *, backward: bool) -> bool:
    text = str(buffer.text or "")
    cursor = int(buffer.cursor_position or 0)
    span = _find_image_token_span(text, cursor, backward=backward)
    if span is None:
        return False
    token_start, token_end = span
    token_text = text[token_start:token_end]
    delete_start, delete_end = _expanded_image_delete_span(text, token_start, token_end)
    new_text = text[:delete_start] + text[delete_end:]
    buffer.text = new_text
    buffer.cursor_position = min(delete_start, len(new_text))

    remaining: list[_PastedImage] = []
    removed = False
    for item in pasted_images:
        if not removed and item.token == token_text:
            _discard_pasted_image(item)
            removed = True
            continue
        remaining.append(item)
    pasted_images[:] = remaining
    return True


def _paste_clipboard_into_buffer(buffer, pasted_images: list[_PastedImage]) -> bool:
    image_path = _read_clipboard_image()
    if image_path:
        _insert_image_tokens(buffer, [image_path], pasted_images, owned=True)
        return True
    clipboard_text = _read_clipboard_text()
    if clipboard_text:
        image_paths = _extract_image_paths_from_text(clipboard_text)
        if image_paths:
            _insert_image_tokens(buffer, image_paths, pasted_images)
            return True
        _insert_buffer_text(buffer, clipboard_text)
        return True
    return False


def _insert_image_tokens(
    buffer,
    image_paths: list[str],
    pasted_images: list[_PastedImage],
    *,
    owned: bool = False,
) -> None:
    tokens: list[str] = []
    for image_path in image_paths:
        token = f"[Image #{_next_image_number(pasted_images)}]"
        pasted_images.append(_PastedImage(token=token, path=image_path, owned=owned))
        tokens.append(token)
    _insert_buffer_text(buffer, " ".join(tokens), pad=True)


def _normalize_prompt_input(raw_text: str, pasted_images: list[_PastedImage]) -> _PromptInput:
    text = str(raw_text or "").strip()
    image_paths, cleanup_paths = _finalize_pasted_images(text, pasted_images)
    if image_paths:
        return _PromptInput(text=text, image_paths=image_paths, cleanup_paths=cleanup_paths)
    direct_paths = _extract_image_paths_from_text(text)
    if direct_paths:
        tokens = [f"[Image #{idx}]" for idx in range(1, len(direct_paths) + 1)]
        return _PromptInput(text=" ".join(tokens), image_paths=direct_paths, cleanup_paths=[])
    return _PromptInput(text=text, image_paths=[], cleanup_paths=[])


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
        return str(args.get("url") or "").strip()
    return str(args)[:160]


def _display_tool_name(name: str, args: dict) -> str:
    return str(args.get("_display_name") or name).strip() or str(name or "").strip()


def _tool_text_line(name: str, args: dict) -> str:
    display_name = _display_tool_name(name, args)
    formatted = _fmt_args(name, args)
    line = f"> {display_name}"
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

    line = Text()
    line.append("  > ", style=f"bold {ACCENT}")
    line.append_text(_gradient_title(display_name))
    if formatted:
        line.append(f"({formatted})", style=TEXT_MUTED)
    if ok is False:
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
    if elapsed_s not in (None, ""):
        try:
            line.append(f" {float(elapsed_s):.1f}s", style=TEXT_MUTED)
        except Exception:
            pass
    return line


# ── Clean answer ──────────────────────────────────────────────
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
    # 清理残留的 XML 工具标签
    cleaned = re.sub(r"</?(?:search|wiki|page|tool_results|result)\b[^>]*>", "", cleaned)
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
        return [("", label)]
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


def _apply_model_state(mode_state: dict, index: int) -> None:
    models = mode_state.get("models") or []
    if not isinstance(models, list) or not models:
        mode_state["model_index"] = 0
        mode_state["model_count"] = 0
        mode_state["model"] = ""
        mode_state["model_id"] = ""
        mode_state["model_provider"] = ""
        return

    idx = int(index or 0) % len(models)
    current = models[idx] if isinstance(models[idx], dict) else {}
    provider = _model_provider(current)
    label = _display_model_label(current)
    mode_state["model_index"] = idx
    mode_state["model_count"] = len(models)
    mode_state["model"] = label
    mode_state["model_id"] = str(current.get("model") or "").strip()
    mode_state["model_provider"] = provider


def _cycle_model(mode_state: dict, delta: int) -> None:
    models = mode_state.get("models") or []
    if not isinstance(models, list) or len(models) <= 1:
        return
    current = int(mode_state.get("model_index") or 0)
    _apply_model_state(mode_state, current + delta)
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

    if session is not None and _PT_AVAILABLE and PTStyle is not None and mode_state is not None:
        kb = KeyBindings()

        @kb.add("left", filter=Condition(lambda: not session.app.current_buffer.text))
        def _prev_model(event):
            _cycle_model(mode_state, -1)
            event.app.invalidate()

        @kb.add("right", filter=Condition(lambda: not session.app.current_buffer.text))
        def _next_model(event):
            _cycle_model(mode_state, 1)
            event.app.invalidate()

        @kb.add("up", filter=Condition(lambda: not session.app.current_buffer.text))
        @kb.add("down", filter=Condition(lambda: not session.app.current_buffer.text))
        def _toggle_session(event):
            mode_state["multi_turn"] = not mode_state["multi_turn"]
            event.app.invalidate()

        @kb.add("c-v", eager=True)
        def _paste(event):
            _paste_clipboard_into_buffer(event.current_buffer, pasted_images)

        @kb.add("escape", "v", eager=True)
        def _paste_alt(event):
            _paste_clipboard_into_buffer(event.current_buffer, pasted_images)

        @kb.add("backspace", eager=True)
        def _delete_prev(event):
            if _delete_image_token(event.current_buffer, pasted_images, backward=True):
                return
            event.current_buffer.delete_before_cursor(count=1)

        @kb.add("delete", eager=True)
        def _delete_next(event):
            if _delete_image_token(event.current_buffer, pasted_images, backward=False):
                return
            event.current_buffer.delete(count=1)

        @kb.add(Keys.BracketedPaste, eager=True)
        def _paste_bracketed(event):
            pasted_text = str(event.data or "")
            image_paths = _extract_image_paths_from_text(pasted_text)
            if image_paths:
                _insert_image_tokens(event.current_buffer, image_paths, pasted_images, owned=False)
                return
            _insert_buffer_text(event.current_buffer, pasted_text)

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
            }
        )
        try:
            raw_text = session.prompt(
                _get_prompt,
                rprompt=_get_rprompt,
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
        return _normalize_prompt_input(str(raw_text or ""), pasted_images)

    if session is not None and _PT_AVAILABLE and PTStyle is not None:
        input_style = PTStyle.from_dict(
            {
                "": INPUT_TEXT,
                "prompt": f"bold {INPUT_ACCENT}",
                "mode": "bold",
                "placeholder": f"fg:{INPUT_PLACEHOLDER}",
                "placeholder_hint": f"fg:{INPUT_PLACEHOLDER}",
            }
        )
        raw_text = session.prompt(
            _prompt_parts(mode_state),
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
        return _normalize_prompt_input(str(raw_text or ""), pasted_images)
    runtime_prewarm = (mode_state or {}).get("_runtime_prewarm")
    if callable(runtime_prewarm):
        try:
            runtime_prewarm()
        except Exception:
            pass
    return _PromptInput(text=input(_PROMPT_PREFIX).strip(), image_paths=[], cleanup_paths=[])


# ── Spinner (lightweight, for runtime status) ─────────────────
class _Spinner:
    """Simple inline spinner using \\r overwrites."""

    def __init__(self) -> None:
        self._text = ""
        self._active = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self, text: str = "") -> None:
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


# ── Slash commands ────────────────────────────────────────────
def _open_config(console: Console) -> None:
    ensure_config_file()
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    try:
        subprocess.run([editor, str(CONFIG_PATH)], check=False)
    except FileNotFoundError:
        console.print(f"  [dim]编辑器 '{editor}' 未找到, 请设置 $EDITOR[/dim]")


# ── Streaming answer with spinner + Live panel ───────────────
def _run_streaming(
    question: str,
    *,
    config: dict,
    stats: Stats,
    console: Console,
    spinner: _Spinner,
    images: list[str] | None = None,
    context: str | None = None,
    multi_turn: bool = True,
) -> str:
    """Run one question with streaming: spinner -> tool lines -> Live panel. Returns final answer text."""

    chunks: list[str] = []
    live: Live | None = None
    last_update = [0.0]
    title = _gradient_title("HY-WebSearch")
    t_start = time.monotonic()

    def _on_status(text: str) -> None:
        spinner.start(text)

    def _on_chunk(delta: str) -> None:
        nonlocal live
        if live is None:
            spinner.stop()
            console.print()
            live = Live(
                _make_panel(
                    _render_stream_preview("..."),
                    title=title,
                    subtitle=_stats_subtitle(stats, multi_turn=multi_turn, elapsed=time.monotonic() - t_start),
                ),
                console=console,
                refresh_per_second=8,
                transient=True,
            )
            live.start()
        chunks.append(delta)
        now = time.monotonic()
        if now - last_update[0] > 0.15:
            live.update(_make_panel(
                _render_stream_preview(_clean_answer("".join(chunks))),
                title=title,
                subtitle=_stats_subtitle(stats, multi_turn=multi_turn, elapsed=now - t_start),
            ))
            last_update[0] = now

    def _on_rewind(thinking: str, tools: list[tuple[str, dict]] | None = None) -> None:
        nonlocal live
        rewind_source = str(thinking or "")
        if not rewind_source.strip():
            rewind_source = "".join(chunks)
        if live:
            live.transient = True
            live.stop()
            live = None
        # 将模型思考内容 (去掉 XML 标签) 作为永久输出保留
        clean = _clean_answer(rewind_source) if rewind_source else ""
        planned_block = _planned_tool_block(tools)
        if planned_block:
            clean = f"{clean}\n{planned_block}".strip() if clean else planned_block
        if clean.strip():
            console.print(
                _make_panel(
                    _render_stream_preview(clean),
                    title=title,
                    subtitle=_stats_subtitle(stats, multi_turn=multi_turn, elapsed=time.monotonic() - t_start),
                )
            )
        chunks.clear()
        last_update[0] = 0.0

    def _on_tool(name: str, args: dict) -> None:
        spinner.stop()
        console.print(_tool_line(name, args))

    try:
        answer = run_stream(
            question,
            config=config,
            stats=stats,
            on_chunk=_on_chunk,
            on_tool=_on_tool,
            on_status=_on_status,
            on_rewind=_on_rewind,
            images=images,
            context=context,
        )
    except KeyboardInterrupt:
        spinner.stop()
        if live:
            live.transient = True
            live.stop()
        console.print(f"\n  [{TEXT_MUTED}]已中断[/{TEXT_MUTED}]")
        raise
    except Exception as e:
        spinner.stop()
        if live:
            live.transient = True
            live.stop()
        console.print(f"  [red]错误: {e}[/red]")
        return ""

    spinner.stop()

    elapsed = time.monotonic() - t_start

    # Final static render with complete stats
    text = _clean_answer(answer) if answer else "未获得有效回答。"
    final_panel = _make_panel(
        _render_markdown(text),
        title=title,
        subtitle=_stats_subtitle(stats, multi_turn=multi_turn, elapsed=elapsed),
    )
    if live:
        live.update(final_panel, refresh=True)
        live.transient = False
        live.stop()
    else:
        console.print(final_panel)
    console.print()
    return text


# ── Main ──────────────────────────────────────────────────────
def main(argv: list[str] | None = None):
    _suppress_logs()

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
        console.print(
            _make_panel(
                _render_markdown(answer),
                title=_gradient_title("HY-WebSearch"),
                subtitle=_stats_subtitle(session_stats),
            )
        )
        shutdown_tools(config)
        _cleanup_cache_dir()
        return

    # ── Interactive mode ──
    mode_state = {
        "multi_turn": True,
        "models": models,
        "model_index": config.get("active_model_index") or 0,
        "name": config.get("name") or DEFAULT_NAME,
        "config_issue": config_issue,
        "config_error": config_error,
        "tools_ready": True,
        "tools_failed": False,
    }
    _apply_model_state(mode_state, int(mode_state["model_index"]))

    def _active_request_config() -> dict:
        return build_model_config(config, model_index=int(mode_state.get("model_index") or 0))

    if config_issue:
        mode_state["_runtime_prewarm"] = lambda: None
        mode_state["_runtime_label"] = lambda: ""
    else:
        mode_state["_runtime_prewarm"] = lambda: start_runtime_prewarm(_active_request_config())
        mode_state["_runtime_label"] = lambda: get_runtime_prewarm_label(_active_request_config())

    spinner = _Spinner()
    last_context: str | None = None
    last_was_multi: bool = True

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
                mode_state["models"] = get_model_profiles(config)
                config_issue, config_error = _config_issue_from(config)
                mode_state["config_issue"] = config_issue
                mode_state["config_error"] = config_error
                _apply_model_state(mode_state, int(config.get("active_model_index") or 0))
                headless = config.get("headless") not in (False, "false", "0", 0)
                if config_issue:
                    mode_state["_runtime_prewarm"] = lambda: None
                    mode_state["_runtime_label"] = lambda: ""
                    console.print(f"  [{TEXT_MUTED}]{config_issue}[/{TEXT_MUTED}]")
                    if config_error:
                        console.print(f"  [{TEXT_MUTED}]{config_error}[/{TEXT_MUTED}]")
                else:
                    mode_state["_runtime_prewarm"] = lambda: start_runtime_prewarm(_active_request_config())
                    mode_state["_runtime_label"] = lambda: get_runtime_prewarm_label(_active_request_config())
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
                model_id = str(mode_state.get("model_id") or "").strip()
                normalized_label = label.replace(" / ", "/").strip().lower()
                normalized_model_id = model_id.strip().lower()
                if model_id and normalized_model_id and normalized_label not in {normalized_model_id, _compact_model_id(model_id, str(mode_state.get("model_provider") or "")).lower(),}:
                    console.print(f"  [{TEXT_MUTED}]{label} -> {model_id}[/{TEXT_MUTED}]")
                else:
                    console.print(f"  [{TEXT_MUTED}]{label}[/{TEXT_MUTED}]")
                continue

            multi = mode_state["multi_turn"]
            # If previous round was single-turn, forget context
            if not last_was_multi:
                last_context = None

            ctx = last_context if multi else None

            try:
                request_config = build_model_config(config, model_index=int(mode_state.get("model_index") or 0))
                answer = _run_streaming(
                    question,
                    config=request_config,
                    stats=session_stats,
                    console=console,
                    spinner=spinner,
                    images=image_paths,
                    context=ctx,
                    multi_turn=multi,
                )
            except KeyboardInterrupt:
                console.print()
                break
            finally:
                _cleanup_image_paths(cleanup_paths)

            # Build context for next round
            if answer:
                last_context = f"User: {question}\nAnswer: {answer}"
            last_was_multi = multi
    finally:
        shutdown_tools(config)
        _cleanup_cache_dir()


if __name__ == "__main__":
    main()
