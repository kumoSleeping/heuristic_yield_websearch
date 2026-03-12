"""
hyw/cli.py - Rich-styled CLI for hyw (Heuristic Yield Web_search)

用法:
    python -m hyw              # 交互模式
    python -m hyw -q "问题"    # 单次问答
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from .main import Stats, load_config, run, run_stream, startup_tools, shutdown_tools, CONFIG_PATH, DEFAULT_NAME

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style as PTStyle

    _PT_AVAILABLE = True
except Exception:
    PromptSession = None  # type: ignore[assignment,misc]
    Condition = None  # type: ignore[assignment,misc]
    KeyBindings = None  # type: ignore[assignment,misc]
    PTStyle = None  # type: ignore[assignment,misc]

# ── Color constants ───────────────────────────────────────────
TEXT_MUTED = "rgb(194,145,92)"
TEXT_SOFT = "rgb(232,189,138)"
BORDER = "rgb(255,108,60)"
ACCENT = "rgb(255,108,60)"
INPUT_ACCENT = "#D4AF37"
INPUT_TEXT = "#E8BD8A"
INPUT_HINT = "#D4AF37"

_QUIT = {"/exit", "/quit", "exit", "quit", "q"}
_DUMB = os.environ.get("TERM", "").lower() in {"", "dumb", "unknown"}
_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\n]+)\)")
_PROMPT_PREFIX = "➜  "

INTRO_PROMPT_ANSI = "\x1b[38;2;255;108;60m"
INTRO_INPUT_ANSI = "\x1b[38;2;232;189;138m"
INTRO_HINT_ANSI = "\x1b[38;2;160;132;98m"

_IO_LOCK = threading.Lock()
_INPUT_RENDERED = threading.Event()
_STATUS_MAX_CELLS = 24


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
            "markdown.h2": f"bold {ACCENT}",
            "markdown.h3": f"bold {ACCENT}",
            "markdown.h4": f"bold {ACCENT}",
            "markdown.h5": f"bold {ACCENT}",
            "markdown.h6": f"bold {ACCENT}",
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
            "markdown.code": TEXT_SOFT,
            "markdown.code_block": TEXT_SOFT,
            "markdown.hr": TEXT_MUTED,
            "markdown.em": TEXT_SOFT,
            "markdown.strong": f"bold {TEXT_SOFT}",
        }
    )


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


# ── Tool call formatting ─────────────────────────────────────
def _fmt_args(name: str, args: dict) -> str:
    if name in ("web_search", "web_search_wiki"):
        q = str(args.get("query") or "").strip()
        df = str(args.get("df") or "").strip()
        if q and df:
            return f"{q} | [{df}]"
        return q
    return str(args)[:160]


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
    cleaned = re.sub(r"</?(?:search|wiki|tool_results|result)\b[^>]*>", "", cleaned)
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


# ── prompt_toolkit input ──────────────────────────────────────
def _ask(session, *, mode_state: dict | None = None) -> str:
    """Read user input. mode_state allows left/right arrow to toggle multi_turn."""
    if session is not None and _PT_AVAILABLE and PTStyle is not None and mode_state is not None:
        kb = KeyBindings()

        @kb.add("left", filter=Condition(lambda: not session.app.current_buffer.text))
        @kb.add("right", filter=Condition(lambda: not session.app.current_buffer.text))
        def _toggle(event):
            mode_state["multi_turn"] = not mode_state["multi_turn"]
            event.app.invalidate()

        def _get_rprompt():
            return []

        # Spinner invalidation thread — keeps prompt animated while loading
        _spinner_stop = threading.Event()

        def _spinner_loop():
            while not _spinner_stop.wait(0.09):
                if mode_state.get("tools_ready"):
                    # One final invalidate to show ✓, then stop
                    try:
                        session.app.invalidate()
                    except Exception:
                        pass
                    break
                try:
                    session.app.invalidate()
                except Exception:
                    pass

        if not mode_state.get("tools_ready"):
            threading.Thread(target=_spinner_loop, daemon=True).start()

        def _get_prompt():
            arrow = _PROMPT_PREFIX
            model = mode_state.get("model", "")
            multi = mode_state.get("multi_turn", True)
            ready = mode_state.get("tools_ready")
            status_color = f"bold fg:#{255:02x}{108:02x}{60:02x}"
            if ready:
                status = "✓"
            else:
                status = _FRAMES[int(time.time() * 12) % len(_FRAMES)]
            mode_label = "Multi Turn" if multi else "New Session"
            parts: list[tuple[str, str]] = [("class:prompt", arrow)]
            parts.extend(_gradient_name(mode_state.get("name", "")))
            parts.append(("class:brand", f"({model}) "))
            parts.append(("", "("))
            parts.extend(_gradient_label(mode_label))
            parts.append(("", ") "))
            parts.append((status_color, f"{status} "))
            return parts

        input_style = PTStyle.from_dict(
            {
                "": INPUT_TEXT,
                "prompt": f"bold {INPUT_ACCENT}",
                "brand": "#c2915c",
                "status": INPUT_HINT,
                "placeholder": f"dim {INPUT_HINT}",
            }
        )
        try:
            text = session.prompt(
                _get_prompt,
                rprompt=_get_rprompt,
                multiline=False,
                style=input_style,
                placeholder=[("class:placeholder", "")],
                default="",
                enable_open_in_editor=False,
                key_bindings=kb,
            )
        finally:
            _spinner_stop.set()
        return str(text or "").strip()

    if session is not None and _PT_AVAILABLE and PTStyle is not None:
        model = (mode_state or {}).get("model", "")
        status_color = f"bold fg:#{255:02x}{108:02x}{60:02x}"
        prompt_parts: list[tuple[str, str]] = [("class:prompt", _PROMPT_PREFIX)]
        prompt_parts.extend(_gradient_name(mode_state.get("name", "")))
        prompt_parts.append(("class:brand", f"({model}) "))
        prompt_parts.append((status_color, "✓ "))
        input_style = PTStyle.from_dict(
            {
                "": INPUT_TEXT,
                "prompt": f"bold {INPUT_ACCENT}",
                "brand": "#c2915c",
                "placeholder": f"dim {INPUT_HINT}",
            }
        )
        text = session.prompt(
            prompt_parts,
            multiline=False,
            style=input_style,
            placeholder=[("class:placeholder", "")],
            default="",
            enable_open_in_editor=False,
        )
        return str(text or "").strip()
    return input(_PROMPT_PREFIX).strip()


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
    mode_label = "Multi Turn" if multi_turn else "New Session"
    out.append(f"{mode_label}  ", style=TEXT_MUTED)
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
    out.append("calls ", style=f"bold {ACCENT}")
    out.append(str(stats.calls), style=TEXT_MUTED)
    return out


# ── Slash commands ────────────────────────────────────────────
def _open_config(console: Console) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text("# hyw config\nlanguage: zh-CN\n", encoding="utf-8")
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
    context: str | None = None,
    multi_turn: bool = True,
) -> str:
    """Run one question with streaming: spinner -> tool lines -> Live panel. Returns final answer text."""

    chunks: list[str] = []
    live: Live | None = None
    last_update = [0.0]
    title = _gradient_title("HY-Websearch")
    t_start = time.monotonic()
    is_final_round = [False]  # 只有最终回答轮才用 transient Live

    def _on_status(text: str) -> None:
        spinner.start(text)

    def _on_chunk(delta: str) -> None:
        nonlocal live
        if live is None:
            spinner.stop()
            console.print()
            live = Live(
                _make_panel(Text("..."), title=title),
                console=console,
                refresh_per_second=8,
                transient=True,
            )
            live.start()
        chunks.append(delta)
        now = time.monotonic()
        if now - last_update[0] > 0.15:
            live.update(_make_panel(
                Markdown(_clean_answer("".join(chunks))),
                title=title,
            ))
            last_update[0] = now

    def _on_rewind(thinking: str) -> None:
        nonlocal live
        if live:
            live.stop()
            live = None
        # 将模型思考内容 (去掉 XML 标签) 作为永久输出保留
        clean = _clean_answer(thinking) if thinking else ""
        if clean.strip():
            console.print(
                _make_panel(
                    Markdown(clean),
                    title=title,
                )
            )
        chunks.clear()
        last_update[0] = 0.0

    def _on_tool(name: str, args: dict) -> None:
        spinner.stop()
        formatted = _fmt_args(name, args)
        line = Text()
        line.append("  > ", style=f"bold {ACCENT}")
        line.append(name, style=TEXT_MUTED)
        if formatted:
            line.append(f"({formatted})", style=TEXT_MUTED)
        console.print(line)

    try:
        answer = run_stream(
            question,
            config=config,
            stats=stats,
            on_chunk=_on_chunk,
            on_tool=_on_tool,
            on_status=_on_status,
            on_rewind=_on_rewind,
            context=context,
        )
    except KeyboardInterrupt:
        spinner.stop()
        if live:
            live.stop()
        console.print(f"\n  [{TEXT_MUTED}]已中断[/{TEXT_MUTED}]")
        return ""
    except Exception as e:
        spinner.stop()
        if live:
            live.stop()
        console.print(f"  [red]错误: {e}[/red]")
        return ""

    spinner.stop()

    if live:
        live.stop()

    elapsed = time.monotonic() - t_start

    # Final static render with complete stats
    text = _clean_answer(answer) if answer else "未获得有效回答。"
    console.print(
        _make_panel(
            Markdown(text),
            title=title,
            subtitle=_stats_subtitle(stats, multi_turn=multi_turn, elapsed=elapsed),
        )
    )
    console.print()
    return text


# ── Main ──────────────────────────────────────────────────────
def main(argv: list[str] | None = None):
    _suppress_logs()

    parser = argparse.ArgumentParser(description="Heuristic Yield Websearch")
    parser.add_argument("-q", "--question", help="单次提问后退出")
    args = parser.parse_args(argv)

    config = load_config()
    model = config.get("model") or "openai/gpt-4o-mini"
    headless = config.get("headless") not in (False, "false", "0", 0)

    console = Console(theme=_build_markdown_theme())
    session_stats = Stats()
    is_tty = sys.stdin.isatty() and sys.stdout.isatty() and not _DUMB
    session = PromptSession(erase_when_done=False) if (is_tty and _PT_AVAILABLE) else None

    # ── Single-shot mode ──
    if args.question:
        try:
            startup_tools(headless=headless)
        except Exception:
            pass
        answer = run(args.question, config=config, stats=session_stats,
                     on_tool=lambda n, a: console.print(Text.assemble(
                         ("  > ", f"bold {ACCENT}"), (f"{n}({_fmt_args(n, a)})", TEXT_MUTED))))
        answer = _clean_answer(answer)
        console.print(
            _make_panel(
                Markdown(answer),
                title=_gradient_title("HY-Websearch"),
                subtitle=_stats_subtitle(session_stats),
            )
        )
        shutdown_tools()
        return

    # ── Interactive mode ──
    mode_state = {"multi_turn": True, "model": model, "tools_ready": False}

    # Background warmup — 完成后 prompt 自动从 ✗ 变 ✓
    def _warmup():
        try:
            startup_tools(headless=headless)
            mode_state["tools_ready"] = True
        except Exception:
            pass

    threading.Thread(target=_warmup, name="hyw-warmup", daemon=True).start()

    spinner = _Spinner()
    last_context: str | None = None
    last_was_multi: bool = True

    try:
        while True:
            try:
                question = _ask(session, mode_state=mode_state)
            except KeyboardInterrupt:
                console.print()
                continue
            except EOFError:
                console.print()
                break

            if not question:
                continue
            if question.lower() in _QUIT:
                break
            if question.lower() == "/stats":
                console.print(f"  [{TEXT_MUTED}]{session_stats.summary()}[/{TEXT_MUTED}]")
                continue
            if question.lower() == "/config":
                _open_config(console)
                continue

            multi = mode_state["multi_turn"]
            # If previous round was single-turn, forget context
            if not last_was_multi:
                last_context = None

            ctx = last_context if multi else None

            answer = _run_streaming(
                question,
                config=config,
                stats=session_stats,
                console=console,
                spinner=spinner,
                context=ctx,
                multi_turn=multi,
            )

            # Build context for next round
            if answer:
                last_context = f"User: {question}\nAnswer: {answer}"
            last_was_multi = multi
    finally:
        shutdown_tools()


if __name__ == "__main__":
    main()
