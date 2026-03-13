from __future__ import annotations

import asyncio
import base64
import contextlib
import ctypes.util
import html
import io
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict
from importlib.util import find_spec

RenderResult = Dict[str, Any]
_BLOCK_MATH_RE = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", flags=re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)(?<!\\)\$(.+?)(?<!\\)\$(?!\$)", flags=re.DOTALL)
_RUNTIME_LOCK = threading.RLock()
_RUNTIME_PREPARED = False
_PREWARM_DONE = False
_DLL_DIR_HANDLES: dict[str, Any] = {}
_REQUIRED_LIBRARIES = {
    "darwin": ("gobject-2.0", "pango-1.0", "pangocairo-1.0", "gdk_pixbuf-2.0"),
    "linux": ("gobject-2.0", "pango-1.0", "pangocairo-1.0", "gdk_pixbuf-2.0"),
    "win32": ("libgobject-2.0-0", "libpango-1.0-0", "libpangocairo-1.0-0", "libgdk_pixbuf-2.0-0"),
}


def _missing_dependency(name: str, package: str) -> RuntimeError:
    return RuntimeError(
        f"{name} is required for websearch markdown rendering. "
        f"Install it with `pip install {package}`."
    )


def _split_env_paths(name: str) -> list[str]:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        return []
    return [part for part in value.split(os.pathsep) if part]


def _dedupe_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        path = str(raw or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _existing_paths(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths if path.is_dir()]


def _macos_library_candidates() -> list[str]:
    return _existing_paths([Path("/opt/homebrew/lib"), Path("/usr/local/lib")])


def _windows_library_candidates() -> list[str]:
    candidates: list[Path] = []
    env_dir = str(os.environ.get("WEASYPRINT_DLL_DIRECTORIES") or "").strip()
    for raw in env_dir.split(os.pathsep):
        path = Path(str(raw or "").strip())
        if raw:
            candidates.append(path)

    roots = []
    msys_root = str(os.environ.get("MSYS2_ROOT") or "").strip()
    if msys_root:
        roots.append(Path(msys_root))
    roots.extend([Path("C:/msys64"), Path("C:/tools/msys64")])

    for root in roots:
        for suffix in ("mingw64/bin", "ucrt64/bin", "clang64/bin"):
            candidates.append(root / suffix)

    return _existing_paths(candidates)


def _platform_library_candidates() -> list[str]:
    if sys.platform == "darwin":
        return _macos_library_candidates()
    if sys.platform == "win32":
        return _windows_library_candidates()
    return []


def _merge_env_path(name: str, paths: list[str]) -> list[str]:
    merged = _dedupe_paths(paths + _split_env_paths(name))
    if merged:
        os.environ[name] = os.pathsep.join(merged)
    return merged


def _configure_macos_weasyprint_runtime() -> list[str]:
    return _merge_env_path("DYLD_FALLBACK_LIBRARY_PATH", _platform_library_candidates())


def _configure_windows_weasyprint_runtime() -> list[str]:
    global _DLL_DIR_HANDLES
    merged = _merge_env_path("WEASYPRINT_DLL_DIRECTORIES", _platform_library_candidates())
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if callable(add_dll_directory):
        for path in merged:
            if path in _DLL_DIR_HANDLES:
                continue
            with contextlib.suppress(Exception):
                _DLL_DIR_HANDLES[path] = add_dll_directory(path)
    return merged


def _configure_weasyprint_runtime() -> list[str]:
    if sys.platform == "darwin":
        return _configure_macos_weasyprint_runtime()
    if sys.platform == "win32":
        return _configure_windows_weasyprint_runtime()
    return []


def _find_missing_system_libraries() -> list[str]:
    names = _REQUIRED_LIBRARIES.get(sys.platform, _REQUIRED_LIBRARIES.get("linux", ()))
    missing: list[str] = []
    for name in names:
        if ctypes.util.find_library(name):
            continue
        missing.append(name)
    return missing


def _weasyprint_install_hint() -> str:
    if sys.platform == "darwin":
        return (
            "macOS: install system libraries with `brew install weasyprint`. "
            "If you installed them with Homebrew in a custom prefix, make sure "
            "`DYLD_FALLBACK_LIBRARY_PATH` includes that `lib` directory."
        )
    if sys.platform == "win32":
        return (
            "Windows: install MSYS2, then run "
            "`pacman -S mingw-w64-x86_64-pango` in the MSYS2 shell. "
            "If DLLs are not in the default location, set "
            "`WEASYPRINT_DLL_DIRECTORIES=C:\\msys64\\mingw64\\bin`."
        )
    return (
        "Linux: install your distro's Pango runtime first. "
        "For Ubuntu/Debian venv wheels, WeasyPrint documents "
        "`apt install python3-pip libpango-1.0-0 libharfbuzz0b "
        "libpangoft2-1.0-0 libharfbuzz-subset0`."
    )


def _format_weasyprint_runtime_error(exc: Exception, searched_paths: list[str]) -> RuntimeError:
    missing = _find_missing_system_libraries()
    detail = str(exc or "").strip()
    if len(detail) > 280:
        detail = detail[:280] + "..."

    parts = [
        "WeasyPrint Python package is installed, but its native runtime libraries are missing or unreachable.",
        _weasyprint_install_hint(),
    ]
    if searched_paths:
        parts.append("Auto-detected library paths: " + ", ".join(searched_paths))
    if missing:
        parts.append("Missing libraries detected: " + ", ".join(missing))
    if detail:
        parts.append("Original error: " + detail)
    return RuntimeError(" ".join(parts))


def ensure_non_browser_render_ready(*, prewarm: bool = False) -> dict[str, Any]:
    global _RUNTIME_PREPARED, _PREWARM_DONE

    with _RUNTIME_LOCK:
        searched_paths = _configure_weasyprint_runtime()

        if not _RUNTIME_PREPARED:
            if find_spec("weasyprint") is None:
                raise _missing_dependency("WeasyPrint", "weasyprint")
            try:
                _load_weasyprint_html()
                _load_fitz_module()
                _load_pillow_modules()
                _load_markdown_module()
                _load_pygments_html_formatter()
                _load_math_to_image()
            except RuntimeError:
                raise
            except Exception as exc:
                raise _format_weasyprint_runtime_error(exc, searched_paths) from exc
            _RUNTIME_PREPARED = True

        if prewarm and not _PREWARM_DONE:
            try:
                _render_markdown_non_browser_sync(
                    "# HYW Warmup\n\n预热渲染器。",
                    "Warmup",
                    "#ef4444",
                )
            except RuntimeError:
                raise
            except Exception as exc:
                raise _format_weasyprint_runtime_error(exc, searched_paths) from exc
            _PREWARM_DONE = True

        return {
            "platform": sys.platform,
            "paths": list(searched_paths),
            "prewarmed": bool(_PREWARM_DONE),
        }


def _load_markdown_module() -> Any:
    try:
        import markdown as markdown_module

        return markdown_module
    except Exception as exc:
        raise _missing_dependency("markdown", "markdown") from exc


def _load_weasyprint_html() -> Any:
    try:
        from weasyprint import HTML

        return HTML
    except Exception as exc:
        if find_spec("weasyprint") is None:
            raise _missing_dependency("WeasyPrint", "weasyprint") from exc
        searched_paths = []
        if sys.platform == "darwin":
            searched_paths = _split_env_paths("DYLD_FALLBACK_LIBRARY_PATH")
        elif sys.platform == "win32":
            searched_paths = _split_env_paths("WEASYPRINT_DLL_DIRECTORIES")
        raise _format_weasyprint_runtime_error(exc, searched_paths) from exc


def _load_fitz_module() -> Any:
    try:
        import fitz

        return fitz
    except Exception as exc:
        raise _missing_dependency("PyMuPDF", "PyMuPDF") from exc


def _load_pillow_modules() -> Any:
    try:
        from PIL import Image, ImageChops

        return Image, ImageChops
    except Exception as exc:
        raise _missing_dependency("Pillow", "Pillow") from exc


def _load_pygments_html_formatter() -> Any:
    try:
        from pygments.formatters import HtmlFormatter

        return HtmlFormatter
    except Exception as exc:
        raise _missing_dependency("Pygments", "Pygments") from exc


def _load_math_to_image() -> Any:
    try:
        from matplotlib.mathtext import math_to_image

        return math_to_image
    except Exception as exc:
        raise _missing_dependency("matplotlib", "matplotlib") from exc


def _normalize_text(markdown_text: str) -> str:
    return str(markdown_text or "").replace("\r\n", "\n").strip()


def _render_math(markdown_text: str) -> str:
    if "$" not in markdown_text:
        return markdown_text

    math_to_image = _load_math_to_image()

    def _replace_block(match: re.Match[str]) -> str:
        latex = str(match.group(1) or "").strip()
        if not latex:
            return match.group(0)
        buffer = io.BytesIO()
        math_to_image(f"${latex}$", buffer, dpi=220, format="png", color="#1f2937")
        data = base64.b64encode(buffer.getvalue()).decode()
        return (
            "\n<div class=\"math-display\">"
            f"<img src=\"data:image/png;base64,{data}\" alt=\"{html.escape(latex)}\" />"
            "</div>\n"
        )

    def _replace_inline(match: re.Match[str]) -> str:
        latex = str(match.group(1) or "").strip()
        if not latex:
            return match.group(0)
        buffer = io.BytesIO()
        math_to_image(f"${latex}$", buffer, dpi=180, format="png", color="#1f2937")
        data = base64.b64encode(buffer.getvalue()).decode()
        return (
            "<span class=\"math-inline\">"
            f"<img class=\"math-inline-img\" src=\"data:image/png;base64,{data}\" alt=\"{html.escape(latex)}\" />"
            "</span>"
        )

    text = _BLOCK_MATH_RE.sub(_replace_block, markdown_text)
    return _INLINE_MATH_RE.sub(_replace_inline, text)


def _markdown_to_html(markdown_text: str) -> str:
    markdown_module = _load_markdown_module()
    _load_pygments_html_formatter()
    return markdown_module.markdown(
        _render_math(_normalize_text(markdown_text)),
        extensions=["extra", "fenced_code", "codehilite", "tables", "sane_lists", "nl2br"],
        extension_configs={
            "codehilite": {
                "guess_lang": False,
                "noclasses": False,
                "linenums": False,
                "css_class": "codehilite",
            }
        },
    )


def _build_html_document(markdown_text: str, title: str, theme_color: str) -> str:
    HtmlFormatter = _load_pygments_html_formatter()
    body = _markdown_to_html(markdown_text)
    pygments_css = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    safe_color = html.escape(str(theme_color or "#ef4444").strip() or "#ef4444")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
	    :root {{
	      --theme: {safe_color};
	      --bg: #f6f4ef;
	      --paper: #fffdf8;
	      --text: #3f342a;
	      --muted: #8a7a6a;
	      --code: #f4efe7;
	    }}
    @page {{
      size: 960px 1800px;
      margin: 0;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
    }}
    .page {{
      width: 880px;
      margin: 40px auto;
      background: var(--paper);
      padding: 44px 52px 48px 52px;
      box-sizing: border-box;
      border: 1px solid rgba(0, 0, 0, 0.05);
    }}
	    .content {{
	      font-size: 20px;
	      line-height: 1.82;
	      font-weight: 500;
	      letter-spacing: 0.01em;
	      word-break: break-word;
	    }}
    .content h1, .content h2, .content h3 {{
      line-height: 1.3;
      margin: 1.2em 0 0.45em 0;
    }}
    .content p {{
      margin: 0.4em 0;
    }}
    .content pre {{
      white-space: pre-wrap;
      margin: 0;
      padding: 0;
      background: transparent;
      border-radius: 0;
      font-family: "SFMono-Regular", "Menlo", monospace;
      font-size: 16px;
      line-height: 1.6;
    }}
    .content .codehilite {{
      margin: 1em 0;
      width: 100%;
      max-width: 100%;
      padding: 16px 18px;
      box-sizing: border-box;
      overflow: hidden;
      border-radius: 12px;
      background: var(--code);
    }}
    .content .codehilite pre,
    .content .codehilite code {{
      max-width: 100%;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      box-sizing: border-box;
    }}
    .content code {{
      background: var(--code);
      border-radius: 6px;
      padding: 2px 6px;
      font-family: "SFMono-Regular", "Menlo", monospace;
      font-size: 0.92em;
    }}
    .content a {{
      color: var(--theme);
      text-decoration: none;
    }}
    .content blockquote {{
      margin: 1em 0;
      padding: 0.3em 0 0.3em 1em;
      border-left: 4px solid rgba(0, 0, 0, 0.12);
      color: #475569;
    }}
    .content .math-display {{
      margin: 1em 0;
      text-align: center;
    }}
    .content .math-inline {{
      display: inline-block;
      vertical-align: middle;
      padding: 0 0.12em;
    }}
    .content .math-display img {{
      max-width: 100%;
      height: auto;
    }}
    .content .math-inline img {{
      height: 1.3em;
      width: auto;
      vertical-align: middle;
    }}
    .meta {{
      margin-top: 28px;
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }}
    {pygments_css}
  </style>
</head>
<body>
  <div class="page">
    <div class="content">{body}</div>
    <div class="meta">HY-WebSearch</div>
  </div>
</body>
</html>
"""


def _trim_bottom_whitespace(image: Any) -> Any:
    Image, ImageChops = _load_pillow_modules()
    bg = Image.new("RGB", image.size, "#f6f4ef")
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if not bbox:
        return image
    bottom = min(image.height, bbox[3] + 32)
    return image.crop((0, 0, image.width, bottom))


def _render_markdown_non_browser_sync(markdown_text: str, title: str, theme_color: str) -> RenderResult:
    ensure_non_browser_render_ready()
    HTML = _load_weasyprint_html()
    fitz = _load_fitz_module()
    Image, _ = _load_pillow_modules()
    html_doc = _build_html_document(markdown_text, title, theme_color)
    pdf_bytes = HTML(string=html_doc).write_pdf()
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        scale = 96.0 / 72.0
        page_images = []
        for page_index in range(len(document)):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            page_image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            if page_index == len(document) - 1:
                page_image = _trim_bottom_whitespace(page_image)
            page_images.append(page_image)
    finally:
        document.close()

    if not page_images:
        raise RuntimeError("weasyprint render produced no pages")

    width = max(image.width for image in page_images)
    height = sum(image.height for image in page_images)
    canvas = Image.new("RGB", (width, height), "#f6f4ef")
    offset_y = 0
    for page_image in page_images:
        canvas.paste(page_image, (0, offset_y))
        offset_y += page_image.height

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    return {
        "ok": True,
        "renderer": "weasyprint",
        "mime_type": "image/png",
        "base64": base64.b64encode(png_bytes).decode(),
    }


async def render_markdown_non_browser_result(
    markdown_text: str,
    title: str = "Assistant Response",
    theme_color: str = "#ef4444",
    **_: Any,
) -> RenderResult:
    return await asyncio.to_thread(
        _render_markdown_non_browser_sync,
        _normalize_text(markdown_text),
        str(title or "Assistant Response").strip() or "Assistant Response",
        str(theme_color or "#ef4444").strip() or "#ef4444",
    )


__all__ = [
    "RenderResult",
    "ensure_non_browser_render_ready",
    "render_markdown_non_browser_result",
]
