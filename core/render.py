from __future__ import annotations

import asyncio
import base64
import html
import io
import re
from typing import Any, Callable, Dict

RenderCallable = Callable[..., Any]
RenderResult = Dict[str, Any]
_BLOCK_MATH_RE = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", flags=re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)(?<!\\)\$(.+?)(?<!\\)\$(?!\$)", flags=re.DOTALL)


def _missing_dependency(name: str, package: str) -> RuntimeError:
    return RuntimeError(
        f"{name} is required for websearch markdown rendering. "
        f"Install it with `pip install {package}`."
    )


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
        raise _missing_dependency("WeasyPrint", "weasyprint") from exc


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


def _build_html_document(markdown_text: str, title: str, theme_color: str) -> str:
    HtmlFormatter = _load_pygments_html_formatter()
    body = _markdown_to_html(markdown_text)
    pygments_css = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    safe_title = html.escape(str(title or "Assistant Response").strip() or "Assistant Response")
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
      --text: #1f2937;
      --muted: #6b7280;
      --code: #f3efe6;
    }}
    @page {{
      size: 1440px 1800px;
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
      width: 1320px;
      margin: 40px auto;
      background: var(--paper);
      padding: 44px 52px 48px 52px;
      box-sizing: border-box;
      border: 1px solid rgba(0, 0, 0, 0.05);
    }}
    .title {{
      margin: 0 0 24px 0;
      padding-left: 16px;
      border-left: 8px solid var(--theme);
      font-size: 34px;
      line-height: 1.25;
      font-weight: 800;
    }}
    .content {{
      font-size: 20px;
      line-height: 1.75;
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
      margin-top: 24px;
      color: var(--muted);
      font-size: 13px;
    }}
    {pygments_css}
  </style>
</head>
<body>
  <div class="page">
    <h1 class="title">{safe_title}</h1>
    <div class="content">{body}</div>
    <div class="meta">render (weasyprint)</div>
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


def _render_markdown_sync(markdown_text: str, title: str, theme_color: str) -> RenderResult:
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


async def render_markdown_result(
    markdown_text: str,
    title: str = "Assistant Response",
    theme_color: str = "#ef4444",
) -> RenderResult:
    return await asyncio.to_thread(
        _render_markdown_sync,
        _normalize_text(markdown_text),
        str(title or "Assistant Response").strip() or "Assistant Response",
        str(theme_color or "#ef4444").strip() or "#ef4444",
    )


async def render_markdown_base64(
    markdown_text: str,
    title: str = "Assistant Response",
    theme_color: str = "#ef4444",
) -> str:
    payload = await render_markdown_result(
        markdown_text=markdown_text,
        title=title,
        theme_color=theme_color,
    )
    return str(payload.get("base64") or "").strip()


__all__ = [
    "RenderCallable",
    "RenderResult",
    "render_markdown_base64",
    "render_markdown_result",
]
