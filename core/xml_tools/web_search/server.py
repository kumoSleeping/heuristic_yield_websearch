from __future__ import annotations

import asyncio
import base64
import html as html_lib
import hashlib
import io
import os
import re
import tempfile
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import markdown
from loguru import logger
from PIL import Image as PILImage

from ._public.browser.renderer import get_content_renderer
from ._public.search.service import DuckDuckGoSearchService

_VALID_TIME_RANGE_CODES = {"a", "d", "w", "m", "y"}
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
_HTML_IMAGE_RE = re.compile(r"""<img[^>]+src=["']([^"']+)["']""", flags=re.IGNORECASE)
_MD_IMAGE_TAG_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
_HTML_IMAGE_TAG_RE = re.compile(r"<img[^>]*>", flags=re.IGNORECASE)
_META_IMAGE_RE = re.compile(
    r"""<meta[^>]+(?:property|name)=["'](?:og:image|twitter:image|og:image:url)["'][^>]+content=["']([^"']+)["']""",
    flags=re.IGNORECASE,
)
_TAG_ATTR_RE = re.compile(r"""([a-zA-Z_:][\w:.-]*)\s*=\s*(".*?"|'.*?'|[^\s>]+)""", flags=re.DOTALL)
_LOW_VALUE_IMAGE_HINT_RE = re.compile(
    r"""(?:favicon|icon|logo)""",
    flags=re.IGNORECASE,
)
_DDG_IMAGES_BLOCK_MARKER = 'data-layout="images"'
_DEFAULT_IMAGE_FETCH_TIMEOUT = 12.0
_VALID_SEARCH_MODES = {"text", "images"}
_LITE_RESULT_LINK_RE = re.compile(
    r"<a\b(?=[^>]*class=['\"]result-link['\"])(?=[^>]*href=['\"](?P<href>[^'\"]+)['\"])[^>]*>(?P<title>.*?)</a>",
    flags=re.IGNORECASE | re.DOTALL,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_query(query: str) -> str:
    return str(query or "").replace('"', " ").replace("“", " ").replace("”", " ").strip()


def _normalize_url(url: str) -> str:
    normalized = str(url or "").strip()
    if not normalized:
        return ""
    parsed = urllib.parse.urlparse(normalized)
    if not parsed.scheme:
        normalized = f"https://{normalized}"
    return normalized


def _normalize_time_range(time_range: str) -> Optional[str]:
    raw = str(time_range or "").strip().lower()
    if not raw or raw == "a":
        return None
    if raw in _VALID_TIME_RANGE_CODES:
        return raw
    return None


def _normalize_search_mode(mode: str) -> str:
    raw = str(mode or "text").strip().lower() or "text"
    if raw not in _VALID_SEARCH_MODES:
        raise ValueError(f"unsupported mode: {mode}")
    return raw


def _safe_hostname(url: str) -> str:
    target = str(url or "").strip()
    if not target:
        return ""
    try:
        return urllib.parse.urlparse(target).hostname or ""
    except Exception:
        return ""


def _host_variants(hostname: str) -> List[str]:
    host = str(hostname or "").strip().lower()
    if not host:
        return []
    variants = [host]
    if host.startswith("www."):
        variants.append(host[4:])
    else:
        variants.append(f"www.{host}")
    return variants


def _normalize_html_url(url: str, base_url: str = "https://duckduckgo.com") -> str:
    raw = html_lib.unescape(str(url or "").strip())
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = f"https:{raw}"
    elif not raw.startswith(("http://", "https://", "data:")):
        raw = urllib.parse.urljoin(base_url, raw)
    return raw


def _clean_html_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(raw or ""))
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_html_attrs(tag_html: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    text = str(tag_html or "")
    if not text:
        return attrs
    for key, value in _TAG_ATTR_RE.findall(text):
        cleaned = str(value or "").strip()
        if cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')) and len(cleaned) >= 2:
            cleaned = cleaned[1:-1]
        attrs[key.lower()] = html_lib.unescape(cleaned)
    return attrs


def _extract_image_links(markdown_text: str, html_text: str, base_url: str) -> List[str]:
    links: List[str] = []
    seen: set[str] = set()

    def _push(raw_url: str) -> None:
        candidate = str(raw_url or "").strip()
        if not candidate:
            return
        if candidate.startswith("data:") or candidate.startswith("javascript:"):
            return
        absolute = urllib.parse.urljoin(base_url, candidate)
        parsed = urllib.parse.urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            return
        if absolute in seen:
            return
        seen.add(absolute)
        links.append(absolute)

    for item in _MD_IMAGE_RE.findall(markdown_text or ""):
        _push(item)
    for item in _HTML_IMAGE_RE.findall(html_text or ""):
        _push(item)

    return links


def _extract_meta_image_links(html_text: str, base_url: str) -> List[str]:
    links: List[str] = []
    seen: set[str] = set()
    for raw in _META_IMAGE_RE.findall(html_text or ""):
        candidate = str(raw or "").strip()
        if not candidate:
            continue
        absolute = urllib.parse.urljoin(base_url, candidate)
        parsed = urllib.parse.urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)
    return links


def _decode_ddg_redirect(href: str) -> str:
    resolved = href
    if resolved.startswith("/"):
        resolved = "https://duckduckgo.com" + resolved
    if "uddg=" in resolved:
        try:
            parsed = urllib.parse.urlparse(resolved)
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                resolved = qs["uddg"][0]
        except Exception:
            pass
    return resolved


def _decode_external_content_image_url(url: str) -> str:
    target = str(url or "").strip()
    if not target:
        return ""
    try:
        parsed = urllib.parse.urlparse(target)
    except Exception:
        return target
    host = str(parsed.hostname or "").lower()
    if "external-content.duckduckgo.com" not in host or not parsed.path.startswith("/iu/"):
        return target
    try:
        query = urllib.parse.parse_qs(parsed.query)
        raw = query.get("u", [None])[0]
        if not raw:
            return target
        return urllib.parse.unquote(raw)
    except Exception:
        return target


def _canonical_image_key(url: str) -> str:
    target = str(url or "").strip()
    if not target:
        return ""
    decoded = _decode_external_content_image_url(target)
    try:
        parsed = urllib.parse.urlparse(decoded)
    except Exception:
        return decoded.lower()
    params = urllib.parse.parse_qs(parsed.query)
    for key in {
        "f",
        "ipt",
        "ipo",
        "pid",
        "cb",
        "w",
        "h",
        "width",
        "height",
        "quality",
        "format",
    }:
        params.pop(key, None)
    normalized_query = urllib.parse.urlencode(
        sorted((k, v) for k, values in params.items() for v in values),
        doseq=True,
    )
    normalized = urllib.parse.urlunparse(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", normalized_query, "")
    )
    return normalized


def _is_low_value_image_ref(url: str, *, alt: str = "", context: str = "") -> bool:
    raw_url = str(url or "").strip()
    if not raw_url:
        return True
    probe = " ".join(
        [
            raw_url.lower(),
            _decode_external_content_image_url(raw_url).lower(),
            str(alt or "").lower(),
            str(context or "").lower(),
        ]
    )
    if "/ip3/" in probe or ".ico" in probe:
        return True
    if _LOW_VALUE_IMAGE_HINT_RE.search(probe):
        return True
    return False


def _default_image_cache_dir() -> Path:
    configured = str(os.environ.get("HYW_WEB_IMAGE_CACHE_DIR", "")).strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (_PROJECT_ROOT / "output" / "web" / "search_image_cache").resolve()


def _compress_image_bytes(raw: bytes, *, quality: int = 85, max_width: int = 1440) -> tuple[bytes, str]:
    """
    Compress raster image bytes with PIL while preserving visual quality.
    Returns (compressed_bytes, ext).
    """
    try:
        img = PILImage.open(io.BytesIO(raw))
    except Exception:
        return raw, "jpg"

    try:
        if getattr(img, "width", 0) and img.width > max_width:
            ratio = max_width / float(img.width)
            new_height = max(1, int((img.height or 1) * ratio))
            img = img.resize((max_width, new_height), PILImage.Resampling.LANCZOS)
    except Exception:
        pass

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    output = io.BytesIO()
    try:
        img.save(output, format="JPEG", quality=int(quality), optimize=True)
        return output.getvalue(), "jpg"
    except Exception:
        return raw, "jpg"


def _strip_image_tags(markdown_text: str) -> str:
    text = str(markdown_text or "")
    text = _MD_IMAGE_TAG_RE.sub("", text)
    text = _HTML_IMAGE_TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _as_dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _ensure_h1_title(markdown_text: str, title: str) -> str:
    body = str(markdown_text or "").strip()
    if not body:
        return f"# {title}\n"
    if re.search(r"^\s*#\s+", body, flags=re.MULTILINE):
        return body
    return f"# {title}\n\n{body}"


def _extract_ddg_images_module_block(html_text: str) -> str:
    page = str(html_text or "")
    if not page:
        return ""
    marker_index = page.find(_DDG_IMAGES_BLOCK_MARKER)
    if marker_index < 0:
        return ""
    start = page.rfind("<li", 0, marker_index)
    if start < 0:
        start = marker_index

    end_candidates: List[int] = []
    for marker in ('data-layout="related_searches"', 'id="more-results"', "</section>"):
        found = page.find(marker, marker_index + 1)
        if found > marker_index:
            end_candidates.append(found)
    end = min(end_candidates) if end_candidates else min(len(page), marker_index + 220000)
    if end <= start:
        return ""
    return page[start:end]


def _extract_ddg_module_image_entries(html_text: str) -> List[Dict[str, str]]:
    block = _extract_ddg_images_module_block(html_text)
    if not block:
        return []

    entries: List[Dict[str, str]] = []
    seen: set[str] = set()
    anchor_re = re.compile(
        r'(?P<open><a[^>]+href=["\'](?P<href>[^"\']*iax=images[^"\']*)["\'][^>]*>)(?P<body>.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )

    for match in anchor_re.finditer(block):
        body = str(match.group("body") or "")
        img_match = re.search(r"<img[^>]*>", body, flags=re.IGNORECASE)
        if not img_match:
            continue
        img_attrs = _extract_html_attrs(img_match.group(0))
        image_url = _normalize_html_url(img_attrs.get("src", ""))
        if not image_url or not image_url.startswith(("http://", "https://")):
            continue
        open_attrs = _extract_html_attrs(match.group("open") or "")
        alt = str(img_attrs.get("alt") or open_attrs.get("aria-label") or "").strip()
        if _is_low_value_image_ref(image_url, alt=alt):
            continue

        tail = block[match.end(): min(len(block), match.end() + 2600)]
        source_match = re.search(
            r"<figcaption[^>]*>.*?<a[^>]+href=[\"'](?P<url>https?://[^\"']+)[\"']",
            tail,
            flags=re.IGNORECASE | re.DOTALL,
        )
        source_url = html_lib.unescape(source_match.group("url")).strip() if source_match else ""

        dedupe_key = _canonical_image_key(image_url)
        if not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        entries.append(
            {
                "image_url": image_url,
                "source_url": source_url,
                "source_host": _safe_hostname(source_url),
                "alt": alt,
                "source_type": "ddg_images_module",
            }
        )

    return entries


class WebToolSuite:
    def __init__(self, headless: bool = False):
        self._headless = bool(headless)
        self._search_service = DuckDuckGoSearchService(headless=self._headless)
        self._search_index_counter = 0
        self._image_cache_dir = _default_image_cache_dir()
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)

    def _next_search_index(self) -> int:
        self._search_index_counter += 1
        return self._search_index_counter

    @staticmethod
    def _notify_progress(
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        payload: Dict[str, Any],
    ) -> None:
        if not callable(progress_callback):
            return
        try:
            progress_callback(payload)
        except Exception:
            pass

    def _build_serp_image_context(self, html_text: str) -> Dict[str, Any]:
        module_entries = _extract_ddg_module_image_entries(html_text)

        merged: List[Dict[str, str]] = []
        seen: set[str] = set()
        for item in module_entries:
            if not isinstance(item, dict):
                continue
            image_url = str(item.get("image_url") or "").strip()
            if not image_url:
                continue
            key = _canonical_image_key(image_url)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)

        by_host: Dict[str, List[Dict[str, str]]] = {}
        for item in merged:
            host = str(item.get("source_host") or "").strip().lower()
            if not host:
                continue
            for variant in _host_variants(host):
                by_host.setdefault(variant, []).append(item)

        return {
            "global": merged,
            "by_host": by_host,
            "has_images_module": bool(module_entries),
            "module_count": len(module_entries),
            "article_count": 0,
        }

    def _parse_ddg_image_dom_rows(self, raw_rows: Any) -> List[Dict[str, str]]:
        if not isinstance(raw_rows, list):
            return []

        entries: List[Dict[str, str]] = []
        seen: set[str] = set()
        for item in raw_rows:
            if not isinstance(item, dict):
                continue
            image_url = _normalize_html_url(str(item.get("image_url") or item.get("image") or "").strip())
            data_url = str(item.get("data_url") or "").strip()
            source_url = _decode_ddg_redirect(str(item.get("source_url") or item.get("url") or "").strip())
            title = _clean_html_text(str(item.get("title") or ""))
            alt = _clean_html_text(str(item.get("alt") or title))
            source_host = _safe_hostname(source_url or image_url)
            context = " ".join(part for part in (source_url, title, alt, source_host) if part)
            if not image_url or not image_url.startswith(("http://", "https://", "data:")):
                continue
            if data_url and not data_url.startswith("data:image/"):
                data_url = ""
            if _is_low_value_image_ref(image_url, alt=alt, context=context):
                continue
            dedupe_key = _canonical_image_key(image_url) or (source_url or image_url)
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            entry = {
                "image_url": image_url,
                "source_url": source_url,
                "source_host": source_host,
                "alt": alt,
                "title": title,
                "source_type": "ddg_images_dom",
            }
            if data_url:
                entry["data_url"] = data_url
            entries.append(entry)
        return entries

    def _scrape_ddg_image_entries_sync(self, search_url: str, max_candidates: int) -> List[Dict[str, str]]:
        browser = getattr(self._search_service, "_browser", None)
        if browser is None:
            return []

        js_code = f"""
(() => {{
  const limit = {max(8, int(max_candidates))};
  const rows = [];
  const seen = new Set();
  const toDataUrl = (img) => {{
    try {{
      const width = img.naturalWidth || img.width || 0;
      const height = img.naturalHeight || img.height || 0;
      if (width < 120 || height < 120) return '';
      const maxWidth = 1280;
      const ratio = width > maxWidth ? (maxWidth / width) : 1;
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(width * ratio));
      canvas.height = Math.max(1, Math.round(height * ratio));
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/jpeg', 0.86);
    }} catch (e) {{
      return '';
    }}
  }};
  const figures = Array.from(document.querySelectorAll('figure'));
  for (const fig of figures) {{
    const img = fig.querySelector('img');
    if (!img) continue;
    const src = img.currentSrc || img.src || '';
    if (!src) continue;
    const width = img.naturalWidth || img.width || 0;
    const height = img.naturalHeight || img.height || 0;
    if (width < 120 || height < 120) continue;
    const parent = fig.parentElement;
    const link = parent ? parent.querySelector('a[href]') : null;
    const href = link ? link.href : '';
    const title = link ? (link.innerText || '').trim() : '';
    const alt = (img.alt || '').trim();
    const key = `${{href}}|${{src}}`;
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push({{
      image_url: src,
      data_url: src.startsWith('data:image/') ? src : toDataUrl(img),
      source_url: href,
      title,
      alt,
    }});
    if (rows.length >= limit) break;
  }}
  return rows;
}})()
"""

        tab = None
        try:
            browser._ensure_ready()
            page = getattr(browser._manager, "page", None)
            if not page:
                return []
            tab = page.new_tab()
            browser._navigate_tab(tab, search_url, timeout=20.0)

            clicked_images = False
            raw_rows: Any = []
            deadline = time.time() + 20.0
            while time.time() < deadline:
                current_url = str(getattr(tab, "url", "") or "")
                if not clicked_images and "ia=images" not in current_url:
                    try:
                        clicked_images = bool(
                            tab.run_js(
                                "const a = document.querySelector('a[href*=\"ia=images\"]'); if (a) { a.click(); true; } else { false; }",
                                as_expr=True,
                            )
                        )
                    except Exception:
                        clicked_images = False
                try:
                    raw_rows = tab.run_js(js_code, as_expr=True) or []
                except Exception:
                    raw_rows = []
                parsed = self._parse_ddg_image_dom_rows(raw_rows)
                if parsed:
                    return parsed[: max(1, int(max_candidates))]
                try:
                    tab.run_js("window.scrollTo(0, Math.min(document.body.scrollHeight, window.scrollY + window.innerHeight * 1.25));")
                except Exception:
                    pass
                time.sleep(0.4)
            return self._parse_ddg_image_dom_rows(raw_rows)[: max(1, int(max_candidates))]
        except Exception as exc:
            logger.debug("web_search image DOM scrape failed for {}: {}", search_url, exc)
            return []
        finally:
            if tab is not None:
                try:
                    browser._release_tab(tab)
                except Exception:
                    try:
                        tab.close()
                    except Exception:
                        pass

    async def _scrape_ddg_image_entries(self, search_url: str, max_candidates: int) -> List[Dict[str, str]]:
        browser = getattr(self._search_service, "_browser", None)
        loop = asyncio.get_running_loop()
        executor = getattr(browser, "_executor", None)
        return await loop.run_in_executor(executor, self._scrape_ddg_image_entries_sync, search_url, max_candidates)

    def _cache_binary_image(self, raw: bytes, *, seed: str, ext: str = "jpg") -> str:
        digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:16]
        safe_ext = str(ext or "jpg").lower()
        if safe_ext not in {"jpg", "png", "webp", "gif", "bmp", "svg"}:
            safe_ext = "jpg"

        write_bytes = raw
        write_ext = safe_ext
        if safe_ext not in {"gif", "svg"}:
            compressed, compressed_ext = _compress_image_bytes(raw, quality=85, max_width=1440)
            if compressed:
                write_bytes = compressed
                write_ext = compressed_ext or "jpg"

        path = self._image_cache_dir / f"{digest}.{write_ext}"
        path.write_bytes(write_bytes)
        return str(path.resolve())

    async def _cache_image_source(self, candidate: Dict[str, str], *, seed: str) -> Optional[Dict[str, str]]:
        image_url = str(candidate.get("image_url") or "").strip()
        data_url = str(candidate.get("data_url") or "").strip()
        source = str(candidate.get("source") or candidate.get("source_url") or image_url).strip()
        source_type = str(candidate.get("source_type") or "").strip() or "unknown"
        if not image_url and not data_url:
            return None

        image_url = _normalize_html_url(image_url)
        if image_url and _is_low_value_image_ref(image_url, context=source):
            return None

        inline_url = data_url if data_url.startswith("data:") else image_url
        if inline_url.startswith("data:"):
            try:
                header, b64 = inline_url.split(",", 1)
            except ValueError:
                return None
            if ";base64" not in header.lower():
                return None
            mime_hint = header.split(";", 1)[0].lower()
            ext = "png" if "png" in mime_hint else ("webp" if "webp" in mime_hint else "jpg")
            try:
                raw = base64.b64decode(b64, validate=False)
            except Exception:
                return None
            if not raw:
                return None
            local_path = self._cache_binary_image(raw, seed=f"{seed}|data", ext=ext)
            return {
                "local_path": local_path,
                "source": source,
                "source_type": source_type if source_type != "unknown" else "data_uri",
            }

        if not image_url:
            return None

        try:
            page = await self._search_service.fetch_page(
                image_url,
                timeout=float(_DEFAULT_IMAGE_FETCH_TIMEOUT),
                include_screenshot=True,
            )
        except Exception as exc:
            logger.debug("web_search image browser fetch failed for {}: {}", image_url, exc)
            return None

        screenshot_b64 = str(page.get("raw_screenshot_b64") or page.get("screenshot_b64") or "").strip()
        if not screenshot_b64:
            return None
        try:
            raw = base64.b64decode(screenshot_b64, validate=False)
        except Exception:
            return None
        if not raw:
            return None

        local_path = self._cache_binary_image(
            raw,
            seed=f"{seed}|{_canonical_image_key(image_url) or image_url}",
            ext="jpg",
        )
        return {
            "local_path": local_path,
            "source": source,
            "source_type": source_type if source_type != "unknown" else "browser_capture",
        }


    def _build_text_search_url(
        self,
        query: str,
        kl: Optional[str],
        time_range: Optional[str],
    ) -> str:
        base = "https://lite.duckduckgo.com/lite/"
        params: Dict[str, str] = {
            "q": query,
        }
        if kl:
            params["kl"] = kl
        if time_range:
            params["df"] = time_range
        return f"{base}?{urllib.parse.urlencode(params)}"

    def _build_image_search_url(
        self,
        query: str,
        kl: Optional[str],
        time_range: Optional[str],
    ) -> str:
        base = "https://duckduckgo.com/"
        params: Dict[str, str] = {
            "q": query,
            "t": "h_",
            "ia": "images",
            "iax": "images",
        }
        if kl:
            params["kl"] = kl
        if time_range:
            params["df"] = time_range
        return f"{base}?{urllib.parse.urlencode(params)}"

    async def _search_via_ddg_lite(
        self,
        query: str,
        kl: Optional[str],
        time_range: Optional[str],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        search_url = self._build_text_search_url(query, kl, time_range)
        page = await self._search_service.fetch_page(search_url, timeout=20.0, include_screenshot=False)
        html = str(page.get("html") or page.get("content") or "")
        results: List[Dict[str, Any]] = []
        seen: set[str] = set()
        matches = list(_LITE_RESULT_LINK_RE.finditer(html))

        for index, match in enumerate(matches, start=1):
            href = _decode_ddg_redirect(match.group("href"))
            if not href.startswith("http"):
                continue
            if "duckduckgo.com" in href:
                continue
            if href in seen:
                continue
            seen.add(href)

            title = _clean_html_text(match.group("title")) or "No Title"
            next_offset = matches[index].start() if index < len(matches) else min(len(html), match.end() + 2400)
            snippet_window = html[match.end():next_offset]
            snippet_match = re.search(
                r"<td\s+class=['\"]result-snippet['\"]>(?P<snippet>.*?)</td>",
                snippet_window,
                flags=re.IGNORECASE | re.DOTALL,
            )
            snippet = _clean_html_text(snippet_match.group("snippet") if snippet_match else snippet_window)[:500]

            results.append(
                {
                    "title": title,
                    "url": href,
                    "snippet": snippet,
                    "images": [],
                    "images_local": [],
                    "image_details": [],
                }
            )
            if len(results) >= max_results:
                break
        return results

    async def _search_via_ddg_images(
        self,
        query: str,
        kl: Optional[str],
        time_range: Optional[str],
        max_results: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        search_url = self._build_image_search_url(query, kl, time_range)
        probe_limit = max(int(max_results) * 4, 10)
        entries = await self._scrape_ddg_image_entries(search_url, max_candidates=probe_limit)
        if not entries:
            page = await self._search_service.fetch_page(search_url, timeout=20.0, include_screenshot=False)
            html = str(page.get("html") or page.get("content") or "")
            serp_image_context = self._build_serp_image_context(html)
            entries = serp_image_context.get("global") if isinstance(serp_image_context.get("global"), list) else []

        self._notify_progress(
            progress_callback,
            {
                "phase": "serp_images",
                "status": "ready",
                "query": query,
                "module_count": len(entries),
            },
        )

        results: List[Dict[str, Any]] = []
        total = min(max(1, int(max_results)), len(entries))
        for idx, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                continue
            image_url = str(entry.get("image_url") or "").strip()
            source_url = str(entry.get("source_url") or "").strip()
            if not image_url:
                continue
            self._notify_progress(
                progress_callback,
                {
                    "phase": "image_download",
                    "status": "start",
                    "result_index": idx,
                    "result_total": total,
                    "title": str(entry.get("alt") or source_url or query).strip(),
                    "required": 1,
                },
            )
            cached = await self._cache_image_source(
                {
                    "image_url": image_url,
                    "data_url": str(entry.get("data_url") or "").strip(),
                    "source": source_url or image_url,
                    "source_url": source_url,
                    "source_type": str(entry.get("source_type") or "ddg_images_module"),
                },
                seed=f"{query}|images|{idx}",
            )
            if not cached:
                continue
            local_path = str(cached.get("local_path") or "").strip()
            if not local_path:
                continue

            title = _clean_html_text(str(entry.get("alt") or ""))
            if not title:
                title = _safe_hostname(source_url) or f"{query} 图片"
            snippet = _safe_hostname(source_url) or ""
            result_url = source_url or _decode_external_content_image_url(image_url) or image_url
            source_host = _safe_hostname(source_url or image_url)
            source_type = str(cached.get("source_type") or entry.get("source_type") or "ddg_images_module").strip() or "ddg_images_module"
            image_detail = {
                "order": 1,
                "local_path": local_path,
                "source": str(cached.get("source") or source_url or image_url).strip(),
                "source_host": source_host,
                "source_type": source_type,
                "intro": title,
            }
            results.append(
                {
                    "index": self._next_search_index(),
                    "title": title,
                    "url": result_url,
                    "intro": snippet,
                    "snippet": snippet,
                    "images": [local_path],
                    "images_local": [local_path],
                    "image_details": [image_detail],
                    "image_count": 1,
                }
            )
            self._notify_progress(
                progress_callback,
                {
                    "phase": "image_download",
                    "status": "done",
                    "result_index": idx,
                    "result_total": total,
                    "title": title,
                    "cached": 1,
                    "required": 1,
                },
            )
            if len(results) >= max_results:
                break

        return results

    async def web_search(
        self,
        query: str,
        mode: str = "text",
        kl: str = "",
        time_range: str = "a",
        max_results: int = 5,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        normalized_query = _normalize_query(query)
        if not normalized_query:
            raise ValueError("query is empty")

        normalized_mode = _normalize_search_mode(mode)
        region = str(kl or "").strip() or None
        normalized_time = _normalize_time_range(time_range)
        limit = max(1, min(int(max_results), 10))
        self._notify_progress(
            progress_callback,
            {
                "phase": "search",
                "status": "start",
                "query": normalized_query,
                "mode": normalized_mode,
            },
        )

        if normalized_mode == "text":
            results = await self._search_via_ddg_lite(normalized_query, region, normalized_time, limit)
            results = [
                {
                    "index": self._next_search_index(),
                    "title": str(row.get("title") or "").strip() or "No Title",
                    "url": str(row.get("url") or "").strip(),
                    "intro": str(row.get("snippet") or row.get("intro") or "").strip(),
                    "snippet": str(row.get("snippet") or row.get("intro") or "").strip(),
                    "images": [],
                    "images_local": [],
                    "image_details": [],
                    "image_count": 0,
                }
                for row in results
                if isinstance(row, dict)
            ]
        else:
            results = await self._search_via_ddg_images(
                normalized_query,
                region,
                normalized_time,
                limit,
                progress_callback,
            )

        self._notify_progress(
            progress_callback,
            {
                "phase": "search",
                "status": "results_ready",
                "query": normalized_query,
                "mode": normalized_mode,
                "count": len(results),
            },
        )
        self._notify_progress(
            progress_callback,
            {
                "phase": "search",
                "status": "done",
                "query": normalized_query,
                "mode": normalized_mode,
                "count": len(results),
            },
        )

        return {
            "query": normalized_query,
            "mode": normalized_mode,
            "filters": {
                "kl": region,
                "time_range": normalized_time,
            },
            "count": len(results),
            "results": results,
        }

    async def web_fetch(
        self,
        url: str,
        include_screenshot: bool = True,
        include_images: bool = True,
        timeout_seconds: float = 20.0,
        max_images: int = 8,
        strip_images_from_markdown: bool = True,
        max_content_chars: int = 12000,
    ) -> Dict[str, Any]:
        normalized_url = _normalize_url(url)
        if not normalized_url:
            raise ValueError("url is empty")

        timeout = max(3.0, min(float(timeout_seconds), 60.0))
        limit = max(0, min(int(max_images), 16))
        include_screenshot = bool(include_screenshot)
        include_images = bool(include_images)

        page = await self._search_service.fetch_page(
            normalized_url,
            timeout=timeout,
            include_screenshot=include_screenshot,
        )

        final_url = str(page.get("url") or normalized_url).strip()
        title = str(page.get("title") or "").strip()
        html = str(page.get("html") or "")
        content_markdown_raw = str(page.get("content") or "").strip()

        if not content_markdown_raw and html:
            try:
                import trafilatura

                content_markdown_raw = (
                    trafilatura.extract(
                        html,
                        include_links=True,
                        include_images=True,
                        include_comments=False,
                        include_tables=True,
                        favor_precision=False,
                        output_format="markdown",
                    )
                    or ""
                ).strip()
            except Exception as exc:
                logger.warning("web_fetch fallback trafilatura extract failed: {}", exc)
                content_markdown_raw = ""

        content_markdown = content_markdown_raw
        if strip_images_from_markdown:
            content_markdown = _strip_image_tags(content_markdown)

        content_truncated = False
        if max_content_chars > 0 and len(content_markdown) > max_content_chars:
            content_markdown = content_markdown[:max_content_chars].rstrip() + "\n\n[content truncated]"
            content_truncated = True

        images: List[Dict[str, Any]] = []
        if include_images and limit > 0:
            image_links = _extract_image_links(content_markdown_raw, html, final_url)
            for item in image_links[:limit]:
                images.append(
                    {
                        "id": f"url_{len(images) + 1}",
                        "source": "content_link",
                        "url": item,
                    }
                )

            dom_images = page.get("images")
            if isinstance(dom_images, list):
                for raw_b64 in dom_images:
                    if len(images) >= limit:
                        break
                    b64 = str(raw_b64 or "").strip()
                    if not b64:
                        continue
                    images.append(
                        {
                            "id": f"img_{len(images) + 1}",
                            "source": "dom_capture",
                            "mime_type": "image/jpeg",
                            "base64": b64,
                        }
                    )

        screenshot = None
        screenshot_b64 = str(page.get("raw_screenshot_b64") or page.get("screenshot_b64") or "").strip()
        if include_screenshot and screenshot_b64:
            screenshot = {
                "mime_type": "image/jpeg",
                "base64": screenshot_b64,
            }

        return {
            "requested_url": normalized_url,
            "url": final_url,
            "title": title,
            "content_markdown": content_markdown,
            "content_markdown_raw": content_markdown_raw,
            "content_truncated": content_truncated,
            "images": images,
            "raw_screenshot_b64": screenshot_b64 if screenshot_b64 else None,
            "screenshot": screenshot,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _render_markdown_plain(self, markdown_text: str, title: str, theme_color: str) -> Optional[str]:
        safe_title = str(title or "Assistant Response").strip()
        safe_color = str(theme_color or "#ef4444").strip()
        html_body = markdown.markdown(
            str(markdown_text or "").strip(),
            extensions=["extra", "fenced_code", "tables", "sane_lists", "nl2br"],
        )

        html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{
      --theme: {safe_color};
      --bg: #f4f4f3;
      --text: #222;
      --card: #ffffff;
      --muted: #6b7280;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "PingFang SC", "Noto Sans SC", "Helvetica Neue", Arial, sans-serif;
    }}
    #card {{
      width: 980px;
      margin: 40px auto;
      background: var(--card);
      border: 2px solid #e8e8e8;
      box-shadow: 0 10px 40px rgba(0,0,0,0.08);
      padding: 36px 40px;
      box-sizing: border-box;
    }}
    .title {{
      font-size: 30px;
      line-height: 1.25;
      font-weight: 800;
      margin: 0 0 18px 0;
      border-left: 8px solid var(--theme);
      padding-left: 14px;
    }}
    .content {{
      font-size: 18px;
      line-height: 1.75;
      word-break: break-word;
    }}
    .content h1, .content h2, .content h3 {{
      margin-top: 1.2em;
      margin-bottom: 0.5em;
      line-height: 1.3;
    }}
    .content p {{
      margin: 0.45em 0;
    }}
    .content pre {{
      background: #0f172a;
      color: #f8fafc;
      border-radius: 10px;
      padding: 16px;
      overflow-x: auto;
    }}
    .content code {{
      background: #f1f5f9;
      padding: 2px 6px;
      border-radius: 4px;
    }}
    .content a {{
      color: var(--theme);
      text-decoration: none;
    }}
    .meta {{
      margin-top: 20px;
      font-size: 13px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div id="card">
    <h1 class="title">{safe_title}</h1>
    <div class="content">{html_body}</div>
    <div class="meta">render (plain)</div>
  </div>
</body>
</html>
"""
        with tempfile.NamedTemporaryFile(suffix=".html", mode="w", encoding="utf-8", delete=False) as temp_html:
            temp_html.write(html_doc)
            html_path = Path(temp_html.name)

        try:
            # Reuse browser screenshot path via search service to avoid another runtime type.
            result = await self._search_service.screenshot_with_content(
                html_path.as_uri(),
                max_content_length=0,
            )
            return str(result.get("screenshot_b64") or "").strip() or None
        finally:
            try:
                html_path.unlink(missing_ok=True)
            except Exception:
                pass

    async def _render_markdown_vue(self, render_data: Dict[str, Any]) -> Optional[str]:
        renderer = await get_content_renderer(headless=self._headless)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
            image_path = Path(temp_image.name)

        try:
            success = await renderer.render(
                markdown_content=str(render_data.get("markdown") or ""),
                output_path=str(image_path),
                stats=render_data.get("stats") or {},
                references=_as_dict_list(render_data.get("references")),
                page_references=_as_dict_list(render_data.get("page_references")),
                image_references=_as_dict_list(render_data.get("image_references")),
                stages_used=_as_dict_list(render_data.get("stages")),
                theme_color=str(render_data.get("theme_color") or "#ef4444"),
            )
            if not success or not image_path.exists():
                return None

            data = image_path.read_bytes()
            if not data:
                return None
            return base64.b64encode(data).decode()
        finally:
            try:
                image_path.unlink(missing_ok=True)
            except Exception:
                pass

    async def markdown_llm_render(
        self,
        markdown_text: str,
        title: str = "Assistant Response",
        theme_color: str = "#ef4444",
        use_vue: bool = True,
        references: Optional[List[Dict[str, Any]]] = None,
        page_references: Optional[List[Dict[str, Any]]] = None,
        image_references: Optional[List[Dict[str, Any]]] = None,
        stages: Optional[List[Dict[str, Any]]] = None,
        stats: Optional[Dict[str, Any]] = None,
        total_time: float = 0.0,
    ) -> Dict[str, Any]:
        safe_title = str(title or "Assistant Response").strip()
        safe_color = str(theme_color or "#ef4444").strip()
        markdown_with_title = _ensure_h1_title(str(markdown_text or ""), safe_title)

        if use_vue:
            render_data = {
                "markdown": markdown_with_title,
                "total_time": float(total_time or 0.0),
                "stages": _as_dict_list(stages),
                "references": _as_dict_list(references),
                "page_references": _as_dict_list(page_references),
                "image_references": _as_dict_list(image_references),
                "stats": stats if isinstance(stats, dict) else {},
                "theme_color": safe_color,
            }

            try:
                vue_b64 = await self._render_markdown_vue(render_data)
            except Exception as exc:
                logger.warning("render vue mode failed: {}", exc)
                vue_b64 = None

            if vue_b64:
                return {
                    "ok": True,
                    "message": "render success",
                    "renderer": "vue",
                    "image": {
                        "mime_type": "image/jpeg",
                        "base64": vue_b64,
                    },
                }

        plain_b64 = await self._render_markdown_plain(markdown_with_title, safe_title, safe_color)
        if plain_b64:
            return {
                "ok": True,
                "message": "render success",
                "renderer": "plain",
                "image": {
                    "mime_type": "image/webp",
                    "base64": plain_b64,
                },
            }

        return {"ok": False, "message": "render failed", "image": None}
