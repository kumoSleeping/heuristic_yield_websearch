#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


JSON_BLOCK_RE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
PAGE_LINE_RE = re.compile(r"^L?(?P<line>\d+)\s*\|\s*(?P<text>.+)$")

TITLE_SPLIT_RE = re.compile(r"\s+-\s+|\s+\|\s+|\s+｜\s+|[_｜|·•]+")
LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
SIGNAL_CHAR_RE = re.compile(r"[\w\u4e00-\u9fff\u3040-\u30ff]")
MARKDOWN_BULLET_RE = re.compile(r"^\s*(?:[*+-]|\d+\.)\s+")
MARKDOWN_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")


@dataclass
class PageEntry:
    source_log: Path
    url: str
    host: str
    raw_title: str
    title_core: str
    path_label: str
    mode: str
    matched_lines: list[tuple[int, str]]
    skeleton_lines: list[str]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _iter_json_blocks(text: str) -> list[dict]:
    rows: list[dict] = []
    for match in JSON_BLOCK_RE.finditer(text):
        block = str(match.group(1) or "").strip()
        if not block:
            continue
        try:
            payload = json.loads(block)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _parse_page_markdown(payload: dict, *, source_log: Path) -> PageEntry | None:
    markdown = str(payload.get("_model_markdown") or "").strip()
    if not markdown.startswith("# Page:"):
        return None
    provider = str(
        payload.get("provider")
        or ((payload.get("_meta") or {}).get("provider") if isinstance(payload.get("_meta"), dict) else "")
        or ""
    ).strip()
    if provider and provider != "jina_ai":
        return None

    lines = markdown.splitlines()
    url = str(lines[0].removeprefix("# Page:").strip())
    if not url:
        return None
    raw_title = ""
    mode = "preview" if any(line.startswith("Preview: ") for line in lines) else "page"
    matched_lines: list[tuple[int, str]] = []
    for line in lines:
        if line.startswith("Title: "):
            raw_title = str(line.removeprefix("Title: ").strip())
            continue
        match = PAGE_LINE_RE.match(line)
        if match is None:
            continue
        matched_lines.append((int(match.group("line")), match.group("text")))
    if not matched_lines:
        return None

    host = urlparse(url).netloc.strip().lower()
    title_core = _title_core(raw_title, url=url)
    path_label = _path_label(url)
    skeleton_lines = _build_skeleton_lines(matched_lines)
    if not skeleton_lines:
        return None
    return PageEntry(
        source_log=source_log,
        url=url,
        host=host,
        raw_title=raw_title,
        title_core=title_core,
        path_label=path_label,
        mode=mode,
        matched_lines=matched_lines,
        skeleton_lines=skeleton_lines,
    )


def _path_label(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path or "").strip("/")
    if not path:
        return parsed.netloc or url
    parts = [part for part in path.split("/") if part]
    tail = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return tail or parsed.netloc or url


def _title_core(title: str, *, url: str) -> str:
    clean = _normalize_space(title)
    if not clean:
        return _path_label(url)

    candidates = [clean]
    for part in TITLE_SPLIT_RE.split(clean):
        part = _normalize_space(part)
        if part:
            candidates.append(part)
            for sep in (":", "："):
                if sep in part:
                    head = _normalize_space(part.split(sep, 1)[0])
                    if 2 <= len(head) <= 80:
                        candidates.append(head)

    path_tokens = {
        token.casefold()
        for token in re.split(r"[/_.-]+", _path_label(url))
        if token.strip()
    }
    best = clean
    best_score: tuple[int, int, int, int, str] | None = None
    for item in candidates:
        low = item.casefold()
        overlap = sum(1 for token in path_tokens if token and token in low)
        length_penalty = 0
        if len(item) > 90:
            length_penalty += 2
        elif len(item) > 50:
            length_penalty += 1
        if len(item) < 3:
            length_penalty += 2
        signal_chars = len(re.findall(r"[\w\u4e00-\u9fff\u3040-\u30ff]", item))
        punctuation_penalty = len(re.findall(r"[/|_·•\-:：]", item))
        score = (length_penalty, -overlap, -signal_chars, punctuation_penalty, item)
        if best_score is None or score < best_score:
            best_score = score
            best = item
    return best


def _strip_markdown_links(text: str) -> str:
    value = str(text or "")
    prev = None
    while prev != value:
        prev = value
        value = re.sub(
            r"\[!\[([^\]]*)\]\([^)]+\)\]\([^)]+\)",
            lambda match: f"[IMG:{_normalize_space(match.group(1) or 'image')}]",
            value,
        )
        value = re.sub(
            r"!\[([^\]]*)\]\([^)]+\)",
            lambda match: f"[IMG:{_normalize_space(match.group(1) or 'image')}]",
            value,
        )
        value = re.sub(
            r"\[([^\]]+)\]\([^)]+\)",
            lambda match: _normalize_space(match.group(1)),
            value,
        )
    value = re.sub(r"\[\]\([^)]+\)", "", value)
    value = re.sub(r"<https?://[^>]+>", "", value)
    value = re.sub(r"\b(?:https?://|blob:http://|mailto:|javascript:)\S+", "", value)
    value = re.sub(r"\[\[(\d+(?:[-,]\d+)*)\]\]\(?", "", value)
    value = re.sub(r"\[(\d+(?:[-,]\d+)*)\]", "", value)
    value = re.sub(r"(^|\s)>+\s*", r"\1", value)
    value = re.sub(r"(^|\s)[*_~]{1,3}(?=\S)", r"\1", value)
    value = re.sub(r"(?<=\S)[*_~]{1,3}(?=\s|$)", "", value)
    value = re.sub(r"\[[xX ]\]", "", value)
    value = value.replace("[](", "")
    value = value.strip(" |")
    value = _normalize_space(value)
    return value


def _resolve_image_tokens(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        body = _normalize_space(match.group(1) or "")
        body = re.sub(r"^Image\s+\d+\s*:?\s*", "", body)
        if _signal_chars(body) < 4:
            return ""
        return body

    value = re.sub(r"\[IMG:([^\]]+)\]", repl, str(text or ""))
    value = value.replace("](", " ")
    return _normalize_space(value)


def _signal_chars(text: str) -> int:
    return len(SIGNAL_CHAR_RE.findall(text))


def _latin_word_count(text: str) -> int:
    return len(LATIN_WORD_RE.findall(text))


def _cjk_char_count(text: str) -> int:
    return len(CJK_CHAR_RE.findall(text))


def _wrapper_ratio(text: str) -> float:
    if not text:
        return 1.0
    wrapper_chars = sum(ch in "[](){}<>*_`|" for ch in text)
    return wrapper_chars / max(len(text), 1)


def _collapse_breadcrumbs(text: str) -> str:
    if not any(sep in text for sep in (">", "›", "»")):
        return text
    parts = [_normalize_space(part) for part in re.split(r"\s*[>›»]+\s*", text) if _normalize_space(part)]
    if len(parts) < 2:
        return text
    tail = parts[-1]
    if _signal_chars(tail) >= max(8, _signal_chars(text) // 4):
        return tail
    return text


def _looks_command_like(raw: str, clean: str) -> bool:
    raw = str(raw or "").strip()
    if not raw or not clean:
        return False
    if "`" in raw or "`" in clean:
        return True
    if re.search(r"(^|\s)(?:\./|\.\./|/[\w./-]+|--[\w-]+|\w+=\S+)", raw):
        return True
    if any(token in raw for token in (" | ", " && ", " || ")):
        return True
    tokens = clean.split()
    if 1 <= len(tokens) <= 8 and any(part.startswith("-") or "/" in part or "." in part for part in tokens[1:]):
        if _latin_word_count(clean) >= 1:
            return True
    return False


def _classify_line(raw: str, clean: str) -> str:
    raw = str(raw or "").lstrip()
    if MARKDOWN_HEADING_RE.match(raw):
        return "heading"
    if raw.startswith("|") and raw.count("|") >= 2:
        return "table"
    if MARKDOWN_BULLET_RE.match(raw):
        return "bullet"
    if _looks_command_like(raw, clean):
        return "command"
    return "text"


def _table_cells(text: str) -> list[str]:
    if not text.startswith("|"):
        return []
    parts = [_normalize_space(part) for part in text.strip("|").split("|")]
    return [part for part in parts if part and not re.fullmatch(r"[:\- ]+", part)]


def _looks_weak_surface(text: str) -> bool:
    if not text:
        return True
    if text in {"?", "？"}:
        return True
    non_image = _normalize_space(re.sub(r"\[IMG:[^\]]+\]", "", text))
    if text.startswith("[IMG:") and _signal_chars(non_image) < 6:
        return True
    if text.count("[IMG:") >= 2 and _signal_chars(non_image) < 12:
        return True
    if re.fullmatch(r"[^\w\u4e00-\u9fff\u3040-\u30ff]+", text):
        return True
    if re.fullmatch(r"(?:\d+[./:-]?)+", text):
        return True
    if _signal_chars(text) <= 1:
        return True
    if _wrapper_ratio(text) >= 0.35:
        return True
    if " " not in text and len(re.findall(r"[a-z][A-Z]", text)) >= 2:
        return True
    return False


def _signal_score(kind: str, text: str) -> int:
    if _looks_weak_surface(text):
        return -100
    signal = _signal_chars(text)
    cjk = _cjk_char_count(text)
    latin_words = _latin_word_count(text)
    has_sentence = bool(re.search(r"[。！？.!?:：;；]", text))
    has_code = bool(re.search(r"`[^`]+`|(^|\s)(?:--[\w-]+|\w+=\S+|/[\w./-]+)", text))
    wrapper_ratio = _wrapper_ratio(text)

    score = 0
    if kind == "heading":
        if signal < 6:
            return -100
        score += 5
        if cjk >= 6 or latin_words >= 3:
            score += 2
        if has_sentence:
            score += 1
    elif kind == "command":
        if signal < 6:
            return -100
        score += 5
        if has_code:
            score += 2
    elif kind == "table":
        cells = _table_cells(text)
        if len(cells) < 2:
            return -100
        if sum(_signal_chars(cell) for cell in cells) < 8:
            return -100
        score += 4
        if any("`" in cell for cell in cells):
            score += 1
    elif kind == "bullet":
        if signal < 8:
            return -100
        if cjk < 6 and latin_words < 5 and not has_sentence and not has_code:
            return -100
        score += 4
        if has_sentence or cjk >= 10 or latin_words >= 8:
            score += 2
    else:
        if signal < 24:
            return -100
        if cjk < 14 and latin_words < 8 and not has_sentence and not has_code:
            return -100
        score += 4
        if has_sentence:
            score += 2

    if cjk >= 8 or latin_words >= 6:
        score += 1
    if "`" in text:
        score += 1
    if text.startswith("[IMG:") or text.count("[IMG:") >= 2:
        score -= 4
    if re.fullmatch(r"[A-Z][A-Z0-9 /&+-]{3,}", text) and latin_words <= 2:
        score -= 4
    if len(text) <= 3:
        score -= 3
    if wrapper_ratio >= 0.24:
        score -= 4
    if kind != "heading" and latin_words <= 3 and cjk == 0 and not has_sentence and len(text) <= 24:
        score -= 5
    return score


def _normalize_skeleton_text(kind: str, clean: str) -> str:
    text = clean
    if kind == "heading":
        text = re.sub(r"^#{1,6}\s*", "", text)
    elif kind == "bullet":
        text = MARKDOWN_BULLET_RE.sub("", text)
    text = _collapse_breadcrumbs(text)
    return _normalize_space(text)


def _build_skeleton_lines(rows: list[tuple[int, str]], *, limit: int = 10) -> list[str]:
    candidates: list[tuple[int, int, str, str]] = []
    seen: set[str] = set()
    for line_no, raw in rows:
        clean = _strip_markdown_links(raw)
        clean = _resolve_image_tokens(clean)
        if not clean or _looks_weak_surface(clean):
            continue
        kind = _classify_line(raw, clean)
        text = _normalize_skeleton_text(kind, clean)
        if not text:
            continue
        dedupe_key = f"{kind}:{text.casefold()}"
        if dedupe_key in seen:
            continue
        score = _signal_score(kind, text)
        if score < 2:
            continue
        seen.add(dedupe_key)
        candidates.append((score, line_no, kind, text[:160]))

    if not candidates:
        return []

    candidates = _focus_candidates(candidates)
    selected: list[str] = []
    for _, line_no, kind, text in sorted(
        sorted(candidates, key=lambda item: (-item[0], item[1]))[: max(limit * 2, limit)],
        key=lambda item: item[1],
    ):
        selected.append(f"- {kind}@L{line_no}: {text}")
        if len(selected) >= limit:
            break
    return selected


def _focus_candidates(candidates: list[tuple[int, int, str, str]]) -> list[tuple[int, int, str, str]]:
    if len(candidates) <= 3:
        return candidates
    line_numbers = [line_no for _, line_no, _, _ in candidates]
    spread = max(line_numbers) - min(line_numbers)
    window = min(max(40, spread // 3), 120)
    half_window = max(20, window // 2)

    best_score: int | None = None
    best_range: tuple[int, int] | None = None
    for _, anchor, _, _ in candidates:
        start = anchor - half_window
        end = anchor + half_window
        score = sum(item_score for item_score, line_no, _, _ in candidates if start <= line_no <= end)
        if best_score is None or score > best_score:
            best_score = score
            best_range = (start, end)

    if best_range is None:
        return candidates

    start, end = best_range
    focused = [item for item in candidates if start <= item[1] <= end]
    if len(focused) >= max(3, min(6, len(candidates) // 2)):
        return focused
    return candidates


def _iter_page_entries(log_dir: Path) -> list[PageEntry]:
    entries: list[PageEntry] = []
    for path in sorted(log_dir.glob("*.md"), reverse=True):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for payload in _iter_json_blocks(text):
            entry = _parse_page_markdown(payload, source_log=path)
            if entry is not None:
                entries.append(entry)
    return entries


def _pick_entries(entries: list[PageEntry], *, limit: int) -> list[PageEntry]:
    def quality(entry: PageEntry) -> tuple[int, int, int]:
        textish = 0
        commandish = 0
        long_lines = 0
        for line in entry.skeleton_lines:
            head, _, body = line.partition(": ")
            kind = head.removeprefix("- ").split("@", 1)[0]
            if kind in {"text", "table"}:
                textish += 2
            elif kind == "command":
                commandish += 2
            elif kind == "bullet" and len(body) >= 48:
                textish += 1
            if len(body) >= 72:
                long_lines += 1
        return (textish + commandish, long_lines, len(entry.skeleton_lines))

    picked: list[PageEntry] = []
    seen_urls: set[str] = set()
    seen_hosts: dict[str, int] = {}
    ranked = sorted(entries, key=lambda item: (quality(item), item.source_log.name), reverse=True)
    for entry in ranked:
        if entry.url in seen_urls:
            continue
        if len(entry.skeleton_lines) < 4:
            continue
        contentish, long_lines, _ = quality(entry)
        if contentish < 3:
            continue
        if long_lines == 0 and not any(line.startswith("- command@") for line in entry.skeleton_lines):
            continue
        host_count = seen_hosts.get(entry.host, 0)
        if host_count >= 2:
            continue
        seen_urls.add(entry.url)
        seen_hosts[entry.host] = host_count + 1
        picked.append(entry)
        if len(picked) >= limit:
            break
    return picked


def _render(entries: list[PageEntry]) -> str:
    lines = ["# Page Skeleton Samples", ""]
    for index, entry in enumerate(entries, start=1):
        lines.append(f"## {index}. {entry.title_core}")
        lines.append(f"- url: {entry.url}")
        lines.append(f"- host: {entry.host}")
        lines.append(f"- mode: {entry.mode}")
        lines.append(f"- source_log: {entry.source_log}")
        lines.append(f"- raw_title: {entry.raw_title or '(none)'}")
        lines.append(f"- title_core: {entry.title_core}")
        lines.append(f"- path_label: {entry.path_label}")
        lines.append(f"- matched_lines: {len(entry.matched_lines)}")
        lines.append("- skeleton:")
        lines.extend(entry.skeleton_lines)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract and clean page skeleton samples from local hyw logs.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("~/.hyw/logs").expanduser(),
        help="Directory containing markdown logs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of samples to emit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown output path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    entries = _iter_page_entries(args.log_dir)
    entries = _pick_entries(entries, limit=max(1, int(args.limit)))
    rendered = _render(entries)
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
