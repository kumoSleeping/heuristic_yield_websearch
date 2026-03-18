from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from .config import DEFAULT_MODEL


def _completion_api_key(cfg: dict[str, Any]) -> str:
    api_key = str(cfg.get("api_key") or "").strip()
    if api_key.startswith("os.environ/"):
        env_name = api_key.partition("/")[2].strip()
        if env_name:
            try:
                import os

                return str(os.getenv(env_name) or "").strip()
            except Exception:
                return ""
    if api_key:
        return api_key
    env_name = str(cfg.get("api_key_env") or "").strip()
    if env_name:
        try:
            import os

            return str(os.getenv(env_name) or "").strip()
        except Exception:
            return ""
    return ""


def _normalized_reasoning_effort(cfg: dict[str, Any]) -> str:
    value = str(cfg.get("reasoning_effort") or "").strip().lower()
    return value if value in ("minimal", "low", "medium", "high", "xhigh", "none") else ""


def should_use_codex_mirror_transport(cfg: dict[str, Any]) -> bool:
    provider_name = str(cfg.get("model_provider") or "").strip().lower()
    api_base = str(cfg.get("api_base") or "").strip().lower()
    custom_llm_provider = str(cfg.get("custom_llm_provider") or "").strip().lower()
    wire_api = str(cfg.get("wire_api") or "").strip().lower()
    if provider_name == "mirror":
        return True
    return bool(api_base.endswith("/codex") and custom_llm_provider == "openai" and wire_api == "responses")


def _workspace_root() -> Path:
    return Path.cwd().resolve()


def _git_metadata(root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"has_changes": False}
    try:
        origin = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            text=True,
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        ).strip()
        if origin:
            metadata["associated_remote_urls"] = {"origin": origin}
    except Exception:
        pass
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        ).strip()
        if commit:
            metadata["latest_git_commit_hash"] = commit
    except Exception:
        pass
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        ).strip()
        metadata["has_changes"] = bool(status)
    except Exception:
        pass
    return metadata


def _request_headers(*, cfg: dict[str, Any], session_id: str | None = None) -> tuple[str, dict[str, str]]:
    del cfg
    request_id = str(session_id or uuid.uuid4())
    root = _workspace_root()
    metadata = {
        "turn_id": str(uuid.uuid4()),
        "workspaces": {
            str(root): _git_metadata(root),
        },
        "sandbox": "seatbelt",
    }
    headers = {
        "accept": "text/event-stream",
        "content-type": "application/json",
        "user-agent": "Codex Desktop/0.115.0 (Mac OS 26.3.1; arm64) dumb (codex-exec; 0.115.0)",
        "originator": "Codex Desktop",
        "x-client-request-id": request_id,
        "session_id": request_id,
        "x-codex-turn-metadata": json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
    }
    return request_id, headers


def _convert_content(content: Any, role: str) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        text_type = "output_text" if role == "assistant" else "input_text"
        return [{"type": text_type, "text": content}]
    if isinstance(content, list):
        items: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                items.extend(_convert_content(item, role))
                continue
            if not isinstance(item, dict):
                items.extend(_convert_content(str(item), role))
                continue
            item_type = str(item.get("type") or "").strip()
            if item_type == "text":
                items.extend(_convert_content(str(item.get("text") or ""), role))
                continue
            if item_type == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    url = str(image_url.get("url") or "").strip()
                    detail = str(image_url.get("detail") or "").strip()
                else:
                    url = str(image_url or "").strip()
                    detail = ""
                if url:
                    payload: dict[str, Any] = {"type": "input_image", "image_url": url}
                    if detail:
                        payload["detail"] = detail
                    items.append(payload)
                continue
            text_value = str(item.get("text") or "").strip()
            if text_value:
                items.extend(_convert_content(text_value, role))
        return items
    return _convert_content(str(content), role)


def _convert_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or str(tool.get("type") or "").strip() != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        converted_tool: dict[str, Any] = {
            "type": "function",
            "name": name,
            "parameters": function.get("parameters") or {},
        }
        description = str(function.get("description") or "").strip()
        if description:
            converted_tool["description"] = description
        if function.get("strict") is not None:
            converted_tool["strict"] = bool(function.get("strict"))
        converted.append(converted_tool)
    return converted


def _messages_to_input(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    input_items: list[dict[str, Any]] = []
    instructions: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip()
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        if role == "system":
            if isinstance(content, str):
                text = content.strip()
                if text:
                    instructions.append(text)
            else:
                parts = _convert_content(content, "system")
                if parts:
                    input_items.append({"type": "message", "role": "system", "content": parts})
            continue
        if role == "tool":
            output_parts = _convert_content(content, "user")
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(msg.get("tool_call_id") or "").strip(),
                    "output": output_parts,
                }
            )
            continue
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": str(tool_call.get("id") or "").strip(),
                        "name": str(function.get("name") or "").strip(),
                        "arguments": str(function.get("arguments") or ""),
                    }
                )
            continue
        input_items.append(
            {
                "type": "message",
                "role": role or "user",
                "content": _convert_content(content, role or "user"),
            }
        )
    return input_items, "\n\n".join(part for part in instructions if part).strip()


def build_request_body(
    *,
    cfg: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    session_id: str,
    stream: bool,
) -> dict[str, Any]:
    input_items, instructions = _messages_to_input(messages)
    body: dict[str, Any] = {
        "model": str(cfg.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "input": input_items,
        "tools": _convert_tools(tools),
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "store": False,
        "stream": stream,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": session_id,
        "text": {"verbosity": "low"},
    }
    if instructions:
        body["instructions"] = instructions
    effort = _normalized_reasoning_effort(cfg)
    if effort:
        body["reasoning"] = {"effort": effort}
    return body


def _iter_sse_events(response: httpx.Response):
    event_name = ""
    data_parts: list[str] = []
    for raw_line in response.iter_lines():
        line = str(raw_line or "")
        if not line:
            if data_parts:
                yield event_name or "message", "\n".join(data_parts)
            event_name = ""
            data_parts = []
            continue
        if line.startswith("event:"):
            event_name = line.partition(":")[2].strip()
            continue
        if line.startswith("data:"):
            data_parts.append(line.partition(":")[2].strip())
    if data_parts:
        yield event_name or "message", "\n".join(data_parts)


def _error_text(response: httpx.Response) -> str:
    try:
        text = response.read().decode(errors="replace").strip()
    except Exception:
        text = ""
    if text:
        return text
    return f"http {response.status_code}"


def stream_response(
    *,
    cfg: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    on_text_delta: Any | None = None,
) -> dict[str, Any]:
    request_id, headers = _request_headers(cfg=cfg)
    api_key = _completion_api_key(cfg)
    if not api_key:
        raise RuntimeError("mirror provider is missing api_key")
    headers["authorization"] = f"Bearer {api_key}"
    body = build_request_body(
        cfg=cfg,
        messages=messages,
        tools=tools,
        session_id=request_id,
        stream=True,
    )
    url = str(cfg.get("api_base") or "").rstrip("/") + "/responses"
    with httpx.stream("POST", url, headers=headers, json=body, timeout=120) as response:
        final_response: dict[str, Any] | None = None
        if response.status_code >= 400:
            raise RuntimeError(_error_text(response))
        content_type = str(response.headers.get("content-type") or "").lower()
        if "text/event-stream" not in content_type:
            raise RuntimeError(_error_text(response))
        for _, data_text in _iter_sse_events(response):
            try:
                payload = json.loads(data_text)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            payload_type = str(payload.get("type") or "").strip()
            if payload_type == "response.output_text.delta":
                delta = str(payload.get("delta") or "")
                if delta and callable(on_text_delta):
                    on_text_delta(delta)
                continue
            if payload_type == "response.completed":
                final_response = dict(payload.get("response") or {})
        if not isinstance(final_response, dict) or not final_response:
            raise RuntimeError("mirror provider returned no completed response")
        return final_response


def message_payload(response_payload: dict[str, Any]) -> dict[str, Any]:
    content_parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_details: list[dict[str, Any]] = []
    for item in response_payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if str(part.get("type") or "").strip() != "output_text":
                    continue
                content_parts.append({"text": str(part.get("text") or "")})
            continue
        if item_type == "function_call":
            tool_calls.append(
                {
                    "id": str(item.get("call_id") or item.get("id") or "").strip(),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or "").strip(),
                        "arguments": str(item.get("arguments") or ""),
                    },
                }
            )
            continue
        if item_type == "reasoning":
            reasoning_details.append(dict(item))
    payload: dict[str, Any] = {"content": content_parts}
    if tool_calls:
        payload["tool_calls"] = tool_calls
    if reasoning_details:
        payload["reasoning_details"] = reasoning_details
    return payload


def response_text(response_payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for part in message_payload(response_payload).get("content") or []:
        if isinstance(part, dict):
            text = str(part.get("text") or "")
        else:
            text = str(part or "")
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def model_response(response_payload: dict[str, Any]) -> Any:
    choice = SimpleNamespace(message=message_payload(response_payload))
    return SimpleNamespace(
        choices=[choice],
        usage=dict(response_payload.get("usage") or {}),
        model=str(response_payload.get("model") or ""),
        response=response_payload,
    )
