<div align="center">

# HYW — Heuristic Yield Websearch

**An LLM-powered terminal assistant that searches, cross-validates, then answers.**

[![Python](https://img.shields.io/badge/python-≥3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](../LICENSE)

[English](README_en.md) · [中文](README_zh.md)

<!-- TODO: add a terminal screenshot or GIF demo -->
<!-- ![demo](../assets/demo.gif) -->

</div>

---

## Why

LLMs have a knowledge cutoff. Ask "what happened today" and they can only guess.

HYW lets the model decide **what to search, how many rounds, and how to cross-validate** — then gives you the answer. Not a simple "search and paste" — it's a multi-round heuristic search loop.

## Features

- **Multi-round autonomous search** — The model breaks down questions, crafts search queries, and validates results across up to 6 iterations
- **XML tag tool calling** — No function calling dependency; works with any LLM provider
- **Streaming output** — Think and display in real-time; search progress visible as it happens
- **Built-in websearch service** — DuckDuckGo-based search, zero config, no API key required
- **Rich terminal UI** — Gradient titles, Markdown rendering, live spinners
- **Multi-turn conversation** — Context auto-carried; toggle mode with arrow keys
- **Any model via LiteLLM** — OpenAI / Anthropic / Google / OpenRouter / local models

<!-- TODO: add more features you consider important -->

## Quickstart

```bash
# CLI (recommended)
pip install "hyw[cli]"

# CLI + entari plugin support
pip install "hyw[cli,entari]"

# Built-in websearch service only
pip install "hyw[websearch]"

# Interactive mode
hyw

# Single question
hyw -q "What's the latest in tech news?"
```

The `hyw` command is available after installing the `cli` extra.

## Configuration

Config file: `~/.hyw/config.yml`. Use `/config` in interactive mode to edit directly.
In interactive mode, `← / →` switches models, and `↑ / ↓` toggles multi-turn vs new session.
Legacy single-model fields (`model` / `api_key` / `api_base`) still work.

```yaml
# Shared defaults for profiles that don't override them
api_key: sk-or-xxx
api_base: https://openrouter.ai/api/v1

# Active profile for startup / single-shot mode
active_model: gemini-lite

models:
  - name: gemini-lite
    model: openrouter/google/gemini-3.1-flash-lite-preview
  - name: kimi-k2.5
    model: openrouter/moonshotai/kimi-k2.5
  - name: cerebras-gpt-oss
    model: cerebras/gpt-oss-120b
    api_key: csk-xxx
    api_base: https://api.cerebras.ai/v1

# Preferences
language: zh-CN
max_rounds: 6          # max search iterations
headless: true         # headless browser mode

# Custom system prompt (appended)
system_prompt: ""
```

<!-- TODO: add more config options -->

## How It Works

```
User Question
  │
  ▼
┌─────────────────────────────────────┐
│  LLM analyzes & decomposes question │
│  Outputs <search>/<wiki> XML tags   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Concurrent search (up to 4/round)  │
│  DuckDuckGo → Parse → Structured   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  LLM validates results & decides:   │
│  ├─ Insufficient → new queries, go  │
│  └─ Sufficient  → final answer      │
└─────────────────────────────────────┘
```

<!-- TODO: add more technical details or architecture notes -->

## Commands

| Command | Description |
|---------|-------------|
| `/config` | Open config file in editor |
| `/stats` | Show session statistics |
| `/exit` | Exit |
| `←` / `→` | Toggle Multi Turn / New Session mode |

## Project Structure

```
core/
├── main.py                 # Conversation loop, tool calls, LLM interaction
├── cli.py                  # Rich terminal UI, streaming output
├── __main__.py             # python -m core entry point
├── web_search.py           # WebToolSuite + service runtime
└── render.py               # Standalone markdown render service
```

## Requirements

- Python ≥ 3.12
- Base deps: `litellm` · `pyyaml` · `loguru`
- `websearch`: `ddgs` · `markdown` · `Pygments` · `matplotlib` · `weasyprint` · `PyMuPDF` · `Pillow`
- `ddgs`: same as `websearch` (compat alias)
- `cli`: `rich` · `prompt-toolkit` · `ddgs` · `Pygments` · `matplotlib` · `weasyprint` · `PyMuPDF` · `Pillow` · `markdown`
- `entari`: `arclet-alconna` · `arclet-entari` · `ddgs` · `Pygments` · `matplotlib` · `weasyprint` · `PyMuPDF` · `Pillow` · `markdown` · `httpx`

<!-- TODO: add system-level dependencies (e.g. Chrome/Chromium) if needed -->

## Roadmap

<!-- TODO: fill in your development plans -->

- [ ] ...
- [ ] ...
- [ ] ...

## Contributing

<!-- TODO: add contributing guidelines -->

Issues and PRs welcome.

## License

[MIT](../LICENSE)

---

<div align="center">
<sub>Built with curiosity and caffeine.</sub>
</div>
