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
- **Capability-based tool registry** — `search / page_extract / render` are selected by capability instead of hard-coded branches
- **Built-in websearch service** — Ships with `ddgs`, Jina AI search/page extraction, and non-browser Markdown render
- **Rich terminal UI** — Gradient titles, Markdown rendering, live spinners
- **Multi-turn conversation** — Context auto-carried; toggle mode with arrow keys
- **Any model via LiteLLM** — OpenAI / Anthropic / Google / OpenRouter / local models

<!-- TODO: add more features you consider important -->

## Quickstart

```bash
# Default install: CLI + ddgs + Jina AI + non-browser render
pip install hyw

# Add entari plugin support
pip install "hyw[entari]"

# Interactive mode
hyw

# Single question
hyw -q "What's the latest in tech news?"
```

The `hyw` command is available in the default install.

## Configuration

Config file: `~/.hyw/config.yml`. Use `/config` in interactive mode to edit directly.
An example based on the multi-model layout lives at `config.example.yml`.
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
max_rounds: 6
headless: true

tools:
  index:
    ddgs:
      search: core.search_ddgs:ddgs_search
    jina_ai:
      search: core.search_jina_ai:jina_ai_search
      page_extract: core.search_jina_ai:jina_ai_page_extract
    render:
      render: core.render_non_browser:render_markdown_non_browser_result
  config:
    jina_ai:
      page_extract:
        prefer_free: true
  use:
    search: ddgs
    page_extract: jina_ai
    render: render

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
| `←` / `→` | Switch active model |
| `↑` / `↓` | Toggle Multi Turn / New Session mode |

## Project Structure

```
core/
├── config.py               # Model config + tool capability registry
├── main.py                 # Conversation loop, tool calls, LLM interaction
├── cli.py                  # Rich terminal UI, streaming output
├── __main__.py             # python -m core entry point
├── search_ddgs.py          # DDGS search provider
├── search_jina_ai.py       # Jina AI search + page extract provider
├── render_non_browser.py   # WeasyPrint-based markdown render provider
├── web_search.py           # WebToolSuite + service runtime
└── render.py               # Standalone markdown render service
```

## Requirements

- Python ≥ 3.12
- Default deps: `litellm` · `pyyaml` · `loguru` · `rich` · `prompt-toolkit` · `ddgs` · `httpx` · `markdown` · `Pygments` · `matplotlib` · `weasyprint` · `PyMuPDF` · `Pillow`
- `entari`: `arclet-alconna` · `arclet-entari`

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
