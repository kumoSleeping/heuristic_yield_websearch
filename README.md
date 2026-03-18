<div align="center">

# HYW — Heuristic Yield Websearch

**An LLM-powered terminal assistant that searches, cross-validates, then answers.**

[![Python](https://img.shields.io/badge/python-≥3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[English](README.md) · [中文](docs/README_zh.md)

<!-- TODO: add a terminal screenshot or GIF demo -->
<!-- ![demo](assets/demo.gif) -->

</div>

---

## Why

LLMs have a knowledge cutoff. Ask "what happened today" and they can only guess.

HYW lets the model decide **what to search, how many rounds, and how to cross-validate** — then gives you the answer. Not a simple "search and paste" — it's a multi-round heuristic search loop.

## Features

- **Multi-round autonomous search** — The model breaks down questions, crafts search queries, and validates results across up to 6 iterations
- **XML tag tool calling** — No function calling dependency; works with any LLM provider
- **Streaming output** — Think and display in real-time; search progress visible as it happens
- **Pluggable tool backends** — Search / page extract / render are selected by capability, not hard-coded per module
- **Built-in websearch service** — Ships with `ddgs`, Jina AI search/page extraction, and non-browser Markdown render
- **Rich terminal UI** — Gradient titles, Markdown rendering, live spinners
- **Multi-turn conversation** — Context auto-carried; toggle mode with arrow keys
- **Any model via LiteLLM** — OpenAI / Anthropic / Google / OpenRouter / local models

<!-- TODO: add more features you consider important -->

## Quickstart

```bash
# Default install: CLI + ddgs + Jina AI + md2png-lite render
pip install hyw

# Add entari plugin support
pip install "hyw[entari]"

# Add Entari + Noto font sync support
pip install "hyw[entari,notosans]"

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
You can also define named transport presets via `model_provider` / `model_providers` for OpenAI-compatible relays.

```yaml
# Shared provider defaults.
# `models[*]` and `sub_agent.*` inherit these unless they override them.
api_key: sk-or-xxx
api_base: https://openrouter.ai/api/v1

# Optional LiteLLM transport preset.
# `requires_openai_auth: true` means "use OPENAI_API_KEY if api_key is omitted".
# model_provider: mirror
# model_providers:
#   mirror:
#     base_url: https://chat.soruxgpt.com/codex
#     wire_api: responses
#     requires_openai_auth: true
#     custom_llm_provider: openai

# Main controller model used at startup / single-shot mode.
# You can set this to either a profile `name` or a raw model id.
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

# Runtime options actually used by the app
language: zh-CN
# Set `false` if the upstream provider has streaming / tool-call compatibility issues.
stream: true
headless: true
# Maximum main-loop rounds. Default is 8.
max_rounds: 8
# Custom system prompt appended to the main controller prompt.
system_prompt: ""

# Child model overrides. Leave empty to inherit the active main model config.
sub_agent:
  websearch:
    model: ""
  page:
    model: ""
  vision:
    model: ""

# Tool capability registry + default provider selection
tools:
  index:
    ddgs:
      search: core.search_ddgs:ddgs_search
    jina_ai:
      search: core.search_jina_ai:jina_ai_search
      page_extract: core.search_jina_ai:jina_ai_page_extract
    md2png_lite:
      render: md2png_lite.provider:render_md2png_lite_result
  config:
    jina_ai:
      page_extract:
        prefer_free: true
  use:
    search: ddgs
    page_extract: jina_ai
    render: md2png_lite

# Legacy stage-specific model slots kept only for compatibility.
# The current main loop does not read them.
stages:
  search:
    model: ""
  fetch:
    model: ""
  summary:
    model: ""
```

What each block does now:

- `api_key` / `api_base`: shared defaults inherited by `models[*]` and `sub_agent.*`.
- `model_provider` / `model_providers`: named transport presets that expand into LiteLLM fields such as `api_base`, `custom_llm_provider`, and `api_key_env`.
- `active_model`: the main controller model currently selected; can match either a profile `name` or a raw model id.
- `models`: switchable main-model profiles for CLI left/right model selection.
- `language` / `stream` / `headless` / `system_prompt`: active runtime options used by the current flow.
- `max_rounds`: maximum main-loop rounds; default is `8`.
- `sub_agent.websearch`: child model that generates 2-6 search terms and runs internal search.
- `sub_agent.page`: child model that compresses pages and extracts evidence.
- `sub_agent.vision`: image understanding helper kept for independent vision flows; not part of the main search loop.
- `tools.index`: capability-to-provider registry.
- `tools.config`: per-provider extra options such as headers or free-route preferences.
- `tools.use`: which provider is selected by default for each capability.
- `stages.*`: legacy stage-specific model slots kept only for old configs; the current main loop does not use them.

## How It Works

```
User Question
  │
  ▼
┌─────────────────────────────────────┐
│  Main model plans the next step     │
│  Outputs <sub_agent ...> XML tags   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  websearch sub-agent builds 2-6     │
│  queries and runs internal search   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Main model chooses concrete pages  │
│  page sub-agent compresses them     │
│  then main model returns the answer │
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
├── web_search.py           # WebToolSuite + service runtime
└── render.py               # md2png-lite render dispatch
```

## Requirements

- Python ≥ 3.12
- Default deps: `litellm` · `pyyaml` · `loguru` · `rich` · `prompt-toolkit` · `ddgs` · `httpx` · `md2png-lite` · `Pillow`
- `entari`: `arclet-alconna` · `arclet-entari` · `md2png-lite`
- `notosans`: `md2png-lite[notosans]`

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

[MIT](LICENSE)

---

<div align="center">
<sub>Built with curiosity and caffeine.</sub>
</div>
