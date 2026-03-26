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
- **Built-in retrieval runtime** — Ships with `ddgs`, optional `jina_ddgs` search rendering, Jina AI page extraction, and non-browser Markdown render
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
# `models[*]` inherit these unless they override them.
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
# models:
#   - name: codex-fast
#     model: gpt-5.4
#     model_provider: mirror
#     reasoning_effort: xhigh
#     # `fast` is normalized to `priority` for Codex mirror on wire.
#     service_tier: fast

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
# Optional provider hint, for example Codex mirror can be configured as `fast`
# and will be normalized to the transport value it accepts on wire.
# service_tier: fast
# Per-search-provider timeout before falling back. Default is 4s.
search_handler_timeout_s: 4
# Model-call retries only; search / page_extract tools are not retried here.
model_retries: 2
model_retry_base_delay_s: 1.0
model_retry_max_delay_s: 8.0
# Custom system prompt appended to the main controller prompt.
system_prompt: ""

# Tool capability registry + default provider selection
tools:
  index:
    ddgs:
      search: core.search_ddgs:ddgs_search
    jina_ddgs:
      search: core.search_ddgs:jina_ddgs_search
    jina_ai:
      page_extract: core.search_jina_ai:jina_ai_page_extract
    md2png_lite:
      render: md2png_lite.provider:render_md2png_lite_result
  config:
    jina_ddgs:
      search:
        headers:
          Accept: text/plain
          X-Engine: browser
          X-Return-Format: markdown
    jina_ai:
      page_extract:
        headers:
          # Authorization: Bearer jina_xxx
          Accept: text/plain
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

- `api_key` / `api_base`: shared defaults inherited by `models[*]`.
- `model_provider` / `model_providers`: named transport presets that expand into LiteLLM fields such as `api_base`, `custom_llm_provider`, and `api_key_env`.
- `active_model`: the main controller model currently selected; can match either a profile `name` or a raw model id.
- `models`: switchable main-model profiles for CLI left/right model selection.
- `language` / `stream` / `headless` / `system_prompt`: active runtime options used by the current flow.
- `max_rounds`: maximum main-loop rounds; default is `8`.
- `service_tier`: optional provider hint. For Codex mirror, `fast` is normalized to the accepted wire value `priority`.
- `search_handler_timeout_s`: per-search-provider timeout before fallback; default is `4`.
- `model_retries` / `model_retry_base_delay_s` / `model_retry_max_delay_s`: model-only retry budget and exponential backoff; tool providers are not retried here.
- `tools.index`: capability-to-provider registry.
- `tools.config`: per-provider extra options such as headers.
- `tools.use`: which provider is selected by default for each capability.
- `stages.*`: legacy stage-specific model slots kept only for old configs; the current main loop does not use them.

Runtime context carryover now keeps only `Latest Round Raw`:

- `Latest Round Raw`: the previous round's full raw search/page results stay visible for the next round.

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
┌─────────────────────────────────────┐
│  Main model chooses concrete pages  │
│  page_extract returns numbered lines│
│  and the main model answers         │
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
├── search_ddgs.py          # DDGS + jina_ddgs search providers
├── search_jina_ai.py       # Jina AI page extract provider
├── web_runtime.py          # WebToolSuite + retrieval runtime
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
