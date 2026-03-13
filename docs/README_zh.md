<div align="center">

# HYW — Heuristic Yield Websearch

**让 LLM 主动搜索、交叉验证、再回答的终端 AI 助手**

[![Python](https://img.shields.io/badge/python-≥3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](../LICENSE)

[English](../README.md) · [中文](README_zh.md)

<!-- TODO: 补充一张终端截图或 GIF 演示 -->
<!-- ![demo](../assets/demo.gif) -->

</div>

---

## 为什么做这个

大模型的知识有截止日期。当你问"今天发生了什么"，它只能猜。

HYW 让模型自己决定**搜什么、搜几轮、怎么交叉验证**，然后才给你答案。不是简单的"搜一下贴上去"——是多轮启发式搜索循环。

## 特性

- **多轮自主搜索** — 模型自行拆解问题、构造搜索词、验证结果，最多 6 轮迭代
- **XML 标签工具调用** — 不依赖 function calling，兼容任意 LLM provider
- **流式输出** — 边思考边显示，搜索过程实时可见
- **工具能力可插拔** — 按 `search / page_extract / render` 能力选择 provider，而不是写死在主流程
- **内置 websearch 服务** — 默认提供 `ddgs`、Jina AI 搜索 / 页面提取，以及非浏览器 Markdown 渲染
- **Rich 终端 UI** — 渐变标题、Markdown 渲染、实时 spinner
- **多轮对话** — 上下文自动传递，左右方向键切换模式
- **通过 LiteLLM 支持任意模型** — OpenAI / Anthropic / Google / OpenRouter / 本地模型

<!-- TODO: 补充你认为重要的特性 -->

## 快速开始

```bash
# 默认安装：CLI + ddgs + Jina AI + 非浏览器渲染
pip install hyw

# 增加 entari 插件支持
pip install "hyw[entari]"

# 交互模式
hyw

# 单次提问
hyw -q "最近有什么科技新闻？"
```

默认安装后即可从命令行直接使用 `hyw`。

## 配置

配置文件位于 `~/.hyw/config.yml`，交互模式下可用 `/config` 命令直接编辑。
仓库根目录提供了一份可直接参考的 `config.example.yml`。
交互模式里 `← / →` 用于切换模型，`↑ / ↓` 用于切换多轮/新会话。
旧的单模型写法（`model` / `api_key` / `api_base`）仍然兼容。

```yaml
# 给未覆写的模型复用一套默认 OpenRouter 配置
api_key: sk-or-xxx
api_base: https://openrouter.ai/api/v1

# 启动 / 单次问答默认使用的模型档位
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

# 偏好
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

# 自定义系统提示词 (追加)
system_prompt: ""
```

<!-- TODO: 补充更多配置项说明 -->

## 工作原理

```
用户提问
  │
  ▼
┌─────────────────────────────────┐
│  LLM 分析问题，拆解搜索计划      │
│  输出 <search>/<wiki> XML 标签   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  并发执行搜索（最多 4 个/轮）     │
│  DuckDuckGo → 解析 → 结构化结果  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  LLM 验证结果，决定：            │
│  ├─ 信息不足 → 构造新搜索词，继续 │
│  └─ 信息充分 → 输出最终回答       │
└─────────────────────────────────┘
```

<!-- TODO: 补充技术细节或架构说明 -->

## 命令

| 命令 | 说明 |
|------|------|
| `/config` | 打开配置文件 |
| `/stats` | 显示本次会话统计 |
| `/exit` | 退出 |
| `←` / `→` | 切换当前模型 |
| `↑` / `↓` | 切换 Multi Turn / New Session 模式 |

## 项目结构

```
core/
├── config.py               # 模型配置 + 工具能力索引
├── main.py                 # 对话循环、工具调用、LLM 交互
├── cli.py                  # Rich 终端 UI、流式输出
├── __main__.py             # python -m core 入口
├── search_ddgs.py          # DDGS 搜索 provider
├── search_jina_ai.py       # Jina AI 搜索 / 页面提取 provider
├── render_non_browser.py   # 非浏览器 Markdown 渲染 provider
├── web_search.py           # WebToolSuite + 服务运行时
└── render.py               # 独立 markdown 渲染服务
```

## 依赖

- Python ≥ 3.12
- 默认依赖：`litellm` · `pyyaml` · `loguru` · `rich` · `prompt-toolkit` · `ddgs` · `httpx` · `markdown` · `Pygments` · `matplotlib` · `weasyprint` · `PyMuPDF` · `Pillow`
- `entari`：`arclet-alconna` · `arclet-entari`

<!-- TODO: 补充系统级依赖（如 Chrome/Chromium）说明 -->

## 路线图

<!-- TODO: 补充你的开发计划 -->

- [ ] ...
- [ ] ...
- [ ] ...

## 贡献

<!-- TODO: 补充贡献指南 -->

欢迎提交 Issue 和 PR。

## 许可证

[MIT](../LICENSE)

---

<div align="center">
<sub>Built with curiosity and caffeine.</sub>
</div>
