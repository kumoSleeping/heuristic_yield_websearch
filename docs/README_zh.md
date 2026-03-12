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
- **内置 websearch 服务** — 基于 DuckDuckGo，零配置，无需 API key
- **Rich 终端 UI** — 渐变标题、Markdown 渲染、实时 spinner
- **多轮对话** — 上下文自动传递，左右方向键切换模式
- **通过 LiteLLM 支持任意模型** — OpenAI / Anthropic / Google / OpenRouter / 本地模型

<!-- TODO: 补充你认为重要的特性 -->

## 快速开始

```bash
# CLI（推荐）
pip install "hyw[cli]"

# CLI + entari 插件支持
pip install "hyw[cli,entari]"

# 仅安装内置 websearch 服务
pip install "hyw[websearch]"

# 交互模式
hyw

# 单次提问
hyw -q "最近有什么科技新闻？"
```

安装 `cli` extra 后即可从命令行直接使用 `hyw`。

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
max_rounds: 6          # 最大搜索轮数
headless: true         # 浏览器无头模式

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
| `←` / `→` | 切换 Multi Turn / New Session 模式 |

## 项目结构

```
core/
├── main.py                 # 对话循环、工具调用、LLM 交互
├── cli.py                  # Rich 终端 UI、流式输出
├── __main__.py             # python -m core 入口
├── web_search.py           # WebToolSuite + 服务运行时
└── render.py               # 独立 markdown 渲染服务
```

## 依赖

- Python ≥ 3.12
- 基础依赖：`litellm` · `pyyaml` · `loguru`
- `websearch`：`ddgs` · `markdown` · `Pygments` · `matplotlib` · `weasyprint` · `PyMuPDF` · `Pillow`
- `ddgs`：与 `websearch` 相同（兼容别名）
- `cli`：`rich` · `prompt-toolkit` · `ddgs` · `Pygments` · `matplotlib` · `weasyprint` · `PyMuPDF` · `Pillow` · `markdown`
- `entari`：`arclet-alconna` · `arclet-entari` · `ddgs` · `Pygments` · `matplotlib` · `weasyprint` · `PyMuPDF` · `Pillow` · `markdown` · `httpx`

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
