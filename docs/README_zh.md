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
# 默认安装：CLI + ddgs + Jina AI + md2png-lite 渲染
pip install hyw

# 增加 entari 插件支持
pip install "hyw[entari]"

# 增加 entari + Noto 字体同步支持
pip install "hyw[entari,notosans]"

# 交互模式
hyw

# 单次提问
hyw -q "最近有什么科技新闻？"
```

默认安装后即可从命令行直接使用 `hyw`。

## 配置

配置文件位于 `~/.hyw/config.yml`，交互模式下可用 `/config` 命令直接编辑。
仓库根目录提供了一份可直接参考的 `config.example.yml`。
交互模式里空输入时，`←` 用于切换 stage1，`→` 用于切换 stage2，`↑ / ↓` 用于切换多轮/新会话。
旧的单模型写法（`model` / `api_key` / `api_base`）仍然兼容。
也支持通过 `model_provider` / `model_providers` 定义命名 transport 预设，方便接 OpenAI 兼容中转。

```yaml
# 共享 provider 默认值。
# `models[*]` 和 `sub_agent.*` 没有单独覆写时都会继承这里。
api_key: sk-or-xxx
api_base: https://openrouter.ai/api/v1

# 可选的 LiteLLM transport 预设。
# `requires_openai_auth: true` 表示未显式填写 api_key 时，读取 OPENAI_API_KEY。
# model_provider: mirror
# model_providers:
#   mirror:
#     base_url: https://chat.soruxgpt.com/codex
#     wire_api: responses
#     requires_openai_auth: true
#     custom_llm_provider: openai

models:
  - name: gemini-lite
    model: openrouter/google/gemini-3.1-flash-lite-preview
  - name: kimi-k2.5
    model: openrouter/moonshotai/kimi-k2.5
  - name: cerebras-gpt-oss
    model: cerebras/gpt-oss-120b
    api_key: csk-xxx
    api_base: https://api.cerebras.ai/v1

# 当前主流程真实会读取的运行参数
language: zh-CN
# 如果上游 provider 的流式 + tool call 兼容性不好，改成 false。
stream: true
headless: true
# 主循环最大轮次，默认 8。
max_rounds: 8
# 追加到主控制模型系统提示词末尾。
system_prompt: ""

# 历史遗留子代理覆写；主流程不再读取这些位
sub_agent:
  websearch:
    model: ""
  page:
    model: ""

# 工具能力注册表 + 默认 provider 选择
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

# 两阶段主流程默认直接来自 `models` 顺序：
# `models[0]` 作为 stage1；
# 如果有 `models[1]`，则作为 stage2；否则 stage2 回退到 `models[0]`。
```

现在可以直接这样理解这些配置：

- `api_key` / `api_base`：顶层默认值，给 `models[*]` 和 `sub_agent.*` 继承。
- `model_provider` / `model_providers`：命名 transport 预设，会展开成 LiteLLM 能识别的 `api_base`、`custom_llm_provider`、`api_key_env` 等字段。
- `models`：两阶段主模型候选列表。默认 `models[0]` 是 stage1，`models[1]` 是 stage2。
- `language` / `stream` / `headless` / `system_prompt`：当前主流程真实生效。
- `max_rounds`：主循环最大轮次，默认是 `8`。
- CLI 快捷键：空输入时按左键轮换 stage1，按右键轮换 stage2。
- `sub_agent.*`：历史兼容位；当前两阶段主流程不再读取。
- `tools.index`：能力到实现的注册表。
- `tools.config`：某个 provider 的附加配置，比如 header、免费优先策略。
- `tools.use`：每种能力默认选哪个 provider。
- 图片策略：图片只在第一阶段直接发送给主模型；不会再经过独立视觉总结模型。

## 工作原理

```
用户提问
  │
  ▼
┌─────────────────────────────────┐
│  主模型规划下一步动作            │
│  输出 <sub_agent ...> XML 标签   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  websearch 子代理生成 2-6 条搜索词│
│  并在内部完成多次搜索            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  主模型挑选具体页面              │
│  page 子代理负责压缩 / 找线索    │
│  最后由主模型输出答案            │
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
├── web_search.py           # WebToolSuite + 服务运行时
└── render.py               # md2png-lite 渲染分发
```

## 依赖

- Python ≥ 3.12
- 默认依赖：`litellm` · `pyyaml` · `loguru` · `rich` · `prompt-toolkit` · `ddgs` · `httpx` · `md2png-lite` · `Pillow`
- `entari`：`arclet-alconna` · `arclet-entari` · `md2png-lite`
- `notosans`：`md2png-lite[notosans]`

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
