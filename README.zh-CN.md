# Codex LiteLLM Adapter

[![CI](https://github.com/nekopara-ai/codex-litellm-adapter/actions/workflows/ci.yml/badge.svg)](https://github.com/nekopara-ai/codex-litellm-adapter/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

一个连接 Codex 客户端与 LiteLLM 网关的 Python 协议适配器，提供模型目录、工具命名空间转换、HTTP/SSE/WebSocket 支持，以及无状态 Responses 历史恢复。

[English](README.md) · [配置说明](docs/configuration.md) · [架构](docs/architecture.md)

仅修改 API 地址，有时不足以让客户端与网关顺利协作：两边可能使用不同的模型目录格式、工具结构、事件结束标志或历史保存机制。本项目集中处理这些协议差异，认证、模型路由、计费和限流仍由 LiteLLM 负责。

当前为首个 alpha 版本。自动测试覆盖合成协议场景和故障恢复；实际模型兼容性取决于你的网关与后端能力。由 NekoPara AI 维护，与 OpenAI、LiteLLM 官方无隶属关系。

## 功能

- 将已认证的模型列表及元数据转换为 Codex 模型目录，支持模板和 ETag。
- 按配置展开工具命名空间，还原 function/custom 工具调用，修正可选 item ID，同时保持 `call_id` 配对。
- 将 Responses WebSocket 请求桥接到 HTTP SSE 上游；支持多行 SSE，保留 Chat 的 `[DONE]` 结束语义。
- 为 `store=false` 与 `previous_response_id` 恢复完整历史，按认证信息和模型隔离。
- 提供有容量与过期边界的内存/SQLite replay，支持失败写入重试和恢复。
- 使用独立健康检查连接池，保留 URL 编码，处理上游重定向及响应截断错误。

## 快速开始

准备 Python 3.11+，以及已经配置好模型和认证的 LiteLLM 网关。目标模型需要具有可用的 Responses 接口。

```bash
git clone https://github.com/nekopara-ai/codex-litellm-adapter.git
cd codex-litellm-adapter
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
export CODEX_ADAPTER_UPSTREAM=http://127.0.0.1:4000
export CODEX_ADAPTER_REPLAY_DB="$PWD/state/replay.sqlite3"
codex-litellm-adapter
```

默认监听 `127.0.0.1:4001`。不设置 replay 数据库路径时使用内存模式，重启后历史会丢失。配置通过环境变量读取，程序不会自动加载 `.env`。

将以下示例合并到 Codex 配置中，并替换模型名：

```toml
model = "your-gateway-model"
model_provider = "litellm_adapter"

[model_providers.litellm_adapter]
name = "LiteLLM Adapter"
base_url = "http://127.0.0.1:4001/v1"
wire_api = "responses"
env_key = "LITELLM_API_KEY"
```

在启动客户端的环境中设置你自己的 `LITELLM_API_KEY`。适配器透传客户端认证，不内置网关主密钥。配置字段参见 [Codex 官方文档](https://developers.openai.com/codex/config-advanced/)。

使用 Docker：

```bash
cp .env.example .env
# 将 CODEX_ADAPTER_UPSTREAM 改为容器内能访问的网关地址。
docker compose up --build -d
```

Compose 示例仅向宿主机回环地址发布端口，使用非 root 用户和独立状态卷。

## 兼容策略

默认透传 `gpt-*` 模型的原生命名空间，其他模型使用 function 命名空间展开。需要 custom 工具桥接时，显式配置：

```bash
export CODEX_ADAPTER_NS_FORCE_FLATTEN='^your-gateway-model$'
```

custom 桥接保留 custom 工具定义，将历史中的 namespaced custom 调用和输出转换为配对的 function 历史。只接受 function 工具的后端仍需单独验证。reasoning 参数别名、上下文预算重试默认关闭；详见[配置文档](docs/configuration.md)。

仓库不包含生产模型别名、内部地址、用户信息、访问凭证、运行日志或历史数据库。模板也不会改变模型真实能力。

## 开发

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python scripts/check_public_tree.py
python -m build
```

测试只使用本机假上游、合成 Key 和临时数据库。CI 配置覆盖 Python 3.11–3.14、打包与 Docker 构建。

replay 数据包含会话内容，使用压缩存储，**并非加密存储**。请保护状态目录；提交问题时只提供脱敏或合成请求。详细说明见 [SECURITY.md](SECURITY.md)。

许可证：[MIT](LICENSE)。
